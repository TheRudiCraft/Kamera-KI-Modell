import math
import sys

import coremltools as ct
import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig
from peft.tuners.tuners_utils import BaseTunerLayer
from transformers import CLIPTextModel, CLIPTokenizer

REPO = "Manojb/stable-diffusion-2-1-base"
TILE = 512
PROMPT = "high quality, highly detailed, sharp focus, clean, natural photo"


def merge_all(model):
    for name, module in list(model.named_modules()):
        if isinstance(module, BaseTunerLayer):
            adapters = list(module.lora_A.keys())
            module.merge(adapter_names=adapters)
            if "." in name:
                parent_name, child = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
            else:
                parent, child = model, name
            setattr(parent, child, module.get_base_layer())
    return model


def load_osediff(unet, vae, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    rank = ckpt["rank_unet"]
    unet.add_adapter(LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=ckpt["unet_lora_encoder_modules"]), adapter_name="default_encoder")
    unet.add_adapter(LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=ckpt["unet_lora_decoder_modules"]), adapter_name="default_decoder")
    unet.add_adapter(LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=ckpt["unet_lora_others_modules"]), adapter_name="default_others")
    loaded = 0
    for n, p in unet.named_parameters():
        if "lora" in n or "conv_in" in n:
            p.data.copy_(ckpt["state_dict_unet"][n])
            loaded += 1
    unet.set_adapter(["default_encoder", "default_decoder", "default_others"])
    vae.add_adapter(LoraConfig(r=ckpt["rank_vae"], init_lora_weights="gaussian", target_modules=ckpt["vae_lora_encoder_modules"]), adapter_name="default_encoder")
    for n, p in vae.named_parameters():
        if "lora" in n:
            p.data.copy_(ckpt["state_dict_vae"][n])
            loaded += 1
    vae.set_adapter(["default_encoder"])
    print("geladene Gewichte:", loaded)
    return merge_all(unet), merge_all(vae)


class Generative(torch.nn.Module):
    def __init__(self, vae, unet, embeds, alpha_bar):
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.register_buffer("embeds", embeds)
        self.register_buffer("timestep", torch.tensor([999], dtype=torch.long))
        self.sa = math.sqrt(alpha_bar)
        self.sb = math.sqrt(1 - alpha_bar)
        self.scale = vae.config.scaling_factor

    def forward(self, image):
        x = image * 2 - 1
        moments = self.vae.quant_conv(self.vae.encoder(x))
        z = moments[:, :4] * self.scale
        eps = self.unet(z, self.timestep, encoder_hidden_states=self.embeds, return_dict=False)[0]
        x0 = (z - self.sb * eps) / self.sa
        decoded = self.vae.decoder(self.vae.post_quant_conv(x0 / self.scale))
        return (decoded.clamp(-1, 1) + 1) * 127.5


def main():
    out_path, ckpt_path = sys.argv[1], sys.argv[2]
    torch.set_grad_enabled(False)
    tokenizer = CLIPTokenizer.from_pretrained(REPO, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(REPO, subfolder="text_encoder")
    ids = tokenizer(PROMPT, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
    embeds = text_encoder(ids)[0]
    del text_encoder
    alpha_bar = float(DDPMScheduler.from_pretrained(REPO, subfolder="scheduler").alphas_cumprod[999])
    vae = AutoencoderKL.from_pretrained(REPO, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(REPO, subfolder="unet")
    unet, vae = load_osediff(unet, vae, ckpt_path)
    model = Generative(vae, unet, embeds, alpha_bar).eval()

    sample = torch.rand(1, 3, TILE, TILE)
    reference = model(sample)
    print("Testlauf PyTorch:", reference.shape, float(reference.min()), float(reference.max()))
    traced = torch.jit.trace(model, sample, check_trace=False)
    del model, unet, vae

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.ImageType(name="image", shape=(1, 3, TILE, TILE), scale=1 / 255.0, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.ImageType(name="output", color_layout=ct.colorlayout.RGB)],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    del traced
    config = ct.optimize.coreml.OptimizationConfig(
        global_config=ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=2048)
    )
    mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config)
    mlmodel.short_description = "OSEDiff one-step generative super resolution, 512 px tiles"
    mlmodel.save(out_path)
    print("gespeichert:", out_path)


if __name__ == "__main__":
    main()
