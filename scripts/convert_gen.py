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


def reference_input():
    from PIL import Image
    from skimage import data
    hr = Image.fromarray(data.astronaut()).convert("RGB").resize((TILE, TILE), Image.BICUBIC)
    return hr.resize((TILE // 4, TILE // 4), Image.BICUBIC).resize((TILE, TILE), Image.BICUBIC)


def main():
    import os
    from PIL import Image
    import safe_fp16
    out_dir, ckpt_path, ref_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)
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

    lr = reference_input()
    lr.save(os.path.join(ref_dir, "input.png"))
    sample = torch.from_numpy(np.asarray(lr).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    reference = model(sample)
    Image.fromarray(reference[0].permute(1, 2, 0).clamp(0, 255).numpy().astype(np.uint8)).save(os.path.join(ref_dir, "torch_out.png"))

    print("überlaufsichere Normierungen:", safe_fp16.make_norms_safe(model))
    model.unet.set_attn_processor(safe_fp16.SafeAttnProcessor())
    model.vae.set_attn_processor(safe_fp16.SafeAttnProcessor())
    check = model(sample)
    print("Abweichung nach Umbau (0..255):", float((check - reference).abs().mean()))

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
    mlmodel.short_description = "OSEDiff one-step generative super resolution, 512 px tiles, overflow-safe fp16"
    mlmodel.save(os.path.join(out_dir, "fp16.mlpackage"))
    print("fp16 gespeichert")
    config = ct.optimize.coreml.OptimizationConfig(
        global_config=ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=2048)
    )
    quantized = ct.optimize.coreml.linear_quantize_weights(mlmodel, config)
    quantized.short_description = mlmodel.short_description
    quantized.save(os.path.join(out_dir, "int8.mlpackage"))
    print("int8 gespeichert")


if __name__ == "__main__":
    main()
