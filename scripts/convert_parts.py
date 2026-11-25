import gc
import os
import sys

import coremltools as ct
import numpy as np
import torch
from PIL import Image
from skimage import data

sys.path.insert(0, os.path.dirname(__file__))
import convert_gen as cg
import safe_fp16
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


class Encoder(torch.nn.Module):
    def __init__(self, g):
        super().__init__()
        self.g = g

    def forward(self, x):
        moments = self.g.vae.quant_conv(self.g.vae.encoder(x * 2 - 1))
        return moments[:, :4] * self.g.scale


class Denoiser(torch.nn.Module):
    def __init__(self, g):
        super().__init__()
        self.g = g

    def forward(self, z):
        return self.g.unet(z, self.g.timestep, encoder_hidden_states=self.g.embeds, return_dict=False)[0]


class Decoder(torch.nn.Module):
    def __init__(self, g):
        super().__init__()
        self.g = g

    def forward(self, x0):
        return self.g.vae.decoder(self.g.vae.post_quant_conv(x0 / self.g.scale))


def tensor_model(module, sample, name, path):
    traced = torch.jit.trace(module, sample, check_trace=False)
    ml = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name=name, shape=tuple(sample.shape))],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    ml.save(path)
    del traced, ml
    gc.collect()


def image_model(module, sample, path):
    traced = torch.jit.trace(module, sample, check_trace=False)
    ml = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.ImageType(name="image", shape=(1, 3, cg.TILE, cg.TILE), scale=1 / 255.0, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.ImageType(name="output", color_layout=ct.colorlayout.RGB)],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    ml.save(path)
    del traced, ml
    gc.collect()


def main():
    ckpt_path, out_dir, ref_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)
    torch.set_grad_enabled(False)

    hr = Image.fromarray(data.astronaut()).convert("RGB").resize((512, 512), Image.BICUBIC)
    lr = hr.resize((128, 128), Image.BICUBIC).resize((512, 512), Image.BICUBIC)
    lr.save(os.path.join(ref_dir, "input.png"))

    tokenizer = CLIPTokenizer.from_pretrained(cg.REPO, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cg.REPO, subfolder="text_encoder")
    ids = tokenizer(cg.PROMPT, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
    embeds = text_encoder(ids)[0]
    del text_encoder
    alpha_bar = float(DDPMScheduler.from_pretrained(cg.REPO, subfolder="scheduler").alphas_cumprod[999])
    vae = AutoencoderKL.from_pretrained(cg.REPO, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(cg.REPO, subfolder="unet")
    unet, vae = cg.load_osediff(unet, vae, ckpt_path)
    g = cg.Generative(vae, unet, embeds, alpha_bar).eval()

    x = torch.from_numpy(np.asarray(lr).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    z = Encoder(g)(x)
    eps = Denoiser(g)(z)
    x0 = (z - g.sb * eps) / g.sa
    decoded = Decoder(g)(x0)
    for name, value in [("x", x), ("z", z), ("eps", eps), ("x0", x0), ("decoded", decoded)]:
        np.save(os.path.join(ref_dir, f"{name}.npy"), value.numpy())
    out = g(x)
    Image.fromarray(out[0].permute(1, 2, 0).clamp(0, 255).numpy().astype(np.uint8)).save(os.path.join(ref_dir, "torch_out.png"))
    print("Referenz fertig")

    tensor_model(Encoder(g).eval(), x, "x", os.path.join(out_dir, "enc.mlpackage"))
    print("Encoder fertig")
    tensor_model(Decoder(g).eval(), x0, "x0", os.path.join(out_dir, "dec.mlpackage"))
    print("Decoder fertig")
    tensor_model(Denoiser(g).eval(), z, "z", os.path.join(out_dir, "unet.mlpackage"))
    print("UNet fertig")

    count = safe_fp16.make_norms_safe(g)
    print("sichere Normierungen:", count)
    check = g(x)
    print("PyTorch nach Umbau, Abweichung:", float((check - out).abs().mean()))
    image_model(g, x, os.path.join(out_dir, "safe.mlpackage"))
    print("safe fertig")

    g.unet.set_attn_processor(safe_fp16.SafeAttnProcessor())
    g.vae.set_attn_processor(safe_fp16.SafeAttnProcessor())
    check = g(x)
    print("PyTorch mit eigener Attention, Abweichung:", float((check - out).abs().mean()))
    image_model(g, x, os.path.join(out_dir, "safeattn.mlpackage"))
    print("safeattn fertig")


if __name__ == "__main__":
    main()
