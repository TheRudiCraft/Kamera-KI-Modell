import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from skimage import data

sys.path.insert(0, os.path.dirname(__file__))
import convert_gen as cg
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


def stats(name, t):
    a = t.detach().float().numpy().ravel()
    info = {
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "absmax": float(np.abs(a).max()),
    }
    print(name, json.dumps(info))
    return info


def image_stats(arr):
    lum = arr.astype(np.float32).mean(axis=2)
    return {
        "mean": float(lum.mean()),
        "p1": float(np.percentile(lum, 1)),
        "p99": float(np.percentile(lum, 99)),
    }


def main():
    ckpt_path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    torch.set_grad_enabled(False)

    hr = Image.fromarray(data.astronaut()).convert("RGB").resize((512, 512), Image.BICUBIC)
    lr = hr.resize((128, 128), Image.BICUBIC).resize((512, 512), Image.BICUBIC)
    lr.save(os.path.join(out_dir, "input.png"))
    hr.save(os.path.join(out_dir, "original.png"))

    tokenizer = CLIPTokenizer.from_pretrained(cg.REPO, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cg.REPO, subfolder="text_encoder")
    ids = tokenizer(cg.PROMPT, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
    embeds = text_encoder(ids)[0]
    del text_encoder
    alpha_bar = float(DDPMScheduler.from_pretrained(cg.REPO, subfolder="scheduler").alphas_cumprod[999])
    vae = AutoencoderKL.from_pretrained(cg.REPO, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(cg.REPO, subfolder="unet")
    unet, vae = cg.load_osediff(unet, vae, ckpt_path)
    model = cg.Generative(vae, unet, embeds, alpha_bar).eval()

    x = torch.from_numpy(np.asarray(lr).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    report = {"alpha_bar": alpha_bar, "sa": model.sa, "sb": model.sb}
    xin = x * 2 - 1
    moments = model.vae.quant_conv(model.vae.encoder(xin))
    z = moments[:, :4] * model.scale
    eps = model.unet(z, model.timestep, encoder_hidden_states=model.embeds, return_dict=False)[0]
    x0 = (z - model.sb * eps) / model.sa
    decoded = model.vae.decoder(model.vae.post_quant_conv(x0 / model.scale))
    report["z"] = stats("z", z)
    report["eps"] = stats("eps", eps)
    report["z_minus_sb_eps"] = stats("z - sb*eps", z - model.sb * eps)
    report["x0"] = stats("x0", x0)
    report["decoded"] = stats("decoded", decoded)

    out = model(x)
    arr = out[0].permute(1, 2, 0).clamp(0, 255).numpy().astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(out_dir, "torch_out.png"))
    report["input_image"] = image_stats(np.asarray(lr))
    report["torch_image"] = image_stats(arr)
    print("Bild rein", json.dumps(report["input_image"]))
    print("Bild raus PyTorch fp32", json.dumps(report["torch_image"]))

    half = model.half()
    try:
        out16 = half(x.half()).float()
        arr16 = out16[0].permute(1, 2, 0).clamp(0, 255).numpy().astype(np.uint8)
        Image.fromarray(arr16).save(os.path.join(out_dir, "torch_fp16_out.png"))
        report["torch_fp16_image"] = image_stats(arr16)
        print("Bild raus PyTorch fp16", json.dumps(report["torch_fp16_image"]))
    except Exception as error:
        report["torch_fp16_error"] = str(error)
        print("fp16 nicht möglich:", error)

    with open(os.path.join(out_dir, "referenz.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
