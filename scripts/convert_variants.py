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
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


def convert(traced, precision):
    return ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.ImageType(name="image", shape=(1, 3, cg.TILE, cg.TILE), scale=1 / 255.0, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.ImageType(name="output", color_layout=ct.colorlayout.RGB)],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=precision,
    )


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
    model = cg.Generative(vae, unet, embeds, alpha_bar).eval()

    x = torch.from_numpy(np.asarray(lr).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    out = model(x)
    arr = out[0].permute(1, 2, 0).clamp(0, 255).numpy().astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(ref_dir, "torch_out.png"))
    print("Referenz fertig")

    traced = torch.jit.trace(model, x, check_trace=False)
    del model, unet, vae
    gc.collect()

    ml = convert(traced, ct.precision.FLOAT32)
    ml.save(os.path.join(out_dir, "fp32.mlpackage"))
    print("fp32 gespeichert")
    del ml
    gc.collect()

    ml = convert(traced, ct.precision.FLOAT16)
    del traced
    gc.collect()
    ml.save(os.path.join(out_dir, "fp16.mlpackage"))
    print("fp16 gespeichert")

    config = ct.optimize.coreml.OptimizationConfig(
        global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric", granularity="per_block", block_size=32, weight_threshold=2048
        )
    )
    q = ct.optimize.coreml.linear_quantize_weights(ml, config)
    q.save(os.path.join(out_dir, "int8block.mlpackage"))
    print("int8block gespeichert")


if __name__ == "__main__":
    main()
