import copy
import gc
import os
import sys

import coremltools as ct
import numpy as np
import torch
from diffusers import UNet2DConditionModel
from diffusers.models.autoencoders.vae import Decoder
from huggingface_hub import hf_hub_download
from PIL import Image
from skimage import data

REPO = "Manojb/stable-diffusion-2-1-base"
ADC_REPO = "Guaishou74851/AdcSR"
LR = 128
SCALE = 4


class Wrapper(torch.nn.Module):
    def __init__(self, body):
        super().__init__()
        self.body = body

    def forward(self, image):
        x = image * 2 - 1
        y = self.body(x)
        return (y.clamp(-1, 1) + 1) * 127.5


def build(adc_dir):
    sys.path.insert(0, adc_dir)
    from model import Net

    unet = UNet2DConditionModel.from_pretrained(REPO, subfolder="unet")
    decoder = Decoder(
        in_channels=4,
        out_channels=3,
        up_block_types=["UpDecoderBlock2D" for _ in range(4)],
        block_out_channels=[64, 128, 256, 256],
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        norm_type="group",
        mid_block_add_attention=True,
    )
    half = hf_hub_download(ADC_REPO, "weight/pretrained/halfDecoder.ckpt")
    state = torch.load(half, map_location="cpu", weights_only=False)["state_dict"]
    decoder.load_state_dict({k.replace("decoder.", ""): v for k, v in state.items() if "decoder" in k}, strict=True)

    net = Net(unet, copy.deepcopy(decoder))
    weights = hf_hub_download(ADC_REPO, "weight/net_params_200.pkl")
    state = torch.load(weights, map_location="cpu", weights_only=False)
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    print("Gewichte laden:", net.load_state_dict(state, strict=True))
    del state

    body = torch.nn.Sequential(net, *decoder.up_blocks, decoder.conv_norm_out, decoder.conv_act, decoder.conv_out)
    return Wrapper(body).eval()


def main():
    import safe_fp16

    adc_dir, out_dir, ref_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)
    torch.set_grad_enabled(False)

    model = build(adc_dir)
    print("Parameter (Mio):", round(sum(p.numel() for p in model.parameters()) / 1e6, 1))

    hr = Image.fromarray(data.astronaut()).convert("RGB").resize((LR * SCALE, LR * SCALE), Image.BICUBIC)
    lr = hr.resize((LR, LR), Image.BICUBIC)
    hr.save(os.path.join(ref_dir, "hr.png"))
    lr.save(os.path.join(ref_dir, "input.png"))
    sample = torch.from_numpy(np.asarray(lr).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    reference = model(sample)
    print("Ausgabeform:", tuple(reference.shape))
    Image.fromarray(reference[0].permute(1, 2, 0).clamp(0, 255).numpy().astype(np.uint8)).save(os.path.join(ref_dir, "torch_out.png"))

    print("überlaufsichere Normierungen:", safe_fp16.make_norms_safe(model))
    attention = 0
    for module in model.modules():
        if hasattr(module, "set_processor"):
            module.set_processor(safe_fp16.SafeAttnProcessor())
            attention += 1
    print("Attention-Schichten umgestellt:", attention)
    check = model(sample)
    print("Abweichung nach Umbau (0..255):", float((check - reference).abs().mean()))

    traced = torch.jit.trace(model, sample, check_trace=False)
    del model
    gc.collect()

    ml = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.ImageType(name="image", shape=(1, 3, LR, LR), scale=1 / 255.0, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.ImageType(name="output", color_layout=ct.colorlayout.RGB)],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    del traced
    gc.collect()
    ml.short_description = "AdcSR one-step real-world super resolution, 128 px in, 512 px out, overflow-safe fp16"
    ml.save(os.path.join(out_dir, "fp16.mlpackage"))
    print("fp16 gespeichert")

    config = ct.optimize.coreml.OptimizationConfig(
        global_config=ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=2048)
    )
    quantized = ct.optimize.coreml.linear_quantize_weights(ml, config)
    quantized.short_description = ml.short_description
    quantized.save(os.path.join(out_dir, "int8.mlpackage"))
    print("int8 gespeichert")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    main()
