import gc
import os
import sys

import coremltools as ct
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import convert_adcsr as ca
import safe_fp16
from PIL import Image
from skimage import data


def main():
    adc_dir, out_dir, ref_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)
    torch.set_grad_enabled(False)
    wrapper = ca.build(adc_dir)
    ks = ca.calibrate_norm_scales(wrapper, ca.calibration_samples())
    safe_fp16.make_norms_safe(wrapper, ks)
    for module in wrapper.modules():
        if hasattr(module, "set_processor"):
            module.set_processor(safe_fp16.SafeAttnProcessor())

    body = wrapper.body
    net = body[0]
    stages = [
        ("s1", torch.nn.Sequential(net.body[0], net.body[1])),
        ("s2", net.body[2]),
        ("u0", body[1]),
        ("u1", body[2]),
        ("u2", body[3]),
        ("u3", body[4]),
        ("tail", torch.nn.Sequential(body[5], body[6], body[7])),
    ]

    lr = Image.fromarray(data.astronaut()).convert("RGB").resize((512, 512), Image.BICUBIC).resize((128, 128), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(lr).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    for name, stage in stages:
        stage.eval()
        y = stage(x)
        np.save(os.path.join(ref_dir, f"{name}_in.npy"), x.numpy())
        np.save(os.path.join(ref_dir, f"{name}_out.npy"), y.numpy())
        print(name, tuple(x.shape), "->", tuple(y.shape), "max %.1f" % float(y.abs().max()), flush=True)
        traced = torch.jit.trace(stage, x, check_trace=False)
        for precision, tag in ((ct.precision.FLOAT32, "fp32"), (ct.precision.FLOAT16, "fp16")):
            ml = ct.convert(
                traced,
                convert_to="mlprogram",
                inputs=[ct.TensorType(name="x", shape=tuple(x.shape))],
                minimum_deployment_target=ct.target.iOS18,
                compute_precision=precision,
            )
            ml.save(os.path.join(out_dir, f"{name}_{tag}.mlpackage"))
            del ml
            gc.collect()
        del traced
        x = y


if __name__ == "__main__":
    main()
