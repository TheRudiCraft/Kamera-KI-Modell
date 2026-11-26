import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import data

sys.path.insert(0, os.path.dirname(__file__))
import convert_adcsr as ca
import safe_fp16


def collect(model, sample, dtype):
    records = []
    hooks = []

    def make(name):
        def hook(module, inputs, output):
            if not isinstance(output, torch.Tensor):
                return
            out = output.detach()
            small = out.float()
            if small.dim() == 4:
                small = F.adaptive_avg_pool2d(small, 8)
            elif small.dim() == 3:
                small = small.mean(dim=1, keepdim=True)
            records.append((name, type(module).__name__, small.cpu().numpy().copy(),
                            float(out.float().abs().max()), bool(torch.isfinite(out.float()).all())))
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, safe_fp16.SafeGroupNorm, safe_fp16.SafeLayerNorm)) or hasattr(module, "set_processor"):
            hooks.append(module.register_forward_hook(make(name)))
    with torch.no_grad():
        out = model(sample.to(dtype))
    for h in hooks:
        h.remove()
    return out.float(), records


def main():
    adc_dir, ref_dir = sys.argv[1], sys.argv[2]
    os.makedirs(ref_dir, exist_ok=True)
    torch.set_grad_enabled(False)
    model = ca.build(adc_dir)
    lr = Image.open(os.path.join(ref_dir, "input.png")).convert("RGB") if os.path.exists(os.path.join(ref_dir, "input.png")) else \
        Image.fromarray(data.astronaut()).convert("RGB").resize((512, 512), Image.BICUBIC).resize((128, 128), Image.BICUBIC)
    sample = torch.from_numpy(np.asarray(lr).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    reference = model(sample)
    print("Original: Mittel %.1f, Spanne %.1f bis %.1f" % (reference.mean(), reference.min(), reference.max()))

    safe_fp16.make_norms_safe(model)
    for module in model.modules():
        if hasattr(module, "set_processor"):
            module.set_processor(safe_fp16.SafeAttnProcessor())

    out32, rec32 = collect(model, sample, torch.float32)
    print("Sicher fp32: Abweichung zum Original %.2f (0..255), Mittel %.1f" % (float((out32 - reference).abs().mean()), out32.mean()))
    for i, (n, t, a, m, f) in enumerate(rec32):
        pass

    top = sorted(rec32, key=lambda r: -r[3])[:12]
    print("Größte Werte in fp32 (Schicht, Typ, max):")
    for n, t, a, m, f in top:
        print("  %-60s %-16s %.1f" % (n, t, m))

    sim_log = []

    def make_round(name):
        def hook(module, inputs, output):
            if not isinstance(output, torch.Tensor):
                return output
            big = float(output.abs().max())
            half = output.half()
            bad = not bool(torch.isfinite(half).all())
            sim_log.append((name, type(module).__name__, big, bad))
            return half.float()
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, safe_fp16.SafeGroupNorm, safe_fp16.SafeLayerNorm)) or hasattr(module, "set_processor"):
            hooks.append(module.register_forward_hook(make_round(name)))
    with torch.no_grad():
        out16 = model(sample)
    for h in hooks:
        h.remove()
    print("fp16-Rundung simuliert: Abweichung zu fp32 %.2f, Mittel %.1f, endlich %s" % (float((out16 - out32).abs().mean()), out16.mean(), bool(torch.isfinite(out16).all())), flush=True)
    bad = [r for r in sim_log if r[3]]
    print("Schichten mit Überlauf (inf):", len(bad), flush=True)
    for n, t, m, b in bad[:10]:
        print("  %-60s %-16s max %.0f" % (n, t, m), flush=True)
    print("Größte Werte insgesamt:", flush=True)
    for n, t, m, b in sorted(sim_log, key=lambda r: -r[2])[:8]:
        print("  %-60s %-16s max %.0f" % (n, t, m), flush=True)


if __name__ == "__main__":
    main()
