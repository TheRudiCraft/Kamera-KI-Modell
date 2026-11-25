import json
import os
import sys
import time

import coremltools as ct
import numpy as np
from PIL import Image


def image_stats(arr):
    lum = arr.astype(np.float32).mean(axis=2)
    return {
        "mean": float(lum.mean()),
        "p1": float(np.percentile(lum, 1)),
        "p99": float(np.percentile(lum, 99)),
        "extreme_percent": float(((lum <= 2) | (lum >= 253)).mean() * 100),
    }


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float(10 * np.log10(255 * 255 / max(mse, 1e-6)))


def main():
    package, ref_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    source = Image.open(os.path.join(ref_dir, "input.png")).convert("RGB")
    torch_out = np.asarray(Image.open(os.path.join(ref_dir, "torch_out.png")).convert("RGB"))
    report = {"input": image_stats(np.asarray(source)), "torch": image_stats(torch_out)}
    units = {
        "cpu": ct.ComputeUnit.CPU_ONLY,
        "cpu_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_ne": ct.ComputeUnit.CPU_AND_NE,
        "all": ct.ComputeUnit.ALL,
    }
    for name, unit in units.items():
        entry = {}
        try:
            begin = time.time()
            model = ct.models.MLModel(package, compute_units=unit)
            entry["load_seconds"] = round(time.time() - begin, 1)
            begin = time.time()
            result = model.predict({"image": source})
            entry["predict_seconds"] = round(time.time() - begin, 1)
            out = result["output"]
            arr = np.asarray(out.convert("RGB")) if isinstance(out, Image.Image) else np.asarray(out)
            Image.fromarray(arr.astype(np.uint8)).save(os.path.join(out_dir, f"coreml_{name}.png"))
            entry["image"] = image_stats(arr)
            entry["psnr_vs_torch"] = psnr(arr, torch_out)
            del model
        except Exception as error:
            entry["error"] = str(error)
        report[name] = entry
        print(name, json.dumps(entry))
    with open(os.path.join(out_dir, "coreml.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
