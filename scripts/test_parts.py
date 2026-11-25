import json
import os
import sys

import coremltools as ct
import numpy as np


def compare(name, package, input_name, source, target):
    entry = {}
    try:
        model = ct.models.MLModel(package, compute_units=ct.ComputeUnit.CPU_ONLY)
        result = model.predict({input_name: source})
        out = np.asarray(list(result.values())[0], dtype=np.float32)
        entry["nan_or_inf"] = int((~np.isfinite(out)).sum())
        finite = np.where(np.isfinite(out), out, 0)
        entry["rel_error"] = float(np.linalg.norm(finite - target) / max(np.linalg.norm(target), 1e-9))
        entry["out_std"] = float(finite.std())
        entry["ref_std"] = float(target.std())
        entry["out_absmax"] = float(np.abs(finite).max())
        entry["ref_absmax"] = float(np.abs(target).max())
    except Exception as error:
        entry["error"] = str(error)
    print(name, json.dumps(entry))
    return entry


def main():
    parts_dir, ref_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    ref = {n: np.load(os.path.join(ref_dir, f"{n}.npy")) for n in ["x", "z", "eps", "x0", "decoded"]}
    report = {
        "encoder": compare("encoder", os.path.join(parts_dir, "enc.mlpackage"), "x", ref["x"], ref["z"]),
        "unet": compare("unet", os.path.join(parts_dir, "unet.mlpackage"), "z", ref["z"], ref["eps"]),
        "decoder": compare("decoder", os.path.join(parts_dir, "dec.mlpackage"), "x0", ref["x0"], ref["decoded"]),
    }
    with open(os.path.join(out_dir, "teile.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
