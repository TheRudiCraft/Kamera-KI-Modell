import os
import sys
import time

import coremltools as ct
import numpy as np


def main():
    parts_dir, ref_dir = sys.argv[1], sys.argv[2]
    for name in ["s1", "s2", "u0", "u1", "u2", "u3", "tail"]:
        x = np.load(os.path.join(ref_dir, f"{name}_in.npy"))
        expected = np.load(os.path.join(ref_dir, f"{name}_out.npy"))
        for tag in ("fp32", "fp16"):
            path = os.path.join(parts_dir, f"{name}_{tag}.mlpackage")
            if not os.path.exists(path):
                print(name, tag, "fehlt", flush=True)
                continue
            try:
                begin = time.time()
                model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
                out = list(model.predict({"x": x}).values())[0]
                rel = float(np.abs(out - expected).mean() / (np.abs(expected).mean() + 1e-6))
                print("%-5s %s rel.Fehler %.4f  Mittel %.3f gegen %.3f  max %.1f gegen %.1f  endlich %s  %.1f s" % (
                    name, tag, rel, out.mean(), expected.mean(), np.abs(out).max(), np.abs(expected).max(), bool(np.isfinite(out).all()), time.time() - begin), flush=True)
                del model
            except Exception as error:
                print(name, tag, "Fehler", str(error)[:200], flush=True)


if __name__ == "__main__":
    main()
