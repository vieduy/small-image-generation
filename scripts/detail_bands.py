"""Fine-detail energy of student outputs relative to the klein teacher (v2 cartoon3d avatars, face region 320x256 of 512).

  $VENV/bin/python scripts/detail_bands.py student_c3_dmd3 student_dl1_dmd ... [--style cartoon3d] [--eval-set v2]
Bands are radial FFT rings (fraction of Nyquist): coarse 0.05-0.15, mid 0.15-0.3, fine 0.3-0.6, finest 0.6-1.0.
Output: printed table + results/detail_bands.jsonl (one row per tag). 1.00 = as much energy as klein in that band.
Note: energy above klein in the finest band usually means noise/artifacts, not detail (dl1: 1.45, scratchy streaks).
"""
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RINGS = ((0.05, 0.15), (0.15, 0.3), (0.3, 0.6), (0.6, 1.0))


def bands(g):
    F = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    h, w = g.shape
    y, x = np.ogrid[:h, :w]
    r = np.hypot(y - h / 2, x - w / 2) / (h / 2)
    return [F[(r >= a) & (r < b)].sum() for a, b in RINGS]


def face(p):
    return cv2.resize(cv2.imread(str(p), 0), (512, 512))[96:416, 128:384].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--style", default="cartoon3d")
    ap.add_argument("--eval-set", default="v2")
    a = ap.parse_args()
    od = ROOT / "outputs"
    sub = Path(a.style) / "avatar"
    dirs = {t: od / t / sub for t in ["klein_personal"] + a.tags}
    names = sorted(set.intersection(*[set(os.listdir(d)) for d in dirs.values()]))
    k = np.mean([bands(face(dirs["klein_personal"] / n)) for n in names], 0)
    print(f"n={len(names)} {a.style} avatars; energy vs klein: coarse  mid  fine  finest | lapvar")
    with open(ROOT / "results/detail_bands.jsonl", "a") as f:
        for t in a.tags:
            B = [bands(face(dirs[t] / n)) for n in names]
            L = [cv2.Laplacian(face(dirs[t] / n).astype(np.uint8), cv2.CV_64F).var() for n in names]
            r = (np.mean(B, 0) / k).round(3).tolist()
            print(f"{t:28s} " + "  ".join(f"{x:.2f}" for x in r) + f" | {np.median(L):.0f}")
            f.write(json.dumps({"tag": t, "style": a.style, "eval_set": a.eval_set, "n": len(names), "bands_vs_klein": r,
                                "lapvar_median": round(float(np.median(L)), 1), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")


if __name__ == "__main__":
    main()
