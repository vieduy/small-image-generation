"""Sharpness of the student through a DMD run (2026-11-18): Laplacian variance of the face crop of the val sample tiles
that train_dmd.py saves every val_every steps (left = klein target, right = student, 256 px). No GPU.
  $VENV/bin/python scripts/dmd_sharpness_series.py dl2_dmd probeA_syn [...]"""
import sys
from pathlib import Path
import cv2
import numpy as np
W = Path(__file__).resolve().parent.parent / "weights/student"
for tag in sys.argv[1:]:
    out = []
    for p in sorted((W / tag).glob("samples_[0-9]*.jpg")):
        a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        tiles = [a[i * 256:(i + 1) * 256] for i in range(a.shape[0] // 256)]
        lv = lambda x: cv2.Laplacian(x[48:208, 64:192], cv2.CV_32F).var()
        out.append((int(p.stem.split("_")[1]), np.median([lv(t[:, 256:]) for t in tiles]), np.median([lv(t[:, :256]) for t in tiles])))
    if out:
        print(f"{tag:14s} klein {out[0][2]:5.0f} | student by DMD step: " + "  ".join(f"{s}:{v:.0f}" for s, v, _ in out), flush=True)
