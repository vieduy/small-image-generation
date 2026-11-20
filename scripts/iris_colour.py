"""Iris colour on v2 cartoon3d avatars. Per image: YuNet eye landmarks of the image itself, disc of radius 0.05 x face width
around each eye, pixels with Lab L < 40 (iris + pupil + lash line, not sclera/highlight). Reports the median over images of
  chroma = mean sqrt(a*^2 + b*^2) of those pixels (brown iris high, black iris ~0),  a*, b* means (brown: both > 0).
Calibration (2026-11-17, 150 v2 faces): klein 7.5, SDXS c3_dmd3 6.1, dl1_dmd 3.6 (black irises), b1 14 (blurry brown + skin bleed).
  $VENV/bin/python scripts/iris_colour.py student_c3_dmd3 student_dl1_dmd ...   (klein_personal always included)
Writes results/iris_colour.jsonl. Images: names present for every tag (v2 face_* / fe2_*), every 2nd one."""
import json, sys, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from metrics import FaceID
ROOT = Path(__file__).resolve().parent.parent
det = FaceID().det; det.setInputSize((512, 512))
tags = ["klein_personal"] + sys.argv[1:]
dirs = {t: ROOT / "outputs" / t / "cartoon3d/avatar" for t in tags}
names = sorted(set.intersection(*[{p.name for p in d.glob("f*.png")} for d in dirs.values()]))[::2]
yy, xx = np.mgrid[:512, :512]
with open(ROOT / "results/iris_colour.jsonl", "a") as fo:
    for t in tags:
        ch, aa, bb = [], [], []
        for n in names:
            bgr = cv2.resize(cv2.imread(str(dirs[t] / n)), (512, 512))
            _, f = det.detect(bgr)
            if f is None or len(f) == 0:
                continue
            f = max(f, key=lambda r: r[2] * r[3])
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            L, A, B = lab[..., 0] * 100 / 255, lab[..., 1] - 128, lab[..., 2] - 128
            near = np.zeros((512, 512), bool)
            for k in range(2):
                near |= (xx - f[4 + 2 * k]) ** 2 + (yy - f[5 + 2 * k]) ** 2 < (0.05 * f[2]) ** 2
            m = near & (L < 40)
            if m.sum() > 10:
                ch.append(float(np.hypot(A[m], B[m]).mean())); aa.append(float(A[m].mean())); bb.append(float(B[m].mean()))
        r = {"tag": t, "n_images": len(names), "n_scored": len(ch), "iris_chroma": round(float(np.median(ch)), 2),
             "a": round(float(np.median(aa)), 2), "b": round(float(np.median(bb)), 2), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        print(r, flush=True); fo.write(json.dumps(r) + "\n")
