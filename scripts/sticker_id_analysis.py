"""Item 3: why do stickers lose identity (student 0.24-0.28 top-5 vs klein 0.40)? CPU-only analysis on v1 eval outputs.
Per output: face box size (fraction of image side), identity rank among the 100 v1 eval inputs (same metric as eval),
and the rank after cropping the face region and upscaling it (does the metric fail because faces are small?).
  source env.sh; CUDA_VISIBLE_DEVICES= $VENV/bin/python scripts/sticker_id_analysis.py
"""
import json, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
import metrics as M  # noqa: E402
ROOT = M.ROOT
fid = M.FaceID()
faces = sorted((ROOT / "data/faces").glob("*.png"))[:100]
F = {p.stem: fid.feat(Image.open(p)) for p in faces}
gal = [s for s in F if F[s] is not None]
G = np.stack([F[s].ravel() / np.linalg.norm(F[s]) for s in gal])


def face_box(img):
    a = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = a.shape[:2]
    fid.det.setInputSize((w, h))
    _, fs = fid.det.detect(a)
    if fs is None or len(fs) == 0:
        return None
    return max(fs, key=lambda r: r[2] * r[3])


def rank_of(feat, stem):
    if feat is None:
        return None
    sims = G @ (feat.ravel() / np.linalg.norm(feat))
    return int((sims > sims[gal.index(stem)]).sum()) + 1


def crop_rank(img, box, stem, scale=2.2):
    x, y, w, h = box[:4]
    cx, cy, s = x + w / 2, y + h / 2, max(w, h) * scale
    c = img.crop((int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2))).resize((512, 512), Image.LANCZOS)
    return rank_of(fid.feat(c), stem)


res = {}
for tag in ("klein_personal", "student_s3d", "student_s5d"):
    for task in ("avatar", "sticker"):
        rows = []
        for p in faces[:50]:
            o = Image.open(ROOT / "outputs" / tag / "cartoon3d" / task / p.name).convert("RGB")
            b = face_box(o)
            r = rank_of(fid.feat(o), p.stem)
            rows.append({"size": None if b is None else float(max(b[2], b[3]) / o.width), "rank": r,
                         "crop_rank": None if b is None else crop_rank(o, b, p.stem)})
        sz = [x["size"] for x in rows if x["size"]]
        t5 = np.mean([x["rank"] is not None and x["rank"] <= 5 for x in rows])
        c5 = np.mean([x["crop_rank"] is not None and x["crop_rank"] <= 5 for x in rows])
        small = [x for x in rows if x["size"] and x["size"] < np.median(sz)]
        big = [x for x in rows if x["size"] and x["size"] >= np.median(sz)]
        res[f"{tag}/{task}"] = {"n": 50, "face_side_frac_median": round(float(np.median(sz)), 3), "top5": round(float(t5), 2),
                                "top5_face_crop": round(float(c5), 2),
                                "top5_small_faces": round(float(np.mean([x["rank"] is not None and x["rank"] <= 5 for x in small])), 2),
                                "top5_big_faces": round(float(np.mean([x["rank"] is not None and x["rank"] <= 5 for x in big])), 2)}
        print(f"{tag}/{task}", res[f"{tag}/{task}"], flush=True)
(ROOT / "results/sticker_id_analysis.json").write_text(json.dumps(res, indent=1))
