"""Skin HUE shift by input skin-tone quartile (2026-11-18): L* says outputs are lighter for everyone, but users see light
(East Asian) faces come out "darker / South Asian". Same face box as metrics.FaceID.skin_L; median a* (red), b* (yellow),
chroma C* and hue angle h (deg; lower = redder/orange, higher = yellower) in vs out, per input L* quartile.
  EVAL_SET=real $VENV/bin/python scripts/skin_hue_by_tone.py [extra input:output pairs...]
"""
import json, os, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
import metrics as M
ROOT = M.ROOT
TAGS = os.environ.get("TAGS", "klein_personal student_c3_dmd3 student_dl1_dmd").split()
fid = M.FaceID()


def lab(img):
    a = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = a.shape[:2]
    fid.det.setInputSize((w, h)); _, fs = fid.det.detect(a)
    if fs is None or len(fs) == 0:
        return None
    x, y, fw, fh = max(fs, key=lambda r: r[2] * r[3])[:4]
    cx, cy = x + fw / 2, y + fh * 0.55
    x0, x1, y0, y1 = int(max(0, cx - .2 * fw)), int(min(w, cx + .2 * fw)), int(max(0, cy - .2 * fh)), int(min(h, cy + .2 * fh))
    if x1 <= x0 or y1 <= y0:
        return None
    L = cv2.cvtColor(a[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    Lm, am, bm = np.median(L[:, 0]) * 100 / 255, np.median(L[:, 1]) - 128, np.median(L[:, 2]) - 128
    return np.array([Lm, am, bm, np.hypot(am, bm), np.degrees(np.arctan2(bm, am))])


def main():
    from eval_quality import eval_items, EVAL_SET
    if len(sys.argv) > 1:  # explicit input:output pairs (e.g. the owner's private selfie; nothing leaves this machine)
        for pr in sys.argv[1:]:
            i, o = pr.split(":")
            a, b = lab(Image.open(i)), lab(Image.open(o))
            print(f"{Path(o).name:28s} in L {a[0]:.1f} a {a[1]:.1f} b {a[2]:.1f} hue {a[4]:.0f} | out L {b[0]:.1f} a {b[1]:.1f} b {b[2]:.1f} hue {b[4]:.0f}")
        return
    faces = [p for p, t in eval_items() if t == "avatar"]
    LI = {p.stem: lab(Image.open(p)) for p in faces}
    ok = sorted([p for p in faces if LI[p.stem] is not None], key=lambda p: LI[p.stem][0])
    q = np.array_split(np.array([p.stem for p in ok]), 4)
    print(f"{EVAL_SET}: input per quartile Q1(dark)..Q4(light): L {[round(float(np.median([LI[s][0] for s in g])), 1) for g in q]}, "
          f"hue {[round(float(np.median([LI[s][4] for s in g]))) for g in q]} deg, chroma {[round(float(np.median([LI[s][3] for s in g])), 1) for g in q]}")
    res = {}
    for t in TAGS:
        d = {}
        for p in ok:
            o = ROOT / "outputs" / t / "cartoon3d/avatar" / p.name
            v = lab(Image.open(o)) if o.exists() else None
            if v is not None:
                d[p.stem] = v - LI[p.stem]
        if len(d) < 20:
            continue
        dh = [round(float(np.median([d[s][4] for s in g if s in d]))) for g in q]
        dc = [round(float(np.median([d[s][3] for s in g if s in d])), 1) for g in q]
        db = [round(float(np.median([d[s][2] for s in g if s in d])), 1) for g in q]
        res[t] = {"d_hue_deg": dh, "d_chroma": dc, "d_b_yellow": db}
        print(f"  {t:22s} change per quartile Q1..Q4: hue {dh} deg | chroma {dc} | b*(yellow) {db}")
    (ROOT / f"results/skin_hue_by_tone_{EVAL_SET}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
