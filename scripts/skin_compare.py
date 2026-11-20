"""Skin-tone shift (dL* = L*(out) - L*(in), central face box) per method on the 100 avatar eval faces + side-by-side grid.
2026-11-18: also skin COLOUR per input skin-tone quartile (Q1 darkest .. Q4 lightest): median change of chroma C*
(saturation) and b* (yellowness). Brightness alone missed "East Asian faces come out tanned / South Asian" (owner report):
klein and the students add saturation + yellowness to medium tones (scripts/skin_hue_by_tone.py).
  source env.sh; $VENV/bin/python scripts/skin_compare.py
"""
import json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).parent))
import metrics as M
from skin_hue_by_tone import lab as skin_lab  # [L, a, b, chroma, hue] of the same face box
ROOT = M.ROOT
TAGS = os.environ.get("TAGS", "klein_text klein_personal student_s1 student_s1d student_s1d_sdvae").split()
fid = M.FaceID()
from eval_quality import eval_items, EVAL_SET  # EVAL_SET=v2 -> 400 faces
STYLE = os.environ.get("STYLE", "cartoon3d")
faces = [p for p, t in eval_items() if t == "avatar"]
Lin = {p.stem: fid.skin_L(Image.open(p)) for p in faces}
LABin = {p.stem: skin_lab(Image.open(p)) for p in faces}
order = sorted([p for p in faces if Lin[p.stem] is not None], key=lambda p: Lin[p.stem])
dark = set(p.stem for p in order[:len(order) // 4])  # darkest quartile (25 of 100 in v1)
quart = np.array_split(np.array([p.stem for p in order]), 4)
res = {}
for t in TAGS:
    d, dd, col = [], [], {}
    for p in faces:
        o = ROOT / "outputs" / t / STYLE / "avatar" / p.name
        lo = fid.skin_L(Image.open(o)) if o.exists() else None
        if lo is None or Lin[p.stem] is None: continue
        d.append(lo - Lin[p.stem])
        if p.stem in dark: dd.append(lo - Lin[p.stem])
        vo = skin_lab(Image.open(o))
        if vo is not None and LABin[p.stem] is not None:
            col[p.stem] = vo - LABin[p.stem]
    qmed = lambda k: [round(float(np.median([col[s][k] for s in g if s in col])), 1) if any(s in col for s in g) else None for g in quart]
    res[t] = {"eval_set": EVAL_SET, "n": len(d), "dL_median": round(float(np.median(d)), 1), "absdL_median": round(float(np.median(np.abs(d))), 1),
              "dL_dark25_median": round(float(np.median(dd)), 1),
              "dChroma_Q1_to_Q4": qmed(3), "dYellow_b_Q1_to_Q4": qmed(2)}
    print(t, res[t], flush=True)
(ROOT / f"results/skin_compare{os.environ.get('OUT', '')}.json").write_text(json.dumps(res, indent=1))
sel = order[::max(1, len(order) // 12)][:12]
c = 160
g = Image.new("RGB", ((len(TAGS) + 1) * c, len(sel) * c + 16), "white")
ImageDraw.Draw(g).text((4, 2), "input | " + " | ".join(TAGS) + "  (rows sorted dark->light skin)", fill="black")
for i, p in enumerate(sel):
    g.paste(Image.open(p).convert("RGB").resize((c, c)), (0, 16 + i * c))
    for j, t in enumerate(TAGS):
        g.paste(Image.open(ROOT / "outputs" / t / STYLE / "avatar" / p.name).convert("RGB").resize((c, c)), ((j + 1) * c, 16 + i * c))
g.save(ROOT / f"grids/compare_teacher_student_avatar{os.environ.get('OUT', '')}.jpg", quality=88)
# sticker grid (faces + subjects)
sk = sorted((ROOT / "outputs/klein_personal" / STYLE / "sticker").glob("*.png"))
sk = sk[::max(1, len(sk) // 10)][:10]
g = Image.new("RGB", ((len(TAGS)) * c, len(sk) * c + 16), "white")
ImageDraw.Draw(g).text((4, 2), "input | " + " | ".join(TAGS[1:]) + " (raw sticker outputs)", fill="black")
for i, p in enumerate(sk):
    src = next((ROOT / "data" / d / p.name for d in ("faces", "faces_eval2", "subjects") if (ROOT / "data" / d / p.name).exists()), None)
    if src: g.paste(Image.open(src).convert("RGB").resize((c, c)), (0, 16 + i * c))
    for j, t in enumerate(TAGS[1:]):
        f = ROOT / "outputs" / t / STYLE / "sticker" / p.name
        if f.exists(): g.paste(Image.open(f).convert("RGB").resize((c, c)), ((j + 1) * c, 16 + i * c))
g.save(ROOT / f"grids/compare_teacher_student_sticker{os.environ.get('OUT', '')}.jpg", quality=88)
