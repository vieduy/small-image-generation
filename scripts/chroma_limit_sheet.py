"""Visual sheet for choosing the skin colour limit CHROMA_MAX (2026-11-18). Synthetic data v2 pairs that pass likeness +
sticker, grouped by skin saturation change d_chroma (CIELAB chroma of the central face box, output - input). Each row is
one band; each cell = input | output with skin swatches (input left, output right) and the numbers.
  $VENV/bin/python scripts/chroma_limit_sheet.py [--per 6]  -> grids/chroma_limit_sheet.jpg
"""
import argparse, json, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import cv2

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument("--per", type=int, default=6); a = ap.parse_args()
out = ROOT / "data/pairs_v2/cartoon3d"
rows = [json.loads(l) for l in open(out / "manifest.jsonl")]
base = lambda r: r["id_pass"] and (r["task"] == "avatar" or r.get("pass_sticker")) and r["extra_ok"] and r.get("d_chroma") is not None
# medium tones first (where the problem was reported): input chroma in the middle 50%
ci = sorted(r["skin_in"][3] for r in rows if base(r))
lo, hi = ci[len(ci) // 4], ci[3 * len(ci) // 4]
bands = [(-99, 5), (8, 12), (13, 17), (18, 22), (23, 27), (30, 99)]
rnd = random.Random(7)
S = 192
def lab2rgb(l):
    x = np.array([[[l[0] * 255 / 100, l[1] + 128, l[2] + 128]]], np.uint8)
    return tuple(int(v) for v in cv2.cvtColor(x, cv2.COLOR_LAB2RGB)[0, 0])
def src(i):
    return (ROOT / "data/pairs/cartoon3d/inputs" / f"{i}.png")
def outp(r):
    v = r.get("var", 0)
    return out / r["task"] / (f"{r['id']}_b.png" if v == 1 else f"{r['id']}.png")
cells_w, cells_h = 2 * S + 8, S + 40
W = 150 + a.per * cells_w; H = len(bands) * cells_h
sheet = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(sheet)
for bi, (b0, b1) in enumerate(bands):
    pool = [r for r in rows if base(r) and b0 <= r["d_chroma"] <= b1 and lo <= r["skin_in"][3] <= hi and r["task"] in ("avatar", "sticker")]
    n_all = sum(1 for r in rows if base(r) and b0 <= r["d_chroma"] <= b1)
    y = bi * cells_h
    lab = f"{'<= +5' if b0 < 0 else ('>= +30' if b1 > 90 else f'+{b0}..+{b1}')}"
    d.text((8, y + 8), "saturation", fill="black"); d.text((8, y + 24), lab, fill="black"); d.text((8, y + 44), f"{n_all} pairs", fill="gray")
    for k, r in enumerate(rnd.sample(pool, min(a.per, len(pool)))):
        x = 150 + k * cells_w
        try:
            im = Image.open(src(r["id"])).convert("RGB").resize((S, S)); om = Image.open(outp(r)).convert("RGB").resize((S, S))
        except FileNotFoundError:
            continue
        sheet.paste(im, (x, y)); sheet.paste(om, (x + S, y))
        d.rectangle([x + S - 28, y + S - 28, x + S - 1, y + S - 1], fill=lab2rgb(r["skin_in"]), outline="white")
        d.rectangle([x + S, y + S - 28, x + S + 27, y + S - 1], fill=lab2rgb(r["skin_out"]), outline="white")
        d.text((x + 2, y + S + 4), f"{r['id']} {r['task'][:3]}  sat {r['d_chroma']:+.0f}  yellow {r['d_b']:+.0f}  L {r['dL']:+.0f}", fill="black")
(ROOT / "grids").mkdir(exist_ok=True)
sheet.save(ROOT / "grids/chroma_limit_sheet.jpg", quality=88)
print("saved grids/chroma_limit_sheet.jpg", W, H)
