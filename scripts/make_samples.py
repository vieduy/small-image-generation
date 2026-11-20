"""Product-flow sample sheets: user photo -> (1) styled avatar, (2) chat sticker (final post-processed WebP, shown on a
checkerboard so transparency + white outline are visible). Teacher (klein-4B, server) vs small student (0.33B, phone path).
Inputs: synthetic eval selfies (never trained on), chosen to span skin tone / gender / age; plus pet/object photos.
  source env.sh; EVAL_SET=v2 CUDA_VISIBLE_DEVICES= $VENV/bin/python scripts/make_samples.py cartoon3d student_s5d student_s6d
Out: grids/samples_<style>.jpg
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).parent))
import metrics as M  # noqa: E402
from eval_quality import eval_items, checker  # noqa: E402
ROOT = M.ROOT
style, tags = sys.argv[1], ["klein_personal"] + sys.argv[2:]
names = {"klein_personal": "klein-4B (server)"} | {t: t.replace("student_", "small ") for t in tags[1:]}
items = eval_items()
faces = [p for p, t in items if t == "sticker" and p.parent.name.startswith("faces")]
subj = [p for p, t in items if t == "sticker" and p.parent.name == "subjects"]
fid = M.FaceID()
L = {p: fid.skin_L(Image.open(p)) for p in faces}
order = sorted([p for p in faces if L[p] is not None], key=lambda p: L[p])
pick = [order[int(i)] for i in np.linspace(0, len(order) - 1, 8)]          # dark -> light skin spread
pick_s = subj[::max(1, len(subj) // 4)][:4]
c, hdr = 200, 40
cols = ["selfie / photo"] + [f"avatar: {names[t]}" for t in tags] + [f"sticker: {names[t]}" for t in tags]
W, H = len(cols) * c, hdr + (len(pick) + len(pick_s)) * c
g = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(g)
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 13)
except OSError:
    font = ImageFont.load_default()
d.text((6, 4), f"style: {style}  |  synthetic eval inputs (not in training)  |  stickers = final 512px WebP (transparent bg, 8px white outline) on checkerboard",
       fill="black", font=font)
for j, s in enumerate(cols):
    d.text((j * c + 4, 22), s, fill="black", font=font)
ck = checker()


def tile(p):
    if not p.exists():
        return Image.new("RGB", (c, c), (235, 235, 235))
    if p.suffix == ".webp":
        return Image.alpha_composite(ck, Image.open(p).convert("RGBA")).convert("RGB").resize((c, c))
    return Image.open(p).convert("RGB").resize((c, c))


for i, p in enumerate(pick + pick_s):
    y = hdr + i * c
    g.paste(tile(p), (0, y))
    for k, t in enumerate(tags):
        base = ROOT / "outputs" / t / style
        if p.parent.name.startswith("faces"):
            g.paste(tile(base / "avatar" / p.name), ((1 + k) * c, y))
        g.paste(tile(base / "sticker" / f"{p.stem}.webp"), ((1 + len(tags) + k) * c, y))
g.save(ROOT / f"grids/samples_{style}.jpg", quality=90)
print("wrote", ROOT / f"grids/samples_{style}.jpg")
