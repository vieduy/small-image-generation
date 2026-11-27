"""Comic sticker sheet (2026-11-27): same faces x options from outputs/expr_eval_<tag>_decg2/comic_line, one row per model.
  python scripts/s16_sheet.py pk_comic_m dl9_comic_line pkC1_m"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parent.parent
tags = sys.argv[1:]
faces, opts, c = ["face_002", "face_034", "face_053", "face_086"], ["happy", "lol", "cry", "love", "thumbs", "wave", "ok", "sleepy", "night"], 150
g = Image.new("RGB", (150 + len(opts) * c, 20 + len(faces) * (len(tags) + 1) * (c + 2) + 10 * len(faces)), "white")
d = ImageDraw.Draw(g)
d.text((4, 4), "COMIC stickers (decoder decg2): " + ", ".join(opts), fill="black")
y = 20
for f in faces:
    ph = sorted((ROOT / "data/faces").glob(f"{f}.png"))
    if ph:
        g.paste(Image.open(ph[0]).convert("RGB").resize((c, c)), (0, y))
    y += c + 2
    for t in tags:
        d.text((4, y + c // 2), t, fill="black")
        for j, o in enumerate(opts):
            p = ROOT / f"outputs/expr_eval_{t}_decg2/comic_line/{f}_{o}.png"
            if p.exists():
                g.paste(Image.open(p).convert("RGB").resize((c, c)), (150 + j * c, y))
        y += c + 2
    y += 10
g.save(ROOT / "grids/comic_students.jpg", quality=88)
print("[s16] grids/comic_students.jpg", flush=True)
