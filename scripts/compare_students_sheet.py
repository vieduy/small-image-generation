"""Side-by-side sheet: photo | klein | students, on fixed eval photos (2026-11-18, data v2 review).
  $VENV/bin/python scripts/compare_students_sheet.py --tags klein_personal,student_dl1_dmd,student_dl2 --out grids/x.jpg
Rows: 4 synthetic (v2), 4 real, 4 selfie avatars (fixed, evenly spaced picks), then 3 stickers."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument("--tags", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--per", type=int, default=4); a = ap.parse_args()
tags = a.tags.split(",")
sets = [("synthetic", sorted((ROOT / "data/faces_eval2").glob("*.png"))[:300]),
        ("real photo", sorted((ROOT / "data/real_p2a/eval").glob("*.png"))[:300]),
        ("selfie", sorted((ROOT / "data/faces_selfie").glob("*.png"))[:200])]
rows = []
for name, ps in sets:
    step = len(ps) // a.per
    rows += [(name, p, "avatar") for p in ps[7::step][:a.per]]
rows += [("sticker", sorted((ROOT / "data/faces_selfie").glob("*.png"))[k], "sticker") for k in (3, 41, 77)]
S = 220
im = Image.new("RGB", (S * (1 + len(tags)), 26 + (S + 18) * len(rows)), "white"); d = ImageDraw.Draw(im)
for j, t in enumerate(["photo"] + tags):
    d.text((j * S + 6, 7), t.replace("student_", ""), fill="black")
for i, (name, p, task) in enumerate(rows):
    y = 26 + i * (S + 18)
    im.paste(Image.open(p).convert("RGB").resize((S, S)), (0, y))
    d.text((4, y + S + 3), f"{name} {task} {p.stem}", fill="black")
    for j, t in enumerate(tags):
        q = ROOT / f"outputs/{t}/cartoon3d/{task}/{p.name}"
        if q.exists():
            o = Image.open(q).convert("RGBA").resize((S, S))
            bg = Image.new("RGBA", (S, S), (235, 235, 235, 255)); bg.alpha_composite(o)
            im.paste(bg.convert("RGB"), ((j + 1) * S, y))
        else:
            d.text(((j + 1) * S + 6, y + S // 2), "(no output)", fill="gray")
im.save(ROOT / a.out, quality=88); print("saved", a.out, im.size)
