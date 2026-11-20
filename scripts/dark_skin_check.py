"""Visual check of skin lightening on the REAL eval set (2026-11-18, dl2 skin_compare: darkest-25% inputs dL +32).
Rows = real eval photos sorted by input skin L* (darkest first); columns = input | klein | dl1_dmd | dl2 (+ more tags).
  $VENV/bin/python scripts/dark_skin_check.py [--n 10] [--tags klein_personal,student_dl1_dmd,student_dl2]"""
import argparse, sys
from pathlib import Path
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).parent))
import gen_pairs_v2 as G  # _skin_lab: median CIELAB of the central face box

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=10)
ap.add_argument("--tags", default="klein_personal,student_dl1_dmd,student_dl2"); ap.add_argument("--task", default="avatar")
a = ap.parse_args()
tags = a.tags.split(",")
ins = sorted((ROOT / "data/real_p2a/eval").glob("*.png"))[:300]
lab = [(G._skin_lab(p), p) for p in ins]
lab = sorted([(l, p) for l, p in lab if l is not None], key=lambda x: x[0][0])
pick = lab[:a.n] + lab[len(lab) // 2: len(lab) // 2 + 3]
S = 200
im = Image.new("RGB", (S * (1 + len(tags)), (S + 22) * len(pick) + 24), "white"); d = ImageDraw.Draw(im)
for j, t in enumerate(["input"] + tags):
    d.text((j * S + 4, 4), t, fill="black")
for i, (l, p) in enumerate(pick):
    y = 24 + i * (S + 22)
    im.paste(Image.open(p).convert("RGB").resize((S, S)), (0, y))
    d.text((4, y + S + 4), f"{p.stem} skin L {l[0]:.0f}", fill="black")
    for j, t in enumerate(tags):
        q = ROOT / f"outputs/{t}/cartoon3d/{a.task}/{p.name}"
        if q.exists():
            im.paste(Image.open(q).convert("RGB").resize((S, S)), ((j + 1) * S, y))
            lo = G._skin_lab(q)
            if lo is not None:
                d.text(((j + 1) * S + 4, y + S + 4), f"L {lo[0]:.0f} ({lo[0] - l[0]:+.0f})", fill="black")
out = ROOT / "grids/dark_skin_check_real.jpg"
im.save(out, quality=88); print("saved", out)
