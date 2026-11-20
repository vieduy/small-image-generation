"""2x zoom of the face centre (256x256 crop of the 512 output -> 512) to compare fine detail (2026-11-18).
  $VENV/bin/python scripts/zoom_detail_sheet.py --tags klein_personal,student_dl1_dmd,student_dl2_dmd --names fe2_157,rp_232,fs_107 --out grids/x.jpg"""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument("--tags", required=True); ap.add_argument("--names", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--task", default="avatar"); a = ap.parse_args()
tags, names = a.tags.split(","), a.names.split(",")
Z = 400
im = Image.new("RGB", (Z * len(tags), (Z + 20) * len(names) + 22), "white"); d = ImageDraw.Draw(im)
for j, t in enumerate(tags):
    d.text((j * Z + 6, 6), t.replace("student_", ""), fill="black")
for i, n in enumerate(names):
    y = 22 + i * (Z + 20)
    for j, t in enumerate(tags):
        p = ROOT / f"outputs/{t}/cartoon3d/{a.task}/{n}.png"
        if p.exists():
            o = Image.open(p).convert("RGB").resize((512, 512))
            im.paste(o.crop((128, 96, 384, 352)).resize((Z, Z), Image.NEAREST), (j * Z, y))
    d.text((6, y + Z + 3), n, fill="black")
im.save(ROOT / a.out, quality=92); print("saved", a.out)
