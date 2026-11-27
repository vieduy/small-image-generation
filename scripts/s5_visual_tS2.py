"""Visual comparison tS2 vs tS1 vs dl6g2 (+ teacher tB1 on real photos), from existing eval outputs (2026-11-25, user "give me the
visual output"). No inference. Out: grids/tS2_compare_real.jpg, grids/tS2_compare_v2.jpg, grids/tS2_compare_expr.jpg"""
import csv
import sys
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
C = 192
MODELS = sys.argv[1:] or ["dl6g2", "tS1", "tS2"]  # 2026-11-25: e.g. dl6g2 tS1 tS3
NEW = MODELS[-1]
TEACHER = "outputs/teacher_real/tB1_s8" if NEW == "tS2" else None  # tB1_s8 = img3/txt4 setting, only matches tS1/tS2


def idcos(t, s):
    return {r["input"]: float(r["id_cos"]) for r in csv.DictReader(open(ROOT / f"results/quality_student_{t}_decg2_{s}.csv"))
            if r["task"] == "avatar" and r["id_cos"]}


def sheet(rows, title, out):
    g = Image.new("RGB", (max(len(r) for r in rows) * C, 22 + len(rows) * (C + 16)), "white")
    d = ImageDraw.Draw(g)
    d.text((4, 5), title, fill="black")
    for i, r in enumerate(rows):
        y = 22 + i * (C + 16)
        for j, (img, cap) in enumerate(r):
            if img is not None and Path(img).exists():
                g.paste(Image.open(img).convert("RGB").resize((C, C)), (j * C, y))
            d.text((j * C + 3, y + C + 2), cap, fill="black")
    g.save(ROOT / out, quality=90)
    print(out)


def avatars(set_, items, title, out, teacher=None):
    ids = {t: idcos(t, set_) for t in MODELS}
    rows = []
    for key, p in items:
        r = [(p, "photo")]
        if teacher:
            r.append((ROOT / teacher / f"{key}.png", "teacher tB1 (8 steps)"))
        for t in MODELS:
            r.append((ROOT / f"outputs/student_{t}_decg2/cartoon3d/avatar/{key}.png", f"{t}   id {ids[t].get(key, float('nan')):.2f}"))
        rows.append(r)
    sheet(rows, title, out)


real = sorted((ROOT / "data/real_p2a/eval").glob("*.png"))[:300]
avatars("real", [(f"rp_{i:03d}", real[i]) for i in range(5, 300, 37)],
        "REAL photos (legal-approved eval set): photo | " + ("teacher tB1 | " if TEACHER else "") + " | ".join(MODELS) + "      id = SFace cosine to the photo", f"grids/{NEW}_compare_real.jpg",
        teacher=TEACHER)
faces = sorted((ROOT / "data/faces").glob("*.png"))[:100]
e2 = sorted((ROOT / "data/faces_eval2").glob("*.png"))[:300]
avatars("v2", [(f"face_{i:03d}", faces[i]) for i in (3, 27, 58, 89)] + [(f"fe2_{i:03d}", e2[i]) for i in (10, 120, 230, 290)],
        "SYNTHETIC faces (v2 eval): photo | " + " | ".join(MODELS) + "      id = SFace cosine", f"grids/{NEW}_compare_v2.jpg")
EX = ["happy", "lol", "cry", "wink", "love", "thumbs", "wave", "thanks", "sleepy"]
rows = []
for f in ("face_002", "face_053", "face_086"):
    rows.append([(ROOT / f"data/faces/{f}.png", f"photo {f}")] + [None] * 0)
    for t in (MODELS[-2], NEW):
        rows.append([(None, t)] + [(ROOT / f"outputs/expr_eval_{t}_decg2/cartoon3d/{f}_{e}.png", f"{t}: {e}") for e in EX])
rows = [r for r in rows if not (len(r) == 1 and r[0][0] is not None and False)]
# put the photo into the first cell of each tS1 row instead of its own row
R2 = []
for i in range(0, len(rows), 3):
    ph, a, b = rows[i], rows[i + 1], rows[i + 2]
    R2.append([ph[0]] + a[1:])
    R2.append([(None, "")] + b[1:])
sheet(R2, f"EXPRESSION stickers: per face, top row {MODELS[-2]}, bottom row {NEW}  (" + ", ".join(EX) + ")", f"grids/{NEW}_compare_expr.jpg")
