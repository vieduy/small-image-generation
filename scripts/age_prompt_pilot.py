"""Pilot (user request 2026-11-17): stop klein drawing adults as children in gesture stickers (Thank you / Shrug).
Same faces, same seeds as gen_pairs; 3 prompt variants:
  A = current TP.expr_sticker_prompt
  B = + explicit adult age decade, adult head-to-body ratio/shoulders/hands, keep visible signs of age, never a child or baby
  C = B with a chest-up crop instead of "head and upper body"
Out: grids/age_prompt_pilot.jpg + results/age_prompt_pilot.json (SFace cosine input vs output per variant)."""
import json, random, sys, zlib
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).parent))
import gen_pairs as GP  # noqa: E402
import i2i_server as S  # noqa: E402
import metrics as M  # noqa: E402
import teacher_prompt as TP  # noqa: E402
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/pairs/cartoon3d"

rng = random.Random(4242)
fp = [GP.face_prompt(rng) for _ in range(5500)]
old = [i for i in range(2500, 5500) if any(a in fp[i] for a in ("in their fifties", "in their sixties", "in their seventies"))][:8]
EXPR = {e["id"]: e for e in yaml.safe_load((ROOT / "configs/sticker_expressions.yaml").read_text())}
st = S.load_styles()["cartoon3d"]
fc = M.FaceCache(M.FaceID(), D)
man = {}
for l in open(D / "manifest_round3b.jsonl"):
    r = json.loads(l)
    if r["task"] == "avatar": man[r["id"]] = r


def variants(pd, at, phrase):
    a = TP.expr_sticker_prompt(st["description"], pd, phrase)
    who = f"The person is {pd}; the sticker shows exactly this person with the same skin tone, age, gender and hairstyle"
    adult = (f"This is an adult in their {at['age']}: realistic adult head-to-body ratio, adult shoulders and adult hands, "
             f"keep the visible signs of age from the photo (wrinkles, grey hair, facial hair); never draw a child or a baby")
    b = (f"Redraw the person in this photo as a die-cut chat sticker, {st['description']}. {who}, with {phrase}. {adult}. "
         f"Head and upper body, expressive, thick white outline, isolated on a plain flat white background.")
    c = b.replace("Head and upper body, expressive", "Chest-up crop, expressive")
    return {"A": a, "B": b, "C": c}


ad = S.Klein(); ad.load(); ad.p.set_progress_bar_config(disable=True)
fid = M.FaceID()
res, tiles = [], []
for i in old:
    x = "f%05d" % i
    at = TP.parse(fp[i])
    if man.get(x, {}).get("extra_verified") is False:
        at = dict(at, extra=None)
    pd = TP.person_desc(at, fc.get(D / "inputs" / f"{x}.png")[1])
    img = Image.open(D / "inputs" / f"{x}.png").convert("RGB")
    fi = fid.feat(img)
    for g in ("thanks", "shrug"):
        sd = 100_000 + i + (zlib.crc32(f"expr_{g}".encode()) % 997 + 11)
        row = [img]
        for k, p in variants(pd, at, EXPR[g]["phrase"]).items():
            with torch.inference_mode():
                o = ad.run([img], p, st, [sd])[0]
            row.append(o)
            res.append({"id": x, "age": at["age"], "gesture": g, "variant": k, "id_cos": fid.cos(fi, fid.feat(o)), "prompt": p})
        tiles.append((x, at["age"], g, row))
S2 = 200
sheet = Image.new("RGB", (S2 * 4 * 2 + 20, 22 + S2 * ((len(tiles) + 1) // 2)), "white"); d = ImageDraw.Draw(sheet)
for c in range(2):
    for j, lab in enumerate(["photo", "A current", "B adult age", "C B + chest-up"]):
        d.text((c * (4 * S2 + 20) + j * S2 + 4, 5), lab, fill="black")
for k, (x, age, g, row) in enumerate(tiles):
    c, r = divmod(k, (len(tiles) + 1) // 2); x0 = c * (4 * S2 + 20); y = 22 + r * S2
    for j, im in enumerate(row):
        sheet.paste(im.convert("RGB").resize((S2, S2)), (x0 + j * S2, y))
    d.text((x0 + 4, y + 4), f"{x} {age} {g}", fill="white")
sheet.save(ROOT / "grids/age_prompt_pilot.jpg", quality=88)
(ROOT / "results/age_prompt_pilot.json").write_text(json.dumps(res, indent=1))
for k in "ABC":
    v = [r["id_cos"] for r in res if r["variant"] == k and r["id_cos"] is not None]
    print(f"[pilot] {k}: n={len(v)} face-detected, mean SFace cos {np.mean(v):.3f}", flush=True)
print("PILOT_DONE", flush=True)
