"""Prompt/style-description A/B for the klein teacher on the hardest eval faces (darkest skin + random), cartoon avatar.
Metrics: dL* = skin lightness(out) - skin lightness(in) (0 is ideal), identity retrieval top-1/top-5 among 100 eval inputs.

  source env.sh; $VENV/bin/python scripts/prompt_ab.py
"""
import json
import zlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

os.environ["MODEL"] = "klein"
os.environ["KLEIN_REF"] = "0"
sys.path.insert(0, str(Path(__file__).parent))
import metrics as M  # noqa: E402

ROOT = M.ROOT
VARIANTS = [("A_v1prompt_cartoon3d", "klein_text", "cartoon3d"), ("B_v2prompt_cartoon3d", "klein_text_v2", "cartoon3d"),
            ("C_v2prompt_cartoon3d_b", "klein_text_v2", "cartoon3d_b"), ("D_personal_cartoon3d", "personal", "cartoon3d")]
import teacher_prompt as TP  # noqa: E402

fid = M.FaceID()
faces = sorted((ROOT / "data/faces").glob("*.png"))[:100]
L = {p.stem: fid.skin_L(Image.open(p)) for p in faces}
feat = {p.stem: fid.feat(Image.open(p)) for p in faces}
gal = [p.stem for p in faces if feat[p.stem] is not None]
G = np.stack([feat[s].ravel() / np.linalg.norm(feat[s]) for s in gal])
dark = sorted([p for p in faces if L[p.stem] is not None], key=lambda p: L[p.stem])[:24]
rest = [p for p in faces if p not in dark][::10][:8]
sel = dark + rest

import yaml  # noqa: E402
import i2i_server as S  # noqa: E402

P = yaml.safe_load((ROOT / "configs/prompts.yaml").read_text())
ad = S.Klein()
ad.load()
ad.p.set_progress_bar_config(disable=True)
res, outs = {}, {}
for name, fam, style in VARIANTS:
    st = yaml.safe_load((ROOT / "styles" / style / "style.yaml").read_text())
    prompt = None if fam == "personal" else P[fam]["avatar"].format(desc=st["description"], short=st.get("short", ""))
    man = {__import__("json").loads(l)["file"]: __import__("json").loads(l)["prompt"] for l in open(ROOT / "data/faces/manifest.jsonl")}
    od = ROOT / "outputs/prompt_ab" / name
    od.mkdir(parents=True, exist_ok=True)
    dl, ranks = [], []
    for p in sel:
        pr = prompt or TP.avatar_prompt(st["description"], TP.person_desc(TP.parse(man[p.name]), L[p.stem]))
        if (od / f"{p.stem}.png").exists():
            o = Image.open(od / f"{p.stem}.png").convert("RGB")
        else:
            with torch.inference_mode():
                o = ad.run([Image.open(p).convert("RGB")], pr, {}, [zlib.crc32(f"{p.stem}|cartoon3d|avatar".encode())])[0]
            o.save(od / f"{p.stem}.png")
        lo = fid.skin_L(o)
        if lo is not None and L[p.stem] is not None:
            dl.append(lo - L[p.stem])
        fo = fid.feat(o)
        if fo is None:
            ranks.append(None)
        else:
            sims = G @ (fo.ravel() / np.linalg.norm(fo))
            ranks.append(int((sims > sims[gal.index(p.stem)]).sum()) + 1)
    ok = [r for r in ranks if r is not None]
    res[name] = {"n": len(sel), "dL_median": round(float(np.median(dl)), 1), "dL_mean": round(float(np.mean(dl)), 1),
                 "dL_dark24_median": round(float(np.median(dl[:24])), 1), "abs_dL_median": round(float(np.median(np.abs(dl))), 1),
                 "id_top1": round(sum(r == 1 for r in ok) / len(sel), 3), "id_top5": round(sum(r <= 5 for r in ok) / len(sel), 3),
                 "face_det": round(len(ok) / len(sel), 3)}
    print(name, json.dumps(res[name]), flush=True)
c = 160
g = Image.new("RGB", ((len(VARIANTS) + 1) * c, 12 * c + 16), "white")
ImageDraw.Draw(g).text((4, 2), "input | " + " | ".join(v[0] for v in VARIANTS), fill="black")
for i, p in enumerate(sel[:12]):
    g.paste(Image.open(p).convert("RGB").resize((c, c)), (0, 16 + i * c))
    for j, (name, _, _) in enumerate(VARIANTS):
        g.paste(Image.open(ROOT / "outputs/prompt_ab" / name / f"{p.stem}.png").convert("RGB").resize((c, c)), ((j + 1) * c, 16 + i * c))
g.save(ROOT / "grids/prompt_ab_skin.jpg", quality=88)
(ROOT / "results/prompt_ab.json").write_text(json.dumps(res, indent=1))
