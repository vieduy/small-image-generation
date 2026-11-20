"""Smoke test + sample grid for one model adapter (run once per model, separate process).
Rows: 4 faces as avatar, 2 faces + 2 subjects as sticker; cols: styles. Fixed seeds. Writes grids/smoke_<model>.jpg
and appends a line to results/smoke.jsonl (load time, single-image latency median of 5, peak VRAM).

  MODEL=sdturbo $VENV/bin/python scripts/smoke.py
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import i2i_server as S  # noqa: E402

ROOT = S.ROOT
M = os.environ["MODEL"]
faces = sorted((ROOT / "data/faces").glob("*.png"))
subj = sorted((ROOT / "data/subjects").glob("*.png"))
rows = [(faces[i], "avatar") for i in (0, 17, 42, 77)] + [(faces[5], "sticker"), (faces[60], "sticker"),
                                                         (subj[0], "sticker"), (subj[13], "sticker")]
styles = S.load_styles()
ad = S.ADAPTERS[M]()
t = time.perf_counter()
ad.load()
ad.p.set_progress_bar_config(disable=True)
if isinstance(ad, S.SanaSprint):
    for st in styles.values():
        for task in ("avatar", "sticker"):
            ad.embed(st[f"{task}_text"])
    ad.free_text_encoder()
load_s = time.perf_counter() - t
torch.cuda.reset_peak_memory_stats()

c = 192
grid = Image.new("RGB", ((len(styles) + 1) * c, len(rows) * c + 20), "white")
d = ImageDraw.Draw(grid)
for j, n in enumerate(["input"] + list(styles)):
    d.text((j * c + 4, 4), n, fill="black")
lat = []
with torch.inference_mode():
    for i, (p, task) in enumerate(rows):
        img = Image.open(p).convert("RGB").resize((S.SIZE, S.SIZE))
        grid.paste(img.resize((c, c)), (0, 20 + i * c))
        for j, st in enumerate(styles.values()):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = ad.run([img], st[f"{task}_text"], st, [1000 + i])[0]
            torch.cuda.synchronize()
            lat.append(time.perf_counter() - t)
            grid.paste(out.resize((c, c)), ((j + 1) * c, 20 + i * c))
        d.text((4, 20 + i * c + 4), task, fill="red")
grid.save(ROOT / f"grids/smoke_{os.environ.get("TAG", M)}.jpg", quality=90)
rec = {"model": M, "tag": os.environ.get("TAG", M), "env": {k: os.environ[k] for k in ("STEPS", "STRENGTH", "KLEIN_REF", "PROMPTS_FAMILY", "IPA_SCALE") if k in os.environ}, "steps": ad.steps, "load_s": round(load_s, 1), "lat_med_s": round(statistics.median(lat[3:]), 4),
       "lat_first_s": round(lat[0], 3), "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
       "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
print(json.dumps(rec), flush=True)
(ROOT / "results").mkdir(exist_ok=True)
with open(ROOT / "results/smoke.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
