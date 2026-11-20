"""klein per-sample batching check (2026-11-18): (1) batch 1 of KleinBatch == stock i2i_server.Klein (KLEIN_REF=0) pixel for
pixel, (2) batch-B outputs vs batch-1 outputs (mean abs pixel diff, 0-255; bf16 kernels differ slightly with batch size),
(3) throughput img/s for batch 1/4/8/16 on 64 real v4 avatar prompts (after warm-up). One GPU.
  CUDA_VISIBLE_DEVICES=1 $VENV/bin/python scripts/bench_klein_batch.py [--bs 1,4,8,16] [--n 64]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("KLEIN_REF", "0")
os.environ.setdefault("PROMPTS_FAMILY", "klein_text")
sys.path.insert(0, str(Path(__file__).parent))
from klein_batch import KleinBatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--bs", default="1,4,8,16")
ap.add_argument("--n", type=int, default=64)
a = ap.parse_args()

pr = json.load(open(ROOT / "data/consistency/prompts_shard0.json"))
jobs = []
for k, p in pr.items():
    s, i = k.split("/")
    if s != "v4":
        continue
    src = ROOT / ("data/ffhq_norm" if i.startswith("q") else "data/pairs/cartoon3d/inputs") / f"{i}.png"
    if src.exists():
        jobs.append((Image.open(src).convert("RGB"), p, 1000 + len(jobs)))
jobs = jobs[: a.n]
kb = KleinBatch()
arr = lambda im: np.asarray(im, dtype=np.float32)  # noqa: E731

# (1) batch 1 vs the stock adapter
import i2i_server as S  # noqa: E402
st = S.load_styles()["cartoon3d"]
ref = S.Klein()
from diffusers import Flux2KleinPipeline  # noqa: E402
stock = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16,
                                           cache_dir=os.environ["KLEIN_CACHE"] + "/hub").to("cuda")
stock.set_progress_bar_config(disable=True)
ref.p = stock
d1 = []
for im, p, s in jobs[:4]:
    with torch.inference_mode():
        x = ref.run([im], p, st, [s])[0]
    y = kb.run([im], [p], [s])[0]
    d1.append(float(np.abs(arr(x) - arr(y)).mean()))
print(f"[bench] batch-1 KleinBatch vs stock adapter, mean abs diff (0-255): {np.round(d1, 3).tolist()}", flush=True)
del stock, ref
torch.cuda.empty_cache()

# (2) + (3)
single = [kb.run([im], [p], [s])[0] for im, p, s in jobs[:8]]
for bs in [int(b) for b in a.bs.split(",")]:
    out = kb.run([j[0] for j in jobs[:bs]], [j[1] for j in jobs[:bs]], [j[2] for j in jobs[:bs]])  # warm-up + equivalence
    dd = [float(np.abs(arr(x) - arr(y)).mean()) for x, y in zip(out[:8], single[:min(bs, 8)])]
    torch.cuda.synchronize()
    t = time.time()
    for i in range(0, len(jobs), bs):
        ch = jobs[i:i + bs]
        kb.run([j[0] for j in ch], [j[1] for j in ch], [j[2] for j in ch])
    torch.cuda.synchronize()
    dt = time.time() - t
    print(f"[bench] bs {bs:2d}: {len(jobs) / dt:.2f} img/s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB, "
          f"diff vs bs1 (0-255) mean {np.mean(dd):.2f} max {np.max(dd):.2f}", flush=True)
for k, (x, y) in enumerate(zip(single[:2], out[:2])):
    Image.fromarray(np.concatenate([np.asarray(x), np.asarray(y)], 1)).save(ROOT / f"grids/klein_batch_check_{k}.jpg", quality=90)
print("[bench] BENCH_DONE", flush=True)
