"""Plan B2 (2026-11-20): encode the training pairs with Sana-Sprint's DC-AE (full encoder) into its latent space (x scaling_factor).
Same rows/keys/order as train_dl.build_cache_dl (manifest kept rows of the given tasks): input photo + target, plain and
horizontally flipped (pixel-space flip, then encode = "proper flip"). fp16 cache, ~16 KB per latent.
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/sana_encode.py --part 0 --nparts 2
  (then --merge) -> data/pairs/<style>/sana_lat_<manifest stem>.pt {key, ids, task, tasks, zi, zt, zi_f, zt_f}
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import sana_common as SC  # noqa: E402
from train_student import TASKS, expr_tasks  # noqa: E402

ROOT = SC.ROOT


def rows_of(style, manifest, tasks):
    d = ROOT / "data/pairs" / style
    rows = [json.loads(l) for l in open(d / manifest)]
    rows = [r for r in rows if r["keep"] and r["task"] in tasks]
    for r in rows:
        r.setdefault("target", f"{r['task']}/{r['id']}.png")
    return d, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="cartoon3d")
    ap.add_argument("--manifest", default="manifest_v2.jsonl")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    tasks = TASKS + expr_tasks()
    d, rows = rows_of(a.style, a.manifest, tasks)
    st = Path(a.manifest).stem
    out = d / f"sana_lat_{st}.pt"
    if a.merge:
        parts = [torch.load(d / f"sana_lat_{st}.part{i}of{a.nparts}.pt") for i in range(a.nparts)]
        c = {"key": [(r["id"], r["task"], r["target"]) for r in rows], "ids": [(r["id"], r["task"]) for r in rows],
             "task": torch.tensor([tasks.index(r["task"]) for r in rows]), "tasks": tasks}
        for k in ("zi", "zt", "zi_f", "zt_f"):
            c[k] = torch.cat([p[k] for p in parts])
            assert len(c[k]) == len(rows), (k, len(c[k]), len(rows))
        torch.save(c, out)
        for i in range(a.nparts):
            (d / f"sana_lat_{st}.part{i}of{a.nparts}.pt").unlink()
        print(f"[enc] merged {len(rows)} pairs -> {out}", flush=True)
        return
    n = len(rows)
    lo, hi = n * a.part // a.nparts, n * (a.part + 1) // a.nparts
    dc = SC.load_dcae().to("cuda").eval()
    SF = dc.config.scaling_factor
    load = lambda p: torch.from_numpy(np.array(Image.open(p).convert("RGB").resize((512, 512)))).permute(2, 0, 1).float() / 127.5 - 1
    res = {k: [] for k in ("zi", "zt", "zi_f", "zt_f")}
    t0 = time.time()
    with ThreadPoolExecutor(16) as pool, torch.no_grad():
        for i in range(lo, hi, a.bs):
            ch = rows[i:min(i + a.bs, hi)]
            xi = torch.stack(list(pool.map(load, [d / r.get("input", f"inputs/{r['id']}.png") for r in ch]))).to("cuda", torch.bfloat16)
            xt = torch.stack(list(pool.map(load, [d / r["target"] for r in ch]))).to("cuda", torch.bfloat16)
            for k, x in (("zi", xi), ("zt", xt), ("zi_f", xi.flip(-1)), ("zt_f", xt.flip(-1))):
                res[k].append((dc.encode(x).latent.float() * SF).half().cpu())
            if (i - lo) % (a.bs * 100) == 0:
                print(f"[enc] part {a.part}: {i - lo}/{hi - lo} {time.time() - t0:.0f}s", flush=True)
    torch.save({k: torch.cat(v) for k, v in res.items()}, d / f"sana_lat_{st}.part{a.part}of{a.nparts}.pt")
    print(f"[enc] part {a.part} done {hi - lo} pairs {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
