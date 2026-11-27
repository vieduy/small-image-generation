"""REVERSE helper (2026-11-26, user: parallel run on local H100 GPU 0): same as s8_teacher_pairs.py but walks the shards from
the END (flip last shard -> plain first shard), skips shards that exist, writes each shard atomically (tmp + os.replace) and does NOT
merge (the forward process on dev-duyhv merges once all shards exist). Same args / outputs. Copy of:
s8 copy of s3_teacher_pairs.py (2026-11-26): adds --k-img (image guidance only on the first k steps; s7 sample_gs).
Original docstring below.
"""
"""Stage 3 (2026-11-23): deterministic teacher pairs for the one-step student.
For every row of data/pairs/<style>/latents_dl_<manifest>.pt (same rows/keys/order as train_dl.py), run teacher tB1 from the
PHONE's fixed noise (dl_arch.fixed_noise, seed 1234) on the cached photo latent zi (and zi_f = latent of the flipped photo), with
the stage-2 setting (default 8 steps, CFG img 3 / txt 4: expr 0.727 / outside 0.698 / likeness 0.375 vs 28 steps 0.738 / 0.701 /
0.381). Output = teacher latents in the student's TinyVAE space, fp16, written in shards (resumable):
  data/pairs/<style>/teacher_<tag>_s<steps>/{plain,flip}_<i>.pt  -> merged: teacher_<tag>_s<steps>.pt {key, zt, zt_f, setting}
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/s3_teacher_pairs.py --tag tB1
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402
import teacher_dl as TD
from s7_guidance_sweep import sample_gs  # noqa: E402
from train_student import TASKS, expr_tasks  # noqa: E402

ROOT, DEV = TD.ROOT, "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tB1")
    ap.add_argument("--style", default="cartoon3d")
    ap.add_argument("--manifest", default="manifest_v2.jsonl")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--g-img", type=float, default=3.0)
    ap.add_argument("--g-txt", type=float, default=4.0)
    ap.add_argument("--k-img", type=int, default=None, help="img guidance on the first k steps only (default: all)")
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--shard", type=int, default=2048)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--name", default=None, help="output name (default teacher_<tag>_s<steps>); 2026-11-25 for a 2nd CFG setting")
    a = ap.parse_args()
    d = ROOT / "data/pairs" / a.style
    st = Path(a.manifest).stem
    C = torch.load(d / f"latents_dl_{st}.pt", mmap=True)
    FL = torch.load(d / f"latents_dl_{st}_flip.pt", mmap=True)
    assert FL["key"] == C["key"]
    unet, T = TD.load_teacher(a.tag)
    unet = unet.to(DEV).eval()
    assert T["tasks"] == TASKS + expr_tasks()
    emb, mask = T["emb"].to(DEV, torch.float16), T["mask"].to(DEV)
    ne, nm = T["null_emb"].to(DEV, torch.float16), T["null_mask"].to(DEV)
    noise = DLA.fixed_noise(DEV, torch.float32)
    n = a.max_rows or len(C["key"])
    oname = a.name or f"teacher_{a.tag}_s{a.steps}"
    od = d / oname
    od.mkdir(exist_ok=True)
    t0, done = time.time(), 0
    for kind, Z in (("flip", FL["zi_f"]), ("plain", C["zi"])):
        for s0 in reversed(range(0, n, a.shard)):
            f = od / f"{kind}_{s0 // a.shard:04d}.pt"
            if f.exists():
                continue
            out = []
            with torch.inference_mode():
                for i in range(s0, min(s0 + a.shard, n), a.bs):
                    j = slice(i, min(i + a.bs, s0 + a.shard, n))
                    zi = Z[j].to(DEV, torch.float16)
                    k = C["task"][j].to(DEV)
                    out.append(sample_gs(unet, zi, emb[k], mask[k], ne, nm, a.g_img, a.k_img or a.steps, a.g_txt, False, a.steps, noise).half().cpu())
            if f.exists():  # the forward process finished it meanwhile
                continue
            torch.save(torch.cat(out), f.with_suffix('.tmp'))
            os.replace(f.with_suffix('.tmp'), f)
            done += min(s0 + a.shard, n) - s0
            el = time.time() - t0
            left = (2 * n - (len(list(od.glob("*.pt"))) * a.shard)) / max(done / el, 1e-9)
            print(f"[s3] {kind} shard {s0 // a.shard} ({min(s0 + a.shard, n)}/{n}) {done / el:.1f} pairs/s, ~{left / 3600:.1f} h left", flush=True)
    print("[s3r] reverse helper done (no merge)", flush=True)


if __name__ == "__main__":
    main()
