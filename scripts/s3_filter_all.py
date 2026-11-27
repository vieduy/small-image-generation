"""R1 + R3 filter on ALL stage-3 synthetic-face teacher pairs (user 2026-11-25: "yes apply R1 + R3 to tS2").
Same definitions as scripts/s3_filter_examples.py (600-row sample: R1 74.8%, R3 84.2%):
  R1 identity: SFace rank of the photo among itself + the next 99 photos of its source pool (sorted ids; pools: inputs/ f*, c*, r*,
     ffhq_norm q*) vs the teacher output; pass if rank <= 5 (avatar, sticker) or <= 10 (expressions); no face in the output = fail.
  R3 skin colour: d_chroma = skin chroma(output) - chroma(photo), central face box, CIELAB; pass if <= 17 (or not measurable).
Teacher outputs are decoded on the GPU and scored in a CPU process pool (no PNGs written).
Out: data/pairs/cartoon3d/teacher_tB1_s8_filter.json (per row + summary), teacher_tB1_s8_drop.json ([[id, task], ...] for
train_dl.py --drop-list).
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/s3_filter_all.py
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/pairs/cartoon3d"
_F = None


def src_path(i):
    return ROOT / "data/ffhq_norm" / f"{i}.png" if i.startswith("q") else D / "inputs" / f"{i}.png"


def _init():
    global _F
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    import metrics as M
    _F = M.FaceID()


def _lab(arr):
    import cv2
    a = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    h, w = a.shape[:2]
    _F.det.setInputSize((w, h))
    _, fs = _F.det.detect(a)
    if fs is None or len(fs) == 0:
        return None
    x, y, fw, fh = max(fs, key=lambda q: q[2] * q[3])[:4]
    cx, cy = x + fw / 2, y + fh * 0.55
    x0, x1, y0, y1 = int(max(0, cx - .2 * fw)), int(min(w, cx + .2 * fw)), int(max(0, cy - .2 * fh)), int(min(h, cy + .2 * fh))
    if x1 <= x0 or y1 <= y0:
        return None
    v = cv2.cvtColor(a[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    aa, bb = float(np.median(v[:, 1])) - 128, float(np.median(v[:, 2])) - 128
    return float(np.hypot(aa, bb))


def photo_job(pid):
    arr = np.array(Image.open(src_path(pid)).convert("RGB"))
    f = _F.feat(Image.fromarray(arr))
    return pid, (None if f is None else (f.ravel() / np.linalg.norm(f)).astype(np.float32)), _lab(arr)


def out_job(args):
    k, arr = args
    f = _F.feat(Image.fromarray(arr))
    return k, (None if f is None else (f.ravel() / np.linalg.norm(f)).astype(np.float32)), _lab(arr)


def main():
    import dl_arch as DLA
    t0 = time.time()
    C = torch.load(D / "latents_dl_manifest_v2.pt", mmap=True)
    TL = torch.load(D / "teacher_tB1_s8.pt", mmap=True)
    assert TL["key"] == C["key"]
    keys = C["key"]
    pools = {p: sorted(x.stem for x in (D / "inputs").glob(f"{p}*.png")) for p in "fcr"}
    pools["q"] = sorted(x.stem for x in (ROOT / "data/ffhq_norm").glob("q*.png"))
    need = set()
    for pid, _, _ in keys:
        pool = pools[pid[0]]
        k = pool.index(pid) if pid in set(pool) else None
        if k is None:
            continue
        need.update(pool[(k + m) % len(pool)] for m in range(100))
    with Pool(96, initializer=_init) as pool:
        PH = {}
        for n_, (pid, f, c) in enumerate(pool.imap_unordered(photo_job, sorted(need), chunksize=64)):
            PH[pid] = (f, c)
            if n_ % 5000 == 0:
                print(f"[flt] photos {n_}/{len(need)} {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"[flt] photo features {len(PH)} in {(time.time() - t0) / 60:.1f} min", flush=True)
    pidx = {p: {s: i for i, s in enumerate(v)} for p, v in pools.items()}

    def gen():
        vae = DLA.load_vae(torch.float32).to("cuda").eval()
        with torch.no_grad():
            for i in range(0, len(keys), 64):
                y = DLA.decode(vae, TL["zt"][i:i + 64].to("cuda").float()).clamp(-1, 1)
                arrs = ((y + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
                for j, a in enumerate(arrs):
                    yield i + j, a

    rows = [None] * len(keys)
    with Pool(96, initializer=_init) as pool:
        for n_, (k, fo, co) in enumerate(pool.imap_unordered(out_job, gen(), chunksize=16)):
            pid, task, _ = keys[k]
            rank = None
            if fo is not None and pid in pidx[pid[0]] and PH.get(pid, (None,))[0] is not None:
                pool_ = pools[pid[0]]
                q = pidx[pid[0]][pid]
                gal = [pool_[(q + m) % len(pool_)] for m in range(100)]
                G = np.stack([PH[g][0] for g in gal if PH.get(g, (None,))[0] is not None])
                s = G @ fo
                rank = int((s > float(PH[pid][0] @ fo)).sum()) + 1
            ci = PH.get(pid, (None, None))[1]
            dch = None if (co is None or ci is None) else round(co - ci, 1)
            r1 = rank is not None and rank <= (10 if task.startswith("expr_") else 5)
            r3 = dch is None or dch <= 17
            rows[k] = {"id": pid, "task": task, "id_rank": rank, "d_chroma": dch, "R1": r1, "R3": r3, "keep": r1 and r3}
            if n_ % 5000 == 0:
                print(f"[flt] {n_}/{len(keys)} {(time.time() - t0) / 60:.1f} min", flush=True)
    S = {"rows": len(rows), "R1": round(float(np.mean([r["R1"] for r in rows])), 3), "R3": round(float(np.mean([r["R3"] for r in rows])), 3),
         "keep": round(float(np.mean([r["keep"] for r in rows])), 3)}
    for t in ("avatar", "sticker", "expr_"):
        v = [r["keep"] for r in rows if r["task"].startswith(t)]
        S[f"keep_{t.rstrip('_')}"] = round(float(np.mean(v)), 3)
    S["keep_per_task"] = {t: round(float(np.mean([r["keep"] for r in rows if r["task"] == t])), 3) for t in sorted({r["task"] for r in rows})}
    S["minutes"] = round((time.time() - t0) / 60, 1)
    (D / "teacher_tB1_s8_filter.json").write_text(json.dumps({"summary": S, "rows": rows}))
    (D / "teacher_tB1_s8_drop.json").write_text(json.dumps([[r["id"], r["task"]] for r in rows if not r["keep"]]))
    print("[flt] done", json.dumps(S), flush=True)


if __name__ == "__main__":
    main()
