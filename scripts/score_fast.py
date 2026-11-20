"""Fast pair scoring for gen_pairs (same metrics, batched GPU + multi-process CPU).
Slow path (per image, batch 1, single CPU process) scored ~2 rows/s; most time was CPU (OpenCV YuNet/SFace, WebP quality
search, connected components) with the GPU idle.
  GPU : DINOv2/CLIP style embeddings (batch 64), BiRefNet_lite matte at 1024 (batch 16), CLIP subject check (batch 32);
        image decoding in a thread pool.
  CPU : FaceCache misses (YuNet+SFace+skin L*) and sticker_postprocess in a process pool (default 32 workers).
Results are identical in definition to the slow path; WebP size search uses method=4 (faster) instead of 6 -- this only
changes the reported KB slightly (pass threshold 100 KB vs typical 15-25 KB).
"""
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import torch
from PIL import Image

import metrics as M

_FID = None


def _init():
    global _FID
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    _FID = M.FaceID()


def _face_job(args):
    root, rel = args
    p = Path(root) / rel
    stt = p.stat()
    fc = M.FaceCache.__new__(M.FaceCache)  # reuse the exact computation of FaceCache.get without its dict/pickle
    fc.fid, fc.root, fc.d, fc.hits, fc.miss = _FID, Path(root), {}, 0, 0
    v = fc.get(p)
    return (rel, stt.st_mtime_ns, stt.st_size), v


def _sticker_job(args):
    path, alpha_u8, method = args
    o = Image.open(path).convert("RGB")
    _, info = M.sticker_postprocess(o, alpha_u8.astype(np.float32) / 255, webp_method=method)
    return info


def _load(paths, workers=16):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda p: Image.open(p).convert("RGB"), paths))


def precompute(out, new_rows, fc, emb=None, mat=None, sc=None, rdm=None, rd=None, subj_name=None, workers=32, webp_method=4,
               log=print):
    """new_rows: list of (id, task, is_face, prompt). Fills fc (FaceCache) for every input/output image of face rows and
    returns {(id, task): partial row dict} with style_dino, refcopy_dino, sticker info, subject check."""
    out = Path(out)
    res = {(i, t): {} for i, t, _, _ in new_rows}
    ctx = get_context("spawn")  # fork after CUDA/thread-pool init deadlocked (README); spawn = clean worker interpreters
    # 1) faces: missing FaceCache entries in parallel
    rels = []
    for i, t, f, _ in new_rows:
        if f:
            for rel in (f"inputs/{i}.png", f"{t}/{i}.png"):
                p = out / rel
                stt = p.stat()
                if (rel, stt.st_mtime_ns, stt.st_size) not in fc.d:
                    rels.append(rel)
    rels = sorted(set(rels))
    if rels:
        with ctx.Pool(workers, initializer=_init) as pool:
            for k, v in pool.imap_unordered(_face_job, [(str(out), r) for r in rels], chunksize=16):
                fc.d[k] = v
        log(f"[fast] face features {len(rels)}")
    # 2) GPU: style embeddings, mattes, subject checks (batched)
    st_rows = [(i, t) for i, t, f, _ in new_rows if t != "avatar"]
    B = 64
    allr = [(i, t) for i, t, _, _ in new_rows]
    for b in range(0, len(allr), B):
        ch = allr[b:b + B]
        ims = _load([out / t / f"{i}.png" for i, t in ch])
        if emb is not None:
            d, _ = emb(ims)
            sd = (d @ rdm).cpu().numpy(); cp = (d @ rd.T).max(1).values.cpu().numpy()
            for k, (i, t) in enumerate(ch):
                res[(i, t)] |= {"style_dino": round(float(sd[k]), 4), "refcopy_dino": round(float(cp[k]), 4)}
    alphas = {}
    for b in range(0, len(st_rows), 16):
        ch = st_rows[b:b + 16]
        ims = _load([out / t / f"{i}.png" for i, t in ch])
        for (i, t), a in zip(ch, mat.batch(ims)):
            alphas[(i, t)] = (a * 255).round().astype(np.uint8)
    if sc is not None:
        subj = [(i, t, p) for i, t, f, p in new_rows if not f]
        for b in range(0, len(subj), 32):
            ch = subj[b:b + 32]
            ims = _load([out / t / f"{i}.png" for i, t, _ in ch])
            for (i, t, p), r in zip(ch, sc.batch(ims, [subj_name(p) for _, _, p in ch])):
                res[(i, t)] |= r
    log(f"[fast] gpu done: {len(allr)} embeddings, {len(alphas)} mattes")
    # 3) sticker post-process in parallel
    if alphas:
        keys = list(alphas)
        with ctx.Pool(workers, initializer=_init) as pool:
            infos = pool.map(_sticker_job, [(str(out / t / f"{i}.png"), alphas[(i, t)], webp_method) for i, t in keys], chunksize=8)
        for k, info in zip(keys, infos):
            res[k] |= info
        log(f"[fast] sticker post-process {len(keys)}")
    return res
