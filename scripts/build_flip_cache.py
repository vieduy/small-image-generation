"""Proper flip augmentation + decoded-target cache for the DreamLite student (2026-11-19).

Root cause found while profiling: train_dl.py flipped the cached LATENTS (zi, zt) for horizontal-flip augmentation, but the
TinyVAE is not flip-equivariant: decode(flip(z)) vs flip(PNG) = 22.6 dB PSNR (round trip 33.4 dB; scripts/flip_aug_check.py,
grids/flip_aug_check.jpg): blocky edges, colour fringes in the eyes, smeared hair. Half of all regression targets were damaged.
Fix: encode the flipped images (zi_f = enc(flip(photo)), zt_f = enc(flip(target))), same rows/order as the main cache.

Also writes the decoded 512 px target cache (speed: removes 1-2 TinyVAE decodes per training step):
  tgt512_dl_<manifest>.u8   uint8 memmap [n, 3, 512, 512] = decode(zt) (the target the lp512/eye/FFL/critic terms already use;
                            uint8 quantisation 61 dB). Flipped samples use flip(decode(zt)) (both ~33 dB from flip(PNG)).
Out: data/pairs/<style>/latents_dl_<manifest>_flip.pt {key, zi_f, zt_f}  +  tgt512_dl_<manifest>.u8 (+ .json with key/shape).
  source env.sh; $VENV/bin/python scripts/build_flip_cache.py --manifest manifest_v2.jsonl --expr
"""
import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402
from train_student import DEV, TASKS, expr_tasks, to_t  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="cartoon3d")
    ap.add_argument("--manifest", default="manifest_v2.jsonl")
    ap.add_argument("--expr", action="store_true")
    ap.add_argument("--no-tgt512", action="store_true")
    a = ap.parse_args()
    d = ROOT / "data/pairs" / a.style
    st = Path(a.manifest).stem
    c = torch.load(d / f"latents_dl_{st}.pt")          # built by train_dl.build_cache_dl (same rows / order)
    tasks = TASKS + (expr_tasks() if a.expr else [])
    rows = [json.loads(l) for l in open(d / a.manifest)]
    rows = [r for r in rows if r["keep"] and r["task"] in tasks]
    for r in rows:
        r.setdefault("target", f"{r['task']}/{r['id']}.png")
    key = [(r["id"], r["task"], r["target"]) for r in rows]
    assert c["key"] == key, "main latent cache is stale for this manifest: run train_dl.py once to rebuild it"
    n = len(rows)
    kh = hashlib.sha1(json.dumps(key).encode()).hexdigest()[:16]
    vae = DLA.load_vae(torch.bfloat16).to(DEV).eval()

    ff = d / f"latents_dl_{st}_flip.pt"
    tf = d / f"tgt512_dl_{st}.u8"
    do_flip = not (ff.exists() and torch.load(ff).get("key") == key)
    do_tgt = not a.no_tgt512 and not (tf.exists() and json.loads(tf.with_suffix(".json").read_text()).get("key_hash") == kh)
    print(f"[flip-cache] n={n} flip latents: {'build' if do_flip else 'ok'}  tgt512: {'build' if do_tgt else 'ok/skip'}", flush=True)

    t0 = time.time()
    if do_tgt:  # from the cached latents only (GPU), no PNG reads
        mm = np.lib.format.open_memmap(str(tf) + ".tmp.npy", mode="w+", dtype=np.uint8, shape=(n, 3, 512, 512))
        with torch.no_grad():
            for i in range(0, n, 128):
                z = c["zt"][i:i + 128].to(DEV, torch.bfloat16)
                x = DLA.decode(vae, z).float().clamp(-1, 1)
                mm[i:i + len(z)] = ((x + 1) * 127.5).round().to(torch.uint8).cpu().numpy()
                if i % 12800 == 0:
                    print(f"[tgt512] {i}/{n} {time.time() - t0:.0f}s", flush=True)
        mm.flush()
        del mm
        Path(str(tf) + ".tmp.npy").replace(tf)
        tf.with_suffix(".json").write_text(json.dumps({"key_hash": kh, "shape": [n, 3, 512, 512], "dtype": "uint8",
                                                       "format": "npy", "what": "TinyVAE decode(zt), (x+1)*127.5"}))
        print(f"[tgt512] done {time.time() - t0:.0f}s -> {tf}", flush=True)

    if do_flip:  # reads every photo + target PNG once (NFS-bound, 32 threads)
        pool = ThreadPoolExecutor(32)
        load = lambda p: to_t(Image.open(p)).flip(-1)
        zi_f, zt_f = [], []
        with torch.no_grad():
            for i in range(0, n, 64):
                ch = rows[i:i + 64]
                xi = torch.stack(list(pool.map(load, [d / r.get("input", f"inputs/{r['id']}.png") for r in ch]))).to(DEV, torch.bfloat16)
                xt = torch.stack(list(pool.map(load, [d / r["target"] for r in ch]))).to(DEV, torch.bfloat16)
                zi_f.append(DLA.encode(vae, xi).float().cpu())
                zt_f.append(DLA.encode(vae, xt).float().cpu())
                if i % 3200 == 0:
                    print(f"[flip] {i}/{n} {time.time() - t0:.0f}s", flush=True)
        torch.save({"key": key, "zi_f": torch.cat(zi_f), "zt_f": torch.cat(zt_f)}, ff)
        print(f"[flip] done {time.time() - t0:.0f}s -> {ff}", flush=True)
    print("FLIP_CACHE_DONE", flush=True)


if __name__ == "__main__":
    main()
