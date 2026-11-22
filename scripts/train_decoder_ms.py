"""dl8 step 3 (2026-11-21): ONE phone decoder for all styles, with per-style FiLM after every decoder block
(dl_arch.install_decoder_film; ~2 KB per style). Recipe = train_decoder.py (L1 + LPIPS-vgg at 512 + DINOv2 critic), init = an
existing decoder (--init, default decg2), on the joint student's OWN latents for each style (per-style views weights/student/
<student>[_<style>], built with train_decoder.build_student_latents) -> klein targets of that style; style-balanced batches.
Saves weights/decoder/<tag>/ (+ style_film_dec.pt, film_style.json = style 0) and one view per extra style:
weights/decoder/<tag>_<style>/ (same weights, film_style.json = that style) -> every eval script takes --decoder <view>.
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/train_decoder_ms.py --student dl8 --styles cartoon3d,comic_line,chibi --tag dec8
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402
from train_decoder import build_student_latents, chmod_tree  # noqa: E402
from train_dl import build_cache_dl  # noqa: E402
from train_student import DEV, TASKS, expr_tasks, to_t  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class Pairs(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items  # (style index, latent, flipped latent, target path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, j):
        s, z, zf, p = self.items[j]
        t = to_t(Image.open(p))
        f = random.random() < 0.5
        return s, (zf if f else z), (t.flip(-1) if f else t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--styles", default="cartoon3d,comic_line,chibi")
    ap.add_argument("--manifest", default="manifest_v2.jsonl")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--init", default="decg2")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--film-lr", type=float, default=1e-3)
    ap.add_argument("--gan-w", type=float, default=0.1)
    ap.add_argument("--gan-start", type=int, default=500)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-pairs", type=int, default=0, help="smoke test: first N pairs per style only")
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    styles = a.styles.split(",")
    tasks = TASKS + expr_tasks()
    st = Path(a.manifest).stem
    out = ROOT / "weights/decoder" / a.tag
    assert not (out / "style_film_dec.pt").exists(), f"{out} exists; refusing to overwrite"
    out.mkdir(parents=True, exist_ok=True)

    vae = DLA.load_vae(torch.bfloat16).to(DEV).eval()
    items_tr, val = [], []
    for si, s in enumerate(styles):
        view = a.student if si == 0 else f"{a.student}_{s}"
        d = ROOT / "data/pairs" / s
        c = build_cache_dl(s, vae, a.manifest, tasks)
        FL = torch.load(d / f"latents_dl_{st}_flip.pt")
        assert FL["key"] == c["key"], f"{s}: flip cache is stale"
        if a.max_pairs:
            m_ = a.max_pairs
            c = {k_: (v_[:m_] if hasattr(v_, "__len__") else v_) for k_, v_ in c.items()}
            FL = {k_: (v_[:m_] if hasattr(v_, "__len__") else v_) for k_, v_ in FL.items()}
        lf = ROOT / "weights/decoder" / f"_zs_{view}_{st}.pt"
        if not (lf.exists() and torch.load(lf, mmap=True).get("key") == c["key"]):
            build_student_latents(view, c, FL, lf)
        Z = torch.load(lf)
        n = len(c["key"])
        vv = []
        for k in range(len(tasks)):
            vv += [i for i in range(n) if c["task"][i] == k][:8 if k < 2 else 2]
        vs = set(vv)
        paths = [d / k_[2] for k_ in c["key"]]
        items_tr.append([(si, Z["zs"][i], Z["zs_f"][i], paths[i]) for i in range(n) if i not in vs])
        val += [(si, Z["zs"][i], paths[i]) for i in vv]
        print(f"[dms] {s}: view {view}, {n} pairs, train {len(items_tr[-1])}, val {len(vv)}", flush=True)
        del c, FL
    del vae

    dec = DLA.load_vae(torch.float32, ROOT / "weights/decoder" / a.init)
    DLA.install_decoder_film(dec, len(styles), trainable=True)
    dec = dec.to(DEV)
    dec.encoder.requires_grad_(False)
    dmod = dec.decoder
    dmod.train()
    film = [p for n_, p in dmod.named_parameters() if n_.startswith("style_film")]
    base = [p for n_, p in dmod.named_parameters() if not n_.startswith("style_film")]
    opt = torch.optim.AdamW([{"params": base, "lr": a.lr}, {"params": film, "lr": a.film_lr, "weight_decay": 0}], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    lp = lpips.LPIPS(net="vgg", verbose=False).to(DEV).eval().requires_grad_(False)
    if a.gan_w > 0:
        from transformers import AutoModel
        from metrics import _snap
        dino = AutoModel.from_pretrained(str(_snap("facebook/dinov2-base"))).to(DEV).eval().requires_grad_(False)
        gl = [2, 5, 8, 11]
        SN = nn.utils.spectral_norm
        heads = nn.ModuleList([nn.Sequential(SN(nn.Conv2d(768, 256, 3, padding=1)), nn.LeakyReLU(0.2), SN(nn.Conv2d(256, 1, 1))) for _ in gl]).to(DEV)
        d_opt = torch.optim.Adam(heads.parameters(), lr=2e-4, betas=(0.0, 0.99))
        im_mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
        im_std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

        def critic(x):
            i, j = random.randint(0, 64), random.randint(0, 64)
            x = (((x[..., i:i + 448, j:j + 448] + 1) / 2) - im_mean) / im_std
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hs = dino(pixel_values=x, output_hidden_states=True).hidden_states
            return torch.cat([h(hs[l + 1][:, 1:].float().transpose(1, 2).reshape(len(x), 768, 32, 32)).flatten(1) for h, l in zip(heads, gl)], 1)

    def decode(z, s):
        dmod._film_style = s.to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return DLA.decode(dec, z.to(DEV)).float().clamp(-1, 1)

    @torch.no_grad()
    def evaluate(step):
        dmod.eval()
        R = {}
        for si, s in enumerate(styles):
            vv = [x for x in val if x[0] == si]
            l1, lps = [], []
            for i in range(0, len(vv), 16):
                ch = vv[i:i + 16]
                y = decode(torch.stack([x[1] for x in ch]).float(), torch.tensor([si] * len(ch)))
                t = torch.stack([to_t(Image.open(x[2])) for x in ch]).to(DEV)
                l1.append((y - t).abs().mean().item()); lps.append(lp(y, t).mean().item())
            R[s] = {"val_l1": round(float(np.mean(l1)), 4), "val_lpips": round(float(np.mean(lps)), 4)}
        dmod.train()
        return R

    loaders = [iter(torch.utils.data.DataLoader(Pairs(it), batch_size=max(1, a.bs // len(styles)), shuffle=True, drop_last=True,
                                                num_workers=max(1, a.workers // len(styles)), persistent_workers=True)) for it in items_tr]

    def nxt(i):
        try:
            return next(loaders[i])
        except StopIteration:
            loaders[i] = iter(torch.utils.data.DataLoader(Pairs(items_tr[i]), batch_size=max(1, a.bs // len(styles)), shuffle=True,
                                                          drop_last=True, num_workers=max(1, a.workers // len(styles))))
            return next(loaders[i])
    log = open(out / "train_log.jsonl", "w")
    r0 = {"step": 0} | evaluate(0)
    print(json.dumps(r0), flush=True); log.write(json.dumps(r0) + "\n"); log.flush()
    t0 = time.time()
    for step in range(1, a.steps + 1):
        bb = [nxt(i) for i in range(len(styles))]  # style-balanced
        s = torch.cat([b[0] for b in bb]); z = torch.cat([b[1] for b in bb]); t = torch.cat([b[2] for b in bb]).to(DEV, non_blocking=True)
        y = decode(z.float(), s)
        l1, lpl = (y - t).abs().mean(), lp(y, t).mean()
        loss = l1 + lpl
        r = {"l1": l1.item(), "lpips": lpl.item()}
        if a.gan_w > 0 and step > a.gan_start:
            heads.requires_grad_(False)
            g = -critic(y).mean()
            heads.requires_grad_(True)
            loss = loss + a.gan_w * g
            r["g_gan"] = g.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(dmod.parameters()), 1.0)
        opt.step(); sched.step()
        if a.gan_w > 0:
            ld = F.relu(1 - critic(t)).mean() + F.relu(1 + critic(y.detach())).mean()
            d_opt.zero_grad(set_to_none=True); ld.backward(); d_opt.step()
        if not math.isfinite(loss.item()):
            print("[dms] non-finite loss, abort", flush=True)
            break
        if step % 50 == 0:
            rr = {"step": step, "s": round(time.time() - t0, 1)} | {k: round(v, 4) for k, v in r.items()}
            if step % a.val_every == 0 or step == a.steps:
                rr |= evaluate(step)
            print(json.dumps(rr), flush=True); log.write(json.dumps(rr) + "\n"); log.flush()
    fs = DLA.film_state(dmod) if False else {n: (dmod.style_film[n.replace(".", "__") + "__g"].detach().cpu(),
                                                 dmod.style_film[n.replace(".", "__") + "__b"].detach().cpu()) for n in dmod._film_names}
    sd = {k: v for k, v in dec.state_dict().items() if "style_film" not in k}
    from diffusers import AutoencoderTiny
    clean = AutoencoderTiny.from_config(dec.config)
    clean.load_state_dict(sd)
    for si, s in enumerate(styles):
        v = out if si == 0 else ROOT / "weights/decoder" / f"{a.tag}_{s}"
        v.mkdir(parents=True, exist_ok=True)
        clean.save_pretrained(v)
        torch.save({"n_styles": len(styles), "film": fs, "styles": styles}, v / "style_film_dec.pt")
        (v / "film_style.json").write_text(json.dumps({"style": si, "styles": styles}))
        chmod_tree(v)
    (out / "config.json.train").write_text(json.dumps(vars(a) | {"minutes": round((time.time() - t0) / 60, 1)}, indent=1))
    print(f"[dms] done {(time.time() - t0) / 60:.1f} min -> {out} (+ views)", flush=True)


if __name__ == "__main__":
    main()
