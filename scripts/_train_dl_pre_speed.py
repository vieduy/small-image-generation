"""Regression stage for the DreamLite one-step student (bake-off vs SDXS b1; same data, same loss).

  source env.sh; $VENV/bin/python scripts/train_dl.py --style cartoon3d --tag dl1 --manifest manifest_round4.jsonl --expr --steps 16000
Out: weights/student/<tag>/{unet/, task_emb.pt (emb, mask, tasks), config.json (arch=dreamlite), train_log.jsonl, samples_*.jpg}
"""
import argparse
import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402
import teacher_prompt as TP  # noqa: E402
from train_student import DEV, TASKS, expr_tasks, to_t  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GEST = ("think", "thumbs", "wave", "ok", "facepalm", "shrug", "thanks")


@torch.no_grad()
def build_cache_dl(style, vae, manifest, tasks):
    """Same rows/keys as train_student.build_cache, latents from DreamLite's TinyVAE. Parallel PNG loading (NFS-bound)."""
    d = ROOT / "data/pairs" / style
    f = d / f"latents_dl_{Path(manifest).stem}.pt"
    rows = [json.loads(l) for l in open(d / manifest)]
    rows = [r for r in rows if r["keep"] and r["task"] in tasks]
    for r in rows:
        r.setdefault("target", f"{r['task']}/{r['id']}.png")
    key = [(r["id"], r["task"], r["target"]) for r in rows]
    if f.exists():
        c = torch.load(f)
        if c.get("key") == key:
            return c
    pool = ThreadPoolExecutor(32)
    load = lambda p: to_t(Image.open(p))
    zi, zt, img256, tk = [], [], [], []
    t0 = time.time()
    for i in range(0, len(rows), 64):
        ch = rows[i:i + 64]
        xi = torch.stack(list(pool.map(load, [d / r.get("input", f"inputs/{r['id']}.png") for r in ch]))).to(DEV, torch.bfloat16)
        xt = torch.stack(list(pool.map(load, [d / r["target"] for r in ch]))).to(DEV, torch.bfloat16)
        zi.append(DLA.encode(vae, xi).float().cpu())
        zt.append(DLA.encode(vae, xt).float().cpu())
        img256.append(((F.interpolate(xt.float(), size=256, mode="area") + 1) * 127.5).round().to(torch.uint8).cpu())
        tk += [tasks.index(r["task"]) for r in ch]
        if i % 1280 == 0:
            print(f"[dl-cache] {i}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    c = {"ids": [(r["id"], r["task"]) for r in rows], "key": key, "zi": torch.cat(zi), "zt": torch.cat(zt),
         "img256": torch.cat(img256), "task": torch.tensor(tk)}
    torch.save(c, f)
    return c


@torch.no_grad()
def init_tokens(style, tasks, photos, out_file):
    """Task tokens = Qwen3-VL edit-mode hidden states of each task's instruction, averaged over photos (vision tokens of a
    512x512 photo have a fixed count, so the per-task sequences align). Padded across tasks with a fixed mask."""
    import yaml
    from diffusers import DreamLiteMobilePipeline
    pipe = DreamLiteMobilePipeline.from_pretrained(DLA.DL_REPO, revision=DLA.DL_REV, unet=None, vae=None,
                                                   torch_dtype=torch.bfloat16).to(DEV)
    import i2i_server as S
    st = S.load_styles()[style]
    EX = {f"expr_{e['id']}": e["phrase"] for e in yaml.safe_load((ROOT / "configs/sticker_expressions.yaml").read_text())}
    embs = []
    for t in tasks:
        inst = (TP.avatar_prompt(st["description"], None) if t == "avatar" else
                TP.sticker_prompt(st["description"], None) if t == "sticker" else
                TP.expr_sticker_prompt(st["description"], None, EX[t]))
        p = (f"[Edit]: A diptych with two side-by-side images of the same scene. Compared to the right side, the left one has "
             f"been redrawn following this instruction: {inst}")
        acc = None
        for ph in photos:
            e, _ = pipe.encode_prompt(mode="edit", prompts=[p], image=Image.open(ph).convert("RGB"), device=DEV, dtype=torch.bfloat16)
            acc = e.float() if acc is None else acc + e.float()
        embs.append((acc / len(photos))[0].cpu())
        print(f"[dl-tokens] {t}: L={embs[-1].shape[0]}", flush=True)
    L = max(e.shape[0] for e in embs)
    emb = torch.zeros(len(tasks), L, embs[0].shape[1])
    mask = torch.zeros(len(tasks), L, dtype=torch.long)
    for i, e in enumerate(embs):
        emb[i, :e.shape[0]] = e
        mask[i, :e.shape[0]] = 1
    torch.save({"emb": emb, "mask": mask, "tasks": tasks, "n_photos": len(photos)}, out_file)
    del pipe
    torch.cuda.empty_cache()
    return emb, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="cartoon3d")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--manifest", default="manifest_round4.jsonl")
    ap.add_argument("--expr", action="store_true")
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--emb-lr", type=float, default=1e-3)
    ap.add_argument("--lpips-w", type=float, default=1.0)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--init-from", default=None, help="start from a trained DreamLite student tag (UNet + tokens/mask), e.g. dl1")
    ap.add_argument("--lora-rank", type=int, default=0, help=">0 (needs --init-from): freeze that UNet, train a LoRA style pack + tokens")
    ap.add_argument("--eye-w", type=float, default=0.0, help="L1 on 2x2-pooled RGB at 512 px inside soft eye masks (YuNet eye landmarks "
                    "of the klein target); target = TinyVAE decode of the target latent (round-trip keeps iris colour)")
    ap.add_argument("--eye-lpips-w", type=float, default=0.0, help="LPIPS on ROI-aligned eye crops of the 512 px output, upsampled to "
                    "--eye-crop px (GFPGAN-style component loss without the discriminator): an iris is ~10 px at the 256 px global LPIPS")
    ap.add_argument("--eye-crop", type=int, default=192)
    ap.add_argument("--lp512-w", type=float, default=0.0, help="LPIPS on a random --lp512-crop px crop of the FULL-RES 512 px output vs "
                    "the TinyVAE decode of the target latent (keeps 0.97 of klein's fine detail). The global LPIPS only sees 256 px, so "
                    "detail between 256 and 512 px was supervised by latent MSE alone (2026-11-19)")
    ap.add_argument("--lp512-crop", type=int, default=256)
    a = ap.parse_args()
    assert a.lora_rank == 0 or a.init_from, "--lora-rank needs --init-from"
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = ROOT / "weights/student" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    tasks = TASKS + (expr_tasks() if a.expr else [])

    vae = DLA.load_vae(torch.bfloat16).to(DEV).eval().requires_grad_(False)
    c = build_cache_dl(a.style, vae, a.manifest, tasks)
    n = len(c["task"])
    tok_file = ROOT / f"weights/student/_dl_tokens_{a.style}_{len(tasks)}.pt"
    if a.init_from:  # tokens of the trained student (its first len(tasks) options), trained further
        t = torch.load(ROOT / "weights/student" / a.init_from / "task_emb.pt")
        assert t["tasks"][:len(tasks)] == tasks, f"{a.init_from} task order {t['tasks'][:len(tasks)]} != {tasks}"
        emb0, mask = t["emb"][:len(tasks)], t["mask"][:len(tasks)]
    elif tok_file.exists():
        t = torch.load(tok_file)
        emb0, mask = t["emb"], t["mask"]
    else:
        d = ROOT / "data/pairs" / a.style
        photos = [d / "inputs" / f"f{i:05d}.png" for i in range(0, 1600, 100)]  # 16 synthetic faces
        emb0, mask = init_tokens(a.style, tasks, photos, tok_file)
    emb = torch.nn.Parameter(emb0.float().to(DEV))
    mask = mask.to(DEV)
    unet = DLA.load_unet(ROOT / "weights/student" / a.init_from / "unet" if a.init_from else None)
    if a.lora_rank > 0:  # same adapter targets as the DreamLite DMD packs (train_dmd.py)
        from peft import LoraConfig, get_peft_model
        unet.requires_grad_(False)
        tm = r".*(to_q|to_k|to_v|to_out\.0|proj_in|proj_out|ff\.net\.0\.proj|ff\.net\.2|conv1\.pointwise|conv2\.pointwise)$"
        unet = get_peft_model(unet, LoraConfig(r=a.lora_rank, lora_alpha=a.lora_rank, target_modules=tm, init_lora_weights="gaussian"))
        print(f"[dl] LoRA r={a.lora_rank}: {sum(p.numel() for p in unet.parameters() if p.requires_grad) / 1e6:.1f}M trainable", flush=True)
    unet = unet.to(DEV).train()
    if a.grad_ckpt:
        unet.enable_gradient_checkpointing()
    lp = lpips.LPIPS(net="vgg", verbose=False).to(DEV).eval().requires_grad_(False)

    val = []
    for k in range(len(tasks)):
        val += [i for i in range(n) if c["task"][i] == k][:32 if k < 2 else 4]
    vs = set(val)
    tr = [i for i in range(n) if i not in vs]
    print(f"[dl] pairs={n} train={len(tr)} val={len(val)} tokens L={emb.shape[1]} "
          f"unet={sum(p.numel() for p in unet.parameters()) / 1e6:.0f}M", flush=True)
    opt = torch.optim.AdamW([{"params": [p for p in unet.parameters() if p.requires_grad], "lr": a.lr},
                             {"params": [emb], "lr": a.emb_lr, "weight_decay": 0}],
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    log = open(out / "train_log.jsonl", "w")

    # ---------- eye geometry (YuNet 5-point landmarks on the klein target at 256 px), only with --eye-w / --eye-lpips-w ----------
    eyes = torch.zeros(n, 2, 2)
    eye_sig = torch.zeros(n)
    if a.eye_w > 0 or a.eye_lpips_w > 0:
        import cv2
        from metrics import FaceID
        det = FaceID().det
        det.setInputSize((256, 256))
        for i in range(n):
            arr = cv2.cvtColor(np.ascontiguousarray(c["img256"][i].permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR)
            _, f = det.detect(arr)
            if f is None or len(f) == 0:
                continue
            f = max(f, key=lambda r: r[2] * r[3])
            eyes[i] = torch.tensor([[float(f[4]), float(f[5])], [float(f[6]), float(f[7])]])
            eye_sig[i] = 0.07 * float(f[2])
        print(f"[dl] eye landmarks on {(eye_sig > 0).sum().item()}/{n} targets", flush=True)
    gy, gx = torch.meshgrid(torch.arange(256, device=DEV) * 2 + 1.0, torch.arange(256, device=DEV) * 2 + 1.0, indexing="ij")

    def eye_geom(idx, flip):  # eye centres / sigma in 512 px coords of the (possibly flipped) sample
        e, sg = eyes[idx].to(DEV) * 2, eye_sig[idx].to(DEV) * 2
        if flip is not None:
            e[..., 0] = torch.where(flip.view(-1, 1), 512 - e[..., 0], e[..., 0])
        return e, sg

    def eye_losses(dec512, zt, idx, flip):
        e, sg = eye_geom(idx, flip)
        ok = sg > 0
        out = {}
        if ok.sum() == 0:
            return out
        with torch.no_grad():
            tgt = DLA.decode(vae, zt.to(torch.bfloat16)).float().clamp(-1, 1)
        if a.eye_w > 0:  # soft mask on a 256 grid (2x2 pooling of the 512 px image)
            m = torch.zeros(len(idx), 256, 256, device=DEV)
            for k in range(2):
                d2 = (gx - e[:, k, 0].view(-1, 1, 1)) ** 2 + (gy - e[:, k, 1].view(-1, 1, 1)) ** 2
                m = torch.maximum(m, torch.exp(-d2 / (2 * sg.clamp(min=1e-3).view(-1, 1, 1) ** 2)))
            m = (m * ok.view(-1, 1, 1)).unsqueeze(1)
            df = F.avg_pool2d(dec512, 2) - F.avg_pool2d(tgt, 2)
            out["eye"] = (m * df.abs()).sum() / (3 * m.sum())
        if a.eye_lpips_w > 0:  # two crops per face, box half-size 2.5 sigma (~iris + lids + brow edge)
            from torchvision.ops import roi_align
            boxes = []
            for b in torch.nonzero(ok).flatten().tolist():
                for k in range(2):
                    h = 2.5 * sg[b].item()
                    cx, cy = e[b, k, 0].item(), e[b, k, 1].item()
                    boxes.append([b, cx - h, cy - h, cx + h, cy + h])
            bx = torch.tensor(boxes, device=DEV, dtype=torch.float32)
            cp = roi_align(dec512, bx, a.eye_crop, aligned=True)
            ct = roi_align(tgt, bx, a.eye_crop, aligned=True)
            out["eye_lpips"] = lp(cp, ct).mean()
        return out

    def lp512_loss(dec512, zt):  # one random face-area crop per sample, same box on output and target, native 512 px
        from torchvision.ops import roi_align
        with torch.no_grad():
            tgt = DLA.decode(vae, zt.to(torch.bfloat16)).float().clamp(-1, 1)
        k = a.lp512_crop
        x0 = torch.randint(48, 512 - 48 - k + 1, (len(zt),)).float()
        y0 = torch.randint(32, 512 - 64 - k + 1, (len(zt),)).float()
        bx = torch.stack([torch.arange(len(zt)).float(), x0, y0, x0 + k, y0 + k], 1).to(DEV)
        return lp(roi_align(dec512, bx, k, aligned=True), roi_align(tgt, bx, k, aligned=True)).mean()

    def batch(idx, aug):
        zi, zt = c["zi"][idx].to(DEV), c["zt"][idx].to(DEV)
        im = c["img256"][idx].to(DEV).float() / 127.5 - 1
        tk = c["task"][idx].to(DEV)
        if aug:
            flip = torch.rand(len(idx), device=DEV) < 0.5
            zi, zt, im = (torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x) for x in (zi, zt, im))
            batch.flip = flip
        else:
            batch.flip = None
        return zi, zt, im, tk

    def losses(zi, zt, im, tk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zp = DLA.forward(unet, zi, emb[tk], mask[tk])
            mse = F.mse_loss(zp.float(), zt)
            dec512 = DLA.decode(vae, zp.to(torch.bfloat16)).float().clamp(-1, 1)
            dec = F.interpolate(dec512, size=256, mode="bilinear", antialias=True)
            lpl = lp(dec, im).mean()
        losses.dec512 = dec512
        losses.lp512 = lp512_loss(dec512, zt) if a.lp512_w > 0 else None
        return mse, lpl, dec

    def samples(step):
        gi = [i for g in GEST if f"expr_{g}" in tasks for i in [j for j in val if c["task"][j] == tasks.index(f"expr_{g}")][:1]]
        vi = torch.tensor(val[:3] + val[32:35] + gi)
        unet.eval()
        with torch.no_grad():
            zi, zt, im, tk = batch(vi, False)
            _, _, dec = losses(zi, zt, im, tk)
            src = DLA.decode(vae, zi.to(torch.bfloat16)).float().clamp(-1, 1)
        unet.train()
        t8 = lambda x: ((F.interpolate(x, size=256, mode="area").clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
        rows = np.concatenate([t8(src), t8(im), t8(dec)], 2)  # photo | klein target | student
        Image.fromarray(np.concatenate(list(rows), 0)).save(out / f"samples_{step:05d}.jpg", quality=88)

    samples(0)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        idx = torch.tensor(random.sample(tr, a.bs))
        zi, zt, im, tk = batch(idx, True)
        mse, lpl, _ = losses(zi, zt, im, tk)
        loss = mse + a.lpips_w * lpl
        if losses.lp512 is not None:
            loss = loss + a.lp512_w * losses.lp512
        el = eye_losses(losses.dec512, zt, idx, batch.flip) if (a.eye_w > 0 or a.eye_lpips_w > 0) else {}
        if "eye" in el:
            loss = loss + a.eye_w * el["eye"]
        if "eye_lpips" in el:
            loss = loss + a.eye_lpips_w * el["eye_lpips"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in unet.parameters() if p.requires_grad] + [emb], 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            r = {"step": step, "mse": round(mse.item(), 5), "lpips": round(lpl.item(), 4), "lr": sched.get_last_lr()[0],
                 "s": round(time.time() - t0, 1)} | {k: round(v.item(), 4) for k, v in el.items()} \
                | ({"lp512": round(losses.lp512.item(), 4)} if losses.lp512 is not None else {})
            if step % a.val_every == 0 or step == a.steps:
                unet.eval()
                with torch.no_grad():
                    vm, vl = [], []
                    for i in range(0, len(val), 16):
                        m_, l_, _ = losses(*batch(torch.tensor(val[i:i + 16]), False))
                        vm.append(m_.item())
                        vl.append(l_.item())
                unet.train()
                r |= {"val_mse": round(sum(vm) / len(vm), 5), "val_lpips": round(sum(vl) / len(vl), 4)}
                samples(step)
            print(json.dumps(r), flush=True)
            log.write(json.dumps(r) + "\n")
            log.flush()
        if not math.isfinite(loss.item()):
            print("[dl] non-finite loss, abort", flush=True)
            break
    unet.save_pretrained(out / ("unet_lora" if a.lora_rank > 0 else "unet"))
    torch.save({"emb": emb.detach().cpu(), "mask": mask.cpu(), "tasks": tasks}, out / "task_emb.pt")
    (out / "config.json").write_text(json.dumps(vars(a) | {"arch": "dreamlite", "init": a.init_from or f"{DLA.DL_REPO}@{DLA.DL_REV}", "param": "flow_x0",
                                                           "T": 1000, "decoder": "DreamLite TinyVAE", "train_pairs": len(tr),
                                                           "minutes": round((time.time() - t0) / 60, 1)}, indent=1))
    print(f"[dl] done {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
