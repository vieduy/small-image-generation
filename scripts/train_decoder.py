"""Phone-decoder fine-tune on the frozen student's own latents (user-approved plan 2026-11-19 ~13:40 VN).

Why: the DreamLite TinyVAE decoder was trained to decode CLEAN latents (enc(sharp image)); the one-step student outputs SOFT
latents, which the decoder faithfully decodes as a soft image. Published precedent: YONOS-SR (ECCV 2024, arXiv 2401.17258)
freezes a one-step diffusion U-Net and fine-tunes only the VAE decoder on its outputs; CCSR (2401.00877) fine-tunes the existing
decoder for detail; stabilityai/sd-vae-ft-mse is a decoder-only drop-in fine-tune for faces.

Training pairs: zs = student(photo latent) for every kept v2 pair (+ the properly flipped photo, zi_f from build_flip_cache.py),
target = klein's ORIGINAL 512 px PNG (full detail; flipped in pixel space). Only the decoder trains (init: DreamLite TinyVAE);
the student and the encoder stay frozen, so on the phone only the decoder weights change (same size and speed).
Losses: L1 + LPIPS(vgg) at full 512 px vs klein, optional frozen-DINOv2 critic (as train_dl.py --gan-w; hinge, 448 crops).

  source env.sh
  $VENV/bin/python scripts/train_decoder.py --student dl5g --build-latents-only          # once per student (shared cache)
  $VENV/bin/python scripts/train_decoder.py --student dl5g --tag dec1 [--gan-w 0.1]
  torchrun --nproc_per_node 2 scripts/train_decoder.py ...   # data-parallel: --bs is the GLOBAL batch, split over the GPUs
Out: weights/decoder/<tag>/ (AutoencoderTiny, loadable by DLA-compatible code; student_infer.py --decoder <tag>),
     train_log.jsonl, samples_*.jpg (klein | original decoder | new decoder, full 256 + native 512 face crops), config.json.
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
from train_dl import build_cache_dl  # noqa: E402
from train_student import DEV, TASKS, expr_tasks, to_t  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GEST = ("think", "thumbs", "wave", "ok", "facepalm", "shrug", "thanks")


def chmod_tree(p):  # cross-node readability (dev-duyhv vs dev-duyhv-2, root-squashed NFS)
    for r, ds, fs in os.walk(p):
        for x in ds + fs:
            try:
                os.chmod(os.path.join(r, x), 0o775 if x in ds else 0o664)
            except OSError:
                pass
    try:
        os.chmod(p, 0o775)
    except OSError:
        pass


@torch.inference_mode()
def build_student_latents(student, c, FL, out):
    w = ROOT / "weights/student" / student
    cfg = json.loads((w / "config.json").read_text())
    assert cfg.get("arch") == "dreamlite", "DreamLite students only"
    dt = torch.float16  # = student_infer.py (phone path is fp16)
    if cfg.get("lora_rank", 0) > 0:  # style pack (2026-11-20, comic decoder): base UNet + LoRA, merged (= student_infer.py)
        from peft import PeftModel
        base = DLA.load_unet(ROOT / "weights/student" / cfg["init_from"] / "unet")
        unet = PeftModel.from_pretrained(base, w / "unet_lora").merge_and_unload().to(DEV, dt).eval()
    else:
        unet = DLA.load_unet(w / "unet").to(DEV, dt).eval()
    te = torch.load(w / "task_emb.pt")
    emb, mask = te["emb"].to(DEV, dt), te["mask"].to(DEV)
    MS = cfg.get("ms_sigmas")
    n = len(c["task"])
    res = {}
    t0 = time.time()
    for name, zsrc in (("zs", c["zi"]), ("zs_f", FL["zi_f"])):
        zs = torch.empty(n, 4, DLA.LAT, DLA.LAT, dtype=torch.float16)
        for i in range(0, n, 128):
            tk = c["task"][i:i + 128].to(DEV)
            zs[i:i + 128] = DLA.forward(unet, zsrc[i:i + 128].to(DEV, dt), emb[tk], mask[tk], sigmas=MS).cpu()
            if i % 12800 == 0:
                print(f"[dec-lat] {name} {i}/{n} {time.time() - t0:.0f}s", flush=True)
        res[name] = zs
    torch.save({"key": c["key"], "student": student, "zs": res["zs"], "zs_f": res["zs_f"]}, out)
    os.chmod(out, 0o664)
    print(f"[dec-lat] done {time.time() - t0:.0f}s -> {out}", flush=True)


@torch.no_grad()
def grow_decoder(old, width=0, extra=None, out_ch=3):
    """2b/2a (2026-11-20): copy a TinyVAE into a wider (uniform `width`; DecoderTiny has one width for all stages), deeper
    (`extra` blocks appended per stage) and/or RGBA (out_ch=4) decoder that computes EXACTLY the same RGB at init:
    new channels read old+new channels but old channels never read new ones (zero weights); extra blocks start as identity
    (last conv zero, input is post-ReLU); the alpha output starts at 0.5."""
    from diffusers import AutoencoderTiny
    from diffusers.models.autoencoders.vae import AutoencoderTinyBlock
    cfg = {k: v for k, v in old.config.items() if not k.startswith("_") and k != "block_out_channels"}
    W = width or cfg["decoder_block_out_channels"][0]
    nb = [n + e for n, e in zip(cfg["num_decoder_blocks"], extra or [0] * len(cfg["num_decoder_blocks"]))]
    cfg.update(decoder_block_out_channels=[W] * len(nb), num_decoder_blocks=nb, out_channels=out_ch)
    new = AutoencoderTiny.from_config(cfg)
    new.encoder.load_state_dict(old.encoder.state_dict())
    ol, nl = list(old.decoder.layers), list(new.decoder.layers)
    i = 0
    for n_ in nl:
        o_ = ol[i] if i < len(ol) else None
        if isinstance(n_, AutoencoderTinyBlock) and not isinstance(o_, AutoencoderTinyBlock):  # extra block -> identity
            n_.conv[4].weight.zero_(); n_.conv[4].bias.zero_()
            continue
        for (_, pn), (_, po) in zip(n_.named_parameters(), o_.named_parameters()):
            if pn.dim() == 4:
                O, I = po.shape[:2]
                pn[O:].mul_(0.1)          # new output channels: small random (read everything)
                pn[:O, I:] = 0            # old outputs ignore new inputs
                pn[:O, :I] = po
            else:
                pn.zero_(); pn[:po.shape[0]] = po
        i += 1
    assert i == len(ol), "layer alignment failed"
    if out_ch > old.config.out_channels:  # alpha starts at 0.5 (raw 0.5 -> 0 after x*2-1): off the clamp edge, gradients flow
        nl[-1].bias[old.config.out_channels:] = 0.5
    z = torch.randn(2, 4, 16, 16)
    d = (new.decode(z).sample[:, :3] - old.decode(z).sample).abs().max().item()
    assert d < 1e-4, f"grown decoder is not function-preserving ({d})"
    return new


@torch.no_grad()
def build_alpha(dec, zs, zs_f, idx, out):
    """2a alpha labels: BiRefNet_lite matte (the current phone cut-out model, metrics.Matter) of what the INIT decoder renders
    from the student latent, for idx (unflipped and flipped latents) -> uint8 [2, len(idx), 512, 512]."""
    from metrics import Matter
    mat = Matter()
    A = np.lib.format.open_memmap(str(out) + ".tmp.npy", mode="w+", dtype=np.uint8, shape=(2, len(idx), 512, 512))
    t0 = time.time()
    for f, Z in enumerate((zs, zs_f)):
        for j in range(0, len(idx), 32):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = DLA.decode(dec, Z[idx[j:j + 32]].to(DEV)).float().clamp(-1, 1)[:, :3]
            ims = [Image.fromarray(((v + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()) for v in y]
            for k, m in enumerate(mat.batch(ims)):
                A[f, j + k] = np.round(m * 255).astype(np.uint8)
            if j % 3200 == 0:
                print(f"[dec-alpha] flip={f} {j}/{len(idx)} {time.time() - t0:.0f}s", flush=True)
    A.flush(); del A
    os.replace(str(out) + ".tmp.npy", out)
    os.chmod(out, 0o664)


class Pairs(torch.utils.data.Dataset):  # index j < n: unflipped, j >= n: flipped (student latent of the flipped photo)
    def __init__(self, idx, zs, zs_f, paths, alpha=None, apos=None):
        self.idx, self.zs, self.zs_f, self.paths, self.alpha, self.apos = idx, zs, zs_f, paths, alpha, apos

    def __len__(self):
        return 2 * len(self.idx)

    def __getitem__(self, j):
        f = j >= len(self.idx)
        i = self.idx[j % len(self.idx)]
        t = to_t(Image.open(self.paths[i]))
        z, t = (self.zs_f[i] if f else self.zs[i]), (t.flip(-1) if f else t)
        if self.alpha is None:
            return z, t
        return z, t, torch.from_numpy(np.array(self.alpha[int(f), self.apos[i]]))[None].float() / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--style", default="cartoon3d")
    ap.add_argument("--manifest", default="manifest_v2.jsonl")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--l1-w", type=float, default=1.0)
    ap.add_argument("--lpips-w", type=float, default=1.0)
    ap.add_argument("--gan-w", type=float, default=0.0)
    ap.add_argument("--gan-layers", default="2,5,8,11")
    ap.add_argument("--gan-start", type=int, default=500)
    ap.add_argument("--d-lr", type=float, default=2e-4)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build-latents-only", action="store_true")
    ap.add_argument("--init", default="", help="start from weights/decoder/<tag> (e.g. decg2) instead of DreamLite's TinyVAE")
    ap.add_argument("--width", type=int, default=0, help="2b: uniform decoder width (64 = original)")
    ap.add_argument("--extra-blocks", default="", help="2b: extra blocks per stage, e.g. 1,1,1,1")
    ap.add_argument("--alpha", action="store_true", help="2a: RGBA decoder (4th channel = cut-out alpha, labels = BiRefNet_lite)")
    ap.add_argument("--alpha-n", type=int, default=8000, help="2a: training pairs with alpha labels (random subset)")
    ap.add_argument("--alpha-w", type=float, default=1.0)
    a = ap.parse_args()
    import torch.distributed as dist
    ddp = "LOCAL_RANK" in os.environ
    rank, world = 0, 1
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))   # DEV = "cuda" -> this rank's GPU
    main0 = rank == 0
    if ddp:
        assert not a.build_latents_only, "build latents in a single process first"
    assert a.bs % world == 0
    torch.manual_seed(a.seed + rank)
    random.seed(a.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    tasks = TASKS + expr_tasks()
    d = ROOT / "data/pairs" / a.style
    st = Path(a.manifest).stem

    vae = DLA.load_vae(torch.bfloat16).to(DEV).eval()
    c = build_cache_dl(a.style, vae, a.manifest, tasks)
    FL = torch.load(d / f"latents_dl_{st}_flip.pt")
    assert FL["key"] == c["key"], "flip cache is stale: rerun scripts/build_flip_cache.py"
    lf = ROOT / "weights/decoder" / f"_zs_{a.student}_{st}.pt"
    lf.parent.mkdir(parents=True, exist_ok=True)
    if not (lf.exists() and torch.load(lf, mmap=True).get("key") == c["key"]):
        assert not ddp, f"run --build-latents-only first ({lf})"
        build_student_latents(a.student, c, FL, lf)
    if a.build_latents_only:
        return
    assert a.tag, "--tag required for training"
    Z = torch.load(lf)
    zs, zs_f = Z["zs"], Z["zs_f"]
    rows = [json.loads(l) for l in open(d / a.manifest)]
    rows = [r for r in rows if r["keep"] and r["task"] in tasks]
    for r in rows:
        r.setdefault("target", f"{r['task']}/{r['id']}.png")
    assert [(r["id"], r["task"], r["target"]) for r in rows] == c["key"]
    paths = [d / r["target"] for r in rows]
    n = len(rows)

    val = []  # = train_dl.py split (same held-out pairs as the student)
    for k in range(len(tasks)):
        val += [i for i in range(n) if c["task"][i] == k][:32 if k < 2 else 4]
    vs = set(val)
    tr = [i for i in range(n) if i not in vs]
    del vae, c, FL

    out = ROOT / "weights/decoder" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    dec = DLA.load_vae(torch.float32, ROOT / "weights/decoder" / a.init if a.init else None)
    ex = [int(v) for v in a.extra_blocks.split(",")] if a.extra_blocks else None
    if a.width or ex or a.alpha:
        n_old = sum(p.numel() for p in dec.decoder.parameters())
        dec = grow_decoder(dec, a.width, ex, 4 if a.alpha else 3)
        if main0:
            print(f"[dec] grown decoder: width {a.width or 'same'}, extra blocks {ex}, out {4 if a.alpha else 3} ch; "
                  f"params {n_old / 1e6:.2f}M -> {sum(p.numel() for p in dec.decoder.parameters()) / 1e6:.2f}M", flush=True)
    dec = dec.to(DEV)
    AL = apos = None
    if a.alpha:  # labels for a random training subset + all val rows, rendered by the init decoder
        rng = random.Random(1)
        aidx = sorted(rng.sample(tr, min(a.alpha_n, len(tr)))) + val
        af = ROOT / "weights/decoder" / f"_alpha_{a.student}_{a.init or 'dl'}_{st}_{a.alpha_n}.npy"
        if not af.exists():
            assert not ddp, "build alpha labels single-process first"
            d_init = DLA.load_vae(torch.bfloat16, ROOT / "weights/decoder" / a.init if a.init else None).to(DEV).eval()
            build_alpha(d_init, zs, zs_f, aidx, af)
            del d_init
        AL = np.load(af, mmap_mode="r")
        assert AL.shape[1] == len(aidx)
        apos = {i: k for k, i in enumerate(aidx)}
        tr = aidx[:len(aidx) - len(val)]
        if main0:
            print(f"[dec] alpha: {len(tr)} train pairs with labels ({af.name})", flush=True)
    dec.encoder.requires_grad_(False)
    dec.decoder.train()
    dec0 = DLA.load_vae(torch.bfloat16).to(DEV).eval().requires_grad_(False)  # original decoder (baseline in val/samples)
    npar = sum(p.numel() for p in dec.decoder.parameters())
    dmod = dec.decoder  # raw module (eval / save); training forward goes through DDP when ddp
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        dec.decoder = DDP(dmod, device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
    if main0:
        print(f"[dec] student={a.student} pairs={n} train={len(tr)}x2 (flip) val={len(val)} decoder params={npar / 1e6:.2f}M "
              f"gpus={world} global bs={a.bs}", flush=True)
    lp = lpips.LPIPS(net="vgg", verbose=False).to(DEV).eval().requires_grad_(False)
    opt = torch.optim.AdamW(dmod.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))

    if a.gan_w > 0:  # same critic as train_dl.py --gan-w
        from transformers import AutoModel
        from metrics import _snap
        dino = AutoModel.from_pretrained(str(_snap("facebook/dinov2-base"))).to(DEV).eval().requires_grad_(False)
        gl = [int(x) for x in a.gan_layers.split(",")]
        SN = nn.utils.spectral_norm
        heads = nn.ModuleList([nn.Sequential(SN(nn.Conv2d(768, 256, 3, padding=1)), nn.LeakyReLU(0.2),
                                             SN(nn.Conv2d(256, 1, 1))) for _ in gl]).to(DEV)
        d_opt = torch.optim.Adam(heads.parameters(), lr=a.d_lr, betas=(0.0, 0.99))
        im_mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
        im_std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
        if ddp:  # identical initial critic heads on every rank (grads are all-reduced by hand below)
            for q in heads.parameters():
                dist.broadcast(q.data, 0)
        if main0:
            print(f"[dec] critic: frozen DINOv2-base layers {gl}, gan_w={a.gan_w} from step {a.gan_start}", flush=True)

        def critic(x):
            i, j = random.randint(0, 64), random.randint(0, 64)
            x = (((x[..., i:i + 448, j:j + 448] + 1) / 2) - im_mean) / im_std
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hs = dino(pixel_values=x, output_hidden_states=True).hidden_states
            o = []
            for h, l in zip(heads, gl):
                t = hs[l + 1][:, 1:].float().transpose(1, 2).reshape(len(x), 768, 32, 32)
                o.append(h(t).flatten(1))
            return torch.cat(o, 1)

    def decode(m, z):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return DLA.decode(m, z.to(DEV)).float().clamp(-1, 1)

    vz = zs[val].float()
    vt = torch.stack([to_t(Image.open(paths[i])) for i in val])
    va = torch.stack([torch.from_numpy(np.array(AL[0, apos[i]]))[None].float() / 255 for i in val]) if a.alpha else None

    @torch.no_grad()
    def evaluate(step):
        wrapped = dec.decoder
        dec.decoder = dmod
        dec.decoder.eval()
        l1s, lps, l1b, lpb, amae, aiou = [], [], [], [], [], []
        for i in range(0, len(val), 16):
            z, t = vz[i:i + 16], vt[i:i + 16].to(DEV)
            y4, y0 = decode(dec, z), decode(dec0, z)
            y = y4[:, :3]
            if a.alpha:
                al, at = (y4[:, 3:4] + 1) / 2, va[i:i + 16].to(DEV)
                amae.append((al - at).abs().mean().item())
                inter = ((al > 0.5) & (at > 0.5)).flatten(1).sum(1).float(); uni = ((al > 0.5) | (at > 0.5)).flatten(1).sum(1).float()
                aiou.append((inter / uni.clamp_min(1)).mean().item())
            l1s.append((y - t).abs().mean().item()); lps.append(lp(y, t).mean().item())
            l1b.append((y0 - t).abs().mean().item()); lpb.append(lp(y0, t).mean().item())
        sel = list(range(3)) + list(range(32, 35)) + [j for j in range(64, len(val), 4)][:4]  # 3 avatars, 3 stickers, 4 expressions
        z, t = vz[sel], vt[sel].to(DEV)
        y4, y0 = decode(dec, z), decode(dec0, z)
        y = y4[:, :3]
        t8 = lambda x: ((F.interpolate(x, size=256, mode="area").clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
        c8 = lambda x: t8(x[..., 112:368, 128:384])
        cols = [t8(t), t8(y), c8(t), c8(y0), c8(y)]  # klein | new | klein 1:1 | orig dec 1:1 | new dec 1:1 (| alpha | alpha cut-out on grey)
        if a.alpha:
            al = (y4[:, 3:4] + 1) / 2
            cols += [t8(al.repeat(1, 3, 1, 1) * 2 - 1), t8(y * al + 0.2 * (1 - al))]
        rows_ = np.concatenate(cols, 2)
        Image.fromarray(np.concatenate(list(rows_), 0)).save(out / f"samples_{step:05d}.jpg", quality=88)
        dec.decoder.train()
        dec.decoder = wrapped
        r_ = {"val_alpha_mae": round(float(np.mean(amae)), 4), "val_alpha_iou": round(float(np.mean(aiou)), 4)} if a.alpha else {}
        return r_ | {"val_l1": round(sum(l1s) / len(l1s), 4), "val_lpips": round(sum(lps) / len(lps), 4),
                "base_l1": round(sum(l1b) / len(l1b), 4), "base_lpips": round(sum(lpb) / len(lpb), 4)}

    def save(step):
        wrapped = dec.decoder
        dec.decoder = dmod
        dec.save_pretrained(out)
        dec.decoder = wrapped
        (out / "config.json.train").write_text(json.dumps(vars(a) | {"step": step, "init": f"{DLA.DL_REPO}@{DLA.DL_REV} vae",
                                                                       "decoder_params_M": round(npar / 1e6, 3), "train_pairs": len(tr)}, indent=1))
        chmod_tree(out)

    if main0:
        log = open(out / "train_log.jsonl", "w")
        r0 = {"step": 0} | evaluate(0)
        print(json.dumps(r0), flush=True)
        log.write(json.dumps(r0) + "\n")
    if ddp:
        dist.barrier()
    ds = Pairs(tr, zs, zs_f, paths, AL, apos)
    sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=True, seed=a.seed, drop_last=True) if ddp else None
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs // world, shuffle=sampler is None, sampler=sampler, drop_last=True,
                                     num_workers=a.workers, pin_memory=True, persistent_workers=True)
    epoch = 0
    it = iter(dl)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        try:
            b_ = next(it)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            it = iter(dl)
            b_ = next(it)
        z, t = b_[0], b_[1].to(DEV, non_blocking=True)
        y4 = decode(dec, z)
        y = y4[:, :3]
        l1 = (y - t).abs().mean()
        lpl = lp(y, t).mean()
        loss = a.l1_w * l1 + a.lpips_w * lpl
        r = {}
        if a.alpha:
            at = b_[2].to(DEV, non_blocking=True)
            al = ((y4[:, 3:4] + 1) / 2).clamp(1e-4, 1 - 1e-4)
            r["alpha"] = (al - at).abs().mean() + 0.5 * F.binary_cross_entropy(al, at)
            loss = loss + a.alpha_w * r["alpha"]
        if a.gan_w > 0 and step > a.gan_start:
            heads.requires_grad_(False)
            r["g_gan"] = -critic(y).mean()
            heads.requires_grad_(True)
            loss = loss + a.gan_w * r["g_gan"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dmod.parameters(), 1.0)
        opt.step()
        sched.step()
        if a.gan_w > 0:
            l_d = F.relu(1 - critic(t)).mean() + F.relu(1 + critic(y.detach())).mean()
            d_opt.zero_grad(set_to_none=True)
            l_d.backward()
            if ddp:
                for q in heads.parameters():
                    if q.grad is not None:
                        dist.all_reduce(q.grad)
                        q.grad /= world
            d_opt.step()
            r["d_loss"] = l_d.detach()
        if step % 50 == 0 and main0:
            rr = {"step": step, "l1": round(l1.item(), 4), "lpips": round(lpl.item(), 4), "lr": sched.get_last_lr()[0],
                  "s": round(time.time() - t0, 1)} | {k: round(v.item(), 4) for k, v in r.items()}
            if step % a.val_every == 0 or step == a.steps:
                rr |= evaluate(step)
                save(step)
                rr["gpus"] = world
            print(json.dumps(rr), flush=True)
            log.write(json.dumps(rr) + "\n")
            log.flush()
        if ddp and (step % a.val_every == 0 or step == a.steps):
            dist.barrier()  # ranks 1.. wait while rank 0 evaluates / saves
        bad = torch.tensor(float(not math.isfinite(loss.item())), device=DEV)
        if ddp:
            dist.all_reduce(bad)
        if bad.item() > 0:
            if main0:
                print("[dec] non-finite loss, abort", flush=True)
            break
    if main0:
        save(a.steps)
        print(f"[dec] done {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
