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
import torch.nn as nn
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
    ap.add_argument("--shared-trunk", action="store_true", help="idea B: down/mid cross-attention reads ONE learned context "
                    "(init = avatar token) instead of the task token, so the down path + mid block run once per photo on the phone")
    ap.add_argument("--shared-scope", default="down_mid", choices=["down_mid", "down"], help="B: which blocks are shared")
    ap.add_argument("--refine-sigma", type=float, default=0.0, help="3a (2026-11-20): >0 (needs --lora-rank, --init-from) = train a "
                    "SECOND-step LoRA: step 1 = the init student with the adapter off (unchanged), x = (1-s)*x0 + s*noise(seed 1235), "
                    "step 2 = UNet + adapter at sigma s. Tokens frozen. Inference: student_infer.py (2 steps for --refine-tasks)")
    ap.add_argument("--lora-scope", default="all", choices=["all", "mid_up0"], help="#4 (2026-11-20): mid_up0 = LoRA only in "
                    "mid_block + up_blocks.0 (where the influence map says the option acts) -> smaller style packs")
    ap.add_argument("--prune", default="", help="Plan A: name of a set in configs/prune_sets.json (S70, S60): residual units removed; "
                    "saved as prune.json")
    ap.add_argument("--drop-attn2", default="", help="#5: comma list of groups whose cross-attention is removed (down,up1 ...); "
                    "saved as drop_attn2.json")
    ap.add_argument("--tokens-only", action="store_true", help="#4: freeze the whole UNet, train only the option tokens (smallest pack)")
    ap.add_argument("--only-tasks", default="", help="comma list: train only on these options (e.g. avatar); val is unchanged")
    ap.add_argument("--half-head", type=int, default=-1, help="B part 2: >=0 = from up_blocks.1 on compute only the output half of "
                    "the diptych + this many columns (dl_arch.install_half_head); saved as half_head.json")
    ap.add_argument("--train-decoder", default=None, help="idea A: train this TinyVAE (weights/decoder/<tag>) jointly with the UNet "
                    "on the pixel losses; saved to weights/decoder/<run tag>")
    ap.add_argument("--wide-latent", type=int, default=0, help="idea A: UNet output / decoder input channels (e.g. 16); extra "
                    "channels start so the output equals the init (UNet extra conv_out small random, decoder extra input weights 0)")
    ap.add_argument("--dec-lr", type=float, default=1e-4)
    ap.add_argument("--pix-l1-w", type=float, default=0.0, help="L1 on the full 512 px output vs the klein target")
    ap.add_argument("--drop-list", default=None, help="json list of [id, task] training pairs to leave out (e.g. data/pairs/<style>/"
                    "drop_skin_lighten.json from scripts/make_skin_drop.py); caches and val split unchanged")
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
    ap.add_argument("--ffl-w", type=float, default=0.0, help="Focal Frequency Loss (Jiang et al., ICCV 2021) on the full 512 px output vs "
                    "the TinyVAE decode of the target: squared spectrum distance (fft2, norm=ortho, per RGB channel) weighted by the "
                    "detached, per-image max-normalised error magnitude ^ --ffl-alpha (2026-11-19, user-approved follow-up to dl3)")
    ap.add_argument("--ffl-alpha", type=float, default=1.0)
    ap.add_argument("--ffl-log", type=int, default=1, help="1 = FFL's log_matrix option, w = log(1 + w): flattens the weight range so "
                    "high-frequency errors (small in absolute terms) are not drowned by low-frequency ones")
    ap.add_argument("--gan-w", type=float, default=0.0, help="vision-aided critic (pix2pix-turbo / Kumari et al. 2022 / ADD style): frozen "
                    "DINOv2-base, trainable spectral-norm conv heads on patch tokens of --gan-layers, hinge loss; real = TinyVAE decode of "
                    "the klein target latent, fake = student 512 px output; random 448 px crops (translation aug). 0 = off")
    ap.add_argument("--gan-layers", default="2,5,8,11")
    ap.add_argument("--gan-start", type=int, default=200, help="critic-only warm-up steps before the generator term is used")
    ap.add_argument("--d-lr", type=float, default=2e-4)
    ap.add_argument("--proper-flip", action="store_true", help="flip augmentation from latents of the FLIPPED IMAGES "
                    "(build_flip_cache.py). Default (off) flips the latents, which the TinyVAE does not commute with: "
                    "decode(flip z) vs flip(PNG) 22.6 dB vs 33.4 dB round trip (2026-11-19, scripts/flip_aug_check.py)")
    ap.add_argument("--tgt-cache", action="store_true", help="read the decoded 512 px target (TinyVAE decode(zt), uint8) "
                    "from tgt512_dl_<manifest>.u8 instead of decoding it every step (speed); flipped samples use its pixel flip")
    ap.add_argument("--compile", action="store_true", help="torch.compile the UNet (speed)")
    ap.add_argument("--ddp-find-unused", action="store_true", help="torchrun/DDP: find_unused_parameters=True")
    ap.add_argument("--bench", type=int, default=0, help=">0: time this many steps (first 10 = warm-up), print s/step, exit, no save")
    a = ap.parse_args()
    assert a.lora_rank == 0 or a.init_from, "--lora-rank needs --init-from"
    # data-parallel (torchrun --nproc_per_node N, 2026-11-19): --bs is the GLOBAL batch, split over the GPUs; val/samples/save on rank 0
    import os
    import torch.distributed as dist
    ddp = "LOCAL_RANK" in os.environ
    rank, world = 0, 1
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        assert not a.bench and a.bs % world == 0
    main0 = rank == 0
    torch.manual_seed(a.seed + rank)
    random.seed(a.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = ROOT / "weights/student" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    tasks = TASKS + (expr_tasks() if a.expr else [])

    vae = DLA.load_vae(torch.bfloat16).to(DEV).eval().requires_grad_(False)
    c = build_cache_dl(a.style, vae, a.manifest, tasks)
    n = len(c["task"])
    st = Path(a.manifest).stem
    if a.proper_flip:
        FL = torch.load(ROOT / "data/pairs" / a.style / f"latents_dl_{st}_flip.pt")
        assert FL["key"] == c["key"], "flip cache is stale: rerun scripts/build_flip_cache.py"
        print(f"[dl] proper flip: latents of flipped images ({len(FL['zt_f'])})", flush=True)
    if a.tgt_cache:
        TG = np.load(ROOT / "data/pairs" / a.style / f"tgt512_dl_{st}.u8", mmap_mode="r")
        assert TG.shape[0] == n, f"tgt512 cache rows {TG.shape[0]} != {n}: rerun scripts/build_flip_cache.py"
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
        if a.lora_scope == "mid_up0":
            tm = r"^(mid_block|up_blocks\.0)\." + tm
        unet = get_peft_model(unet, LoraConfig(r=a.lora_rank, lora_alpha=a.lora_rank, target_modules=tm, init_lora_weights="gaussian"))
        print(f"[dl] LoRA r={a.lora_rank}: {sum(p.numel() for p in unet.parameters() if p.requires_grad) / 1e6:.1f}M trainable", flush=True)
    if a.shared_trunk:
        assert not hasattr(unet, "shared_emb"), "init already has a shared trunk"
        print(f"[dl] shared trunk (B): {DLA.install_shared_trunk(unet, emb0[0:1].float(), mask[0:1], trainable=True, scope=a.shared_scope)} {a.shared_scope} "
              f"cross-attention modules read one learned context (init = '{tasks[0]}' token)", flush=True)
    if a.prune:
        assert not hasattr(unet, "_prune"), "init is already pruned"
        n0 = sum(p.numel() for p in unet.parameters())
        DLA.install_prune(unet, json.load(open(ROOT / "configs/prune_sets.json"))[a.prune])
        print(f"[dl] prune {a.prune}: {len(unet._prune)} units removed, params {n0 / 1e6:.0f}M -> "
              f"{sum(p.numel() for p in unet.parameters()) / 1e6:.0f}M", flush=True)
    if a.drop_attn2:
        assert not hasattr(unet, "_drop_attn2"), "init already drops cross-attention"
        print(f"[dl] drop attn2 in {a.drop_attn2}: {DLA.install_drop_attn2(unet, a.drop_attn2.split(','))} modules removed", flush=True)
    if a.tokens_only:
        assert a.lora_rank == 0, "--tokens-only: no LoRA"
        unet.requires_grad_(False)
        print("[dl] tokens only: UNet frozen", flush=True)
    if a.half_head >= 0:
        assert not hasattr(unet, "_half_head"), "init already has a half head"
        DLA.install_half_head(unet, a.half_head)
        print(f"[dl] half head: up_blocks.1+ compute the output half + {a.half_head} columns", flush=True)
    dvae = None
    if a.train_decoder:
        assert not ddp, "joint decoder training is single-GPU"
        dvae = DLA.load_vae(torch.float32, ROOT / "weights/decoder" / a.train_decoder)
        if a.wide_latent > 4:
            from diffusers import AutoencoderTiny
            N, co = a.wide_latent, unet.conv_out
            new = nn.Conv2d(co.in_channels, N, co.kernel_size, padding=co.padding)
            with torch.no_grad():
                new.weight.zero_(); new.bias.zero_()
                new.weight[:co.out_channels] = co.weight; new.bias[:co.out_channels] = co.bias
                new.weight[co.out_channels:] = torch.randn_like(new.weight[co.out_channels:]) * co.weight.std() * 0.1
            unet.conv_out = new
            unet.register_to_config(out_channels=N)
            wv = AutoencoderTiny.from_config({**dvae.config, "latent_channels": N})
            sd_old, sd_new = dvae.state_dict(), wv.state_dict()
            with torch.no_grad():
                for k_, v_ in sd_new.items():
                    o_ = sd_old[k_]
                    if o_.shape == v_.shape:
                        v_.copy_(o_)
                    else:  # decoder first conv [64, 4, 3, 3] -> [64, N, ...]: extra inputs 0; encoder last conv [4, ...] -> [N, ...]: extra outputs 0
                        v_.zero_()
                        v_[tuple(slice(0, d) for d in o_.shape)] = o_
                        print(f"[dl] wide latent: {k_} {tuple(o_.shape)} -> {tuple(v_.shape)}", flush=True)
            dvae = wv
        dvae = dvae.to(DEV).train()
        dvae.encoder.requires_grad_(False)
        print(f"[dl] joint decoder (A): init {a.train_decoder}, latent channels {dvae.config.latent_channels}, "
              f"{sum(p.numel() for p in dvae.decoder.parameters()) / 1e6:.2f}M decoder params, lr {a.dec_lr}", flush=True)
    DV = dvae if dvae is not None else vae
    unet = unet.to(DEV).train()
    unet_raw = unet
    if a.grad_ckpt:
        unet_raw.enable_gradient_checkpointing()
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        unet = DDP(unet_raw, device_ids=[torch.cuda.current_device()], broadcast_buffers=False,
                   find_unused_parameters=a.ddp_find_unused)
    if a.compile:
        unet = torch.compile(unet)
    unet_eval = unet_raw if ddp else unet   # rank-0-only eval must not go through DDP collectives
    NET = {"m": unet}
    lp = lpips.LPIPS(net="vgg", verbose=False).to(DEV).eval().requires_grad_(False)

    val = []
    for k in range(len(tasks)):
        val += [i for i in range(n) if c["task"][i] == k][:32 if k < 2 else 4]
    vs = set(val)
    tr = [i for i in range(n) if i not in vs]
    if a.drop_list:
        dl_ = {tuple(x) for x in json.load(open(ROOT / a.drop_list))}
        n0 = len(tr)
        tr = [i for i in tr if tuple(c["ids"][i]) not in dl_]
        print(f"[dl] drop-list {a.drop_list}: {n0 - len(tr)} training pairs left out ({len(dl_)} listed)", flush=True)
    print(f"[dl] pairs={n} train={len(tr)} val={len(val)} tokens L={emb.shape[1]} "
          f"unet={sum(p.numel() for p in unet_raw.parameters()) / 1e6:.0f}M gpus={world} rank={rank}", flush=True)
    if a.only_tasks:
        keep = {tasks.index(t) for t in a.only_tasks.split(",")}
        tr = [i for i in tr if int(c["task"][i]) in keep]
        print(f"[dl] only tasks {a.only_tasks}: {len(tr)} training pairs", flush=True)
    if a.refine_sigma > 0:
        assert a.lora_rank > 0 and a.init_from, "--refine-sigma needs --lora-rank and --init-from"
        emb.requires_grad_(False)  # step 1 must stay the init student
        print(f"[dl] refiner (3a): step 2 at sigma {a.refine_sigma}, LoRA r{a.lora_rank}; step 1 = {a.init_from} (adapter off), tokens frozen", flush=True)
    tok_params = [emb] + ([unet_raw.shared_emb] if a.shared_trunk else [])
    groups = [{"params": [p for n_, p in unet_raw.named_parameters() if p.requires_grad and n_ != "shared_emb"], "lr": a.lr},
              {"params": tok_params, "lr": a.emb_lr, "weight_decay": 0}]
    if dvae is not None:
        groups.append({"params": list(dvae.decoder.parameters()), "lr": a.dec_lr, "weight_decay": 0})
    groups = [g for g in groups if len(g["params"])]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    log = open(out / "train_log.jsonl", "w") if main0 else None

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

    def eye_losses(dec512, tgt, idx, flip):
        e, sg = eye_geom(idx, flip)
        ok = sg > 0
        out = {}
        if ok.sum() == 0:
            return out
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

    def lp512_loss(dec512, tgt):  # one random face-area crop per sample, same box on output and target, native 512 px
        from torchvision.ops import roi_align
        k = a.lp512_crop
        cx = torch.randint(48, 512 - 48 - k + 1, (len(tgt),)).float()
        cy = torch.randint(32, 512 - 64 - k + 1, (len(tgt),)).float()
        bx = torch.stack([torch.arange(len(tgt)).float(), cx, cy, cx + k, cy + k], 1).to(DEV)
        return lp(roi_align(dec512, bx, k, aligned=True), roi_align(tgt, bx, k, aligned=True)).mean()

    def ffl_loss(x, y):  # Focal Frequency Loss, patch_factor 1, batch_matrix False (paper defaults), optional log matrix
        d = torch.fft.fft2(x.float(), norm="ortho") - torch.fft.fft2(y.float(), norm="ortho")
        dist = d.real ** 2 + d.imag ** 2
        with torch.no_grad():
            w = dist.sqrt() ** a.ffl_alpha
            if a.ffl_log:
                w = torch.log1p(w)
            w = w / w.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        return (w * dist).mean()

    # ---------- vision-aided critic: frozen DINOv2-base + trainable heads (only with --gan-w) ----------
    if a.gan_w > 0:
        from transformers import AutoModel
        from metrics import _snap
        dino = AutoModel.from_pretrained(str(_snap("facebook/dinov2-base"))).to(DEV).eval().requires_grad_(False)
        gl = [int(x) for x in a.gan_layers.split(",")]
        SN = nn.utils.spectral_norm
        heads = nn.ModuleList([nn.Sequential(SN(nn.Conv2d(768, 256, 3, padding=1)), nn.LeakyReLU(0.2),
                                             SN(nn.Conv2d(256, 1, 1))) for _ in gl]).to(DEV)
        if ddp:  # identical critic heads on every rank; their grads are all-reduced by hand
            for q in heads.parameters():
                dist.broadcast(q.data, 0)
        d_opt = torch.optim.Adam(heads.parameters(), lr=a.d_lr, betas=(0.0, 0.99))
        im_mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
        im_std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
        print(f"[dl] critic: frozen DINOv2-base layers {gl}, heads {sum(p.numel() for p in heads.parameters()) / 1e6:.1f}M, "
              f"gan_w={a.gan_w} from step {a.gan_start}", flush=True)

        def critic(x):  # [-1, 1] 512 px -> per-patch logits [B, n_layers * 32 * 32]
            i, j = random.randint(0, 64), random.randint(0, 64)
            x = (((x[..., i:i + 448, j:j + 448] + 1) / 2) - im_mean) / im_std
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hs = dino(pixel_values=x, output_hidden_states=True).hidden_states
            out = []
            for h, l in zip(heads, gl):
                t = hs[l + 1][:, 1:].float()                                   # hidden_states[0] = embeddings; drop CLS
                t = t.transpose(1, 2).reshape(len(x), 768, 32, 32)
                out.append(h(t).flatten(1))
            return torch.cat(out, 1)

    def batch(idx, aug):
        zi, zt = c["zi"][idx].to(DEV), c["zt"][idx].to(DEV)
        im = c["img256"][idx].to(DEV).float() / 127.5 - 1
        tk = c["task"][idx].to(DEV)
        batch.idx = idx
        if aug:
            flip = torch.rand(len(idx), device=DEV) < 0.5
            f4 = flip.view(-1, 1, 1, 1)
            if a.proper_flip:  # latents of the flipped images; the 256 px target flips in pixel space
                zi = torch.where(f4, FL["zi_f"][idx].to(DEV), zi)
                zt = torch.where(f4, FL["zt_f"][idx].to(DEV), zt)
                im = torch.where(f4, im.flip(-1), im)
            else:
                zi, zt, im = (torch.where(f4, x.flip(-1), x) for x in (zi, zt, im))
            batch.flip = flip
        else:
            batch.flip = None
        return zi, zt, im, tk

    def losses(zi, zt, im, tk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if a.refine_sigma > 0:
                zp = DLA.refine_forward(NET["m"], unet_raw, zi, emb[tk], mask[tk], a.refine_sigma)
            else:
                zp = DLA.forward(NET["m"], zi, emb[tk], mask[tk])
            mse = F.mse_loss(zp[:, :zt.shape[1]].float(), zt)
            dec512 = DLA.decode(DV, zp.to(torch.bfloat16)).float().clamp(-1, 1)
            dec = F.interpolate(dec512, size=256, mode="bilinear", antialias=True)
            lpl = lp(dec, im).mean()
        losses.dec512 = dec512
        losses.tgt = None
        if a.lp512_w > 0 or a.ffl_w > 0 or a.gan_w > 0 or a.eye_w > 0 or a.eye_lpips_w > 0 or a.pix_l1_w > 0:
            with torch.no_grad():
                if a.tgt_cache:
                    t_ = torch.from_numpy(TG[batch.idx.numpy()]).to(DEV).float() / 127.5 - 1
                    if batch.flip is not None:
                        t_ = torch.where(batch.flip.view(-1, 1, 1, 1), t_.flip(-1), t_)
                    losses.tgt = t_
                else:
                    losses.tgt = DLA.decode(vae, zt.to(torch.bfloat16)).float().clamp(-1, 1)
        losses.lp512 = lp512_loss(dec512, losses.tgt) if a.lp512_w > 0 else None
        losses.ffl = ffl_loss(dec512, losses.tgt) if a.ffl_w > 0 else None
        losses.pix = F.l1_loss(dec512, losses.tgt) if a.pix_l1_w > 0 else None
        return mse, lpl, dec

    def samples(step):
        gi = [i for g in GEST if f"expr_{g}" in tasks for i in [j for j in val if c["task"][j] == tasks.index(f"expr_{g}")][:1]]
        vi = torch.tensor(val[:3] + val[32:35] + gi)
        unet_raw.eval()
        NET["m"] = unet_eval
        with torch.no_grad():
            zi, zt, im, tk = batch(vi, False)
            _, _, dec = losses(zi, zt, im, tk)
            src = DLA.decode(vae, zi.to(torch.bfloat16)).float().clamp(-1, 1)
            d512 = losses.dec512
            t512 = DLA.decode(vae, zt.to(torch.bfloat16)).float().clamp(-1, 1)
        unet_raw.train()
        NET["m"] = unet
        t8 = lambda x: ((F.interpolate(x, size=256, mode="area").clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
        c8 = lambda x: t8(x[..., 112:368, 128:384])  # native 512 px face-area crop (1:1 pixels; shows the detail the 256 tiles hide)
        rows = np.concatenate([t8(src), t8(im), t8(dec), c8(t512), c8(d512)], 2)  # photo | klein | student | klein 1:1 | student 1:1
        Image.fromarray(np.concatenate(list(rows), 0)).save(out / f"samples_{step:05d}.jpg", quality=88)

    if not a.bench and main0:
        samples(0)
    if ddp:
        dist.barrier()
    t0 = time.time()
    for step in range(1, (a.bench or a.steps) + 1):
        if a.bench and step == 11:
            torch.cuda.synchronize()
            tb = time.time()
        idx = torch.tensor(random.sample(tr, a.bs // world))
        zi, zt, im, tk = batch(idx, True)
        mse, lpl, _ = losses(zi, zt, im, tk)
        loss = mse + a.lpips_w * lpl
        if losses.lp512 is not None:
            loss = loss + a.lp512_w * losses.lp512
        el = eye_losses(losses.dec512, losses.tgt, idx, batch.flip) if (a.eye_w > 0 or a.eye_lpips_w > 0) else {}
        if "eye" in el:
            loss = loss + a.eye_w * el["eye"]
        if "eye_lpips" in el:
            loss = loss + a.eye_lpips_w * el["eye_lpips"]
        if losses.pix is not None:
            el["pix_l1"] = losses.pix
            loss = loss + a.pix_l1_w * losses.pix
        if losses.ffl is not None:
            el["ffl"] = losses.ffl
            loss = loss + a.ffl_w * losses.ffl
        if a.gan_w > 0 and step > a.gan_start:  # hinge generator term through the frozen DINOv2 (heads not updated here)
            heads.requires_grad_(False)
            el["g_gan"] = -critic(losses.dec512).mean()
            heads.requires_grad_(True)
            loss = loss + a.gan_w * el["g_gan"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if ddp and emb.grad is not None:  # task tokens live outside the DDP module
            dist.all_reduce(emb.grad)
            emb.grad /= world
        torch.nn.utils.clip_grad_norm_([p for p in unet_raw.parameters() if p.requires_grad] + [emb], 1.0)
        if dvae is not None:
            torch.nn.utils.clip_grad_norm_(dvae.decoder.parameters(), 1.0)
        opt.step()
        sched.step()
        if a.gan_w > 0:  # one critic update per step: klein target (decoded) = real, student output = fake
            l_d = F.relu(1 - critic(losses.tgt)).mean() + F.relu(1 + critic(losses.dec512.detach())).mean()
            d_opt.zero_grad(set_to_none=True)
            l_d.backward()
            if ddp:
                for q in heads.parameters():
                    if q.grad is not None:
                        dist.all_reduce(q.grad)
                        q.grad /= world
            d_opt.step()
            el["d_loss"] = l_d.detach()
        if step % 50 == 0 and main0:
            r = {"step": step, "mse": round(mse.item(), 5), "lpips": round(lpl.item(), 4), "lr": sched.get_last_lr()[0],
                 "s": round(time.time() - t0, 1)} | {k: round(v.item(), 4) for k, v in el.items()} \
                | ({"lp512": round(losses.lp512.item(), 4)} if losses.lp512 is not None else {})
            if step % a.val_every == 0 or step == a.steps:
                unet_raw.eval()
                NET["m"] = unet_eval
                with torch.no_grad():
                    vm, vl = [], []
                    for i in range(0, len(val), 16):
                        m_, l_, _ = losses(*batch(torch.tensor(val[i:i + 16]), False))
                        vm.append(m_.item())
                        vl.append(l_.item())
                unet_raw.train()
                NET["m"] = unet
                r |= {"val_mse": round(sum(vm) / len(vm), 5), "val_lpips": round(sum(vl) / len(vl), 4)}
                samples(step)
            print(json.dumps(r), flush=True)
            log.write(json.dumps(r) + "\n")
            log.flush()
        if ddp and step % 50 == 0 and (step % a.val_every == 0 or step == a.steps):
            dist.barrier()  # other ranks wait while rank 0 validates
        bad = torch.tensor(float(not math.isfinite(loss.item())), device=DEV)
        if ddp:
            dist.all_reduce(bad)
        if bad.item() > 0:
            if main0:
                print("[dl] non-finite loss, abort", flush=True)
            break
    if a.bench:
        torch.cuda.synchronize()
        print(f"BENCH {(time.time() - tb) / (a.bench - 10):.3f} s/step  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB  "
              f"flags tgt_cache={a.tgt_cache} compile={a.compile} lp512={a.lp512_w} eye={a.eye_w}/{a.eye_lpips_w}", flush=True)
        return
    if not main0:
        dist.barrier()
        dist.destroy_process_group()
        return
    unet_raw.save_pretrained(out / ("unet_lora" if a.lora_rank > 0 else "unet"))
    torch.save({"emb": emb.detach().cpu(), "mask": mask.cpu(), "tasks": tasks}, out / "task_emb.pt")
    if a.shared_trunk:
        torch.save({"emb": unet_raw.shared_emb.detach().cpu(), "mask": unet_raw.shared_mask.cpu(), "scope": a.shared_scope}, out / "shared_ctx.pt")
    elif hasattr(unet_raw, "shared_emb"):  # init had a (frozen) shared trunk: keep it
        torch.save({"emb": unet_raw.shared_emb.detach().cpu(), "mask": unet_raw.shared_mask.cpu(), "scope": unet_raw._shared_scope}, out / "shared_ctx.pt")
    if hasattr(unet_raw, "_prune"):
        (out / "prune.json").write_text(json.dumps({"units": unet_raw._prune}))
    if hasattr(unet_raw, "_drop_attn2"):
        (out / "drop_attn2.json").write_text(json.dumps({"groups": unet_raw._drop_attn2}))
    if hasattr(unet_raw, "_half_head"):
        (out / "half_head.json").write_text(json.dumps({"margin": unet_raw._half_head["margin"], "start": unet_raw._half_head["start"]}))
    if dvae is not None:
        dvae.save_pretrained(ROOT / "weights/decoder" / a.tag)
        os.system(f"chmod -R a+rwX '{ROOT / 'weights/decoder' / a.tag}'")
    (out / "config.json").write_text(json.dumps(vars(a) | {"arch": "dreamlite", "init": a.init_from or f"{DLA.DL_REPO}@{DLA.DL_REV}", "param": "flow_x0",
                                                           "T": 1000, "decoder": "DreamLite TinyVAE", "train_pairs": len(tr),
                                                           "minutes": round((time.time() - t0) / 60, 1)}, indent=1))
    print(f"[dl] done {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)
    os.system(f"chmod -R a+rwX '{out}'")  # cross-node readability (root-squashed NFS)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
