"""tSP0b version (2026-11-27, after tSP0's swap test STOP: pose ignored, pconv |w| 10x below conv_in): a MULTI-LEVEL pose adapter
(pose_student.PoseAdapter: features added to the inputs of down_blocks 0/1/2, zero-init outputs, --pose-lr), --pose-scale (map x scale),
DENSE maps (s26), and --gesture-frac of each batch drawn from rows that carry a pose map. Saved: <out>/pose_adapter.pt.

Earlier docstring (tSP0, conv-in variant):
COPY of train_dl.py + POSE CONDITIONING (2026-11-27, user "yes go with the 2k step check"; hand pose of stickers).
--pose-maps data/pairs/<style>/pose_maps_<pairs>.pt (scripts/s23_pose_maps.py): 3-channel 64 x 64 maps (L hand, R hand, upper arms) of the
TEACHER TARGET of each gesture row and view (plain / flipped photo); all other rows get an all-zero map. The UNet's conv_in is wrapped:
conv_in(x) + pconv(pose) with pconv = Conv2d(3, C, 3) ZERO-initialised (step 0 = the init student exactly); the pose map sits on the
left (canvas) half of the [canvas | photo] diptych, zeros on the photo half. pconv trains at --pose-lr (higher than --lr, new weights).
Saved separately: <out>/pose_conv.pt (the unet dir stays a plain DreamLite UNet). Load: scripts/pose_student.py.

Regression stage for the DreamLite one-step student (bake-off vs SDXS b1; same data, same loss).

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
    ap.add_argument("--shared-scope", default="down_mid", choices=list(DLA.SHARED_SCOPES), help="B: which blocks are shared")
    ap.add_argument("--refine-sigma", type=float, default=0.0, help="3a (2026-11-20): >0 (needs --lora-rank, --init-from) = train a "
                    "SECOND-step LoRA: step 1 = the init student with the adapter off (unchanged), x = (1-s)*x0 + s*noise(seed 1235), "
                    "step 2 = UNet + adapter at sigma s. Tokens frozen. Inference: student_infer.py (2 steps for --refine-tasks)")
    ap.add_argument("--lora-scope", default="all", choices=["all", "mid_up0"], help="#4 (2026-11-20): mid_up0 = LoRA only in "
                    "mid_block + up_blocks.0 (where the influence map says the option acts) -> smaller style packs")
    ap.add_argument("--prune", default="", help="Plan A: name of a set in configs/prune_sets.json (S70, S60): residual units removed; "
                    "saved as prune.json")
    ap.add_argument("--drop-attn2", default="", help="#5: comma list of groups whose cross-attention is removed (down,up1 ...); "
                    "saved as drop_attn2.json")
    ap.add_argument("--joint-styles", default="", help="joint multi-style model (2026-11-20): comma list of extra styles trained "
                    "together with --style in ONE UNet; each extra style gets its own 22 option tokens (task names '<style>/<task>', "
                    "init = the --init-from tokens of the same option). Data: data/pairs/<style>/--joint-manifest (prep_style_data.py), "
                    "needs --proper-flip caches per style; --tgt-cache not supported. Also writes per-style views <tag>_<style>/")
    ap.add_argument("--joint-manifest", default="manifest_v2.jsonl")
    ap.add_argument("--style-tokens", type=int, default=0, help="dl8 (2026-11-21), with --joint-styles: factorised context = "
                    "[N learned style tokens | the 22 option tokens shared by all styles] instead of 22 independent tokens per style")
    ap.add_argument("--style-film", action="store_true", help="dl8: per-style FiLM after the GroupNorms of mid/up blocks (dl_arch.install_style_film)")
    ap.add_argument("--distill-w", type=float, default=0.0, help="dl8: weight of an extra latent MSE to a specialist's output for the same "
                    "photo (--distill-latents), on top of the klein-target losses")
    ap.add_argument("--distill-latents", default="", help="dl8: 'style=weights/decoder/_zs_<student>_<manifest>.pt,...' (train_decoder.py "
                    "--build-latents-only caches: zs, zs_f in the style's row order)")
    ap.add_argument("--style-weights", default="", help="dl9: sampling weight per style (base style first), e.g. 1,2,1; default uniform")
    ap.add_argument("--distill-style-w", default="", help="dl9: per-style multiplier of --distill-w, e.g. 1,3,1")
    ap.add_argument("--teacher-latents", default="", help="stage 4 (2026-11-23): data/pairs/<style>/teacher_<tag>_s<n>.pt (scripts/"
                    "s3_teacher_pairs.py): replace the klein target latents (plain + flipped photo) by the multi-step teacher's, sampled "
                    "from the phone's fixed noise; pixel targets (256 and 512) = TinyVAE decode of those latents")
    ap.add_argument("--extra-pairs", default="", help="with --teacher-latents (2026-11-25): extra teacher pairs on real photos "
                    "(scripts/s5_real_pairs.py -> data/pairs/real_teacher/teacher_tB1_real.pt), appended to the training rows")
    ap.add_argument("--extra-skin-max", type=float, default=10.0, help="keep extra pairs with skin dL* <= this (face found in both)")
    ap.add_argument("--extra-skin-min", type=float, default=-10.0)
    ap.add_argument("--extra-no-filter", action="store_true", help="keep ALL extra pairs (user 2026-11-25: 'remove the skin filter')")
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
    ap.add_argument("--save-at", default="", help="2026-11-26 (tS4 long run): comma list of steps; at each, a plain-recipe snapshot "
                    "(unet + task_emb.pt + config.json) -> weights/student/<tag>_s<step>/ for checkpoint evals. Mid-schedule: cosine lr not annealed")
    ap.add_argument("--pose-maps", required=True)
    ap.add_argument("--pose-lr", type=float, default=1e-3)
    ap.add_argument("--pose-scale", type=float, default=4.0)
    ap.add_argument("--gesture-frac", type=float, default=0.5)
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
    base_tasks = list(tasks)
    if a.joint_styles:
        assert not a.tgt_cache, "--joint-styles: --tgt-cache not supported"
        js = [x for x in a.joint_styles.split(",") if x]
        jst = Path(a.joint_manifest).stem
        for si, s_ in enumerate(js, 1):
            cs = build_cache_dl(s_, vae, a.joint_manifest, base_tasks)
            c = {"ids": c["ids"] + [(f"{s_}/{i}", t) for i, t in cs["ids"]], "key": c["key"] + cs["key"],
                 "zi": torch.cat([c["zi"], cs["zi"]]), "zt": torch.cat([c["zt"], cs["zt"]]),
                 "img256": torch.cat([c["img256"], cs["img256"]]), "task": torch.cat([c["task"], cs["task"] + si * len(base_tasks)])}
            if a.proper_flip:
                F2 = torch.load(ROOT / "data/pairs" / s_ / f"latents_dl_{jst}_flip.pt")
                assert F2["key"] == cs["key"], f"{s_}: flip cache is stale: rerun scripts/build_flip_cache.py --style {s_}"
                FL = {"key": FL["key"] + F2["key"], "zi_f": torch.cat([FL["zi_f"], F2["zi_f"]]), "zt_f": torch.cat([FL["zt_f"], F2["zt_f"]])}
            print(f"[dl] joint style {s_}: {len(cs['task'])} pairs", flush=True)
            del cs
        tasks = base_tasks + [f"{s_}/{t}" for s_ in js for t in base_tasks]
        n = len(c["task"])
        print(f"[dl] joint model: {len(tasks)} options, {n} pairs", flush=True)
    ZS = None
    if a.distill_w > 0:
        dmap = dict(x.split("=", 1) for x in a.distill_latents.split(","))
        zs_, zf_, keys_ = [], [], []
        for s_ in [a.style] + ([x for x in a.joint_styles.split(",") if x] if a.joint_styles else []):
            Zd = torch.load(ROOT / dmap[s_])
            zs_.append(Zd["zs"]); zf_.append(Zd["zs_f"]); keys_ += Zd["key"]
        assert keys_ == c["key"], "distill latents do not match the training rows"
        ZS = {"zs": torch.cat(zs_), "zs_f": torch.cat(zf_)}
        print(f"[dl] distill: specialist latents for {len(keys_)} pairs ({a.distill_latents}), w={a.distill_w}", flush=True)
    if a.teacher_latents:
        assert not a.tgt_cache and not a.joint_styles, "--teacher-latents: no --tgt-cache / --joint-styles"
        TL = torch.load(ROOT / a.teacher_latents)
        assert TL["key"] == c["key"], "teacher latents do not match the training rows"
        c["zt"] = TL["zt"].to(c["zt"].dtype)
        if a.proper_flip:
            FL["zt_f"] = TL["zt_f"].to(FL["zt_f"].dtype)
        print(f"[dl] teacher targets: {a.teacher_latents} ({len(TL['key'])} rows, setting {TL.get('setting')})", flush=True)
        del TL
    if a.extra_pairs:
        assert a.teacher_latents and a.proper_flip, "--extra-pairs needs --teacher-latents --proper-flip"
        X = torch.load(ROOT / a.extra_pairs)
        dL_ = X["dL"]
        sel = torch.arange(len(dL_)) if a.extra_no_filter else \
            torch.nonzero(~torch.isnan(dL_) & (dL_ <= a.extra_skin_max) & (dL_ >= a.extra_skin_min)).flatten()
        c = {"ids": c["ids"] + [X["ids"][i] for i in sel.tolist()], "key": c["key"] + [X["key"][i] for i in sel.tolist()],
             "zi": torch.cat([c["zi"], X["zi"][sel].to(c["zi"].dtype)]), "zt": torch.cat([c["zt"], X["zt"][sel].to(c["zt"].dtype)]),
             "img256": torch.cat([c["img256"], X["img256"][sel]]), "task": torch.cat([c["task"], X["task"][sel].to(c["task"].dtype)])}
        FL = {"key": c["key"], "zi_f": torch.cat([FL["zi_f"], X["zi_f"][sel].to(FL["zi_f"].dtype)]),
              "zt_f": torch.cat([FL["zt_f"], X["zt_f"][sel].to(FL["zt_f"].dtype)])}
        n = len(c["task"])
        print(f"[dl] extra real teacher pairs: {len(sel)}/{len(dL_)} kept ({'no filter' if a.extra_no_filter else f'skin dL* in [{a.extra_skin_min}, {a.extra_skin_max}]'}) -> {n} rows",
              flush=True)
        del X
    if a.tgt_cache:
        TG = np.load(ROOT / "data/pairs" / a.style / f"tgt512_dl_{st}.u8", mmap_mode="r")
        assert TG.shape[0] == n, f"tgt512 cache rows {TG.shape[0]} != {n}: rerun scripts/build_flip_cache.py"
    tok_file = ROOT / f"weights/student/_dl_tokens_{a.style}_{len(tasks)}.pt"
    if a.init_from:  # tokens of the trained student (its first len(tasks) options), trained further
        t = torch.load(ROOT / "weights/student" / a.init_from / "task_emb.pt")
        nb = len(base_tasks)
        assert t["tasks"][:nb] == base_tasks, f"{a.init_from} task order {t['tasks'][:nb]} != {base_tasks}"
        rows_ = [t["tasks"].index(x) if x in t["tasks"] else base_tasks.index(x.split("/", 1)[1]) for x in tasks]
        emb0, mask = t["emb"][rows_], t["mask"][rows_]
    elif tok_file.exists():
        t = torch.load(tok_file)
        emb0, mask = t["emb"], t["mask"]
    else:
        d = ROOT / "data/pairs" / a.style
        photos = [d / "inputs" / f"f{i:05d}.png" for i in range(0, 1600, 100)]  # 16 synthetic faces
        emb0, mask = init_tokens(a.style, tasks, photos, tok_file)
    emb = torch.nn.Parameter(emb0.float().to(DEV))
    mask = mask.to(DEV)
    unet = DLA.load_unet(ROOT / "weights/student" / a.init_from / "unet" if a.init_from else None, style_film=False)
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
    NB = len(base_tasks)
    NS = len(tasks) // NB
    film_init = None
    sfi_ = ROOT / "weights/student" / (a.init_from or "") / "style_film.pt"
    if a.init_from and sfi_.exists():  # init (e.g. dl8) carries per-style FiLM: start from its values (loaded without hooks)
        film_init = torch.load(sfi_)["film"]
        print(f"[dl] init FiLM taken over ({len(film_init)} GroupNorms)", flush=True)
    if a.style_film:
        nm_, npar_ = DLA.install_style_film(unet, NS, film=film_init, trainable=True)
        print(f"[dl] style FiLM: {nm_} GroupNorms x {NS} styles, {npar_} params", flush=True)
    CTX = lambda tk: (emb[tk], mask[tk])
    opt_emb = style_emb = None
    if a.style_tokens > 0:
        assert a.joint_styles, "--style-tokens needs --joint-styles"
        stf_ = ROOT / "weights/student" / (a.init_from or "") / "style_tokens.pt"
        if a.init_from and stf_.exists():  # continue a factorised student (dl8 -> dl9)
            st_ = torch.load(stf_)
            assert st_["styles"] == [a.style] + [x for x in a.joint_styles.split(",") if x] and st_["style_emb"].shape[1] == a.style_tokens
            opt_emb = torch.nn.Parameter(st_["opt_emb"].float().to(DEV))
            opt_mask = st_["opt_mask"].to(DEV)
            style_emb = torch.nn.Parameter(st_["style_emb"].float().to(DEV))
            print(f"[dl] style/option tokens taken over from {a.init_from}", flush=True)
        else:
            opt_emb = torch.nn.Parameter(emb0[:NB].float().to(DEV))
            opt_mask = mask[:NB]
            g_ = torch.Generator().manual_seed(0)
            base_ = emb0[:NB, :a.style_tokens].float().mean(0, keepdim=True)
            style_emb = torch.nn.Parameter((base_ + 0.02 * base_.std() * torch.randn(NS, a.style_tokens, base_.shape[-1], generator=g_)).to(DEV))
        ones_ = torch.ones(a.style_tokens, dtype=mask.dtype, device=DEV)

        def CTX(tk):
            s_, k_ = tk // NB, tk % NB
            return (torch.cat([style_emb[s_], opt_emb[k_]], 1),
                    torch.cat([ones_.expand(len(tk), -1), opt_mask[k_]], 1))
        print(f"[dl] factorised context: {NS} styles x {a.style_tokens} style tokens + {NB} shared option tokens", flush=True)
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
        val += [i for i in range(n) if c["task"][i] == k][:32 if k % len(base_tasks) < 2 else 4]
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
    import pose_student as PS
    pad = PS.install_pose_adapter(unet_raw)
    PMf = torch.load(ROOT / a.pose_maps)
    PM = torch.zeros(len(c["task"]), 2, 3, 64, 64, dtype=torch.uint8)
    PM[torch.tensor(PMf["rows"])] = PMf["maps"]
    assert [tuple(k) for k in PMf["keys"]] == [tuple(c["key"][i]) for i in PMf["rows"]], "pose maps do not match the cache rows"
    if main0:
        print(f"[pose] maps for {len(PMf['rows'])} rows (others zero); adapter zero-init outputs, lr {a.pose_lr}, scale {a.pose_scale}, "
              f"gesture frac {a.gesture_frac}", flush=True)
    tok_params = ([opt_emb, style_emb] if a.style_tokens > 0 else [emb]) + ([unet_raw.shared_emb] if a.shared_trunk else [])
    film_params = [p for n_, p in unet_raw.named_parameters() if n_.startswith("style_film")]
    groups = [{"params": [p for n_, p in unet_raw.named_parameters() if p.requires_grad and n_ != "shared_emb" and not n_.startswith("style_film")
                          and ".pconv." not in n_], "lr": a.lr},
              {"params": list(pad.parameters()), "lr": a.pose_lr, "weight_decay": 0},
              {"params": film_params, "lr": a.emb_lr, "weight_decay": 0},
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
    eye_cache = ROOT / "data/pairs" / a.style / f"_eyes_{Path(a.manifest).stem}.pt"  # 2026-11-27: computed once (scripts/s24_eye_cache.py)
    if (a.eye_w > 0 or a.eye_lpips_w > 0) and eye_cache.exists() and torch.load(eye_cache)["key"] == [tuple(k) for k in c["key"]]:
        E_ = torch.load(eye_cache)
        eyes, eye_sig = E_["eyes"], E_["eye_sig"]
        print(f"[dl] eye landmarks from cache {eye_cache.name}: {(eye_sig > 0).sum().item()}/{n}", flush=True)
    elif a.eye_w > 0 or a.eye_lpips_w > 0:
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
        view = batch.flip.long().cpu() if batch.flip is not None else torch.zeros(len(idx), dtype=torch.long)
        pm = PM[idx, view].to(DEV).float() / 255 * a.pose_scale  # [B, 3, 64, 64]
        PS.set_pose(unet_raw, torch.cat([pm, torch.zeros_like(pm)], -1))  # left (canvas) half only
        if a.teacher_latents:  # the teacher's image for exactly this (photo, flip): decode its latent
            with torch.no_grad():
                im = F.interpolate(DLA.decode(vae, zt.to(torch.bfloat16)).float().clamp(-1, 1), size=256, mode="area")
        return zi, zt, im, tk

    def losses(zi, zt, im, tk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if a.refine_sigma > 0:
                zp = DLA.refine_forward(NET["m"], unet_raw, zi, emb[tk], mask[tk], a.refine_sigma)
            else:
                if a.style_film:
                    unet_raw._film_style = tk // NB
                zp = DLA.forward(NET["m"], zi, *CTX(tk))
            mse = F.mse_loss(zp[:, :zt.shape[1]].float(), zt)
            losses.distill = None
            if ZS is not None:
                fl_ = batch.flip if batch.flip is not None else torch.zeros(len(batch.idx), dtype=torch.bool, device=DEV)
                zs_t = torch.where(fl_.view(-1, 1, 1, 1).cpu(), ZS["zs_f"][batch.idx], ZS["zs"][batch.idx]).to(DEV).float()
                if a.distill_style_w:
                    dsw_ = torch.tensor([float(x) for x in a.distill_style_w.split(",")], device=DEV)[tk // NB]
                    losses.distill = ((zp[:, :zt.shape[1]].float() - zs_t) ** 2).mean((1, 2, 3)).mul(dsw_).mean()
                else:
                    losses.distill = F.mse_loss(zp[:, :zt.shape[1]].float(), zs_t)
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
        try:  # a preview image must never kill a run (NFS I/O error killed pk_chibi at step 6000, 2026-11-20)
            Image.fromarray(np.concatenate(list(rows), 0)).save(out / f"samples_{step:05d}.jpg", quality=88)
        except OSError as e:
            print(f"[dl] samples_{step:05d}.jpg not written: {e}", flush=True)

    if not a.bench and main0:
        samples(0)
    if ddp:
        dist.barrier()
    has_map = set(PMf["rows"])
    tr_g = [i for i in tr if i in has_map]
    if main0:
        print(f"[pose] gesture training rows {len(tr_g)} of {len(tr)}", flush=True)
    t0 = time.time()
    if a.joint_styles:
        nb_ = len(base_tasks)
        tr_by = [[i for i in tr if int(c["task"][i]) // nb_ == si] for si in range(len(tasks) // nb_)]
        print(f"[dl] joint: style-balanced sampling, pairs per style {[len(x) for x in tr_by]}", flush=True)
    for step in range(1, (a.bench or a.steps) + 1):
        if a.bench and step == 11:
            torch.cuda.synchronize()
            tb = time.time()
        if a.joint_styles:  # style-balanced: each sample picks a style uniformly, then a pair of that style (chibi has fewer pairs)
            sw_ = [float(x) for x in a.style_weights.split(",")] if a.style_weights else [1.0] * len(tr_by)
            idx = torch.tensor([random.choice(tr_by[random.choices(range(len(tr_by)), weights=sw_)[0]]) for _ in range(a.bs // world)])
        else:
            ng_ = int(round(a.bs // world * a.gesture_frac))
            idx = torch.tensor(random.sample(tr_g, ng_) + random.sample(tr, a.bs // world - ng_))
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
        if getattr(losses, "distill", None) is not None:
            el["distill"] = losses.distill
            loss = loss + a.distill_w * losses.distill
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
        if ddp:  # task tokens (and dl8 style / shared option tokens) live outside the DDP module
            for tp_ in tok_params:
                if tp_.grad is not None:
                    dist.all_reduce(tp_.grad)
                    tp_.grad /= world
        torch.nn.utils.clip_grad_norm_([p for p in unet_raw.parameters() if p.requires_grad] + tok_params + list(pad.parameters()), 1.0)
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
        if a.save_at and main0 and step in {int(x) for x in a.save_at.split(",") if x} and step != a.steps:
            assert a.style_tokens == 0 and not a.style_film and not a.shared_trunk, "--save-at: plain recipe or LoRA pack only"
            snap = ROOT / "weights/student" / f"{a.tag}_s{step}"
            unet_raw.save_pretrained(snap / "unet")
            torch.save(pad.state_dict(), snap / "pose_adapter.pt")
            torch.save({"emb": emb.detach().cpu(), "mask": mask.cpu(), "tasks": tasks}, snap / "task_emb.pt")
            (snap / "config.json").write_text(json.dumps(vars(a) | {"arch": "dreamlite", "init": a.init_from, "param": "flow_x0", "T": 1000,
                                                                    "decoder": "DreamLite TinyVAE", "train_pairs": len(tr), "snapshot_step": step,
                                                                    "minutes": round((time.time() - t0) / 60, 1)}, indent=1))
            print(f"[dl] snapshot step {step} -> {snap}", flush=True)
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
    unet_raw.save_pretrained(out / "unet")
    torch.save(pad.state_dict(), out / "pose_adapter.pt")
    if a.style_tokens > 0:  # dl8: task_emb.pt = style 0's full context per option (the dir is the base-style view); raw factors kept
        with torch.no_grad():
            e0_, m0_ = CTX(torch.arange(NB, device=DEV))
        torch.save({"emb": e0_.detach().cpu(), "mask": m0_.cpu(), "tasks": base_tasks}, out / "task_emb.pt")
        torch.save({"style_emb": style_emb.detach().cpu(), "opt_emb": opt_emb.detach().cpu(), "opt_mask": opt_mask.cpu(),
                    "styles": [a.style] + [x for x in a.joint_styles.split(",") if x], "tasks": base_tasks}, out / "style_tokens.pt")
    else:
        torch.save({"emb": emb.detach().cpu(), "mask": mask.cpu(), "tasks": tasks}, out / "task_emb.pt")
    if a.style_film:
        torch.save({"n_styles": NS, "film": DLA.film_state(unet_raw)}, out / "style_film.pt")
        (out / "film_style.json").write_text(json.dumps({"style": 0}))
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
    if a.joint_styles:  # per-style views: same UNet, that style's 22 tokens under the plain task names (eval scripts unchanged)
        nb = len(base_tasks)
        for si, s_ in enumerate([x for x in a.joint_styles.split(",") if x], 1):
            v = ROOT / "weights/student" / f"{a.tag}_{s_}"
            v.mkdir(exist_ok=True)
            if not (v / "unet").exists():
                os.symlink(f"../{a.tag}/unet", v / "unet")
            if a.style_tokens > 0:
                with torch.no_grad():
                    es_, ms_ = CTX(torch.arange(NB, device=DEV) + si * NB)
                torch.save({"emb": es_.detach().cpu(), "mask": ms_.cpu(), "tasks": base_tasks}, v / "task_emb.pt")
            else:
                torch.save({"emb": emb.detach().cpu()[si * nb:(si + 1) * nb].clone(), "mask": mask.cpu()[si * nb:(si + 1) * nb].clone(), "tasks": base_tasks},
                           v / "task_emb.pt")
            if a.style_film:
                if not (v / "style_film.pt").exists():
                    os.symlink(f"../{a.tag}/style_film.pt", v / "style_film.pt")
                (v / "film_style.json").write_text(json.dumps({"style": si}))
            (v / "config.json").write_text((out / "config.json").read_text().replace(f'"style": "{a.style}"', f'"style": "{s_}"'))
            print(f"[dl] view {v.name}: {s_} tokens", flush=True)
    print(f"[dl] done {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)
    os.system(f"chmod -R a+rwX '{out}'")  # cross-node readability (root-squashed NFS)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
