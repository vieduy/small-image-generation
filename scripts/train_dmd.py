"""DMD2-style distribution-matching fine-tune of the one-step student (deviation 13, user-approved 2026-11-16).

Why: MSE+LPIPS regression makes the 1-step student predict the conditional MEAN of klein's outputs -> soft, detail-poor
images (the GAN and 4-step tests in README confirm it is the objective, not the decoder or step count).
DMD (Yin et al. 2024, DMD2) instead matches the DISTRIBUTION: for the student image x0 it pushes along
(score_fake - score_real), where
  real score = FLUX.2-klein-base-4B (Apache-2.0, non-distilled), conditioned exactly like the teacher pairs
               (photo as reference image + the per-pair teacher prompt), with CFG;
  fake score = the same transformer + a LoRA, trained online to denoise the student's own outputs.
Both share one copy of the 4B weights (LoRA disabled = real).
Bridge: student SD latent -> TAESD decoder (the phone decoder) -> pixels -> FLUX.2 VAE encoder -> klein latent; the DMD
gradient flows back through both (differentiable).
The regression loss (MSE+LPIPS to the filtered klein targets) is KEPT with weight --reg-w as an identity/skin anchor.
Phone model is unchanged: same 1-step UNet + LoRA style pack + task tokens.

  source env.sh; $VENV/bin/python scripts/train_dmd.py --style anime --init-from an_lora --tag an_dmd --steps 2000
Out: weights/student/<tag>/{unet_lora/, task_emb.pt, config.json, train_log.jsonl, samples_*.jpg}
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
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from train_student import DEV, TASKS, alphas_cumprod, build_cache, expr_tasks, forward, to_t  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
KLEIN = "black-forest-labs/FLUX.2-klein-4B"          # text encoder + VAE (byte-identical in the base repo)
KBASE = "black-forest-labs/FLUX.2-klein-base-4B"     # non-distilled transformer = real score
BF = torch.bfloat16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="anime")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--init-from", required=True, help="a LoRA style-pack student (e.g. an_lora); its LoRA + tokens are trained further")
    ap.add_argument("--manifest", default="manifest.jsonl")
    ap.add_argument("--steps", type=int, default=2000, help="generator updates")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5, help="student LoRA lr")
    ap.add_argument("--emb-lr", type=float, default=1e-4)
    ap.add_argument("--fake-lr", type=float, default=1e-4)
    ap.add_argument("--fake-rank", type=int, default=64)
    ap.add_argument("--fake-updates", type=int, default=2, help="fake-score updates per generator update (DMD2 TTUR)")
    ap.add_argument("--fake-warmup", type=int, default=200, help="fake-only updates before the generator starts")
    ap.add_argument("--gan-w", type=float, default=0.0, help="DMD2-style GAN (0 = off): classifier head on the fake-score "
                    "transformer's single-block features of noised images; real = the klein targets at 512 px")
    ap.add_argument("--gan-block", type=int, default=9, help="single-stream block whose output feeds the GAN head (0..19)")
    ap.add_argument("--d-lr", type=float, default=1e-4, help="GAN head lr")
    ap.add_argument("--dmd-w", type=float, default=1.0)
    ap.add_argument("--reg-w", type=float, default=1.0, help="MSE+LPIPS anchor to the klein targets (0 = pure DMD)")
    ap.add_argument("--lpips-w", type=float, default=1.0)
    ap.add_argument("--cfg", type=float, default=3.0, help="CFG on the real score")
    ap.add_argument("--smin", type=float, default=0.02)
    ap.add_argument("--smax", type=float, default=0.98)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    # colour fix (run 4+): DMD run 2 lost iris colour + muted colours; these anchor LOW-frequency colour only, so DMD keeps the detail
    ap.add_argument("--color-w", type=float, default=0.0, help="L1 between 8x8-pooled student and klein-target pixels (global colour/tone)")
    ap.add_argument("--eye-w", type=float, default=0.0, help="L1 on 2x2-pooled pixels inside soft eye masks (YuNet eye landmarks of the target)")
    ap.add_argument("--eye-chroma", action="store_true", help="eye term on CHROMA only (pixel minus its grey level), 4x4-pooled: "
                    "keeps iris colour but leaves luminance edges/highlights to DMD (run 4 full-RGB 2x2 eye term re-blurred the eyes)")
    ap.add_argument("--eye-luma-w", type=float, default=0.0, help="(needs --eye-w > 0 for the masks) extra L1 on 4x4-pooled LUMINANCE "
                    "inside the eye masks: penalises a solid black eye vs a brown iris + white sclera (DreamLite dl1_dmd), while "
                    "highlights below 4x4 stay with DMD")
    ap.add_argument("--eye-luma-pool", type=int, default=4, choices=[4, 8], help="pooling of the eye luminance term (8 = only "
                    "'the whole eye is too dark'; 4 at weight 4 blurred the highlights, dl1_dmd_eye)")
    ap.add_argument("--eye-no-dmd", action="store_true", help="(needs --eye-w > 0) block the DMD gradient inside the eye masks: "
                    "the DMD term sees the same image, but eye pixels are steered only by the regression + eye terms "
                    "(DMD pushed DreamLite eyes to solid black; eye-loss weights alone could not counter it)")
    ap.add_argument("--expr", action="store_true", help="also train on the sticker-set expression pairs (tokens 2..21; hands/gestures)")
    ap.add_argument("--expr-frac", type=float, default=0.5, help="with --expr and --max-pairs: share of the subset drawn from expression rows")
    ap.add_argument("--ckpt-every", type=int, default=0, help=">0: save a loadable pack weights/student/<tag>_s<step> every N generator steps")
    ap.add_argument("--max-pairs", type=int, default=0, help="random subset of training pairs (0 = all); bounds the Qwen3 embed cache")
    ap.add_argument("--lp512-w", type=float, default=0.0, help="(with --reg-w > 0) LPIPS on a random --lp512-crop px crop of the "
                    "full-res 512 px output vs the decoded target latent (the anchor LPIPS only sees 256 px)")
    ap.add_argument("--lp512-crop", type=int, default=256)
    ap.add_argument("--prompt-manifest", default="", help="take the real/fake-score prompts from this manifest (same rows), "
                    "e.g. manifest_v2_oldprompt.jsonl (probe B); images/targets still from --manifest")
    ap.add_argument("--src-prefixes", default="", help="train only on rows whose photo id starts with one of these letters "
                    "(e.g. fc = synthetic only); val rows unchanged")
    ap.add_argument("--ms-sigmas", default="", help="DreamLite multi-step student, e.g. 1.0,0.5 (2 steps; photo beside the image "
                    "at every step, fixed per-step noise). Each iteration trains a random step k (DMD2 backward simulation)")
    ap.add_argument("--new-lora", type=int, default=32, help="rank of a fresh student LoRA when --init-from is a full student (not a pack)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = ROOT / "weights/student" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "w")

    def P(*x):
        print(*x, flush=True)

    # ---------- data (same cache/filter as train_student) ----------
    from diffusers import AutoencoderKL, AutoencoderTiny, UNet2DConditionModel
    DLARCH = json.loads((ROOT / "weights/student" / a.init_from / "config.json").read_text()).get("arch") == "dreamlite"
    if DLARCH:  # DreamLite student (scripts/dl_arch.py): its own TinyVAE latent space for the cache + the phone decoder
        import dl_arch as DLA
        from train_dl import build_cache_dl
        dlvae = DLA.load_vae(BF).to(DEV).eval().requires_grad_(False)
    sdvae = None if DLARCH else AutoencoderKL.from_pretrained("IDKiro/sdxs-512-0.9", subfolder="vae_large", torch_dtype=BF).to(DEV).eval()
    tasks = TASKS + (expr_tasks() if a.expr else [])
    c = build_cache_dl(a.style, dlvae, a.manifest, tasks) if DLARCH else build_cache(a.style, sdvae, a.manifest, tasks)
    del sdvae
    d = ROOT / "data/pairs" / a.style
    rows = {(r["id"], r["task"]): r for r in map(json.loads, open(d / a.manifest))}
    n = len(c["task"])
    prow = rows if not a.prompt_manifest else {(r["id"], r["task"]): r for r in map(json.loads, open(d / a.prompt_manifest))}
    prompts = [prow[k]["teacher_prompt"] for k in c["ids"]]
    if a.prompt_manifest:
        P(f"[dmd] prompts from {a.prompt_manifest}, e.g.: {prompts[0][:160]}")
    photos = [d / rows[k].get("input", f"inputs/{k[0]}.png") for k in c["ids"]]  # data v2: optional per-row input path (FFHQ)
    val = []
    for k in range(len(tasks)):
        val += [i for i in range(n) if c["task"][i] == k][:32 if k < 2 else 4]
    vs = set(val)
    tr = [i for i in range(n) if i not in vs]
    if a.src_prefixes:  # probe A (2026-11-18): is the DMD softening caused by soft real-photo inputs?
        tr = [i for i in tr if c["ids"][i][0][0] in a.src_prefixes]
        P(f"[dmd] source filter {a.src_prefixes!r}: {len(tr)} training rows")
    if a.max_pairs and len(tr) > a.max_pairs:
        rng = random.Random(a.seed)
        if a.expr:
            te_, tb_ = [i for i in tr if c["task"][i] >= 2], [i for i in tr if c["task"][i] < 2]
            ne = min(len(te_), int(a.max_pairs * a.expr_frac))
            tr = sorted(rng.sample(te_, ne) + rng.sample(tb_, min(len(tb_), a.max_pairs - ne)))
        else:
            tr = sorted(rng.sample(tr, a.max_pairs))
    P(f"[dmd] tasks={len(tasks)} train subset={len(tr)} (expression rows {sum(int(c['task'][i] >= 2) for i in tr)})")
    used = sorted(set(tr) | vs)                              # only these get Qwen3 embeds / reference latents / eye masks
    pos = {i: j for j, i in enumerate(used)}
    prompts = [prompts[i] for i in used]
    photos = [photos[i] for i in used]
    n_used = len(used)
    P(f"[dmd] pairs={n} train={len(tr)} val={len(val)}")

    # ---------- klein: text embeds + reference latents (precomputed), base transformer ----------
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
    kc = os.environ["KLEIN_CACHE"] + "/hub"
    tf = Flux2Transformer2DModel.from_pretrained(KBASE, subfolder="transformer", torch_dtype=BF, cache_dir=kc)
    pipe = Flux2KleinPipeline.from_pretrained(KLEIN, transformer=tf, torch_dtype=BF, cache_dir=kc).to(DEV)
    t0 = time.time()
    with torch.no_grad():
        pe_all = []
        for i in range(0, n_used, 16):
            pe, _ = pipe.encode_prompt(prompts[i:i + 16], device=DEV)
            pe_all.append(pe.cpu())
        pe_all = torch.cat(pe_all)                          # [n, 512, 7680] bf16, CPU
        neg_pe, txt_ids = pipe.encode_prompt([""], device=DEV)
        ref_all = []
        for i in range(0, n_used, 16):
            x = torch.stack([to_t(Image.open(p)) for p in photos[i:i + 16]]).to(DEV, BF)
            ref_all.append(pipe._pack_latents(pipe._encode_vae_image(x, None)).cpu())
        ref_all = torch.cat(ref_all)                        # [n, 1024, 128]
        z_dummy = torch.zeros(1, 128, 32, 32, device=DEV)
        lat_ids = pipe._prepare_latent_ids(z_dummy).to(DEV)                     # (1, 1024, 4)
        img_ids = pipe._prepare_image_ids([z_dummy]).to(DEV)                    # (1, 1024, 4)
        ids = torch.cat([lat_ids, img_ids], 1)
    pipe.text_encoder.to("cpu")
    del pipe.text_encoder
    torch.cuda.empty_cache()
    P(f"[dmd] text/ref precompute {time.time() - t0:.0f}s  pe={tuple(pe_all.shape)}")
    fvae = pipe.vae.eval().requires_grad_(False)
    bn_m = fvae.bn.running_mean.view(1, -1, 1, 1).to(DEV, BF)
    bn_s = torch.sqrt(fvae.bn.running_var.view(1, -1, 1, 1) + fvae.config.batch_norm_eps).to(DEV, BF)

    def to_klein(img):  # [-1,1] pixels -> packed, BN-normalised klein latent [B,1024,128]; differentiable
        z = fvae.encode(img.to(BF)).latent_dist.mode()
        z = (pipe._patchify_latents(z) - bn_m) / bn_s
        return pipe._pack_latents(z)

    from peft import LoraConfig, get_peft_model, PeftModel
    tf.requires_grad_(False)
    fake = get_peft_model(tf, LoraConfig(r=a.fake_rank, lora_alpha=a.fake_rank, init_lora_weights="gaussian",
                                         # double blocks: to_q/k/v, to_out.0, add_*; single blocks: fused qkv+mlp in, to_out
                                         target_modules=r".*(to_q|to_k|to_v|to_out\.0|add_q_proj|add_k_proj|add_v_proj|"
                                                        r"to_add_out|to_qkv_mlp_proj|single_transformer_blocks\.\d+\.attn\.to_out)$"))
    for p in fake.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    n_fake = sum(p.numel() for p in fake.parameters() if p.requires_grad)
    if n_fake == 0:
        raise SystemExit("[dmd] fake LoRA matched no modules - check Flux2 attention names")
    fake.enable_gradient_checkpointing() if hasattr(fake, "enable_gradient_checkpointing") else tf.enable_gradient_checkpointing()
    P(f"[dmd] fake-score LoRA r={a.fake_rank}: {n_fake / 1e6:.1f}M params")

    # ---------- DMD2-style GAN (--gan-w): frozen fake-score features (LADD-like, head only trained) ----------
    if a.gan_w > 0:
        from concurrent.futures import ThreadPoolExecutor
        tpaths = [d / c["key"][i][2] for i in used]                 # klein target PNGs (512 px; the cache only has 256 px)
        pool = ThreadPoolExecutor(16)
        real_all = []
        with torch.no_grad():
            for i in range(0, n_used, 16):
                x = torch.stack(list(pool.map(lambda p: to_t(Image.open(p)), tpaths[i:i + 16]))).to(DEV, BF)
                real_all.append(to_klein(x).cpu())
        real_all = torch.cat(real_all)                               # [n_used, 1024, 128]
        P(f"[dmd] GAN: real klein latents {tuple(real_all.shape)}; head on single block {a.gan_block}")
        NTXT = pe_all.shape[1]
        dim = tf.config.num_attention_heads * tf.config.attention_head_dim

        class _Stop(Exception):
            pass
        feat = {}

        def _hook(mod, inp, out):
            if not feat.get("on"):                                   # only during gan_logits (not the DMD score calls)
                return
            feat["h"] = (out[1] if isinstance(out, tuple) else out)
            raise _Stop                                              # the rest of the transformer is not needed
        tf.single_transformer_blocks[a.gan_block].register_forward_hook(_hook)
        gsize = int(real_all.shape[1] ** 0.5)                        # 32
        head = torch.nn.Sequential(
            torch.nn.Conv2d(dim, 256, 4, 2, 1), torch.nn.GroupNorm(32, 256), torch.nn.SiLU(),
            torch.nn.Conv2d(256, 256, 4, 2, 1), torch.nn.GroupNorm(32, 256), torch.nn.SiLU(),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(256, 1)).to(DEV)
        d_opt = torch.optim.AdamW(head.parameters(), lr=a.d_lr, betas=(0.0, 0.99), weight_decay=0.0)
        P(f"[dmd] GAN head {sum(p.numel() for p in head.parameters()) / 1e6:.1f}M params, gan_w={a.gan_w}")

        def gan_logits(x0, pe, ref, feat_grad=True):
            """noised x0 -> fake-score features of the target tokens -> logit. Gradient checkpointing is off for this call
            (the forward stops at the hooked block, so memory stays small). feat_grad=False: head-only update."""
            xt, sig, _ = noisy(x0)
            gc_ = tf.gradient_checkpointing
            tf.gradient_checkpointing = False
            feat["on"] = True
            try:
                with torch.autocast("cuda", dtype=BF), torch.set_grad_enabled(feat_grad and torch.is_grad_enabled()):
                    score(fake, xt, sig, pe, ref)
            except _Stop:
                pass
            finally:
                tf.gradient_checkpointing = gc_
                feat["on"] = False
            h = feat.pop("h")[:, NTXT:NTXT + x0.shape[1]]               # [B, 1024, dim] target tokens
            h = h.transpose(1, 2).reshape(x0.shape[0], dim, gsize, gsize)
            with torch.autocast("cuda", dtype=BF):
                return head(h).float().squeeze(1)

    def score(model, xt, sig, pe, ref):  # flow-matching velocity for the target tokens
        B = xt.shape[0]
        o = model(hidden_states=torch.cat([xt, ref], 1).to(BF), timestep=sig.to(BF), guidance=None,
                  encoder_hidden_states=pe.to(BF), txt_ids=txt_ids.expand(B, -1, -1), img_ids=ids.expand(B, -1, -1),
                  return_dict=False)[0]
        return o[:, :xt.shape[1]].float()

    def noisy(x0):
        sig = torch.empty(x0.shape[0], device=DEV).uniform_(a.smin, a.smax)
        eps = torch.randn_like(x0)
        s = sig.view(-1, 1, 1)
        return (1 - s) * x0 + s * eps, sig, eps

    # ---------- eye masks (YuNet 5-point landmarks on the klein TARGET at 256 px; output layout follows the photo) ----------
    eyes = torch.zeros(n_used, 2, 2)
    eye_sig = torch.zeros(n_used)
    if a.eye_w > 0:
        import cv2
        from metrics import FaceID
        det = FaceID().det
        det.setInputSize((256, 256))
        for j, i in enumerate(used):
            arr = cv2.cvtColor(np.ascontiguousarray(c["img256"][i].permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR)
            _, f = det.detect(arr)
            if f is None or len(f) == 0:
                continue
            f = max(f, key=lambda r: r[2] * r[3])
            eyes[j] = torch.tensor([[float(f[4]), float(f[5])], [float(f[6]), float(f[7])]])
            eye_sig[j] = 0.07 * float(f[2])                       # ~iris+lids radius; f[2] = face box width (px @256)
        P(f"[dmd] eye landmarks on {(eye_sig > 0).sum().item()}/{n_used} targets")
    gy, gx = torch.meshgrid(torch.arange(128, device=DEV) * 2 + 1.0, torch.arange(128, device=DEV) * 2 + 1.0, indexing="ij")

    def eye_mask(idx):  # [B,1,128,128] soft mask, 0 where no face was found
        j = torch.tensor([pos[int(i)] for i in idx])
        e, sg = eyes[j].to(DEV), eye_sig[j].to(DEV).view(-1, 1, 1)
        m = torch.zeros(len(j), 128, 128, device=DEV)
        for k in range(2):
            d2 = (gx - e[:, k, 0].view(-1, 1, 1)) ** 2 + (gy - e[:, k, 1].view(-1, 1, 1)) ** 2
            m = torch.maximum(m, torch.exp(-d2 / (2 * sg.clamp(min=1e-3) ** 2)))
        return (m * (sg > 0)).unsqueeze(1)

    # ---------- student (generator) ----------
    wi = ROOT / "weights/student" / a.init_from
    icfg = json.loads((wi / "config.json").read_text())
    load_u = (lambda pth: DLA.load_unet(pth)) if DLARCH else (lambda pth: UNet2DConditionModel.from_pretrained(pth))
    if icfg.get("lora_rank", 0) > 0:  # continue a style pack
        base = load_u(ROOT / "weights/student" / icfg["init_from"] / "unet")
        unet = PeftModel.from_pretrained(base, wi / "unet_lora", is_trainable=True).to(DEV).train()
    else:  # full student (e.g. base b1): new zero-start LoRA pack on it (same targets as train_student --lora-rank)
        from peft import LoraConfig as LC, get_peft_model as gpm
        base = load_u(wi / "unet").requires_grad_(False)
        tm = ["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2", "conv1", "conv2"]
        if DLARCH:  # DreamLite resnet convs are DepthwiseSeparableConv containers: adapt their 1x1 pointwise convs
            tm = r".*(to_q|to_k|to_v|to_out\.0|proj_in|proj_out|ff\.net\.0\.proj|ff\.net\.2|conv1\.pointwise|conv2\.pointwise)$"
        unet = gpm(base, LC(r=a.new_lora, lora_alpha=a.new_lora, target_modules=tm, init_lora_weights="gaussian")).to(DEV).train()
        icfg = icfg | {"lora_rank": a.new_lora, "init_from": a.init_from}
    MS = [float(s) for s in a.ms_sigmas.split(",")] if a.ms_sigmas else icfg.get("ms_sigmas")
    if MS:
        assert DLARCH and MS[0] == 1.0, "--ms-sigmas: DreamLite students only, first sigma 1.0"
        icfg = icfg | {"ms_sigmas": MS}
        P(f"[dmd] multi-step student, sigmas {MS}")
    te0 = torch.load(wi / "task_emb.pt")
    emb = torch.nn.Parameter(te0["emb"].float().to(DEV))
    dlmask = te0["mask"].to(DEV) if DLARCH else None
    taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=BF).to(DEV).eval().requires_grad_(False)
    lp = lpips.LPIPS(net="vgg", verbose=False).to(DEV).eval().requires_grad_(False)
    ac = alphas_cumprod().to(DEV)
    g_par = [p for p in unet.parameters() if p.requires_grad]
    g_opt = torch.optim.AdamW([{"params": g_par, "lr": a.lr}, {"params": [emb], "lr": a.emb_lr, "weight_decay": 0}], weight_decay=0.01)
    f_opt = torch.optim.AdamW([p for p in fake.parameters() if p.requires_grad], lr=a.fake_lr, weight_decay=0.0)

    def student(idx, aug=False, k=None):
        zi, zt = c["zi"][idx].to(DEV), c["zt"][idx].to(DEV)
        im = c["img256"][idx].to(DEV).float() / 127.5 - 1
        e = emb[c["task"][idx].to(DEV)]
        with torch.autocast("cuda", dtype=BF):
            if DLARCH:
                zp = DLA.forward(unet, zi, e, dlmask[c["task"][idx].to(DEV)], sigmas=MS, k=k)
                img = DLA.decode(dlvae, zp.to(BF)).float().clamp(-1, 1)          # DreamLite TinyVAE = the phone decoder
            else:
                zp = forward(unet, zi, e, icfg["param"], icfg["T"], ac)
                img = taesd.decode(zp.to(BF)).sample.float().clamp(-1, 1)            # the phone output, 512 px
        return zp, zt, img, im

    def cond(idx):
        j = [pos[int(i)] for i in idx]
        return pe_all[j].to(DEV), ref_all[j].to(DEV)

    def fake_step(x0, pe, ref):
        xt, sig, eps = noisy(x0)
        with torch.autocast("cuda", dtype=BF):
            v = score(fake, xt, sig, pe, ref)
        loss = F.mse_loss(v, eps - x0)
        f_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in fake.parameters() if p.requires_grad], 1.0)
        f_opt.step()
        return loss.item()

    def save_samples(step):
        unet.eval()
        with torch.no_grad():
            vi = torch.tensor(val[:4] + val[32:36])
            _, _, img, im = student(vi)
        unet.train()
        tiles = [((torch.cat([F.interpolate(im[i:i + 1], size=256), F.interpolate(img[i:i + 1], size=256, mode="area")], -1)[0]
                   .clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy() for i in range(len(vi))]
        Image.fromarray(np.concatenate(tiles, 0)).save(out / f"samples_{step:05d}.jpg", quality=90)
        gest = [f"expr_{g}" for g in ("think", "thumbs", "wave", "ok", "facepalm", "shrug", "thanks") if f"expr_{g}" in tasks]
        if gest:  # hand-gesture progress sheet: per gesture 2 val faces, (klein target | student) x 2
            gv = [i for k in gest for i in [j for j in val if c["task"][j] == tasks.index(k)][:2]]
            unet.eval()
            with torch.no_grad():
                _, _, img, im = student(torch.tensor(gv))
            unet.train()
            t = lambda x: ((F.interpolate(x, size=256, mode="area")[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
            rows = [np.concatenate([t(im[r:r + 1]), t(img[r:r + 1]), t(im[r + 1:r + 2]), t(img[r + 1:r + 2])], 1)
                    for r in range(0, len(gv) - 1, 2)]
            Image.fromarray(np.concatenate(rows, 0)).save(out / f"samples_gest_{step:05d}.jpg", quality=90)

    def save_pack(d, tag, extra=None):
        d.mkdir(parents=True, exist_ok=True)
        unet.save_pretrained(d / "unet_lora")
        te = {"emb": emb.detach().cpu(), "tasks": tasks if emb.shape[0] == len(tasks) else TASKS + expr_tasks()}
        if dlmask is not None:
            te["mask"] = dlmask.cpu()
        torch.save(te, d / "task_emb.pt")
        (d / "config.json").write_text(json.dumps(icfg | {"tag": tag, "dmd": vars(a), "dmd_init": a.init_from, "real_score": KBASE}
                                                  | (extra or {}), indent=1))

    save_samples(0)
    t0 = time.time()
    for it in range(1, a.fake_warmup + a.steps + 1):
        gen = it > a.fake_warmup
        idx = torch.tensor(random.sample(tr, a.bs))
        pe, ref = cond(idx)
        ks = random.randint(1, len(MS)) if MS else None   # multi-step: train step ks on the output of steps 1..ks-1
        zp, zt, img, im = student(idx, k=ks)
        if gen and a.eye_no_dmd and a.eye_w > 0:
            # eye mask (128 grid over 256 px) -> 512 px; x2 + clamp widens it to cover iris, sclera and lids
            m512 = (F.interpolate(eye_mask(idx), size=img.shape[-1], mode="bilinear") * 2).clamp(max=1)
            x0 = to_klein(img * (1 - m512) + img.detach() * m512)
        else:
            x0 = to_klein(img) if gen else None
        r = {"it": it} | ({"k": ks} if MS else {})
        if gen:
            with torch.no_grad():
                xt, sig, _ = noisy(x0.detach().float())
                s = sig.view(-1, 1, 1)
                with torch.autocast("cuda", dtype=BF):
                    with fake.disable_adapter():
                        vc = score(fake, xt, sig, pe, ref)
                        vu = score(fake, xt, sig, neg_pe.expand(len(idx), -1, -1), ref)
                    vf = score(fake, xt, sig, pe, ref)
                x0_real = xt - s * (vu + a.cfg * (vc - vu))
                x0_fake = xt - s * vf
                wgt = (x0.detach().float() - x0_real).abs().mean(dim=[1, 2], keepdim=True).clamp(min=1e-4)
                grad = torch.nan_to_num((x0_fake - x0_real) / wgt)
            l_dmd = 0.5 * F.mse_loss(x0.float(), (x0.detach().float() - grad))
            loss = a.dmd_w * l_dmd
            if a.gan_w > 0:  # non-saturating generator loss through the frozen features + head
                head.requires_grad_(False)
                l_g = F.softplus(-gan_logits(x0.float(), pe, ref)).mean()
                head.requires_grad_(True)
                loss = loss + a.gan_w * l_g
                r["g_gan"] = round(l_g.item(), 4)
            if a.reg_w > 0:
                l_mse = F.mse_loss(zp.float(), zt)
                l_lp = lp(F.interpolate(img, size=256, mode="bilinear", antialias=True), im).mean()
                loss = loss + a.reg_w * (l_mse + a.lpips_w * l_lp)
                r |= {"mse": round(l_mse.item(), 4), "lpips": round(l_lp.item(), 4)}
                if a.lp512_w > 0:  # full-res crop LPIPS (2026-11-19)
                    from torchvision.ops import roi_align
                    with torch.no_grad(), torch.autocast("cuda", dtype=BF):
                        tgt = (DLA.decode(dlvae, zt.to(BF)) if DLARCH else taesd.decode(zt.to(BF)).sample).float().clamp(-1, 1)
                    k = a.lp512_crop
                    cx = torch.randint(48, 512 - 48 - k + 1, (len(idx),)).float()   # crop corner (NOT x0: that is the student latent)
                    cy = torch.randint(32, 512 - 64 - k + 1, (len(idx),)).float()
                    bx = torch.stack([torch.arange(len(idx)).float(), cx, cy, cx + k, cy + k], 1).to(DEV)
                    l_512 = lp(roi_align(img, bx, k, aligned=True), roi_align(tgt, bx, k, aligned=True)).mean()
                    loss = loss + a.reg_w * a.lp512_w * l_512
                    r["lp512"] = round(l_512.item(), 4)
            if a.color_w > 0 or a.eye_w > 0:
                s256 = F.interpolate(img, size=256, mode="area")
            if a.color_w > 0:
                l_col = (F.avg_pool2d(s256, 8) - F.avg_pool2d(im, 8)).abs().mean()
                loss = loss + a.color_w * l_col
                r["col"] = round(l_col.item(), 4)
            if a.eye_w > 0:
                m = eye_mask(idx)
                if a.eye_chroma:
                    m = F.avg_pool2d(m, 2)
                    df = F.avg_pool2d(s256 - s256.mean(1, keepdim=True), 4) - F.avg_pool2d(im - im.mean(1, keepdim=True), 4)
                else:
                    df = F.avg_pool2d(s256, 2) - F.avg_pool2d(im, 2)
                if m.sum() > 1:
                    l_eye = (m * df.abs()).sum() / (3 * m.sum())
                    loss = loss + a.eye_w * l_eye
                    r["eye"] = round(l_eye.item(), 4)
                    if a.eye_luma_w > 0:
                        ml = m if a.eye_chroma else F.avg_pool2d(m, 2)  # 64x64, matches 4x4 pooling of the 256 px image
                        if a.eye_luma_pool == 8:
                            ml = F.avg_pool2d(ml, 2)                    # 32x32
                        k = a.eye_luma_pool
                        dlum = F.avg_pool2d(s256.mean(1, keepdim=True), k) - F.avg_pool2d(im.mean(1, keepdim=True), k)
                        l_luma = (ml * dlum.abs()).sum() / ml.sum()
                        loss = loss + a.eye_luma_w * l_luma
                        r["eye_luma"] = round(l_luma.item(), 4)
            g_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(g_par + [emb], 1.0)
            g_opt.step()
            r |= {"dmd": round(l_dmd.item(), 5)}
            x0f = x0.detach().float()
        else:
            with torch.no_grad():
                x0f = to_klein(img).float()
        for _ in range(a.fake_updates):
            r["fake"] = round(fake_step(x0f, pe, ref), 4)
        if a.gan_w > 0:  # one head update per iteration (also during the fake warm-up): klein targets vs student outputs
            j = [pos[int(i)] for i in idx]
            lr_ = gan_logits(real_all[j].to(DEV).float(), pe, ref, feat_grad=False)
            lf_ = gan_logits(x0f, pe, ref, feat_grad=False)
            l_d = F.softplus(-lr_).mean() + F.softplus(lf_).mean()
            d_opt.zero_grad(set_to_none=True)
            l_d.backward()
            d_opt.step()
            r |= {"d_loss": round(l_d.item(), 4), "d_real": round(torch.sigmoid(lr_).mean().item(), 3),
                  "d_fake": round(torch.sigmoid(lf_).mean().item(), 3)}
        r["s"] = round(time.time() - t0, 1)
        if it % 25 == 0:
            P(json.dumps(r))
            log.write(json.dumps(r) + "\n")
            log.flush()
        g = it - a.fake_warmup
        if gen and (g % a.val_every == 0 or g == a.steps):
            save_samples(g)
        if gen and a.ckpt_every and g % a.ckpt_every == 0 and g < a.steps:
            save_pack(ROOT / "weights/student" / f"{a.tag}_s{g}", f"{a.tag}_s{g}", {"ckpt_step": g})
        if not math.isfinite(r.get("dmd", 0.0) + r["fake"]):
            P("[dmd] non-finite loss, abort")
            break
    save_pack(out, a.tag, {"minutes": round((time.time() - t0) / 60, 1)})
    P(f"[dmd] done {(time.time() - t0) / 60:.1f} min, peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} GB -> {out}")


if __name__ == "__main__":
    main()
