"""Step 4: one-step, text-free, image-conditioned student (pix2pix-turbo-style recipe, BlazeEdit-style text removal).

Student  : SDXS-512-0.9 UNet (0.32B, OpenRAIL++), full fine-tune. Input = SD-VAE latent of the user photo; one UNet pass
           at fixed timestep T; per-task learned conditioning tokens (init from the SDXS text encoder on the task prompt,
           then trained) replace the text encoder -> NO text encoder at inference.
Param    : eps_x0  z_out = (z_in - sqrt(1-a_T) * eps) / sqrt(a_T)   (pix2pix-turbo: treat input latent as x_T)
           direct  z_out = UNet(z_in)
Losses   : MSE(z_out, z_target) + lpips_w * LPIPS(TAESD.decode(z_out), target image) at 256 px.
Data     : data/pairs/<style>/manifest.jsonl rows with keep=True (klein teacher). 32 pairs/task held out for val.
Decoder  : TAESD (MIT, ~1M params) - the same tiny decoder SDXS ships with; phone-friendly.

  source env.sh; $VENV/bin/python scripts/train_student.py --style cartoon3d --tag s1 --steps 4000
Out: weights/student/<tag>/{unet/, task_emb.pt, config.json, train_log.jsonl}
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import lpips
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, AutoencoderTiny, UNet2DConditionModel
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

ROOT = Path(__file__).resolve().parent.parent
SDXS = "IDKiro/sdxs-512-0.9"
TASKS = ["avatar", "sticker"]  # token 0/1 for every student (export/infer contract)


def expr_tasks():
    """Sticker-set expression tasks (configs/sticker_expressions.yaml), tokens 2.. when training with --expr."""
    import yaml
    return [f"expr_{e['id']}" for e in yaml.safe_load((ROOT / "configs/sticker_expressions.yaml").read_text())]
DEV = "cuda"


def alphas_cumprod():
    b = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float64) ** 2  # scaled_linear (SDXS scheduler)
    return torch.cumprod(1 - b, 0).float()


def to_t(img, size=512):
    return torch.from_numpy(__import__("numpy").array(img.convert("RGB").resize((size, size)))).permute(2, 0, 1).float() / 127.5 - 1


@torch.no_grad()
def build_cache(style, vae, manifest="manifest.jsonl", tasks=TASKS):
    """Encode all kept pairs once (SD-VAE posterior mean, scaled). Returns dict of tensors.
    Rows may carry "target" (relative image path, e.g. best-of-K candidates); the cache key includes it."""
    d = ROOT / "data/pairs" / style
    f = d / ("latents_sdvae.pt" if manifest == "manifest.jsonl" else f"latents_sdvae_{Path(manifest).stem}.pt")
    rows = [json.loads(l) for l in open(d / manifest)]
    rows = [r for r in rows if r["keep"] and r["task"] in tasks]  # e.g. expression rows are ignored unless --expr
    for r in rows:
        r.setdefault("target", f"{r['task']}/{r['id']}.png")
    key = [(r["id"], r["task"], r["target"]) for r in rows]
    if f.exists():
        c = torch.load(f)
        if c.get("key") == key:
            return c
    zi, zt, img256, tk = [], [], [], []
    sf = vae.config.scaling_factor
    for i in range(0, len(rows), 32):
        ch = rows[i:i + 32]
        xi = torch.stack([to_t(Image.open(d / r.get("input", f"inputs/{r['id']}.png"))) for r in ch]).to(DEV, torch.bfloat16)
        xt = torch.stack([to_t(Image.open(d / r["target"])) for r in ch]).to(DEV, torch.bfloat16)
        zi.append((vae.encode(xi).latent_dist.mean * sf).float().cpu())
        zt.append((vae.encode(xt).latent_dist.mean * sf).float().cpu())
        img256.append(((F.interpolate(xt.float(), size=256, mode="area") + 1) * 127.5).round().to(torch.uint8).cpu())
        tk += [tasks.index(r["task"]) for r in ch]
    c = {"ids": [(r["id"], r["task"]) for r in rows], "key": key, "zi": torch.cat(zi), "zt": torch.cat(zt), "img256": torch.cat(img256),
         "task": torch.tensor(tk)}
    torch.save(c, f)
    return c


@torch.no_grad()
def init_task_emb(style_desc, tasks=TASKS, te_from=None):
    """te_from: repo whose tokenizer + text encoder initialise the tokens (default SDXS, 1024-d; SD 1.5 = CLIP ViT-L, 768-d)."""
    te_from = te_from or SDXS
    tok = CLIPTokenizer.from_pretrained(te_from, subfolder="tokenizer")
    try:
        te = CLIPTextModel.from_pretrained(te_from, subfolder="text_encoder")
    except (OSError, EnvironmentError):  # repos cached with fp16 weights only (e.g. stable-diffusion-v1-5)
        te = CLIPTextModel.from_pretrained(te_from, subfolder="text_encoder", variant="fp16")
    te = te.float().to(DEV)
    import yaml
    P = yaml.safe_load((ROOT / "configs/prompts.yaml").read_text())["default"]
    EX = {f"expr_{e['id']}": e["phrase"] for e in yaml.safe_load((ROOT / "configs/sticker_expressions.yaml").read_text())}
    embs = []
    for t in tasks:
        text = P[t].format(desc=style_desc, short="") if t in P else P["sticker"].format(desc=style_desc, short="") + ", " + EX[t]
        ids = tok(text, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
        embs.append(te(ids.to(DEV))[0][0].float().cpu())
    del te
    torch.cuda.empty_cache()
    return torch.stack(embs)  # [len(tasks), 77, 1024]


class ProjD(torch.nn.Module):
    """Projected discriminator (vision-aided GAN): frozen VGG16 features (the LPIPS backbone, BSD) + small trainable
    patch heads on relu2_2..relu5_3, hinge loss. VGG on purpose: DINOv2/CLIP are the eval style metrics and must not be
    optimized against."""
    def __init__(self, lp):
        super().__init__()
        self.lp = [lp]  # not a submodule: frozen, shared with the LPIPS loss
        sn = torch.nn.utils.parametrizations.spectral_norm
        self.heads = torch.nn.ModuleList([torch.nn.Sequential(sn(torch.nn.Conv2d(c, 128, 1)), torch.nn.LeakyReLU(0.2),
                                                              sn(torch.nn.Conv2d(128, 1, 3, padding=1))) for c in (128, 256, 512, 512)])

    def forward(self, x):
        f = self.lp[0].net(self.lp[0].scaling_layer(x))
        return torch.cat([h(t.float()).flatten(1) for h, t in zip(self.heads, (f.relu2_2, f.relu3_3, f.relu4_3, f.relu5_3))], 1)


def forward(unet, zin, emb, param, T, ac):
    t = torch.full((zin.shape[0],), T, device=DEV, dtype=torch.long)
    out = unet(zin, t, encoder_hidden_states=emb).sample
    if param == "direct":
        return out
    a = ac[T]
    return (zin - (1 - a).sqrt() * out) / a.sqrt()


def msteps_ts(K, T=999):
    """Timesteps of a K-step student: step 0 maps the photo latent at T; steps 1..K-1 refine a re-noised estimate."""
    return [T - i * ((T + 1) // K) for i in range(K)]  # K=4 -> [999, 749, 499, 249]


def renoise(z, t, ac, gen=None):
    """x_t = sqrt(a_t) z + sqrt(1 - a_t) eps (the standard forward process applied to the current estimate)."""
    eps = torch.randn(z.shape, generator=gen, device=z.device, dtype=torch.float32).to(z.dtype)
    a = ac[t]
    return a.sqrt().to(z.dtype) * z + (1 - a).sqrt().to(z.dtype) * eps


def sample_chain(unet, zin, emb, param, ts, ac, gen=None):
    """K-step inference (multistep consistency-style sampling): z = f(photo, ts[0]); z = f(renoise(z, t), t) for t in ts[1:].
    K = 1 is exactly the original one-step student."""
    z = forward(unet, zin, emb, param, ts[0], ac)
    for t in ts[1:]:
        z = forward(unet, renoise(z, t, ac, gen), emb, param, t, ac)
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="cartoon3d")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--param", default="eps_x0", choices=["eps_x0", "direct"])
    ap.add_argument("--T", type=int, default=999)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--emb-lr", type=float, default=1e-3)
    ap.add_argument("--lpips-w", type=float, default=1.0)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", default="manifest.jsonl", help="pairs manifest in data/pairs/<style>/")
    ap.add_argument("--expr", action="store_true", help="also train the sticker-set expression tokens (tasks expr_*)")
    ap.add_argument("--init-from", default=None, help="start from a trained student tag (UNet + task tokens)")
    ap.add_argument("--lora-rank", type=int, default=0, help=">0: freeze the init UNet, train a LoRA add-on (style pack)")
    ap.add_argument("--gan-w", type=float, default=0.0, help="adversarial weight (0 = off)")
    ap.add_argument("--gan-start", type=int, default=2000)
    ap.add_argument("--d-lr", type=float, default=1e-4)
    ap.add_argument("--msteps", type=int, default=1, help="K-step student (K>1: refinement steps at msteps_ts(K)[1:])")
    ap.add_argument("--tf-prob", type=float, default=0.5, help="K>1: prob. a refinement input is the re-noised TEACHER latent "
                    "(else the student's own detached step-0 estimate, which matches inference)")
    ap.add_argument("--base", default=SDXS, help="UNet to start from when not --init-from (default SDXS; e.g. stabilityai/sd-turbo)")
    ap.add_argument("--te-from", default=None, help="text encoder for the token init (default SDXS; must match the base's "
                    "cross-attention width, e.g. stable-diffusion-v1-5/stable-diffusion-v1-5 for 768-d)")
    ap.add_argument("--max-pairs", type=int, default=0, help="random subset of training pairs (0 = all); val set unchanged")
    ap.add_argument("--expr-frac", type=float, default=0.5, help="with --expr and --max-pairs: share of the subset from expression rows")
    a = ap.parse_args()
    assert a.msteps == 1 or a.param == "direct", "multi-step student implemented for --param direct"
    ts = msteps_ts(a.msteps, a.T)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = ROOT / "weights/student" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    import yaml
    style = yaml.safe_load((ROOT / "styles" / a.style / "style.yaml").read_text())

    vae = AutoencoderKL.from_pretrained(SDXS, subfolder="vae_large", torch_dtype=torch.bfloat16).to(DEV).eval()
    tasks = TASKS + (expr_tasks() if a.expr else [])
    c = build_cache(a.style, vae, a.manifest, tasks)
    del vae
    taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.bfloat16).to(DEV).eval().requires_grad_(False)
    lp = lpips.LPIPS(net="vgg", verbose=False).to(DEV).eval().requires_grad_(False)
    if a.init_from:
        wi = ROOT / "weights/student" / a.init_from
        base_cfg = json.loads((wi / "config.json").read_text())
        assert base_cfg["param"] == a.param and base_cfg["T"] == a.T, "init student must use the same param/T"
        unet = UNet2DConditionModel.from_pretrained(wi / "unet").to(DEV)
        e0 = torch.load(wi / "task_emb.pt")["emb"].float()  # new style: tokens re-trained
        if e0.shape[0] < len(tasks):  # add expression tokens to a 2-token student (init from the text encoder)
            e0 = torch.cat([e0, init_task_emb(style["description"], tasks, base_cfg.get("te_from"))[e0.shape[0]:]])
        emb = torch.nn.Parameter(e0.to(DEV))
    else:
        if a.base == SDXS:
            unet = UNet2DConditionModel.from_pretrained(SDXS, subfolder="unet").to(DEV)
        else:  # e.g. stabilityai/sd-turbo: same SD2.x latent space + 1024-d cross-attn, so the cache and tokens carry over
            unet = UNet2DConditionModel.from_pretrained(a.base, subfolder="unet", variant="fp16").float().to(DEV)
        emb = torch.nn.Parameter(init_task_emb(style["description"], tasks, a.te_from).to(DEV))
        assert emb.shape[-1] == unet.config.cross_attention_dim, \
            f"token width {emb.shape[-1]} != UNet cross-attention {unet.config.cross_attention_dim}: pass --te-from <base repo>"
    if a.lora_rank > 0:
        from peft import LoraConfig, get_peft_model
        tm = ["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2", "conv1", "conv2"]
        unet.requires_grad_(False)
        unet = get_peft_model(unet, LoraConfig(r=a.lora_rank, lora_alpha=a.lora_rank, target_modules=tm, init_lora_weights="gaussian"))
        n_tr = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        print(f"[train] LoRA r={a.lora_rank}: {n_tr / 1e6:.2f}M trainable params ({n_tr * 2 / 2**20:.1f} MB fp16)", flush=True)
    unet.train()
    ac = alphas_cumprod().to(DEV)

    n = len(c["task"])
    val = []
    for k in range(len(tasks)):
        val += [i for i in range(n) if c["task"][i] == k][:32 if k < 2 else 4]
    tr = [i for i in range(n) if i not in set(val)]
    if a.max_pairs and len(tr) > a.max_pairs:  # same selection rule as train_dmd.py (seeded, expression share)
        rng = random.Random(a.seed)
        if a.expr:
            te_, tb_ = [i for i in tr if c["task"][i] >= 2], [i for i in tr if c["task"][i] < 2]
            ne = min(len(te_), int(a.max_pairs * a.expr_frac))
            tr = sorted(rng.sample(te_, ne) + rng.sample(tb_, min(len(tb_), a.max_pairs - ne)))
        else:
            tr = sorted(rng.sample(tr, a.max_pairs))
    print(f"[train] pairs={n} train={len(tr)} val={len(val)} per-task={[(c['task'] == k).sum().item() for k in range(len(tasks))]}", flush=True)
    opt = torch.optim.AdamW([{"params": [p for p in unet.parameters() if p.requires_grad], "lr": a.lr}, {"params": [emb], "lr": a.emb_lr, "weight_decay": 0}],
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    log = open(out / "train_log.jsonl", "w")
    D = ProjD(lp).to(DEV)
    d_opt = torch.optim.Adam(D.heads.parameters(), lr=a.d_lr, betas=(0.0, 0.99))

    def batch(idx, aug):
        zi, zt, im = c["zi"][idx].to(DEV), c["zt"][idx].to(DEV), c["img256"][idx].to(DEV).float() / 127.5 - 1
        e = emb[c["task"][idx].to(DEV)]
        if aug:
            flip = torch.rand(len(idx), device=DEV) < 0.5
            zi, zt, im = (torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x) for x in (zi, zt, im))
        return zi, zt, im, e

    def losses(zi, zt, im, e, k=0, chain=False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if chain:  # validation: the full K-step inference chain (fixed noise seed)
                zp = sample_chain(unet, zi, e, a.param, ts, ac, torch.Generator(DEV).manual_seed(0))
            elif k == 0:
                zp = forward(unet, zi, e, a.param, ts[0], ac)
            else:  # refinement step k: re-noise (teacher latent | own detached step-0 estimate) to ts[k], predict the target
                with torch.no_grad():
                    own = forward(unet, zi, e, a.param, ts[0], ac).float()
                m = (torch.rand(len(zt), device=DEV) < a.tf_prob).view(-1, 1, 1, 1)
                zp = forward(unet, renoise(torch.where(m, zt, own), ts[k], ac), e, a.param, ts[k], ac)
            mse = F.mse_loss(zp.float(), zt)
            dec = taesd.decode(zp.to(torch.bfloat16)).sample.float().clamp(-1, 1)
            dec = F.interpolate(dec, size=256, mode="bilinear", antialias=True)
            lpl = lp(dec, im).mean()
        return mse, lpl, dec

    t0 = time.time()
    for step in range(1, a.steps + 1):
        idx = torch.tensor(random.sample(tr, a.bs))
        bt = batch(idx, True)
        mse, lpl, dec = losses(*bt, k=random.randrange(a.msteps))
        loss = mse + a.lpips_w * lpl
        gan = a.gan_w > 0 and step > a.gan_start
        if gan:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                g_adv = -D(dec).mean()
            loss = loss + a.gan_w * g_adv
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in unet.parameters() if p.requires_grad] + [emb], 1.0)
        opt.step()
        sched.step()
        if gan:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                d_loss = F.relu(1 - D(bt[2])).mean() + F.relu(1 + D(dec.detach())).mean()
            d_opt.zero_grad(set_to_none=True)
            d_loss.backward()
            d_opt.step()
        if step % 50 == 0:
            r = {"step": step, "mse": round(mse.item(), 5), "lpips": round(lpl.item(), 4), "lr": sched.get_last_lr()[0],
                 "s": round(time.time() - t0, 1)}
            if gan:
                r |= {"g_adv": round(g_adv.item(), 4), "d_loss": round(d_loss.item(), 4)}
            if step % a.val_every == 0 or step == a.steps:
                unet.eval()
                with torch.no_grad():
                    vm, vl = [], []
                    for i in range(0, len(val), 16):
                        m_, l_, _ = losses(*batch(torch.tensor(val[i:i + 16]), False), chain=True)
                        vm.append(m_.item())
                        vl.append(l_.item())
                unet.train()
                r |= {"val_mse": round(sum(vm) / len(vm), 5), "val_lpips": round(sum(vl) / len(vl), 4)}
            print(json.dumps(r), flush=True)
            log.write(json.dumps(r) + "\n")
            log.flush()
        if not math.isfinite(loss.item()):
            print("[train] non-finite loss, abort", flush=True)
            break
    extra = {}
    if a.lora_rank > 0 and not a.init_from:  # LoRA on a frozen PUBLIC base (base probe): save merged, so it loads as a full student
        unet.merge_and_unload().save_pretrained(out / "unet")
        extra = {"lora_rank": 0, "frozen_base_lora_rank": a.lora_rank}
    else:
        unet.save_pretrained(out / ("unet_lora" if a.lora_rank > 0 else "unet"))
    torch.save({"emb": emb.detach().cpu(), "tasks": tasks}, out / "task_emb.pt")
    (out / "config.json").write_text(json.dumps(vars(a) | extra | {"init": a.init_from or a.base, "decoder": "madebyollin/taesd", "encoder": SDXS + "/vae_large",
                                                           "train_pairs": len(tr), "msteps_ts": ts, "minutes": round((time.time() - t0) / 60, 1)}, indent=1))
    print(f"[train] done {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
