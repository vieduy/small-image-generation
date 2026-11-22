"""Plan B1 (2026-11-20, user "let training sana sprint with decoder tiny"): a phone-sized encoder / decoder for Sana-Sprint's
DC-AE f32c32 latent (DC-AE itself: 153M / 1,295 GFLOPs encoder, 159M / 2,092 GFLOPs decoder at 512 px; results/sana_b0.json).

Tiny model = diffusers AutoencoderTiny (the DreamLite / TAESD family), 64 channels, latent 32 ch, 6 stages (5x down / up: 512 <-> 16):
  encoder blocks [1,3,3,3,3,3], decoder blocks [3,3,3,3,3,1]. Latent space = DC-AE latent * scaling_factor (0.41407), i.e. what
  the Sana transformer reads/writes. Warm start: the 4 high-resolution stages are copied from DreamLite's TinyVAE (same 64-ch
  layers; decoder aligned from the output end, encoder from the input end); the 2 new low-resolution stages and the 32-ch
  latent convs start from scratch.
  --part dec : tiny_dec(DC-AE enc(x)) -> x.  L1 + LPIPS(vgg, 512) + DINOv2 critic (train_decoder.py recipe) on images =
               klein targets of cartoon3d / comic_line / chibi (what the student will output) + photos.
  --part enc : tiny_enc(x) -> DC-AE enc(x) (MSE on scaled latents), on photos (what the phone encodes) + some targets.
Hold-out: the 64 cartoon3d targets of scripts/sana_b0.py (seed 0) are never trained on; eval every --eval-every steps on them:
  PSNR / LPIPS(alex) / sharpness ratio of tiny_dec(dc_enc(x)) [dec], dc_dec(tiny_enc(x)) and tiny pair [enc], vs DC-AE 37.6 /
  0.006 / 1.03 and DreamLite TinyVAE 33.6 / 0.016 / 0.92 (gate). Checkpoints at every eval (weights/tiny_dcae/<tag>/step_*).
Steps/lr: NOT curve-based (no prior run of this kind): 12000 steps (decg2 used 10000 from a pretrained decoder; half of this
model is new) and lr 2e-4 (2x our decoder fine-tunes, for the new layers); checkpoints + evals decide whether to extend.
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/train_tiny_dcae.py --part dec --tag tdec1
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SNAP = Path("/mnt/project/kchat/duyhv/sana/hf_cache/hub/models--Efficient-Large-Model--Sana_Sprint_0.6B_1024px_diffusers/snapshots/aa76e7f4f4928f378716b6716a2130fba3caf5b1")
DEV = "cuda"


def tiny_config():
    return dict(in_channels=3, out_channels=3, encoder_block_out_channels=[64] * 6, decoder_block_out_channels=[64] * 6,
                act_fn="relu", upsample_fn="nearest", latent_channels=32, upsampling_scaling_factor=2,
                num_encoder_blocks=[1, 3, 3, 3, 3, 3], num_decoder_blocks=[3, 3, 3, 3, 3, 1], latent_magnitude=3, latent_shift=0.5,
                force_upcast=False, scaling_factor=1.0)


@torch.no_grad()
def warm_start(tiny, tv):
    """Copy DreamLite TinyVAE layers with identical shapes: decoder aligned from the output end, encoder from the input end."""
    n = 0
    for new_seq, old_seq, rev in ((tiny.decoder.layers, tv.decoder.layers, True), (tiny.encoder.layers, tv.encoder.layers, False)):
        nl, ol = list(new_seq), list(old_seq)
        if rev:
            nl, ol = nl[::-1], ol[::-1]
        for a, b in zip(nl, ol):
            if type(a) is not type(b):
                break
            pa, pb = list(a.parameters()), list(b.parameters())
            if [p.shape for p in pa] != [p.shape for p in pb]:
                break
            for x, y in zip(pa, pb):
                x.copy_(y)
            n += 1
    return n


def image_list(holdout):
    """klein targets (kept rows) of the three production styles + photos (cartoon3d inputs, ffhq_norm)."""
    targets, photos = [], []
    for st in ("cartoon3d", "comic_line", "chibi"):
        d = ROOT / "data/pairs" / st
        for l in open(d / "manifest_v2.jsonl"):
            r = json.loads(l)
            if r.get("keep"):
                p = d / r.get("target", f"{r['task']}/{r['id']}.png")
                if str(p) not in holdout:
                    targets.append(p)
    photos = sorted((ROOT / "data/pairs/cartoon3d/inputs").glob("*.png")) + sorted((ROOT / "data/ffhq_norm").glob("*.png"))
    return targets, photos


class Imgs(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = torch.from_numpy(np.array(Image.open(self.paths[i]).convert("RGB").resize((512, 512)))).permute(2, 0, 1).float() / 127.5 - 1
        return x.flip(-1) if random.random() < 0.5 else x


def holdout_set():
    rows = [json.loads(l) for l in open(ROOT / "data/pairs/cartoon3d/manifest_v2.jsonl")]
    rows = [r for r in rows if r.get("keep")]
    rng = np.random.default_rng(0)  # = scripts/sana_b0.py
    return [ROOT / "data/pairs/cartoon3d" / r.get("target", f"{r['task']}/{r['id']}.png") for r in (rows[i] for i in rng.choice(len(rows), 64, replace=False))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["dec", "enc"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gan-w", type=float, default=0.1)
    ap.add_argument("--gan-start", type=int, default=1000)
    ap.add_argument("--photo-frac", type=float, default=-1, help="share of photos per batch (default: dec 0.25, enc 0.75)")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = ROOT / "weights/tiny_dcae" / a.tag
    assert not (out / "final").exists(), f"{out} already trained; refusing to overwrite"
    out.mkdir(parents=True, exist_ok=True)
    pf = a.photo_frac if a.photo_frac >= 0 else (0.25 if a.part == "dec" else 0.75)

    from diffusers import AutoencoderDC, AutoencoderTiny
    dc = AutoencoderDC.from_pretrained(SNAP / "vae", torch_dtype=torch.bfloat16).to(DEV).eval().requires_grad_(False)
    SF = dc.config.scaling_factor
    tiny = AutoencoderTiny(**tiny_config())
    nwarm = warm_start(tiny, DLA.load_vae(torch.float32))
    tiny = tiny.to(DEV)
    net = tiny.decoder if a.part == "dec" else tiny.encoder
    net.train()
    other = tiny.encoder if a.part == "dec" else tiny.decoder
    other.requires_grad_(False)
    npar = sum(p.numel() for p in net.parameters())
    from torch.utils.flop_counter import FlopCounterMode
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        tiny.decode(torch.zeros(1, 32, 16, 16, device=DEV)) if a.part == "dec" else tiny.encode(torch.zeros(1, 3, 512, 512, device=DEV))
    gfl = fc.get_total_flops() / 1e9

    hold = holdout_set()
    targets, photos = image_list({str(p) for p in hold})
    print(f"[tiny] part={a.part} params {npar / 1e6:.2f}M, {gfl:.0f} GFLOPs at 512; warm-started layers {nwarm}; "
          f"train images: {len(targets)} targets + {len(photos)} photos (photo share {pf}); hold-out {len(hold)}", flush=True)
    mk = lambda ps, bs: iter(torch.utils.data.DataLoader(Imgs(ps), batch_size=bs, shuffle=True, drop_last=True, num_workers=a.workers // 2,
                                                         pin_memory=True, persistent_workers=True))
    n_ph = max(1, int(round(a.bs * pf))) if pf > 0 else 0
    n_tg = a.bs - n_ph
    it_t = mk(targets, n_tg) if n_tg else None
    it_p = mk(photos, n_ph) if n_ph else None

    import lpips
    lp = lpips.LPIPS(net="vgg", verbose=False).to(DEV).eval().requires_grad_(False)
    lpa = lpips.LPIPS(net="alex", verbose=False).to(DEV).eval().requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    if a.part == "dec" and a.gan_w > 0:  # train_decoder.py critic
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

    def dc_lat(x):
        with torch.no_grad():
            return dc.encode(x.to(torch.bfloat16)).latent.float() * SF

    kk = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=DEV).view(1, 1, 3, 3)
    lapv = lambda y: float(F.conv2d(((y.float() + 1) * 127.5).mean(1, keepdim=True), kk).var())
    hx = torch.stack([torch.from_numpy(np.array(Image.open(p).convert("RGB").resize((512, 512)))).permute(2, 0, 1).float() / 127.5 - 1 for p in hold])

    @torch.no_grad()
    def evaluate(step):
        net.eval()
        R = {}
        combos = {"dec": {"tinydec(dcenc)": lambda x: tiny.decode(dc_lat(x)).sample},
                  "enc": {"dcdec(tinyenc)": lambda x: dc.decode((tiny.encode(x).latents / SF).to(torch.bfloat16)).sample.float(),
                          "latent_mse": None}}[a.part]
        for name, fn in combos.items():
            if fn is None:
                R[name] = round(float(np.mean([F.mse_loss(tiny.encode(hx[i:i + 16].to(DEV)).latents, dc_lat(hx[i:i + 16].to(DEV))).item()
                                               for i in range(0, len(hx), 16)])), 5)
                continue
            ps, lps, lr_ = [], [], []
            for i in range(0, len(hx), 16):
                x = hx[i:i + 16].to(DEV)
                y = fn(x).float().clamp(-1, 1)
                mse = ((y - x) ** 2).mean((1, 2, 3)) * 127.5 ** 2
                ps += (10 * torch.log10(255 ** 2 / mse.clamp_min(1e-10))).tolist()
                lps += lpa(y, x).flatten().tolist()
                lr_ += [lapv(y[j:j + 1]) / max(lapv(x[j:j + 1]), 1e-6) for j in range(len(x))]
            R[name] = {"psnr": round(float(np.mean(ps)), 2), "lpips": round(float(np.mean(lps)), 4), "sharp": round(float(np.mean(lr_)), 3)}
        if a.part == "dec":
            x = hx[:4].to(DEV)
            y = tiny.decode(dc_lat(x)).sample.clamp(-1, 1)
            g = torch.cat([torch.cat(list(x), 2), torch.cat(list(y), 2)], 1)
            Image.fromarray(((g + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()).resize((1024, 512)).save(out / f"samples_{step:05d}.jpg", quality=88)
        net.train()
        return R

    log = open(out / "train_log.jsonl", "w")
    r0 = {"step": 0} | evaluate(0)
    print(json.dumps(r0), flush=True); log.write(json.dumps(r0) + "\n"); log.flush()
    t0 = time.time()
    for step in range(1, a.steps + 1):
        xs = []
        for kind in ("t", "p"):  # epoch wrap (bug fixed 2026-11-20 ~22:00 UTC: tdec1r/tenc1r stopped at the end of their first epoch)
            it_ = it_t if kind == "t" else it_p
            if it_ is None:
                continue
            try:
                xs.append(next(it_))
            except StopIteration:
                it_ = mk(targets, n_tg) if kind == "t" else mk(photos, n_ph)
                if kind == "t":
                    it_t = it_
                else:
                    it_p = it_
                xs.append(next(it_))
        x = torch.cat(xs).to(DEV, non_blocking=True)
        z = dc_lat(x)
        r = {}
        if a.part == "dec":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = tiny.decode(z).sample.float().clamp(-1, 1)
            l1, lpl = (y - x).abs().mean(), lp(y, x).mean()
            loss = l1 + lpl
            r |= {"l1": l1.item(), "lpips": lpl.item()}
            if a.gan_w > 0 and step > a.gan_start:
                heads.requires_grad_(False)
                g = -critic(y).mean()
                heads.requires_grad_(True)
                loss = loss + a.gan_w * g
                r["g_gan"] = g.item()
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                zp = tiny.encode(x).latents.float()
            loss = F.mse_loss(zp, z)
            r["mse"] = loss.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if a.part == "dec" and a.gan_w > 0:
            ld = F.relu(1 - critic(x)).mean() + F.relu(1 + critic(y.detach())).mean()
            d_opt.zero_grad(set_to_none=True); ld.backward(); d_opt.step()
            r["d_loss"] = ld.item()
        if not math.isfinite(loss.item()):
            print("[tiny] non-finite loss, abort", flush=True)
            break
        if step % 100 == 0:
            rr = {"step": step, "lr": sched.get_last_lr()[0], "s": round(time.time() - t0, 1)} | {k: round(v, 4) for k, v in r.items()}
            if step % a.eval_every == 0 or step == a.steps:
                rr |= evaluate(step)
                tiny.save_pretrained(out / f"step_{step:05d}")
            print(json.dumps(rr), flush=True); log.write(json.dumps(rr) + "\n"); log.flush()
    tiny.save_pretrained(out / "final")
    (out / "config_train.json").write_text(json.dumps(vars(a) | {"params_M": round(npar / 1e6, 3), "gflops_512": round(gfl, 1),
                                                                 "warm_layers": nwarm, "n_targets": len(targets), "n_photos": len(photos),
                                                                 "dcae_scaling_factor": SF, "minutes": round((time.time() - t0) / 60, 1)}, indent=1))
    print(f"[tiny] done {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
