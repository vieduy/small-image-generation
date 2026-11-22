"""Plan B3 (2026-11-20, user: "queue b3, not to ask me again"): one-step Sana-Sprint 0.6B phone student, text-free option tokens.

Model/forward: scripts/sana_common.py (x0 = eps - F([eps | photo]) or F(cat[eps, photo]), fixed noise, t = pi/2, guidance 4.5).
Data: data/pairs/<style>/sana_lat_<manifest>.pt (scripts/sana_encode.py: DC-AE latents of photo/target, plain + proper flip) and
the 512 px klein target PNGs. Same rows and val split as train_dl.py (first 32 avatar / 32 sticker / 4 per expression).
Losses (= the DreamLite student recipe minus the eye terms): latent MSE + LPIPS-vgg at 256 + LPIPS on one 256 crop at native
512 (lp512) + DINOv2 critic (hinge, frozen DINOv2-base layers 2,5,8,11) from --gan-start; images decoded by the frozen TINY decoder
(--tiny-dec, the phone decoder; scripts/train_tiny_dcae.py). Optimiser AdamW: transformer --lr, option tokens --emb-lr;
cosine schedule, 200 warm-up steps; bf16 autocast; fp32 master weights.
Steps / lr are NOT curve-based (first Sana student): 15000 steps (DreamLite's dl1 regression stage used 16k from an editing
model; Sana must learn editing), lr 5e-5 (5x the DreamLite fine-tunes, as Sana starts without any editing ability); checkpoints
(fp16) + val sheets every --ckpt-every steps decide whether to extend.
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/train_sana.py --tag sD1 --cond diptych --tiny-dec tdec1r
Out: weights/sana_student/<tag>/{step_*/transformer, final/transformer, task_emb.pt, train_log.jsonl, samples_*.jpg, sana_student.json}
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
import sana_common as SC  # noqa: E402

ROOT = SC.ROOT
DEV = "cuda"


class Targets(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return torch.from_numpy(np.array(Image.open(self.paths[i]).convert("RGB").resize((512, 512)))).permute(2, 0, 1).float() / 127.5 - 1, i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cond", required=True, choices=["diptych", "concat"])
    ap.add_argument("--style", default="cartoon3d")
    ap.add_argument("--manifest", default="manifest_v2.jsonl")
    ap.add_argument("--tiny-dec", default="tdec1r")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--emb-lr", type=float, default=1e-3)
    ap.add_argument("--lpips-w", type=float, default=1.0)
    ap.add_argument("--lp512-w", type=float, default=1.0)
    ap.add_argument("--gan-w", type=float, default=0.2)
    ap.add_argument("--gan-start", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0, help="smoke test: use only the first N rows")
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = ROOT / "weights/sana_student" / a.tag
    assert not (out / "final").exists(), f"{out} already trained; refusing to overwrite"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    d = ROOT / "data/pairs" / a.style
    C = torch.load(d / f"sana_lat_{Path(a.manifest).stem}.pt")
    tasks = C["tasks"]
    n = len(C["key"]) if not a.max_rows else a.max_rows
    paths = [d / k[2] for k in C["key"][:n]]
    val = []
    for k in range(len(tasks)):
        val += [i for i in range(n) if C["task"][i] == k][:32 if k < 2 else 4]
    vs = set(val)
    tr_idx = [i for i in range(n) if i not in vs]

    tr = SC.load_transformer(cond=a.cond).to(DEV).train()
    tok = out / "task_emb_init.pt"
    if tok.exists():
        t = torch.load(tok); emb0, mask = t["emb"], t["mask"]
    else:
        emb0, mask = SC.init_tokens(a.style, tasks, DEV)
        torch.save({"emb": emb0, "mask": mask, "tasks": tasks, "prompts": SC.option_prompts(a.style, tasks)}, tok)
    emb = nn.Parameter(emb0.float().to(DEV))
    mask = mask.to(DEV)
    tiny, tiny_step = SC.load_tiny(a.tiny_dec)
    tdec = tiny.to(DEV).eval().requires_grad_(False)
    noise = SC.fixed_noise(DEV)
    print(f"[sana] {a.tag}: cond={a.cond} transformer {sum(p.numel() for p in tr.parameters()) / 1e6:.1f}M, tokens {tuple(emb.shape)}, "
          f"pairs {n} train {len(tr_idx)} val {len(val)}, tiny decoder {a.tiny_dec}/{tiny_step}", flush=True)

    import lpips
    from torchvision.ops import roi_align
    lp = lpips.LPIPS(net="vgg", verbose=False).to(DEV).eval().requires_grad_(False)
    opt = torch.optim.AdamW([{"params": tr.parameters(), "lr": a.lr}, {"params": [emb], "lr": a.emb_lr, "weight_decay": 0}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
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

    def lp512(y, t):
        k = 256
        cx = torch.randint(48, 512 - 48 - k + 1, (len(t),)).float()
        cy = torch.randint(32, 512 - 64 - k + 1, (len(t),)).float()
        bx = torch.stack([torch.arange(len(t)).float(), cx, cy, cx + k, cy + k], 1).to(DEV)
        return lp(roi_align(y, bx, k, aligned=True), roi_align(t, bx, k, aligned=True)).mean()

    def batch_lat(idx, flip):
        f = flip.view(-1, 1, 1, 1)
        zi = torch.where(f, C["zi_f"][idx], C["zi"][idx]).to(DEV).float()
        zt = torch.where(f, C["zt_f"][idx], C["zt"][idx]).to(DEV).float()
        return zi, zt, C["task"][idx].to(DEV)

    def run(zi, tk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x0 = SC.forward(tr, zi, emb[tk], mask[tk], a.cond, noise)
            y = tdec.decode(x0.to(torch.bfloat16)).sample.float().clamp(-1, 1)
        return x0.float(), y

    vt = torch.stack([Targets(paths)[i][0] for i in val])

    @torch.no_grad()
    def evaluate(step):
        tr.eval()
        ls, lps = [], []
        for i in range(0, len(val), 16):
            idx = torch.tensor(val[i:i + 16])
            zi, zt, tk = batch_lat(idx, torch.zeros(len(idx), dtype=torch.bool))
            x0, y = run(zi, tk)
            ls.append(F.mse_loss(x0, zt).item())
            lps.append(lp(F.interpolate(y, 256, mode="bilinear", antialias=True), F.interpolate(vt[i:i + 16].to(DEV), 256, mode="bilinear", antialias=True)).mean().item())
        sel = list(range(3)) + list(range(32, 35)) + [j for j in range(64, len(val), 8)][:4]
        idx = torch.tensor([val[j] for j in sel])
        zi, _, tk = batch_lat(idx, torch.zeros(len(idx), dtype=torch.bool))
        _, y = run(zi, tk)
        src = tdec.decode(zi.float()).sample.float().clamp(-1, 1)
        g = torch.cat([torch.cat(list(src), 2), torch.cat(list(y), 2), torch.cat(list(vt[sel].to(DEV)), 2)], 1)
        Image.fromarray(((g + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()).resize((g.shape[2] // 4, g.shape[1] // 4)).save(out / f"samples_{step:05d}.jpg", quality=85)
        tr.train()
        return {"val_mse": round(float(np.mean(ls)), 5), "val_lpips": round(float(np.mean(lps)), 4)}

    def save(name):
        p = out / name
        SC.save_student(tr, p / "transformer")
        torch.save({"emb": emb.detach().cpu(), "mask": mask.cpu(), "tasks": tasks}, p / "task_emb.pt")

    ds = Targets(paths)
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, tr_idx), batch_size=a.bs, shuffle=True, drop_last=True,
                                     num_workers=a.workers, pin_memory=True, persistent_workers=True)
    log = open(out / "train_log.jsonl", "a")
    r0 = {"step": 0} | evaluate(0)
    print(json.dumps(r0), flush=True); log.write(json.dumps(r0) + "\n"); log.flush()
    it, ts = iter(dl), time.time()
    for step in range(1, a.steps + 1):
        try:
            img, idx = next(it)
        except StopIteration:
            it = iter(dl); img, idx = next(it)
        flip = torch.rand(len(idx)) < 0.5
        tgt = torch.where(flip.view(-1, 1, 1, 1), img.flip(-1), img).to(DEV, non_blocking=True)
        zi, zt, tk = batch_lat(idx, flip)
        x0, y = run(zi, tk)
        mse = F.mse_loss(x0, zt)
        lpl = lp(F.interpolate(y, 256, mode="bilinear", antialias=True), F.interpolate(tgt, 256, mode="bilinear", antialias=True)).mean()
        l512 = lp512(y, tgt)
        loss = mse + a.lpips_w * lpl + a.lp512_w * l512
        r = {"mse": mse.item(), "lpips": lpl.item(), "lp512": l512.item()}
        if a.gan_w > 0 and step > a.gan_start:
            heads.requires_grad_(False)
            g = -critic(y).mean()
            heads.requires_grad_(True)
            loss = loss + a.gan_w * g
            r["g_gan"] = g.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(tr.parameters()) + [emb], 1.0)
        opt.step(); sched.step()
        if a.gan_w > 0:
            ld = F.relu(1 - critic(tgt)).mean() + F.relu(1 + critic(y.detach())).mean()
            d_opt.zero_grad(set_to_none=True); ld.backward(); d_opt.step()
            r["d_loss"] = ld.item()
        if not math.isfinite(loss.item()):
            print("[sana] non-finite loss, abort", flush=True)
            break
        if step % 50 == 0:
            rr = {"step": step, "lr": sched.get_last_lr()[0], "s": round(time.time() - ts, 1)} | {k: round(v, 4) for k, v in r.items()}
            if step % a.ckpt_every == 0 or step == a.steps:
                rr |= evaluate(step)
                save(f"step_{step:05d}")
            print(json.dumps(rr), flush=True); log.write(json.dumps(rr) + "\n"); log.flush()
    save("final")
    SC.save_meta(out, vars(a) | {"tasks": tasks, "train_pairs": len(tr_idx), "tiny_decoder_used": f"{a.tiny_dec}/{tiny_step}",
                                 "minutes": round((time.time() - t0) / 60, 1),
                                 "forward": "x0 = eps - F(...) at t=pi/2 (scm_timestep 1), guidance 4.5x0.1, fixed noise seed 1234"})
    print(f"[sana] done {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
