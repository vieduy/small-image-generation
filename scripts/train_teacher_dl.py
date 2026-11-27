"""Stage 1 (2026-11-23): fine-tune DreamLite-base (28-step, undistilled) into a klein-quality multi-step TEACHER at 512 px.

Data: the student's training cache (data/pairs/<style>/latents_dl_<manifest>.pt + _flip.pt: TinyVAE latents of photo zi and
klein target zt; DreamLite-base uses the byte-identical TinyVAE). Same rows / val split as train_dl.py (first 32 avatar, 32
sticker, 4 per expression of each style).
Loss: flow-matching velocity MSE, || v_hat - (eps - x0) ||^2, sigma ~ shifted logit-normal (teacher_dl.train_sigmas), random eps.
Condition dropout (InstructPix2Pix recipe, keeps both CFGs usable): 5% photo -> zeros, 5% option tokens -> null tokens, 5% both.
Optimiser: AdamW, UNet --lr 1e-5 (dl6g2's fine-tune lr; unverified for this model), tokens --emb-lr 1e-3, cosine, 500 warm-up,
bf16 autocast, grad clip 1, EMA 0.999 of the UNet (sampled + saved). Steps: --steps 20000 is NOT curve-based (scale copied
from dl3F's 16k); val velocity loss + 28-step sample sheets every --ckpt-every steps decide.
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/train_teacher_dl.py --tag tB1
Out: weights/teacher/<tag>/{step_*/, final/}{unet (EMA, fp16), tokens.pt}, train_log.jsonl, samples_*.jpg, config.json
"""
import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402
import teacher_dl as TD  # noqa: E402
from train_student import TASKS, expr_tasks  # noqa: E402

ROOT = TD.ROOT
DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--styles", default="cartoon3d")
    ap.add_argument("--manifest", default="manifest_v2.jsonl")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--emb-lr", type=float, default=1e-3)
    ap.add_argument("--drop", type=float, default=0.05, help="each of: photo-only, tokens-only, both dropped")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--ckpt-every", type=int, default=2500)
    ap.add_argument("--g-img", type=float, default=2.0)
    ap.add_argument("--g-txt", type=float, default=2.0)
    ap.add_argument("--max-rows", type=int, default=0, help="smoke test")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = ROOT / "weights/teacher" / a.tag
    assert not (out / "final").exists(), f"{out} already trained; refusing to overwrite"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    base_tasks = TASKS + expr_tasks()
    styles = a.styles.split(",")
    tasks = base_tasks if len(styles) == 1 else [f"{s}/{t}" if si else t for si, s in enumerate(styles) for t in base_tasks]
    st = Path(a.manifest).stem

    # ---- data: student caches (TinyVAE latents), mmap ----
    ZI, ZT, ZIF, ZTF, TK, VAL, TR, PATHS = [], [], [], [], [], [], [], []
    off = 0
    for si, s in enumerate(styles):
        d = ROOT / "data/pairs" / s
        C = torch.load(d / f"latents_dl_{st}.pt", mmap=True)
        FL = torch.load(d / f"latents_dl_{st}_flip.pt", mmap=True)
        assert FL["key"] == C["key"], f"{s}: flip cache is stale"
        n = len(C["key"]) if not a.max_rows else min(a.max_rows, len(C["key"]))
        ZI.append(C["zi"][:n]); ZT.append(C["zt"][:n]); ZIF.append(FL["zi_f"][:n]); ZTF.append(FL["zt_f"][:n])
        tk = C["task"][:n] + si * len(base_tasks)
        TK.append(tk)
        PATHS += [d / k[2] for k in C["key"][:n]]
        tl = C["task"][:n].tolist()
        vv = []
        for k in range(len(base_tasks)):
            vv += [i for i in range(n) if tl[i] == k][:32 if k < 2 else 4]
        vs = set(vv)
        VAL += [off + i for i in vv]
        TR += [off + i for i in range(n) if i not in vs]
        off += n
        print(f"[teacher] {s}: {n} pairs, val {len(vv)}", flush=True)
    ZI, ZT, ZIF, ZTF, TK = torch.cat(ZI), torch.cat(ZT), torch.cat(ZIF), torch.cat(ZTF), torch.cat(TK)

    # ---- tokens: option tokens from the student's Qwen3-VL init (same text encoder), null tokens = empty edit prompt ----
    emb0, mask = [], []
    for s in styles:
        t = torch.load(ROOT / f"weights/student/_dl_tokens_{s}_{len(base_tasks)}.pt")
        assert t["tasks"] == base_tasks
        emb0.append(t["emb"]); mask.append(t["mask"])
    L = max(e.shape[1] for e in emb0)
    emb0 = torch.cat([F.pad(e, (0, 0, 0, L - e.shape[1])) for e in emb0])
    mask = torch.cat([F.pad(m, (0, L - m.shape[1])) for m in mask])
    photos = [ROOT / "data/pairs/cartoon3d/inputs" / f"f{i:05d}.png" for i in range(0, 1600, 100)]
    ne, nm = TD.null_tokens(photos, DEV)
    emb = torch.nn.Parameter(emb0.float().to(DEV)); mask = mask.to(DEV)
    null_emb = torch.nn.Parameter(ne.float().to(DEV)); null_mask = nm.to(DEV)

    from diffusers import DreamLiteUNetModel
    unet = DreamLiteUNetModel.from_pretrained(TD.BASE_REPO, subfolder="unet", revision=TD.BASE_REV, torch_dtype=torch.float32).to(DEV)
    unet.train()
    ema = copy.deepcopy(unet).eval().requires_grad_(False)
    opt = torch.optim.AdamW([{"params": unet.parameters(), "lr": a.lr},
                             {"params": [emb, null_emb], "lr": a.emb_lr, "weight_decay": 0}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    vae = DLA.load_vae(torch.float32).to(DEV).eval()
    print(f"[teacher] {a.tag}: base UNet {sum(p.numel() for p in unet.parameters()) / 1e6:.0f}M, tokens {tuple(emb.shape)}, "
          f"null {tuple(null_emb.shape)}, train {len(TR)} val {len(VAL)}, styles {styles}", flush=True)

    def batch(idx, flip):
        f = flip.view(-1, 1, 1, 1)
        zi = torch.where(f, ZIF[idx], ZI[idx]).to(DEV).float()
        zt = torch.where(f, ZTF[idx], ZT[idx]).to(DEV).float()
        return zi, zt, TK[idx].to(DEV)

    def ctx(tk, drop_txt):
        L_ = max(emb.shape[1], null_emb.shape[1])
        e_o = F.pad(emb[tk], (0, 0, 0, L_ - emb.shape[1])); m_o = F.pad(mask[tk], (0, L_ - mask.shape[1]))
        e_n = F.pad(null_emb, (0, 0, 0, L_ - null_emb.shape[1])).expand(len(tk), -1, -1)
        m_n = F.pad(null_mask, (0, L_ - null_mask.shape[1])).expand(len(tk), -1)
        d = drop_txt.view(-1, 1, 1)
        return torch.where(d, e_n, e_o), torch.where(drop_txt.view(-1, 1), m_n, m_o)

    g_val = torch.Generator().manual_seed(123)
    v_sig = torch.tensor([0.25, 0.5, 0.75, 1.0])

    @torch.no_grad()
    def evaluate(step):
        ls = []
        for i in range(0, min(len(VAL), 128), 32):
            idx = torch.tensor(VAL[i:i + 32])
            zi, zt, tk = batch(idx, torch.zeros(len(idx), dtype=torch.bool))
            e, m = ctx(tk, torch.zeros(len(idx), dtype=torch.bool, device=DEV))
            for s in v_sig:
                eps = torch.randn(zt.shape, generator=g_val).to(DEV)
                sg = torch.full((len(idx),), float(s), device=DEV)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v = TD.velocity(ema, (1 - s) * zt + s * eps, zi, e, m, sg).float()
                ls.append(F.mse_loss(v, eps - zt).item())
        sel = [VAL[j] for j in list(range(0, 3)) + list(range(32, 35)) + list(range(64, 128, 16)) if j < len(VAL)][:10]
        idx = torch.tensor(sel)
        zi, zt, tk = batch(idx, torch.zeros(len(idx), dtype=torch.bool))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x0 = TD.sample(ema, zi, emb[tk], mask[tk], null_emb, null_mask, a.g_img, a.g_txt, generator=torch.Generator().manual_seed(0))
        rows = [DLA.decode(vae, z.float()).clamp(-1, 1) for z in (zi, x0, zt)]
        g = torch.cat([torch.cat(list(r), 2) for r in rows], 1)
        Image.fromarray(((g + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()).resize((g.shape[2] // 4, g.shape[1] // 4)).save(
            out / f"samples_{step:05d}.jpg", quality=85)
        return {"val_v_mse": round(float(np.mean(ls)), 5)}

    def save(name):
        p = out / name
        p.mkdir(parents=True, exist_ok=True)
        copy.deepcopy(ema).to(torch.float16).save_pretrained(p / "unet")  # fp16 copy; EMA master stays fp32
        torch.save({"emb": emb.detach().cpu(), "mask": mask.cpu(), "null_emb": null_emb.detach().cpu(), "null_mask": null_mask.cpu(),
                    "tasks": tasks, "styles": styles, "g_img": a.g_img, "g_txt": a.g_txt}, p / "tokens.pt")

    log = open(out / "train_log.jsonl", "a")
    r0 = {"step": 0} | evaluate(0)
    print(json.dumps(r0), flush=True); log.write(json.dumps(r0) + "\n"); log.flush()
    ts = time.time()
    for step in range(1, a.steps + 1):
        idx = torch.tensor(random.sample(TR, a.bs))
        zi, zt, tk = batch(idx, torch.rand(a.bs) < 0.5)
        r = torch.rand(a.bs, device=DEV)
        drop_img = r < 2 * a.drop                      # [0, 2d): photo dropped (incl. both)
        drop_txt = (r >= a.drop) & (r < 3 * a.drop)    # [d, 3d): tokens dropped (incl. both)
        zi = torch.where(drop_img.view(-1, 1, 1, 1), torch.zeros_like(zi), zi)
        e, m = ctx(tk, drop_txt)
        sg = TD.train_sigmas(a.bs, DEV)
        eps = torch.randn_like(zt)
        s4 = sg.view(-1, 1, 1, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = TD.velocity(unet, (1 - s4) * zt + s4 * eps, zi, e, m, sg).float()
        loss = F.mse_loss(v, eps - zt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(unet.parameters()) + [emb, null_emb], 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), unet.parameters()):
                pe.lerp_(pm, 1 - a.ema)
        if not math.isfinite(loss.item()):
            print("[teacher] non-finite loss, abort", flush=True)
            break
        if step % 50 == 0:
            rr = {"step": step, "loss": round(loss.item(), 5), "lr": sched.get_last_lr()[0], "s": round(time.time() - ts, 1)}
            if step % a.ckpt_every == 0 or step == a.steps:
                rr |= evaluate(step)
                save(f"step_{step:05d}")
            print(json.dumps(rr), flush=True); log.write(json.dumps(rr) + "\n"); log.flush()
    save("final")
    (out / "config.json").write_text(json.dumps(vars(a) | {"init": f"{TD.BASE_REPO}@{TD.BASE_REV}", "tasks": tasks, "train_pairs": len(TR),
                                                           "minutes": round((time.time() - t0) / 60, 1)}, indent=1))
    print(f"[teacher] done {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
