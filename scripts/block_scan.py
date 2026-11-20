"""Plan A1 (smaller student, 2026-11-20): which blocks of the one-step student can go? Training-free scan.

Each removable unit is bypassed (output := input) on its own, and we measure on synthetic faces x all 22 options:
  dev   LPIPS(output, full-model output)        quality cost of removing it
  sep   pairwise LPIPS among the 20 expressions  (options must stay distinct)
  clip  CLIP-L 20-way expression accuracy
and its exact share of UNet FLOPs (512 px diptych). Units with a residual path, so bypassing keeps shapes:
  transformer layers (…attentions.i.transformer_blocks.j), whole attention modules (…attentions.i),
  resnets whose in/out channels match (down/mid; up resnets eat a skip feature and are handled when building the model).
Ranking: dev per GFLOP saved. Synthetic faces only (data/faces).
  source env.sh; CUDA_VISIBLE_DEVICES=1 $VENV/bin/python scripts/block_scan.py --tag dl6g2 --decoder decg2
Out: results/block_scan_<tag>.json (+ .txt ranking)
"""
import argparse
import itertools
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEV = "cuda"


def units(unet):
    out = []
    for n, m in unet.named_modules():
        if re.fullmatch(r"(down_blocks\.\d+|mid_block|up_blocks\.\d+)\.attentions\.\d+\.transformer_blocks\.\d+", n):
            out.append((n, "tlayer", m))
        elif re.fullmatch(r"(down_blocks\.\d+|mid_block|up_blocks\.\d+)\.attentions\.\d+", n):
            out.append((n, "attention", m))
        elif re.fullmatch(r"(down_blocks\.\d+|mid_block)\.resnets\.\d+", n):
            if getattr(m, "in_channels", None) == getattr(m, "out_channels", -1) and getattr(m, "conv_shortcut", None) is None:
                out.append((n, "resnet", m))
    return out


class Bypass:
    """forward hooks that make one module return its own input (residual path only)."""

    def __init__(self, m):
        self.m, self.x, self.h = m, None, []

    def on(self):
        def pre(mod, args, kwargs):
            self.x = args[0] if args else kwargs.get("hidden_states", kwargs.get("input_tensor"))

        def post(mod, args, kwargs, out):
            if isinstance(out, tuple):
                return (self.x,) + tuple(out[1:])
            if hasattr(out, "sample"):
                out.sample = self.x
                return out
            return self.x
        self.h = [self.m.register_forward_pre_hook(pre, with_kwargs=True), self.m.register_forward_hook(post, with_kwargs=True)]

    def off(self):
        for h in self.h:
            h.remove()
        self.h = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dl6g2")
    ap.add_argument("--decoder", default="decg2")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    t0 = time.time()
    w = ROOT / "weights/student" / a.tag
    te = torch.load(w / "task_emb.pt")
    tasks = te["tasks"]
    emb, mask = te["emb"].to(DEV, torch.float16), te["mask"].to(DEV)
    unet = DLA.load_unet(w / "unet").to(DEV, torch.float16).eval()
    venc = DLA.load_vae(torch.float16).to(DEV).eval()
    vdec = DLA.load_vae(torch.float16, ROOT / "weights/decoder" / a.decoder).to(DEV).eval()
    U = units(unet)

    from torch.utils.flop_counter import FlopCounterMode
    z0 = torch.zeros(1, 4, DLA.LAT, DLA.LAT, device=DEV, dtype=torch.float16)
    fc = FlopCounterMode(display=False, depth=None)
    with torch.no_grad(), fc:
        DLA.forward(unet, z0, emb[:1], mask[:1])
    total = fc.get_total_flops()
    per = {k: sum(v.values()) for k, v in fc.get_flop_counts().items()}
    pre = type(unet).__name__ + "."  # FlopCounterMode names modules '<ClassName>.<path>'
    gf = {n: per.get(pre + n, per.get(n, 0)) / 1e9 for n, _, _ in U}
    miss = [n for n in gf if gf[n] == 0]
    print(f"[scan] {a.tag}: {len(U)} units ({sum(k == 'tlayer' for _, k, _ in U)} transformer layers, "
          f"{sum(k == 'attention' for _, k, _ in U)} attention modules, {sum(k == 'resnet' for _, k, _ in U)} resnets); "
          f"UNet {total / 1e9:.0f} GFLOPs; no FLOP count for {miss[:3]}", flush=True)

    EX = yaml.safe_load((ROOT / "configs/sticker_expressions.yaml").read_text())
    ex_k = [tasks.index(f"expr_{e['id']}") for e in EX]
    faces = sorted((ROOT / "data/faces").glob("*.png"))[:100]
    faces = [faces[int(i)] for i in np.linspace(0, len(faces) - 1, a.n)]
    with torch.no_grad():
        zs = [DLA.encode(venc, torch.from_numpy(np.array(Image.open(f).convert("RGB").resize((512, 512)))).permute(2, 0, 1)[None]
                         .to(DEV, torch.float16) / 127.5 - 1) for f in faces]

    import lpips
    from transformers import CLIPModel, CLIPProcessor
    import metrics as M
    lp = lpips.LPIPS(net="alex", verbose=False).to(DEV).eval()
    cp = M._snap("openai/clip-vit-large-patch14")
    cm, pr = CLIPModel.from_pretrained(str(cp), dtype=torch.float16).to(DEV).eval(), CLIPProcessor.from_pretrained(str(cp))
    with torch.no_grad():
        ti = pr(text=[f"a cartoon sticker of a person, {e['phrase']}" for e in EX], return_tensors="pt", padding=True).to(DEV)
        tf = cm.text_projection(cm.text_model(**ti).pooler_output).float()
        tf = tf / tf.norm(dim=-1, keepdim=True)
    cmean = torch.tensor(pr.image_processor.image_mean, device=DEV).view(1, 3, 1, 1)
    cstd = torch.tensor(pr.image_processor.image_std, device=DEV).view(1, 3, 1, 1)
    pairs = list(itertools.combinations(range(len(ex_k)), 2))
    pi, pj = [ex_k[i] for i, _ in pairs], [ex_k[j] for _, j in pairs]

    def run():
        with torch.no_grad():
            return [torch.nn.functional.interpolate(
                DLA.decode(vdec, DLA.forward(unet, z.expand(len(tasks), -1, -1, -1), emb, mask)).float().clamp(-1, 1),
                size=256, mode="bilinear", antialias=True) for z in zs]

    def clip_acc(outs):
        hit = []
        with torch.no_grad():
            for y in outs:
                im = torch.nn.functional.interpolate((y[ex_k] + 1) / 2, size=224, mode="bicubic", antialias=True).clamp(0, 1)
                f = cm.visual_projection(cm.vision_model(pixel_values=((im - cmean) / cstd).half()).pooler_output).float()
                f = f / f.norm(dim=-1, keepdim=True)
                hit += ((f @ tf.T).argmax(1) == torch.arange(len(EX), device=DEV)).tolist()
        return float(np.mean(hit))

    def score(outs, ref):
        with torch.no_grad():
            return {"dev": round(float(np.mean([lp(x, r).mean().item() for x, r in zip(outs, ref)])), 4),
                    "sep": round(float(np.mean([lp(y[pi], y[pj]).mean().item() for y in outs])), 4), "clip": round(clip_acc(outs), 4)}

    ref = run()
    res = {"full": score(ref, ref) | {"gflops": round(total / 1e9, 1)}}
    print(f"[scan] full {res['full']}", flush=True)
    for i, (n, kind, m) in enumerate(U):
        b = Bypass(m)
        b.on()
        try:
            res[n] = {"kind": kind, "gflops": round(gf[n], 2), "share": round(gf[n] * 1e9 / total, 4)} | score(run(), ref)
        except Exception as e:  # shape mismatch etc.: not bypassable
            res[n] = {"kind": kind, "error": str(e)[:120]}
        b.off()
        if i % 10 == 0 or "error" in res[n]:
            print(f"[scan] {i + 1}/{len(U)} {n} {res[n]}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    rows = sorted([(v["dev"] / max(v["gflops"], 1e-3), k, v) for k, v in res.items() if k != "full" and "dev" in v])
    lines = [f"{'unit':58s} {'kind':9s} {'GFLOPs':>7s} {'dev':>6s} {'sep':>6s} {'clip':>6s} dev/GFLOP",
             f"{'full':58s} {'':9s} {res['full']['gflops']:7.1f} {0:6.3f} {res['full']['sep']:6.3f} {res['full']['clip']:6.3f}"]
    lines += [f"{k:58s} {v['kind']:9s} {v['gflops']:7.2f} {v['dev']:6.3f} {v['sep']:6.3f} {v['clip']:6.3f} {r:.4f}" for r, k, v in rows]
    (ROOT / f"results/block_scan_{a.tag}.json").write_text(json.dumps({"faces": [f.stem for f in faces], "units": res}, indent=1))
    (ROOT / f"results/block_scan_{a.tag}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:25]), flush=True)
    print(f"[scan] done {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
