"""Paper (idea B, 2026-11-19): where in the UNet does the menu option (task token) act? Training-free map.

For a trained one-step student (default dl6g2), every cross-attention module (attn2, 36 of them) can read either its task token
(normal) or one NEUTRAL context (the 'avatar' token, or the mean of all option tokens). We switch modules to neutral and measure
how the 22 outputs per photo change:
  dev      LPIPS(output, normal output)                      how much the option still changes this output
  sep      mean pairwise LPIPS among the 20 expression outputs of one photo   (collapse -> 0)
  clip     CLIP-L 20-way expression accuracy (same texts as expr_eval.py)
Configs: full (reference); prefix-k (first k attn2 modules in EXECUTION order neutral -> everything before module k+1 can run
once per photo); leave-one-out (only module i neutral); per group (down0/down1/down2/mid/up0/up1 neutral, or only that group
conditioned). Each prefix point also gets its exact shareable-FLOPs fraction and the per-set speedup 1/(s/22 + 1 - s).
Synthetic faces only (data/faces).
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/influence_map.py --tag dl6g2 --decoder decg2 --neutral avatar
Out: results/influence_<tag>_<neutral>.json, grids/influence_<tag>_<neutral>.png (curves), grids/influence_<tag>_<neutral>_examples.jpg
"""
import argparse
import itertools
import json
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


class SwitchProc:
    """attn2 processor wrapper: when .neutral is set, the context/mask are replaced by the UNet's neutral context."""

    def __init__(self, inner, unet):
        self.inner, self.unet, self.neutral = inner, [unet], False

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kw):
        if self.neutral and encoder_hidden_states is not None:
            u, B = self.unet[0], hidden_states.shape[0]
            encoder_hidden_states = u._neutral_proj.expand(B, -1, -1)
            attention_mask = u._neutral_bias.expand(B, -1, -1)
        return self.inner(attn, hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, **kw)


def group_of(name):
    p = name.split(".")
    return {"down_blocks": f"down{p[1]}", "up_blocks": f"up{p[1]}", "mid_block": "mid"}[p[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dl6g2")
    ap.add_argument("--decoder", default="decg2")
    ap.add_argument("--neutral", default="avatar", choices=["avatar", "mean"])
    ap.add_argument("--n", type=int, default=16, help="synthetic faces (spread over data/faces[:100])")
    a = ap.parse_args()
    t0 = time.time()
    name = f"{a.tag}_{a.neutral}"
    w = ROOT / "weights/student" / a.tag
    te = torch.load(w / "task_emb.pt")
    tasks = te["tasks"]
    emb, mask = te["emb"].to(DEV, torch.float16), te["mask"].to(DEV)
    unet = DLA.load_unet(w / "unet").to(DEV, torch.float16).eval()
    venc = DLA.load_vae(torch.float16).to(DEV).eval()
    vdec = DLA.load_vae(torch.float16, ROOT / "weights/decoder" / a.decoder).to(DEV).eval() if a.decoder else venc

    if a.neutral == "avatar":
        ne, nm = emb[tasks.index("avatar")][None], mask[tasks.index("avatar")][None].float()
    else:  # mean over all option tokens, union mask
        ne, nm = emb.float().mean(0, keepdim=True).half(), mask.float().max(0, keepdim=True).values
    mods = [(n, m) for n, m in unet.named_modules() if n.endswith("attn2")]
    procs = {}
    for n, m in mods:
        procs[n] = SwitchProc(m.processor, unet)
        m.set_processor(procs[n])

    def pre(module, args, kwargs):
        module._neutral_proj = module.process_encoder_hidden_states(ne, kwargs.get("added_cond_kwargs"))
        module._neutral_bias = ((1 - nm.to(module._neutral_proj.dtype)) * -10000.0).unsqueeze(1)
    unet.register_forward_pre_hook(pre, with_kwargs=True)

    # execution order of attn2 modules + cumulative FLOPs at each one's call (= shareable compute if it is the first conditioned)
    from torch.utils.flop_counter import FlopCounterMode
    order, cum = [], []
    fc = FlopCounterMode(display=False)
    def tap(n):
        def h(mod, args):  # must return None (a returned value replaces the module's args)
            order.append(n)
            cum.append(fc.get_total_flops())
        return h
    hooks = [m.register_forward_pre_hook(tap(n)) for n, m in mods]
    z0 = torch.zeros(1, 4, DLA.LAT, DLA.LAT, device=DEV, dtype=torch.float16)
    with torch.no_grad(), fc:
        DLA.forward(unet, z0, emb[:1], mask[:1])
    total = fc.get_total_flops()
    for h in hooks:
        h.remove()
    assert sorted(order) == sorted(n for n, _ in mods), "an attn2 module ran twice or not at all"
    share = {n: c / total for n, c in zip(order, cum)}  # fraction of UNet FLOPs before module n
    print(f"[infl] {len(order)} attn2 modules, UNet {total / 1e9:.0f} GFLOPs; first in up path at "
          f"{min(share[n] for n in order if n.startswith('up')):.3f}", flush=True)

    EX = yaml.safe_load((ROOT / "configs/sticker_expressions.yaml").read_text())
    ex_k = [tasks.index(f"expr_{e['id']}") for e in EX]
    faces = sorted((ROOT / "data/faces").glob("*.png"))[:100]
    faces = [faces[int(i)] for i in np.linspace(0, len(faces) - 1, a.n)]
    zs = []
    with torch.no_grad():
        for f in faces:
            x = torch.from_numpy(np.array(Image.open(f).convert("RGB").resize((512, 512)))).permute(2, 0, 1)[None].to(DEV, torch.float16) / 127.5 - 1
            zs.append(DLA.encode(venc, x))

    groups = ["down0", "down1", "down2", "mid", "up0", "up1"]
    cfgs = {"full": set()}
    for k in range(1, len(order) + 1):
        cfgs[f"prefix{k:02d}"] = set(order[:k])
    for n in order:
        cfgs[f"only:{n}"] = {n}
    for g in groups:
        cfgs[f"group:{g}"] = {n for n in order if group_of(n) == g}
        cfgs[f"keep:{g}"] = {n for n in order if group_of(n) != g}

    import lpips
    from transformers import CLIPModel, CLIPProcessor
    import metrics as M
    lp = lpips.LPIPS(net="alex", verbose=False).to(DEV).eval()
    cp = M._snap("openai/clip-vit-large-patch14")
    cm, pr = CLIPModel.from_pretrained(str(cp), dtype=torch.float16).to(DEV).eval(), CLIPProcessor.from_pretrained(str(cp))
    with torch.no_grad():
        ti = pr(text=[f"a cartoon sticker of a person, {e['phrase']}" for e in EX], return_tensors="pt", padding=True).to(DEV)
        tf = cm.text_projection(cm.text_model(**ti).pooler_output).float()  # (get_text_features returns an output object here)
        tf = tf / tf.norm(dim=-1, keepdim=True)
    cmean = torch.tensor(pr.image_processor.image_mean, device=DEV).view(1, 3, 1, 1)
    cstd = torch.tensor(pr.image_processor.image_std, device=DEV).view(1, 3, 1, 1)

    def run(neutral_set):
        for n, p in procs.items():
            p.neutral = n in neutral_set
        outs = []
        with torch.no_grad():
            for z in zs:
                y = DLA.decode(vdec, DLA.forward(unet, z.expand(len(tasks), -1, -1, -1), emb, mask)).float().clamp(-1, 1)
                outs.append(y)  # [T, 3, 512, 512] in [-1, 1]
        return outs

    def clip_acc(outs):
        hit = []
        with torch.no_grad():
            for y in outs:
                im = torch.nn.functional.interpolate((y[ex_k] + 1) / 2, size=224, mode="bicubic", antialias=True).clamp(0, 1)
                f = cm.visual_projection(cm.vision_model(pixel_values=((im - cmean) / cstd).half()).pooler_output).float()
                f = f / f.norm(dim=-1, keepdim=True)
                hit += ((f @ tf.T).argmax(1) == torch.arange(len(EX), device=DEV)).tolist()
        return float(np.mean(hit))

    pairs = list(itertools.combinations(range(len(ex_k)), 2))
    pi, pj = [ex_k[i] for i, _ in pairs], [ex_k[j] for _, j in pairs]

    def small(y):
        return torch.nn.functional.interpolate(y, size=256, mode="bilinear", antialias=True)

    ref = run(set())
    ref_s = [small(y) for y in ref]
    res = {}
    for ci, (cn, ns) in enumerate(cfgs.items()):
        outs = ref if cn == "full" else run(ns)
        with torch.no_grad():
            s = [small(y) for y in outs]
            dev = float(np.mean([lp(a_, b_).mean().item() for a_, b_ in zip(s, ref_s)]))
            dev_av = float(np.mean([lp(a_[:1], b_[:1]).item() for a_, b_ in zip(s, ref_s)]))
            sep = float(np.mean([lp(y[pi], y[pj]).mean().item() for y in s]))
        first_cond = next((n for n in order if n not in ns), None)
        sh = share[first_cond] if first_cond else 1.0
        res[cn] = {"n_neutral": len(ns), "dev": round(dev, 4), "dev_avatar": round(dev_av, 4), "sep": round(sep, 4),
                   "clip": round(clip_acc(outs), 4), "share": round(sh, 4), "speedup22": round(1 / (sh / 22 + 1 - sh), 3)}
        if cn.startswith(("full", "prefix", "group", "keep")):
            print(f"[infl] {ci + 1}/{len(cfgs)} {cn:22s} {res[cn]}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if cn in ("prefix14", "prefix18", "group:up0", "group:up1"):  # example sheet rows: face 0, 6 options
            res.setdefault("_examples", {})[cn] = outs[0][[0, 1] + ex_k[:4]].cpu()
    ex = res.pop("_examples")
    rows = [torch.cat(list(ref[0][[0, 1] + ex_k[:4]].cpu()), 2)] + [torch.cat(list(v), 2) for v in ex.values()]
    g = ((torch.cat(rows, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(g).resize((g.shape[1] // 2, g.shape[0] // 2)).save(ROOT / f"grids/influence_{name}_examples.jpg", quality=88)
    meta = {"tag": a.tag, "decoder": a.decoder, "neutral": a.neutral, "faces": [f.stem for f in faces], "order": order,
            "share_before": {n: round(share[n], 4) for n in order}, "gflops": round(total / 1e9, 1),
            "example_rows": ["full"] + list(ex), "minutes": round((time.time() - t0) / 60, 1)}
    (ROOT / f"results/influence_{name}.json").write_text(json.dumps({"meta": meta, "configs": res}, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    pk = sorted(k for k in res if k.startswith("prefix"))
    xs = [res[k]["share"] for k in pk]
    for i_, (m_, lab) in enumerate([("clip", "CLIP expression acc (20-way)"), ("sep", "option separation (pairwise LPIPS)"),
                                    ("dev", "change vs full model (LPIPS)")]):
        ax[i_].plot(xs, [res[k][m_] for k in pk], "o-", ms=3)
        ax[i_].axhline(res["full"][m_], ls="--", c="gray", lw=1)
        ax[i_].set_xlabel("shared (task-free) fraction of UNet FLOPs"); ax[i_].set_title(lab, fontsize=10)
        for k in pk:
            if group_of(order[int(k[6:]) - 1]) != group_of(order[min(int(k[6:]), len(order) - 1)]):
                ax[i_].annotate(group_of(order[int(k[6:]) - 1]) + "|", (res[k]["share"], res[k][m_]), fontsize=7)
    fig.suptitle(f"{a.tag}: first-k cross-attention modules read a neutral '{a.neutral}' context (training-free)", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / f"grids/influence_{name}.png", dpi=130)
    print(f"[infl] done {meta['minutes']} min -> results/influence_{name}.json", flush=True)


if __name__ == "__main__":
    main()
