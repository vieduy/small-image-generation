"""Step 1 of the teacher style fix (2026-11-25, user "yes do step 1"): re-sweep teacher tB1's CFG WITH a style score.
Why: tB1 pairs (CFG img 3 / txt 4) are less cartoon3d than klein's (style_clip on synthetic training rows 0.667 vs 0.771,
results/style_targets_klein_vs_tB1.json); the stage-2 sweep picked CFG on expr + SFace likeness only, and likeness rises with
image guidance (more photo-like).
Setting = the pair-generation setting (s3_teacher_pairs.py): 8 steps, the phone's fixed noise (seed 1234), DreamLite TinyVAE decode.
Grid: g_img {1.0, 1.5, 2.0, 2.5, 3.0} x g_txt {1.5, 2.5, 4.0}.
Tune sets (training inputs, none in any eval set):
  avatars: 100 synthetic (pools c/f/r) + 50 FFHQ (q) random avatar rows of latents_dl_manifest_v2 (seed 0); the klein target of the
           same rows is the style reference level.
  expressions: the stage-2 tune faces (8 faces x 20 edits, data/faces, disjoint from eval_e1's 24 expression faces).
Metrics per setting: avatar style_clip / style_dino (eval_quality.py metric: cosine to the mean of styles/cartoon3d refs) for
synthetic and FFHQ, avatar SFace likeness + skin dL* (synthetic), expression CLIP 20-way accuracy (all / outside) and sticker style_clip.
Pick rule (fixed before running): among settings with synthetic avatar style_clip >= klein level - 0.01, max expr_all + likeness;
if none qualifies -> report the trade-off, no pick.
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/s6_cfg_style_sweep.py
Out: results/teacher/tB1_cfg_style_sweep.json, grids/teacher/tB1_cfg_style_sweep.jpg
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402
import metrics as M  # noqa: E402
import teacher_dl as TD  # noqa: E402
from eval_e1 import OUTSIDE, expr_faces, to_t  # noqa: E402

ROOT, DEV = TD.ROOT, "cuda"
D = ROOT / "data/pairs/cartoon3d"
G_IMG, G_TXT = (1.0, 1.5, 2.0, 2.5, 3.0), (1.5, 2.5, 4.0)


def src(pid):
    return ROOT / "data/ffhq_norm" / f"{pid}.png" if pid.startswith("q") else D / "inputs" / f"{pid}.png"


def pil(y):
    return [Image.fromarray(a) for a in ((y.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()]


def main():
    t0 = time.time()
    C = torch.load(D / "latents_dl_manifest_v2.pt", mmap=True)
    av = [i for i, k in enumerate(C["key"]) if k[1] == "avatar"]
    rng = random.Random(0)
    syn = rng.sample([i for i in av if not C["key"][i][0].startswith("q")], 100)
    ffq = rng.sample([i for i in av if C["key"][i][0].startswith("q")], 50)
    rows = syn + ffq
    is_syn = np.array([True] * len(syn) + [False] * len(ffq))
    photos = [src(C["key"][i][0]) for i in rows]

    unet, T = TD.load_teacher("tB1")
    unet = unet.to(DEV).eval()
    tasks = T["tasks"]
    emb_, mask = T["emb"].to(DEV, torch.float16), T["mask"].to(DEV)
    ne, nm = T["null_emb"].to(DEV, torch.float16), T["null_mask"].to(DEV)
    vae = DLA.load_vae(torch.float16).to(DEV).eval()
    noise = DLA.fixed_noise(DEV, torch.float32)
    fid, E = M.FaceID(), M.Embedder()
    rd, rc = E([Image.open(p) for p in sorted((ROOT / "styles/cartoon3d").glob("ref_*.png"))])
    rdm, rcm = [torch.nn.functional.normalize(x.mean(0), dim=0) for x in (rd, rc)]

    def style(imgs):
        d, c = zip(*[E(imgs[i:i + 50]) for i in range(0, len(imgs), 50)])
        return (torch.cat(d) @ rdm).cpu().numpy(), (torch.cat(c) @ rcm).cpu().numpy()

    EX = yaml.safe_load((ROOT / "configs/sticker_expressions.yaml").read_text())
    ids = [e["id"] for e in EX]
    from transformers import CLIPModel, CLIPProcessor
    cp = M._snap("openai/clip-vit-large-patch14")
    cm, pr = CLIPModel.from_pretrained(str(cp), dtype=torch.float16).to(DEV).eval(), CLIPProcessor.from_pretrained(str(cp))
    with torch.no_grad():
        ti = pr(text=[f"a cartoon sticker of a person, {e['phrase']}" for e in EX], return_tensors="pt", padding=True).to(DEV)
        tf = cm.text_projection(cm.text_model(**ti).pooler_output).float()
        tf = tf / tf.norm(dim=-1, keepdim=True)
    xf_eval = expr_faces(fid)
    allf = sorted((ROOT / "data/faces").glob("*.png"))[:100]
    tune_x = [p for p in allf if p not in set(xf_eval)][::9][:8]  # = teacher_eval.py --sweep

    with torch.no_grad():  # klein reference (same rows, same decoder)
        K = sum((pil(DLA.decode(vae, C["zt"][rows[i:i + 50]].to(DEV, torch.float16))) for i in range(0, len(rows), 50)), [])
    kd, kc = style(K)
    ref = {"klein_style_clip_syn": round(float(kc[is_syn].mean()), 3), "klein_style_clip_ffhq": round(float(kc[~is_syn].mean()), 3),
           "klein_style_dino_syn": round(float(kd[is_syn].mean()), 3)}
    print("[s6] klein reference", ref, flush=True)
    xa = torch.stack([to_t(Image.open(p)) for p in photos]).to(DEV, torch.float16)
    xx = torch.stack([to_t(Image.open(p)) for p in tune_x]).to(DEV, torch.float16)
    with torch.inference_mode():
        za = torch.cat([DLA.encode(vae, xa[i:i + 50]) for i in range(0, len(xa), 50)])
        zx = DLA.encode(vae, xx)
    ka = torch.full((len(rows),), tasks.index("avatar"), device=DEV)
    zxe = zx.repeat_interleave(len(ids), 0)
    kx = torch.tensor([tasks.index(f"expr_{e}") for _ in tune_x for e in ids], device=DEV)
    in_feat = {p: fid.feat(Image.open(p)) for p in photos[:len(syn)]}
    in_L = {p: fid.skin_L(Image.open(p)) for p in photos[:len(syn)]}

    def gen(z, k, gi, gt):
        out = []
        with torch.inference_mode():
            for i in range(0, len(z), 48):
                zz = TD.sample(unet, z[i:i + 48], emb_[k[i:i + 48]], mask[k[i:i + 48]], ne, nm, gi, gt, steps=8, noise=noise)
                out += pil(DLA.decode(vae, zz.to(torch.float16)))
        return out

    res, keep = {}, {}
    for gi in G_IMG:
        for gt in G_TXT:
            oa, ox = gen(za, ka, gi, gt), gen(zxe, kx, gi, gt)
            sd, sc = style(oa)
            _, xc = style(ox)
            cos, dl = [], []
            for p, o in zip(photos[:len(syn)], oa[:len(syn)]):
                fo = fid.feat(o)
                if fo is not None and in_feat[p] is not None:
                    cos.append(fid.cos(in_feat[p], fo))
                lo = fid.skin_L(o)
                if lo is not None and in_L[p] is not None:
                    dl.append(lo - in_L[p])
            hit = np.zeros((len(tune_x), len(ids)))
            with torch.inference_mode():
                for i in range(len(tune_x)):
                    pv = pr.image_processor(images=ox[i * len(ids):(i + 1) * len(ids)], return_tensors="pt")["pixel_values"].to(DEV, torch.float16)
                    f = cm.visual_projection(cm.vision_model(pixel_values=pv).pooler_output).float()
                    f = f / f.norm(dim=-1, keepdim=True)
                    hit[i] = ((f @ tf.T).argmax(1) == torch.arange(len(ids), device=DEV)).cpu().numpy()
            r = {"style_clip_syn": round(float(sc[is_syn].mean()), 3), "style_clip_ffhq": round(float(sc[~is_syn].mean()), 3),
                 "style_dino_syn": round(float(sd[is_syn].mean()), 3), "sticker_style_clip": round(float(xc.mean()), 3),
                 "likeness_syn": round(float(np.mean(cos)), 3), "n_face": len(cos), "skin_dL_syn": round(float(np.median(dl)), 1) if dl else None,
                 "expr_all": round(float(hit.mean()), 3), "expr_outside": round(float(hit[:, [ids.index(e) for e in OUTSIDE]].mean()), 3)}
            k = f"img{gi}_txt{gt}"
            res[k] = r
            keep[k] = oa[:4]
            print(f"[s6] {k} {r} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    thr = ref["klein_style_clip_syn"] - 0.01
    ok = {k: v for k, v in res.items() if v["style_clip_syn"] >= thr and v["n_face"] >= 80}
    pick = max(ok, key=lambda k: ok[k]["expr_all"] + ok[k]["likeness_syn"]) if ok else None
    out = {"setting": "tB1, 8 steps, fixed phone noise (seed 1234), TinyVAE decode", "klein_reference": ref, "grid": res, "pick": pick,
           "rule": f"style_clip_syn >= {thr:.3f} (klein - 0.01) and >= 80/100 faces detected; then max expr_all + likeness_syn",
           "current_pairs_setting": "img3.0_txt4.0", "minutes": round((time.time() - t0) / 60, 1)}
    (ROOT / "results/teacher/tB1_cfg_style_sweep.json").write_text(json.dumps(out, indent=1))
    print("[s6] pick", pick, res.get(pick), flush=True)

    # grid: 4 synthetic faces; cols photo | klein | g_img 1.0..3.0 at the pick's g_txt (or 4.0)
    gt = float(pick.split("txt")[1]) if pick else 4.0
    c = 160
    g = Image.new("RGB", ((2 + len(G_IMG)) * c, 22 + 4 * (c + 14)), "white")
    d = ImageDraw.Draw(g)
    d.text((4, 4), f"teacher tB1, 8 steps, txt {gt}: photo | klein target | img guidance {G_IMG}  (pick {pick})", fill="black")
    for r_ in range(4):
        y = 22 + r_ * (c + 14)
        cells = [(Image.open(photos[r_]), "photo"), (K[r_], f"klein  style {kc[r_]:.2f}")]
        for gi in G_IMG:
            cells.append((keep[f"img{gi}_txt{gt}"][r_], f"img {gi}"))
        for j, (im, cap) in enumerate(cells):
            g.paste(im.convert("RGB").resize((c, c)), (j * c, y))
            d.text((j * c + 3, y + c + 1), cap, fill="black")
    (ROOT / "grids/teacher").mkdir(parents=True, exist_ok=True)
    g.save(ROOT / "grids/teacher/tB1_cfg_style_sweep.jpg", quality=90)
    print("[s6] done", out["minutes"], "min", flush=True)


if __name__ == "__main__":
    main()
