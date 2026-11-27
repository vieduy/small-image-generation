"""Stage-2 follow-up sweep A (2026-11-26, user "yes do A first"): early-step image guidance + an attribute-agreement score.
Why (README 2026-11-26 s7 sweep): "img 2 on the first 2 of 8 steps" kept the klein look and visibly fixed attributes that img 1
loses (cap colour, grey beard, hair colour), but SFace likeness barely moved (+0.016) and style missed the rule by 0.005.
Settings (8 steps, fixed phone noise, txt 2.5): baselines base_img1 (tS3 pairs), base_img3_txt4 (tS1 pairs);
  interval g {1.5, 2, 2.5, 3} x k {1, 2, 3} (img g on the first k steps, 1 after). Sampler = s7 sample_gs.
Rows / metrics = s7 + attr (scripts/attr_metric.py: CLIP zero-shot hair colour / age / beard / headwear / glasses, output label ==
photo label, mean of 5 groups) on the 100 synthetic avatars, + skin dL* on the darkest 25 of them (report only).
Pick rule (fixed before running):
  metric sanity: attr(base_img3_txt4) > attr(base_img1), else no pick (the score does not see what guidance adds).
  qualify: style_clip_syn >= klein - 0.01, tex_syn <= 1.10 x klein tex, >= 80/100 faces, expr_all >= expr_all(base_img1) - 0.05.
  pick: max attr (ties within 0.005 -> higher likeness_syn).
  proceed (pairs + tS4 in scripts/s8_chain.sh) only if pick != base_img1 and attr(pick) - attr(base_img1) >= 0.02.
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/s8_attr_sweep.py
Out: results/teacher/tB1_attr_sweep.json, grids/teacher/tB1_attr_sweep.jpg
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
from attr_metric import Attr  # noqa: E402
from eval_e1 import OUTSIDE, expr_faces, to_t  # noqa: E402

ROOT, DEV = TD.ROOT, "cuda"
D = ROOT / "data/pairs/cartoon3d"
SETTINGS = [("base_img1", 1.0, 8, 2.5, False), ("base_img3_txt4", 3.0, 8, 4.0, False)]
SETTINGS += [(f"int_img{g}_k{k}", g, k, 2.5, False) for g in (1.5, 2.0, 2.5, 3.0) for k in (1, 2, 3)]
EXAMPLE_IDS = ["f03664", "f03766", "c02859", "f00495", "f01482"]


def tex(img):
    g = np.asarray(img.convert("L").resize((256, 256)), dtype=np.float32)[70:190, 70:190]
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(np.abs(lap).mean())


@torch.no_grad()
def sample_gs(unet, z_photo, emb, mask, null_emb, null_mask, g_img, k_img, g_txt, apg, steps, noise):
    """TD.sample with img guidance g_img on the first k_img steps (1.0 after), optional APG on the img term (x0 space)."""
    B, dev, dt = z_photo.shape[0], z_photo.device, z_photo.dtype
    x = noise.expand(B, -1, -1, -1).to(dev, torch.float32)
    L = max(emb.shape[1], null_emb.shape[1])
    pad = lambda e, m: (torch.nn.functional.pad(e, (0, 0, 0, L - e.shape[1])), torch.nn.functional.pad(m, (0, L - m.shape[1])))
    e_t, m_t = pad(emb, mask)
    e_n, m_n = pad(null_emb.expand(B, -1, -1), null_mask.expand(B, -1))
    zp3 = torch.cat([torch.zeros_like(z_photo), z_photo, z_photo])
    e3, m3 = torch.cat([e_n, e_n, e_t]), torch.cat([m_n, m_n, m_t])
    S = [TD.shift(s) for s in torch.linspace(1.0, 1.0 / steps, steps).tolist()] + [0.0]
    for i in range(steps):
        s, s_next = S[i], S[i + 1]
        v_u, v_i, v_t = TD.velocity(unet, torch.cat([x] * 3), zp3, e3, m3, torch.full((3 * B,), s, device=dev)).float().chunk(3)
        g = g_img if i < k_img else 1.0
        if apg and g != 1.0:  # x0 = x - s * v
            x_u, x_i, x_t = (x - s * v for v in (v_u, v_i, v_t))
            d = x_i - x_u
            c = x_i.flatten(1)
            par = ((d.flatten(1) * c).sum(1, keepdim=True) / c.pow(2).sum(1, keepdim=True).clamp(min=1e-8) * c).view_as(d)
            x0 = x_i + g_txt * (x_t - x_i) + (g - 1.0) * (d - par)
            v = (x - x0) / s
        else:
            v = v_u + g_txt * (v_t - v_i) + g * (v_i - v_u)
        x = x + (s_next - s) * v
    return x.to(dt)


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
    ex = [next(i for i in av if C["key"][i][0] == pid) for pid in EXAMPLE_IDS]
    rows = syn + ffq + ex
    NS = len(syn) + len(ffq)  # scored rows; the example rows are for the picture only
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
    kd, kc = style(K[:NS])
    k_tex = float(np.mean([tex(im) for im in K[:len(syn)]]))
    p_tex = float(np.mean([tex(Image.open(p).convert("RGB")) for p in photos[:len(syn)]]))
    ref = {"klein_style_clip_syn": round(float(kc[is_syn].mean()), 3), "klein_style_clip_ffhq": round(float(kc[~is_syn].mean()), 3),
           "klein_style_dino_syn": round(float(kd[is_syn].mean()), 3),
           "klein_tex_syn": round(k_tex, 2), "photo_tex_syn": round(p_tex, 2),
           "klein_likeness_syn": None}
    print("[s8] klein reference", ref, flush=True)
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

    kcos = [fid.cos(in_feat[p], fid.feat(o)) for p, o in zip(photos[:len(syn)], K[:len(syn)])]
    ref["klein_likeness_syn"] = round(float(np.mean([c_ for c_ in kcos if c_ is not None])), 3)
    AT = Attr(DEV, cm, pr)
    lab_p = AT.labels([Image.open(p) for p in photos[:len(syn)]])
    ref["klein_attr"] = {g: round(v, 3) for g, v in AT.agree(lab_p, AT.labels(K[:len(syn)])).items()}
    Ls = np.array([in_L[p] if in_L[p] is not None else np.nan for p in photos[:len(syn)]])
    dark = set(np.argsort(np.where(np.isnan(Ls), 1e9, Ls))[:25].tolist())
    print("[s8] klein reference", ref, flush=True)

    def gen(z, k, gi, ki, gt, apg):
        out = []
        with torch.inference_mode():
            for i in range(0, len(z), 48):
                zz = sample_gs(unet, z[i:i + 48], emb_[k[i:i + 48]], mask[k[i:i + 48]], ne, nm, gi, ki, gt, apg, 8, noise)
                out += pil(DLA.decode(vae, zz.to(torch.float16)))
        return out

    res, keep = {}, {}
    for name, gi, ki, gt, apg in SETTINGS:
        if True:
            oa_all, ox = gen(za, ka, gi, ki, gt, apg), gen(zxe, kx, gi, ki, gt, apg)
            oa = oa_all[:NS]
            sd, sc = style(oa)
            _, xc = style(ox)
            cos, dl = [], []
            dld = []
            for j_, (p, o) in enumerate(zip(photos[:len(syn)], oa[:len(syn)])):
                fo = fid.feat(o)
                if fo is not None and in_feat[p] is not None:
                    cos.append(fid.cos(in_feat[p], fo))
                lo = fid.skin_L(o)
                if lo is not None and in_L[p] is not None:
                    dl.append(lo - in_L[p])
                    if j_ in dark:
                        dld.append(lo - in_L[p])
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
                 "tex_syn": round(float(np.mean([tex(o) for o in oa[:len(syn)]])), 2), "expr_all": round(float(hit.mean()), 3), "expr_outside": round(float(hit[:, [ids.index(e) for e in OUTSIDE]].mean()), 3)}
            k = name
            ag = AT.agree(lab_p, AT.labels(oa[:len(syn)]))
            r.update({"attr": round(ag["attr"], 3), "attr_groups": {g: round(v, 3) for g, v in ag.items() if g != "attr"},
                      "skin_dL_dark25": round(float(np.median(dld)), 1) if dld else None, "g_img": gi, "k_img": ki, "g_txt": gt, "apg": apg})
            res[k] = r
            keep[k] = oa_all[NS:]
            print(f"[s8] {k} {r} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    thr, tmax = ref["klein_style_clip_syn"] - 0.01, 1.10 * ref["klein_tex_syn"]
    b1 = res["base_img1"]
    sane = res["base_img3_txt4"]["attr"] > b1["attr"]
    ok = {k: v for k, v in res.items() if v["style_clip_syn"] >= thr and v["tex_syn"] <= tmax and v["n_face"] >= 80
          and v["expr_all"] >= b1["expr_all"] - 0.05} if sane else {}
    best = max((v["attr"] for v in ok.values()), default=None)
    pick = max([k for k in ok if ok[k]["attr"] >= best - 0.005], key=lambda k: ok[k]["likeness_syn"]) if ok else None
    proceed = bool(pick and pick != "base_img1" and res[pick]["attr"] - b1["attr"] >= 0.02)
    out = {"setting": "tB1, 8 steps, fixed phone noise (seed 1234), TinyVAE decode", "klein_reference": ref, "grid": res, "pick": pick,
           "proceed": proceed, "metric_sane": sane, "qualified": sorted(ok),
           "rule": f"sanity attr(img3) > attr(img1); style_clip_syn >= {thr:.3f}, tex_syn <= {tmax:.2f}, >= 80/100 faces, "
                   f"expr_all >= {b1['expr_all'] - 0.05:.3f}; max attr (ties 0.005 -> likeness); proceed if attr gain >= 0.02 over base_img1",
           "minutes": round((time.time() - t0) / 60, 1)}
    (ROOT / "results/teacher/tB1_attr_sweep.json").write_text(json.dumps(out, indent=1))
    print("[s8] pick", pick, "proceed", proceed, res.get(pick), flush=True)

    # picture: the 5 example rows; cols photo | klein | img1 | img3/txt4 | pick | best-likeness qualified other (or best overall)
    others = sorted([k for k in res if k not in ("base_img1", "base_img3_txt4", pick)], key=lambda k: -res[k]["attr"])
    cols = ["base_img1", "base_img3_txt4"] + ([pick] if pick and pick != "base_img1" else []) + others[:3]
    c = 200
    g = Image.new("RGB", ((2 + len(cols)) * c, 22 + len(EXAMPLE_IDS) * (c + 14)), "white")
    d = ImageDraw.Draw(g)
    d.text((4, 4), f"teacher tB1, 8 steps: photo | klein | settings (pick {pick}); id/tex = sweep means", fill="black")
    for r_ in range(len(EXAMPLE_IDS)):
        y = 22 + r_ * (c + 14)
        cells = [(Image.open(photos[NS + r_]), "photo"), (K[NS + r_], "klein")]
        cells += [(keep[k][r_], f"{k} attr {res[k]['attr']:.2f} tex {res[k]['tex_syn']:.0f}") for k in cols]
        for j, (im, cap) in enumerate(cells):
            g.paste(im.convert("RGB").resize((c, c)), (j * c, y))
            d.text((j * c + 3, y + c + 1), cap, fill="black")
    g.save(ROOT / "grids/teacher/tB1_attr_sweep.jpg", quality=90)
    print("[s8] done", out["minutes"], "min", flush=True)


if __name__ == "__main__":
    main()
