"""Decide between two early-guidance teacher settings (2026-11-26, user "yes run the test on dev-duyhv").
Sweep A (results/teacher/tB1_attr_sweep.json) could not separate int_img3.0_k1 and int_img2.0_k2 except on dark-skin lightening
(+3.1 vs +7.5 on 25 synthetic faces); the real problem is real photos (+32 on the darkest quarter for every model), untested.
Teacher tB1, avatar, 8 steps, fixed phone noise, txt 2.5 (sampler = s7 sample_gs). Settings: base_img1 (tS3 pairs), int_img3.0_k1,
int_img2.0_k2, and base_img3_txt4 (tS1 pairs) as reference.
1) Real photos: data/real_p2a/eval (300, CelebA-derived, legal-approved, internal only): skin dL* = skin_L(out) - skin_L(photo)
   (metrics.FaceID.skin_L, as skin_compare.py), median on all and on the darkest 25% of inputs (75); + SFace likeness, attr, style_clip.
2) Synthetic: 5 example training rows (grids/teacher/stage2_guidance_examples.jpg) + 20 faces of data/faces_eval2 (every 15th) -> sheet.
Decision rule (stated to the user before running): int_img3.0_k1 if it clearly reduces real dark-quarter lightening vs both
base_img1 and int_img2.0_k2 (>= 5 dL* points lower); otherwise int_img2.0_k2.
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/s9_skin_attr_test.py
Out: results/teacher/tB1_skin_attr_test.json, grids/teacher/tB1_skin_attr_synthetic.jpg, grids/teacher/tB1_skin_attr_real_dark.jpg (internal)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402
import metrics as M  # noqa: E402
import teacher_dl as TD  # noqa: E402
from attr_metric import Attr, tex  # noqa: E402
from eval_e1 import to_t  # noqa: E402
from s7_guidance_sweep import EXAMPLE_IDS, sample_gs  # noqa: E402

ROOT, DEV = TD.ROOT, "cuda"
D = ROOT / "data/pairs/cartoon3d"
SETTINGS = [("base_img1", 1.0, 8, 2.5), ("int_img3.0_k1", 3.0, 1, 2.5), ("int_img2.0_k2", 2.0, 2, 2.5), ("base_img3_txt4", 3.0, 8, 4.0)]


def pil(y):
    return [Image.fromarray(a) for a in ((y.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()]


def main():
    t0 = time.time()
    unet, T = TD.load_teacher("tB1")
    unet = unet.to(DEV).eval()
    emb_, mask = T["emb"].to(DEV, torch.float16), T["mask"].to(DEV)
    ne, nm = T["null_emb"].to(DEV, torch.float16), T["null_mask"].to(DEV)
    vae = DLA.load_vae(torch.float16).to(DEV).eval()
    noise = DLA.fixed_noise(DEV, torch.float32)
    ka = T["tasks"].index("avatar")
    fid, E, AT = M.FaceID(), M.Embedder(), Attr()
    _, rc = E([Image.open(p) for p in sorted((ROOT / "styles/cartoon3d").glob("ref_*.png"))])
    rcm = torch.nn.functional.normalize(rc.mean(0), dim=0)

    def gen(paths, gi, ki, gt):
        out = []
        with torch.inference_mode():
            for i in range(0, len(paths), 48):
                x = torch.stack([to_t(Image.open(p)) for p in paths[i:i + 48]]).to(DEV, torch.float16)
                z = DLA.encode(vae, x)
                k = torch.full((len(x),), ka, device=DEV)
                out += pil(DLA.decode(vae, sample_gs(unet, z, emb_[k], mask[k], ne, nm, gi, ki, gt, False, 8, noise).half()))
        return out

    # 1) real photos
    real = sorted((ROOT / "data/real_p2a/eval").glob("*.png"))[:300]
    rimg = [Image.open(p).convert("RGB") for p in real]
    rL = np.array([fid.skin_L(p) if fid.skin_L(p) is not None else np.nan for p in rimg])
    order = np.argsort(np.where(np.isnan(rL), 1e9, rL))
    dark = order[:75]
    rf = [fid.feat(p) for p in rimg]
    lab_p = AT.labels(rimg)
    res, R_out = {}, {}
    for name, gi, ki, gt in SETTINGS:
        o = gen(real, gi, ki, gt)
        R_out[name] = o
        dl = np.array([(fid.skin_L(x) if fid.skin_L(x) is not None else np.nan) - L for x, L in zip(o, rL)])
        cos = [c for c in (fid.cos(f, fid.feat(x)) for f, x in zip(rf, o)) if c is not None]
        _, c = zip(*[E(o[i:i + 50]) for i in range(0, len(o), 50)])
        ag = AT.agree(lab_p, AT.labels(o))
        res[name] = {"skin_dL_median": round(float(np.nanmedian(dl)), 1), "skin_dL_dark25": round(float(np.nanmedian(dl[dark])), 1),
                     "skin_dL_dark25_mean": round(float(np.nanmean(dl[dark])), 1),
                     "skin_dL_lightest25": round(float(np.nanmedian(dl[order[-75:]])), 1),
                     "likeness": round(float(np.mean(cos)), 3), "attr": round(ag["attr"], 3),
                     "attr_groups": {g: round(v, 3) for g, v in ag.items() if g != "attr"},
                     "style_clip": round(float((torch.cat(c) @ rcm).mean()), 3), "tex": round(float(np.mean([tex(x) for x in o])), 2),
                     "dark_dl_values": [None if np.isnan(v) else round(float(v), 1) for v in dl[dark]]}
        print(f"[s9] real {name} " + json.dumps({k: v for k, v in res[name].items() if k != "dark_dl_values"}), flush=True)
    a, b, c0 = (res[k]["skin_dL_dark25"] for k in ("int_img3.0_k1", "int_img2.0_k2", "base_img1"))
    pick = "int_img3.0_k1" if (a <= b - 5 and a <= c0 - 5) else "int_img2.0_k2"
    # paired bootstrap of the dark-quarter difference (img3_k1 - img2_k2)
    da = np.array([np.nan if v is None else v for v in res["int_img3.0_k1"]["dark_dl_values"]])
    db = np.array([np.nan if v is None else v for v in res["int_img2.0_k2"]["dark_dl_values"]])
    m = ~np.isnan(da) & ~np.isnan(db)
    rng = np.random.default_rng(0)
    bs = [np.median((da[m] - db[m])[rng.integers(0, m.sum(), m.sum())]) for _ in range(2000)]
    diff = {"median_paired_diff": round(float(np.median(da[m] - db[m])), 1), "ci95": [round(float(np.percentile(bs, 2.5)), 1),
            round(float(np.percentile(bs, 97.5)), 1)], "n": int(m.sum())}
    out = {"setting": "tB1 avatar, 8 steps, fixed phone noise; real = data/real_p2a/eval 300 (dark quarter = 75 lowest input skin L*)",
           "real": res, "dark_diff_img3k1_minus_img2k2": diff, "pick": pick,
           "rule": "int_img3.0_k1 if its real dark-quarter median dL* is >= 5 lower than both base_img1 and int_img2.0_k2, else int_img2.0_k2"}
    print("[s9] pick", pick, json.dumps(diff), flush=True)

    # 2) synthetic sheet
    C = torch.load(D / "latents_dl_manifest_v2.pt", mmap=True)
    ex = [D / "inputs" / f"{p}.png" for p in EXAMPLE_IDS]
    syn = ex + sorted((ROOT / "data/faces_eval2").glob("*.png"))[::15][:20]
    cols = ["base_img1", "int_img3.0_k1", "int_img2.0_k2", "base_img3_txt4"]
    S_out = {n: gen(syn, gi, ki, gt) for n, gi, ki, gt in SETTINGS}
    s, per = 170, 5
    rows_ = (len(syn) + per - 1) // per
    for kind, photos, outs, fname, idx in (
            ("synthetic", syn, S_out, "tB1_skin_attr_synthetic.jpg", list(range(len(syn)))),
            ("real darkest-skin inputs (internal only)", real, R_out, "tB1_skin_attr_real_dark.jpg", dark[:20].tolist())):
        nrow = len(idx)
        g = Image.new("RGB", ((1 + len(cols)) * s * 2, 24 + ((nrow + 1) // 2) * (s + 14)), "white")
        d = ImageDraw.Draw(g)
        d.text((4, 6), f"teacher tB1 avatar, 8 steps, {kind}: photo | " + " | ".join(cols) + "   (two faces per row)", fill="black")
        for n_, i in enumerate(idx):
            x0 = (n_ % 2) * (1 + len(cols)) * s
            y = 24 + (n_ // 2) * (s + 14)
            ph = Image.open(photos[i]).convert("RGB")
            cells = [(ph, "photo")] + [(outs[c][i], c) for c in cols]
            for j, (im, cap) in enumerate(cells):
                g.paste(im.resize((s, s)), (x0 + j * s, y))
                if kind != "synthetic" and j > 0:
                    v = fid.skin_L(im)
                    L = rL[i]
                    cap = f"{cap} dL {v - L:+.0f}" if v is not None and not np.isnan(L) else cap
                d.text((x0 + j * s + 3, y + s + 1), cap, fill="black")
        g.save(ROOT / "grids/teacher" / fname, quality=88)
    out["minutes"] = round((time.time() - t0) / 60, 1)
    (ROOT / "results/teacher/tB1_skin_attr_test.json").write_text(json.dumps(out, indent=1))
    print("[s9] done", out["minutes"], "min", flush=True)


if __name__ == "__main__":
    main()
