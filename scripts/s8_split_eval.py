"""Avatar evals split by input type (2026-11-26, user: "report evals split by input type").
Uses the saved eval outputs outputs/student_<tag>_decg2/cartoon3d/avatar/<prefix>_<n>.png (written by student_infer.py for
EVAL_SET v2 / real / selfie), no generation. Input types: face = synthetic portraits (data/faces), fe2 = synthetic portraits
(data/faces_eval2), fs = synthetic phone-style selfies (data/faces_selfie), rp = real photos (data/real_p2a/eval, CelebA-derived).
Per model x type: SFace likeness, attr agreement (scripts/attr_metric.py), style_clip (cosine to the cartoon3d refs' mean CLIP
embedding = eval_quality.py), tex (face fine detail) + ratio to dl6g2, share over-detailed (tex > 1.3 x dl6g2's median tex for
that type), skin dL* median and on the darkest 25% of inputs.
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/s8_split_eval.py --tags dl6g2 tS1 tS3 tS4 --out tS4
Out: results/split_eval_<out>.json (+ markdown table on stdout)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import metrics as M  # noqa: E402
from attr_metric import Attr, tex  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TYPES = {"face": ("synthetic portraits (data/faces)", "data/faces"), "fe2": ("synthetic portraits (faces_eval2)", "data/faces_eval2"),
         "fs": ("synthetic selfies", "data/faces_selfie"), "rp": ("real photos (CelebA-derived)", "data/real_p2a/eval")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fid, E, AT = M.FaceID(), M.Embedder(), Attr()
    _, rc = E([Image.open(p) for p in sorted((ROOT / "styles/cartoon3d").glob("ref_*.png"))])
    rcm = torch.nn.functional.normalize(rc.mean(0), dim=0)
    res = {}
    for pre, (name, src) in TYPES.items():
        base = sorted((ROOT / "outputs/student_dl6g2_decg2/cartoon3d/avatar").glob(f"{pre}_*.png"))
        names = [p.name for p in base if (ROOT / src / p.name).exists()]
        if not names:
            print(f"[split] {pre}: no photos matched", flush=True)
            continue
        photos = [Image.open(ROOT / src / b).convert("RGB") for b in names]
        pf = [fid.feat(p) for p in photos]
        pL = np.array([fid.skin_L(p) if fid.skin_L(p) is not None else np.nan for p in photos])
        dark = np.argsort(np.where(np.isnan(pL), 1e9, pL))[:max(1, len(names) // 4)]
        lab_p = AT.labels(photos)
        base_tex = None
        res[pre] = {"name": name, "n": len(names), "photo_tex": round(float(np.mean([tex(p) for p in photos])), 2), "models": {}}
        for tag in a.tags:
            d = ROOT / f"outputs/student_{tag}_decg2/cartoon3d/avatar"
            if not all((d / b).exists() for b in names):
                print(f"[split] {tag} {pre}: outputs missing, skipped", flush=True)
                continue
            outs = [Image.open(d / b).convert("RGB") for b in names]
            tx = np.array([tex(o) for o in outs])
            if tag == "dl6g2":
                base_tex = tx
            cos = [fid.cos(f, fid.feat(o)) for f, o in zip(pf, outs)]
            cos = [c for c in cos if c is not None]
            dl = np.array([(fid.skin_L(o) or np.nan) - L for o, L in zip(outs, pL)])
            _, c = zip(*[E(outs[i:i + 50]) for i in range(0, len(outs), 50)])
            ag = AT.agree(lab_p, AT.labels(outs))
            r = {"likeness": round(float(np.mean(cos)), 3), "attr": round(ag["attr"], 3),
                 "attr_groups": {g: round(v, 3) for g, v in ag.items() if g != "attr"},
                 "style_clip": round(float((torch.cat(c) @ rcm).mean()), 3), "tex": round(float(tx.mean()), 2),
                 "skin_dL_median": round(float(np.nanmedian(dl)), 1), "skin_dL_dark25": round(float(np.nanmedian(dl[dark])), 1)}
            if base_tex is not None:
                r["tex_vs_dl6g2"] = round(float(tx.mean() / base_tex.mean()), 2)
                r["over_detailed_share"] = round(float((tx > 1.3 * np.median(base_tex)).mean()), 2)
            res[pre]["models"][tag] = r
            print(f"[split] {pre} {tag} {r}", flush=True)
    (ROOT / f"results/split_eval_{a.out}.json").write_text(json.dumps(res, indent=1))
    print("\n| input type | model | likeness | attr | style | tex (x dl6g2) | over-detailed | skin dL median / dark25 |")
    print("|---|---|---|---|---|---|---|---|")
    for pre, v in res.items():
        for tag, r in v["models"].items():
            print(f"| {v['name']} (n={v['n']}) | {tag} | {r['likeness']:.3f} | {r['attr']:.3f} | {r['style_clip']:.3f} | "
                  f"{r['tex']:.1f} (x{r.get('tex_vs_dl6g2', 1):.2f}) | {r.get('over_detailed_share', 0):.2f} | "
                  f"{r['skin_dL_median']:+.1f} / {r['skin_dL_dark25']:+.1f} |")


if __name__ == "__main__":
    main()
