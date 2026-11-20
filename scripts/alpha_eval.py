"""2a eval (2026-11-20): is the RGBA decoder's own alpha a drop-in for the BiRefNet_lite cut-out on the phone?

Reads RGBA outputs of a student run with an RGBA decoder (student_infer.py / expr_eval.py --decoder <rgba tag>), and for each
image compares the decoder alpha with BiRefNet_lite (metrics.Matter, the current cut-out model) run on the SAME RGB:
  alpha MAE, IoU (>0.5), boundary MAE (pixels where either alpha is in 0.05..0.95 or they disagree), and the sticker
  post-process pass rate (metrics.sticker_postprocess) with each alpha.
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/alpha_eval.py --dirs outputs/student_dl6g2_decA/cartoon3d/sticker,...
Out: results/alpha_eval_<name>.json, grids/alpha_eval_<name>.jpg
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import metrics as M  # noqa: E402

ROOT = M.ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True, help="comma list of folders with RGBA PNGs")
    ap.add_argument("--name", required=True)
    ap.add_argument("--max", type=int, default=400, help="images per folder")
    a = ap.parse_args()
    mat = M.Matter()
    res, sheet = {}, []
    for d in a.dirs.split(","):
        fs = sorted((ROOT / d).glob("*.png"))[: a.max]
        assert fs, d
        rows = []
        for j in range(0, len(fs), 16):
            ims = [Image.open(f) for f in fs[j:j + 16]]
            assert all(im.mode == "RGBA" for im in ims), f"{d}: not RGBA (decoder without alpha?)"
            rgb = [im.convert("RGBA").split() for im in ims]
            rgb_im = [Image.merge("RGB", c[:3]) for c in rgb]
            ref = mat.batch(rgb_im)
            for f, im, c, r in zip(fs[j:j + 16], rgb_im, rgb, ref):
                al = np.asarray(c[3], dtype=np.float32) / 255
                b = ((al > 0.05) & (al < 0.95)) | ((r > 0.05) & (r < 0.95)) | ((al > 0.5) != (r > 0.5))
                inter, uni = ((al > 0.5) & (r > 0.5)).sum(), ((al > 0.5) | (r > 0.5)).sum()
                p_dec = M.sticker_postprocess(im, al)[1]["pass_sticker"]
                p_ref = M.sticker_postprocess(im, r)[1]["pass_sticker"]
                rows.append({"f": f.stem, "mae": float(np.abs(al - r).mean()), "iou": float(inter / max(uni, 1)),
                             "bmae": float(np.abs(al - r)[b].mean()) if b.any() else 0.0, "pass_dec": bool(p_dec), "pass_ref": bool(p_ref)})
                if len(sheet) < 12 and len(rows) % 9 == 1:
                    g = lambda x: np.repeat((np.clip(x, 0, 1) * 255).astype(np.uint8)[..., None], 3, 2)
                    grey = lambda x: (np.asarray(im, np.float32) * x[..., None] + 160 * (1 - x[..., None])).astype(np.uint8)
                    sheet.append(np.concatenate([np.asarray(im), g(al), g(r), grey(al), grey(r)], 1))
        res[d] = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in ("mae", "iou", "bmae", "pass_dec", "pass_ref")}
        res[d] |= {"n": len(rows), "worst_iou": sorted((round(r["iou"], 3), r["f"]) for r in rows)[:5]}
        print(f"[alpha] {d}: {res[d]}", flush=True)
    (ROOT / f"results/alpha_eval_{a.name}.json").write_text(json.dumps(res, indent=1))
    g = np.concatenate(sheet, 0)  # cols: RGB | decoder alpha | BiRefNet alpha | decoder cut-out | BiRefNet cut-out
    Image.fromarray(g).resize((g.shape[1] // 2, g.shape[0] // 2)).save(ROOT / f"grids/alpha_eval_{a.name}.jpg", quality=88)


if __name__ == "__main__":
    main()
