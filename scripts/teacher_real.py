"""Teacher tB1 avatars on REAL photos (2026-11-25): is the teacher better than klein there too?
Same set and metrics as eval_quality.py EVAL_SET=real (data/real_p2a/eval[:300], legal-approved, identity-disjoint from training):
SFace cosine(photo, avatar) and skin dL* (metrics.FaceID.skin_L, output - input; median over faces, and over the 25% darkest inputs).
Sampling = the stage-3 pair setting: 8 steps, CFG img 3 / txt 4, from the phone's fixed noise; DreamLite TinyVAE decoder.
Reference numbers (results/quality_*_real.csv): klein_personal avatar 0.208, dl6g2 + decg2 0.164.
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/teacher_real.py --tag tB1
Out: results/teacher/<tag>_real_s8.json, outputs/teacher_real/<tag>_s8/*.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402
import metrics as M  # noqa: E402
import teacher_dl as TD  # noqa: E402
from eval_e1 import to_t  # noqa: E402

ROOT, DEV = TD.ROOT, "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tB1")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--g-img", type=float, default=3.0)
    ap.add_argument("--g-txt", type=float, default=4.0)
    a = ap.parse_args()
    unet, T = TD.load_teacher(a.tag)
    unet = unet.to(DEV).eval()
    k = T["tasks"].index("avatar")
    emb, mask = T["emb"][k:k + 1].to(DEV, torch.float16), T["mask"][k:k + 1].to(DEV)
    ne, nm = T["null_emb"].to(DEV, torch.float16), T["null_mask"].to(DEV)
    vae = DLA.load_vae(torch.float16).to(DEV).eval()
    noise = DLA.fixed_noise(DEV, torch.float32)
    faces = sorted((ROOT / "data/real_p2a/eval").glob("*.png"))[:300]
    od = ROOT / f"outputs/teacher_real/{a.tag}_s{a.steps}"
    od.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for i in range(0, len(faces), 16):
            ch = faces[i:i + 16]
            x = torch.stack([to_t(Image.open(p)) for p in ch]).to(DEV, torch.float16)
            z = TD.sample(unet, DLA.encode(vae, x), emb.expand(len(ch), -1, -1), mask.expand(len(ch), -1), ne, nm,
                          a.g_img, a.g_txt, steps=a.steps, noise=noise)
            for yy, p in zip(DLA.decode(vae, z.to(torch.float16)), ch):
                Image.fromarray(((yy.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()).save(od / p.name)
    fid = M.FaceID()
    cos, dl, lin = [], [], []
    for p in faces:
        a_, o_ = Image.open(p), Image.open(od / p.name)
        fa, fo = fid.feat(a_), fid.feat(o_)
        if fa is not None and fo is not None:
            cos.append(fid.cos(fa, fo))
        li, lo = fid.skin_L(a_), fid.skin_L(o_)
        if li is not None and lo is not None:
            dl.append(lo - li); lin.append(li)
    dl, lin = np.array(dl), np.array(lin)
    dark = dl[lin <= np.percentile(lin, 25)] if len(lin) else dl
    res = {"teacher": a.tag, "steps": a.steps, "g_img": a.g_img, "g_txt": a.g_txt, "set": "data/real_p2a/eval[:300]",
           "avatar_likeness": round(float(np.mean(cos)), 4), "n": len(cos),
           "skin_dL_median": round(float(np.median(dl)), 1), "skin_dL_dark25_median": round(float(np.median(dark)), 1),
           "reference": {"klein_personal": 0.208, "dl6g2_decg2": 0.164}}
    (ROOT / f"results/teacher/{a.tag}_real_s{a.steps}.json").write_text(json.dumps(res, indent=1))
    print("[treal]", res, flush=True)


if __name__ == "__main__":
    main()
