"""Pose-conditioning step 1 (2026-11-27, user "yes start step 1 on GPU 0"): does a whole-body pose detector work on our cartoon stickers?
DWPose-style whole-body model via rtmlib 0.0.13 (onnxruntime, CPU here; YOLOX person detector + RTMPose whole-body, 133 keypoints:
0-16 body, 91-111 left hand, 112-132 right hand; models downloaded from OpenMMLab, public weights, run locally).
Images = the student's TRAINING TARGETS for gesture options: teacher pairs teacher_tB1_s8_img3.0_k1.pt (fixed phone noise), decoded with
the TinyVAE; 7 options (thumbs, wave, ok, think, facepalm, shrug, thanks) x 7 random rows + 1 avatar row.
Out: grids/s22_pose_check.jpg (skeleton overlay, caption = mean keypoint score of the arm / left hand / right hand), results/s22_pose_check.json
Go / no-go (by eye on the sheet): arms + wrists placed right on most stickers -> coarse pose maps OK; finger keypoints right on most
-> fine (gesture-shape) maps OK.
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/s22_pose_check.py
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/pairs/cartoon3d"
OPTS = ["thumbs", "wave", "ok", "think", "facepalm", "shrug", "thanks"]
ARM = [5, 6, 7, 8, 9, 10]          # shoulders, elbows, wrists
LH, RH = list(range(91, 112)), list(range(112, 133))


def main():
    from rtmlib import Wholebody, draw_skeleton
    t0 = time.time()
    TP = torch.load(D / "teacher_tB1_s8_img3.0_k1.pt", mmap=True)
    keys = [tuple(k) for k in TP["key"]]
    rng = random.Random(0)
    sel = []
    for o in OPTS:
        sel += rng.sample([i for i, k in enumerate(keys) if k[1] == f"expr_{o}"], 7)
    sel += rng.sample([i for i, k in enumerate(keys) if k[1] == "avatar"], 1)
    vae = DLA.load_vae(torch.float16).to("cuda").eval()
    with torch.inference_mode():
        y = DLA.decode(vae, TP["zt"][sel].to("cuda", torch.float16)).float().clamp(-1, 1)
    ims = [((v + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy() for v in y]
    wb = Wholebody(mode="balanced", backend="onnxruntime", device="cpu", to_openpose=False)
    print(f"[s22] model ready {time.time() - t0:.0f}s", flush=True)
    res, tiles = [], []
    for i, a in zip(sel, ims):
        bgr = np.ascontiguousarray(a[..., ::-1])
        kp, sc = wb(bgr)
        if kp is None or len(kp) == 0:
            res.append({"key": keys[i][:2], "persons": 0})
            tiles.append((a, f"{keys[i][1][5:] if keys[i][1] != 'avatar' else 'avatar'}: no person"))
            continue
        p = int(np.argmax(sc.mean(1)))  # most confident person
        s = sc[p]
        r = {"key": keys[i][:2], "persons": int(len(kp)), "arm": round(float(s[ARM].mean()), 2),
             "lhand": round(float(s[LH].mean()), 2), "rhand": round(float(s[RH].mean()), 2),
             "kp": kp[p].round(1).tolist(), "score": s.round(3).tolist()}
        res.append(r)
        vis = draw_skeleton(bgr.copy(), kp[p:p + 1], sc[p:p + 1], kpt_thr=0.3)
        tiles.append((vis[..., ::-1], f"{keys[i][1][5:] if keys[i][1] != 'avatar' else 'avatar'} arm {r['arm']} L {r['lhand']} R {r['rhand']}"))
    (ROOT / "results/s22_pose_check.json").write_text(json.dumps(res))
    c = 240
    g = Image.new("RGB", (8 * c, 20 + 7 * (c + 16)), "white")
    d = ImageDraw.Draw(g)
    d.text((4, 4), "Whole-body pose (rtmlib balanced) on teacher TRAINING TARGETS; caption = mean keypoint score arm / left hand / right hand (thr 0.3 drawn)", fill="black")
    for n, (a, cap) in enumerate(tiles):
        x0, y0 = (n % 8) * c, 20 + (n // 8) * (c + 16)
        g.paste(Image.fromarray(np.ascontiguousarray(a)).resize((c, c)), (x0, y0))
        d.text((x0 + 3, y0 + c + 2), cap, fill="black")
    g.save(ROOT / "grids/s22_pose_check.jpg", quality=90)
    hs = [max(r["lhand"], r["rhand"]) for r in res if r.get("persons")]
    print(f"[s22] done {time.time() - t0:.0f}s; persons found {sum(1 for r in res if r.get('persons'))}/{len(res)}; "
          f"best-hand score median {np.median(hs):.2f}", flush=True)


if __name__ == "__main__":
    main()
