"""Pose conditioning step 2 (2026-11-27, user: steps 2-4 on the local H100): pose maps for the student's gesture training targets.
Runs with venv_ort (onnxruntime-gpu 1.22 + rtmlib; other packages from the project venv via a .pth link):
  CUDA_VISIBLE_DEVICES=0 venv_ort/bin/python scripts/s23_pose_maps.py
Rows: cartoon3d teacher pairs teacher_tB1_s8_img3.0_k1.pt, options thumbs / wave / ok / think / facepalm / shrug / thanks (all rows), both
views: plain = dec(zt), flipped = dec(zt_f) (the teacher's output for the mirrored photo). Whole-body pose (rtmlib balanced = RTMW-dw-x-l,
CUDA), most confident person. Map = 3 channels at 64 x 64 (the student's latent grid, left panel), rendered at 256 px with 3 px lines and
area-pooled 4x: ch0 = subject's LEFT hand (21-keypoint hand skeleton + forearm elbow->wrist), ch1 = RIGHT hand (same), ch2 = upper arms
(shoulder->elbow, both). A hand is drawn only if its mean keypoint score >= 0.45 (visible ~0.55-0.9, hidden <= 0.39 on the step-1 sheet);
keypoints with score < 0.3 are skipped. Other options get an all-zero map (in the trainer).
Templates (for the phone): per option, the most common visible-hand pattern (L / R / LR) on the plain view, then the per-keypoint median
position over those rows (absolute sticker coordinates: the teacher frames every sticker alike) -> rendered like a training map.
Out: data/pairs/cartoon3d/pose_maps_<pairs>.pt {key_index, maps uint8 [N, 2, 3, 64, 64], kp float16 [N, 2, 133, 2], score [N, 2, 133]},
     data/pairs/cartoon3d/pose_templates_<pairs>.pt {option: {pattern, kp, score, map}}, results/s23_pose_maps.json,
     grids/s23_pose_maps.jpg (targets + maps + templates), grids/s23_ok_fingers.jpg (OK-sign hand zooms with the hand skeleton)
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/pairs/cartoon3d"
NAME = "teacher_tB1_s8_img3.0_k1"
OPTS = ["thumbs", "wave", "ok", "think", "facepalm", "shrug", "thanks"]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
              (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
LH0, RH0 = 91, 112
HAND_THR, KP_THR = 0.45, 0.3


def render(kp, sc, size=256, scale=0.5):
    """kp [133, 2] in 512 px coords -> uint8 [3, 64, 64]."""
    chans = []
    for ch in range(3):
        im = Image.new("L", (size, size), 0)
        d = ImageDraw.Draw(im)
        segs = []
        if ch < 2:
            h0, elbow, wrist = (LH0, 7, 9) if ch == 0 else (RH0, 8, 10)
            if sc[h0:h0 + 21].mean() >= HAND_THR:
                segs += [(h0 + a, h0 + b) for a, b in HAND_EDGES] + [(elbow, wrist)]
        else:
            segs += [(5, 7), (6, 8)]
        for a, b in segs:
            if sc[a] >= KP_THR and sc[b] >= KP_THR:
                d.line([tuple(kp[a] * scale), tuple(kp[b] * scale)], fill=255, width=3)
        chans.append(np.asarray(im.resize((64, 64), Image.BOX)))
    return np.stack(chans)


def pattern(sc):
    return ("L" if sc[LH0:LH0 + 21].mean() >= HAND_THR else "") + ("R" if sc[RH0:RH0 + 21].mean() >= HAND_THR else "") or "-"


def main():
    ort.preload_dlls()
    from rtmlib import Wholebody, draw_skeleton
    t0 = time.time()
    TP = torch.load(D / f"{NAME}.pt", mmap=True)
    keys = [tuple(k) for k in TP["key"]]
    rows = [i for i, k in enumerate(keys) if k[1] in [f"expr_{o}" for o in OPTS]]
    print(f"[s23] {len(rows)} gesture rows x 2 views", flush=True)
    vae = DLA.load_vae(torch.float16).to("cuda").eval()
    wb = Wholebody(mode="balanced", backend="onnxruntime", device="cuda", to_openpose=False)
    N = len(rows)
    maps = np.zeros((N, 2, 3, 64, 64), np.uint8)
    KP = np.zeros((N, 2, 133, 2), np.float16)
    SC = np.zeros((N, 2, 133), np.float16)
    for j in range(0, N, 32):
        idx = rows[j:j + 32]
        for v, Z in enumerate((TP["zt"], TP["zt_f"])):
            with torch.inference_mode():
                y = DLA.decode(vae, Z[idx].to("cuda", torch.float16)).float().clamp(-1, 1)
            ims = ((y + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            for n, a in enumerate(ims):
                kp, sc = wb(np.ascontiguousarray(a[..., ::-1]))
                if kp is None or len(kp) == 0:
                    continue
                p = int(np.argmax(sc.mean(1)))
                KP[j + n, v], SC[j + n, v] = kp[p], sc[p]
                maps[j + n, v] = render(kp[p], sc[p])
        if (j // 32) % 25 == 0:
            el = time.time() - t0
            print(f"[s23] {j + len(idx)}/{N} rows {el:.0f}s (~{el / (j + len(idx)) * (N - j - len(idx)) / 60:.0f} min left)", flush=True)
    torch.save({"rows": rows, "keys": [keys[i] for i in rows], "maps": torch.from_numpy(maps), "kp": torch.from_numpy(KP),
                "score": torch.from_numpy(SC)}, D / f"pose_maps_{NAME}.pt")
    # templates
    stats, temps = {}, {}
    opt_of = np.array([keys[i][1][5:] for i in rows])
    for o in OPTS:
        m = np.where(opt_of == o)[0]
        pats = [pattern(SC[k, 0].astype(np.float32)) for k in m]
        cnt = Counter(pats)
        top = cnt.most_common(1)[0][0]
        sel = [k for k, p in zip(m, pats) if p == top]
        kp_med = np.median(KP[sel, 0].astype(np.float32), 0)
        sc_med = np.median(SC[sel, 0].astype(np.float32), 0)
        temps[o] = {"pattern": top, "n": len(sel), "kp": kp_med.tolist(), "score": sc_med.tolist(), "map": render(kp_med, sc_med).tolist()}
        stats[o] = {"rows": int(len(m)), "patterns": dict(cnt), "template_pattern": top}
    torch.save(temps, D / f"pose_templates_{NAME}.pt")
    out = {"rows": N, "minutes": round((time.time() - t0) / 60, 1), "per_option": stats}
    (ROOT / "results/s23_pose_maps.json").write_text(json.dumps(out, indent=1))
    print("[s23] " + json.dumps(out), flush=True)
    # sheets: per option 2 examples (target + map overlay) + template; OK finger zooms
    c = 200
    g = Image.new("RGB", (7 * c, 20 + len(OPTS) * (c + 14)), "white")
    dr = ImageDraw.Draw(g)
    dr.text((4, 4), "per option: 3 training targets with their pose map (red = L hand, green = R hand, blue = upper arms) | template map", fill="black")
    for r_, o in enumerate(OPTS):
        m = np.where(opt_of == o)[0][:3]
        y0 = 20 + r_ * (c + 14)
        with torch.inference_mode():
            ims = ((DLA.decode(vae, TP["zt"][[rows[k] for k in m]].to("cuda", torch.float16)).float().clamp(-1, 1) + 1) * 127.5
                   ).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        for n, (k, a) in enumerate(zip(m, ims)):
            mp = np.asarray(Image.fromarray(np.moveaxis(maps[k, 0], 0, -1)).resize((512, 512), Image.NEAREST)).astype(np.float32) / 255
            ov = (a * (1 - 0.7 * mp.max(-1, keepdims=True)) + 0.7 * mp.max(-1, keepdims=True) * mp[..., [0, 1, 2]] * 255).astype(np.uint8)
            g.paste(Image.fromarray(a).resize((c, c)), (2 * n * c, y0))
            g.paste(Image.fromarray(ov).resize((c, c)), ((2 * n + 1) * c, y0))
        tm = np.moveaxis(np.array(temps[o]["map"], np.uint8), 0, -1)
        g.paste(Image.fromarray(tm).resize((c, c), Image.NEAREST), (6 * c, y0))
        dr.text((6 * c + 3, y0 + c + 1), f"{o} template ({temps[o]['pattern']}, n={temps[o]['n']})", fill="black")
    g.save(ROOT / "grids/s23_pose_maps.jpg", quality=90)
    m = np.where(opt_of == "ok")[0][:12]
    g2 = Image.new("RGB", (6 * 256, 2 * 256), "white")
    with torch.inference_mode():
        ims = ((DLA.decode(vae, TP["zt"][[rows[k] for k in m]].to("cuda", torch.float16)).float().clamp(-1, 1) + 1) * 127.5
               ).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    for n, (k, a) in enumerate(zip(m, ims)):
        kp, sc = KP[k, 0].astype(np.float32), SC[k, 0].astype(np.float32)
        vis = draw_skeleton(np.ascontiguousarray(a[..., ::-1]).copy(), kp[None], sc[None], kpt_thr=0.3)[..., ::-1]
        h0 = LH0 if sc[LH0:LH0 + 21].mean() >= sc[RH0:RH0 + 21].mean() else RH0
        cx, cy = kp[h0:h0 + 21].mean(0)
        x0, y0 = int(np.clip(cx - 96, 0, 320)), int(np.clip(cy - 96, 0, 320))
        g2.paste(Image.fromarray(np.ascontiguousarray(vis[y0:y0 + 192, x0:x0 + 192])).resize((256, 256)), ((n % 6) * 256, (n // 6) * 256))
    g2.save(ROOT / "grids/s23_ok_fingers.jpg", quality=90)
    print(f"[s23] done {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
