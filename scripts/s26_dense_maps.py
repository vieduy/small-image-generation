"""Dense pose maps for tSP0b (2026-11-27): re-rendered from the keypoints saved by s23_pose_maps.py (no new pose detection).
tSP0's thin 3 px lines were ~1% of the first-layer signal (README 09:58). Dense = per visible hand a FILLED ellipse over its keypoint box
(+25% margin) + 10 px skeleton / forearm lines; upper arms 10 px lines; rendered at 256 px, area-pooled to 64 x 64. Same channels / rules.
Out: data/pairs/cartoon3d/pose_maps_dense_<pairs>.pt, pose_templates_dense_<pairs>.pt, grids/s26_dense_maps.jpg"""
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).parent))
from s23_pose_maps import HAND_EDGES, HAND_THR, KP_THR, LH0, RH0, NAME  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/pairs/cartoon3d"


def render_dense(kp, sc, size=256, scale=0.5, w=10):
    chans = []
    for ch in range(3):
        im = Image.new("L", (size, size), 0)
        d = ImageDraw.Draw(im)
        if ch < 2:
            h0, elbow, wrist = (LH0, 7, 9) if ch == 0 else (RH0, 8, 10)
            if sc[h0:h0 + 21].mean() >= HAND_THR:
                pts = kp[h0:h0 + 21][sc[h0:h0 + 21] >= KP_THR] * scale
                if len(pts):
                    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
                    mx, my = 0.25 * (x1 - x0) + 4, 0.25 * (y1 - y0) + 4
                    d.ellipse([x0 - mx, y0 - my, x1 + mx, y1 + my], fill=255)
                for a, b in [(h0 + a, h0 + b) for a, b in HAND_EDGES] + [(elbow, wrist)]:
                    if sc[a] >= KP_THR and sc[b] >= KP_THR:
                        d.line([tuple(kp[a] * scale), tuple(kp[b] * scale)], fill=255, width=w)
        else:
            for a, b in [(5, 7), (6, 8)]:
                if sc[a] >= KP_THR and sc[b] >= KP_THR:
                    d.line([tuple(kp[a] * scale), tuple(kp[b] * scale)], fill=255, width=w)
        chans.append(np.asarray(im.resize((64, 64), Image.BOX)))
    return np.stack(chans)


PM = torch.load(D / f"pose_maps_{NAME}.pt")
KP, SC = PM["kp"].float().numpy(), PM["score"].float().numpy()
maps = np.zeros(tuple(PM["maps"].shape), np.uint8)
for i in range(len(KP)):
    for v in range(2):
        if SC[i, v].max() > 0:
            maps[i, v] = render_dense(KP[i, v], SC[i, v])
torch.save({k: v for k, v in PM.items() if k != "maps"} | {"maps": torch.from_numpy(maps)}, D / f"pose_maps_dense_{NAME}.pt")
T = torch.load(D / f"pose_templates_{NAME}.pt")
for o, t in T.items():
    t["map"] = render_dense(np.array(t["kp"]), np.array(t["score"])).tolist()
torch.save(T, D / f"pose_templates_dense_{NAME}.pt")
old, new = PM["maps"].float().mean().item() / 255, maps.mean() / 255
print(f"[s26] mean map value thin {old:.4f} -> dense {new:.4f} (x{new / old:.1f})", flush=True)
c = 160
g = Image.new("RGB", (2 * len(T) * c, c + 16), "white")
dr = ImageDraw.Draw(g)
for j, (o, t) in enumerate(T.items()):
    m = np.moveaxis(np.array(t["map"], np.uint8), 0, -1)
    g.paste(Image.fromarray(np.moveaxis(PM["maps"][list(PM["keys"]).index(next(k for k in PM["keys"] if k[1] == f"expr_{o}")), 0].numpy(), 0, -1)).resize((c, c), Image.NEAREST), (2 * j * c, 0))
    g.paste(Image.fromarray(m).resize((c, c), Image.NEAREST), ((2 * j + 1) * c, 0))
    dr.text((2 * j * c + 3, c + 2), f"{o}: thin row | dense template", fill="black")
g.save(ROOT / "grids/s26_dense_maps.jpg", quality=90)
