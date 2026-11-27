"""Per-row detail-consistency check of teacher pairs vs klein (2026-11-26, user: "for the next teacher pairs, add a per-row
consistency check (drop pairs whose detail is far above klein's on the same photo)").
For each training row and both views (plain photo, flipped photo): decode the teacher latent and klein's latent with the same
TinyVAE, detect the face box on klein's 256 px image (YuNet; mirrored for the flipped view), tex = mean |Laplacian| of that box at
128 px (attr_metric.tex_box). ratio = tex(teacher) / tex(klein).
Drop rule (fixed before running): drop the row if ratio > 1.5 on either view. No face found -> kept (ratio NaN).
1.5 = "far above": klein's own spread on the same photos is ~+-15% and tS1's over-detailed outputs were x1.75 (README 2026-11-26).
  python scripts/s8_consistency.py --pairs teacher_tB1_s8_<name> [--max-rows N]
Out: data/pairs/cartoon3d/<pairs>_consistency_drop.json ([id, task] list for train_dl --drop-list),
     <pairs>_consistency.json (drop rates by pool / task group, ratio percentiles), <pairs>_consistency_ratios.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402
from attr_metric import tex_box  # noqa: E402
from metrics import FaceID  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
THR = 1.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--style", default="cartoon3d", help="2026-11-27: comic teacher pairs")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--bs", type=int, default=128)
    a = ap.parse_args()
    t0 = time.time()
    D = ROOT / "data/pairs" / a.style
    C = torch.load(D / "latents_dl_manifest_v2.pt", mmap=True)
    FL = torch.load(D / "latents_dl_manifest_v2_flip.pt", mmap=True)
    TP = torch.load(D / f"{a.pairs}.pt", mmap=True)
    assert [tuple(k) for k in TP["key"]] == [tuple(k) for k in C["key"]] == [tuple(k) for k in FL["key"]]
    n = a.max_rows or len(C["key"])
    vae = DLA.load_vae(torch.float16).to("cuda").eval()
    det = FaceID().det
    det.setInputSize((256, 256))
    cv2.setNumThreads(8)

    def dec(z):
        with torch.inference_mode():
            y = F.interpolate(DLA.decode(vae, z.to("cuda", torch.float16)).float().clamp(-1, 1), size=256, mode="area")
        return [Image.fromarray(x) for x in ((y + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()]

    ratio = np.full((n, 2), np.nan, dtype=np.float32)
    for s0 in range(0, n, a.bs):
        j = slice(s0, min(s0 + a.bs, n))
        kp, tp, kf, tf = dec(C["zt"][j]), dec(TP["zt"][j]), dec(FL["zt_f"][j]), dec(TP["zt_f"][j])
        for i in range(len(kp)):
            _, f = det.detect(cv2.cvtColor(np.asarray(kp[i]), cv2.COLOR_RGB2BGR))
            if f is None or len(f) == 0:
                continue
            x, y, w, h = max(f, key=lambda r: r[2] * r[3])[:4]
            bp, bf = (x, y, w, h), (256 - x - w, y, w, h)
            tkp, tkf = tex_box(kp[i], bp), tex_box(kf[i], bf)
            if tkp > 0:
                ratio[s0 + i, 0] = tex_box(tp[i], bp) / tkp
            if tkf > 0:
                ratio[s0 + i, 1] = tex_box(tf[i], bf) / tkf
        if (s0 // a.bs) % 50 == 0:
            print(f"[s8c] {min(s0 + a.bs, n)}/{n} rows, {(time.time() - t0) / 60:.1f} min", flush=True)

    keys = [tuple(k) for k in C["key"][:n]]
    worst = np.nanmax(np.where(np.isnan(ratio), -1, ratio), axis=1)
    drop = worst > THR
    grp = lambda t: "avatar" if t == "avatar" else ("sticker" if t == "sticker" else "expressions")
    stats = {"pairs": a.pairs, "rows": n, "threshold": THR, "no_face": int(np.isnan(ratio[:, 0]).sum()),
             "dropped": int(drop.sum()), "drop_rate": round(float(drop.mean()), 4),
             "ratio_plain_pct": {p: round(float(np.nanpercentile(ratio[:, 0], p)), 3) for p in (10, 50, 90, 99)},
             "by_pool": {}, "by_task": {}, "minutes": round((time.time() - t0) / 60, 1)}
    for name, fn in (("by_pool", lambda k: k[0][0]), ("by_task", lambda k: grp(k[1]))):
        for v in sorted({fn(k) for k in keys}):
            m = np.array([fn(k) == v for k in keys])
            stats[name][v] = {"rows": int(m.sum()), "drop_rate": round(float(drop[m].mean()), 4),
                              "ratio_median": round(float(np.nanmedian(ratio[m, 0])), 3)}
    (D / f"{a.pairs}_consistency_drop.json").write_text(json.dumps([[k[0], k[1]] for k, d in zip(keys, drop) if d]))
    (D / f"{a.pairs}_consistency.json").write_text(json.dumps(stats, indent=1))
    torch.save({"key": keys, "ratio": torch.from_numpy(ratio)}, D / f"{a.pairs}_consistency_ratios.pt")
    print("[s8c] done", json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
