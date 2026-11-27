"""Next step #2 (2026-11-25, user "do 2 first, use 2 idle gpu"): teacher tB1 pairs on REAL photos, skin-scored.

Photos: data/ffhq_norm (FFHQ 512 px, user-approved exception 2026-11-18) without captioned children (pairs_v2 captions, child=yes)
+ data/real_p2a/train (CelebA-derived, legal-approved 2026-11-16; identity-disjoint from the real eval set data/real_p2a/eval).
Options per photo: "avatar" + one other option drawn uniformly (sticker or one of the 20 expressions), seed 20261125.
Teacher: tB1, 8 steps, CFG img 3 / txt 4, the phone's fixed noise (= stage 3), on the TinyVAE latent of the photo and of the
flipped photo. Scores on the plain output (decoded 512 px): skin dL* = skin_L(output) - skin_L(photo) (metrics.FaceID), and
SFace cosine(photo, output); the input's skin_L is stored so filters can be checked per skin quartile.
Shards: --part i --nparts N (one per GPU) -> data/pairs/real_teacher/part_<i>of<N>.pt; --merge -> data/pairs/real_teacher/
teacher_tB1_real.pt {ids, key, zi, zi_f, zt, zt_f, img256 (uint8, plain teacher output), task, skin_in, dL, id_cos}.
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/s5_real_pairs.py --part 0 --nparts 2
"""
import argparse
import glob
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402
import metrics as M  # noqa: E402
import teacher_dl as TD  # noqa: E402
from eval_e1 import to_t  # noqa: E402
from train_student import TASKS, expr_tasks  # noqa: E402

ROOT, DEV = TD.ROOT, "cuda"
OUT = ROOT / "data/pairs/real_teacher"


def photo_list():
    child = set()
    for f in glob.glob(str(ROOT / "data/pairs_v2/*/captions*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if str(r.get("child", "")).lower() == "yes":
                child.add(r["id"])
    ff = [p for p in sorted((ROOT / "data/ffhq_norm").glob("*.png")) if p.stem not in child]
    rp = sorted((ROOT / "data/real_p2a/train").glob("*.png"))
    return ff + rp, len(child)


def rows_all():
    photos, nchild = photo_list()
    tasks = TASKS + expr_tasks()
    rng = random.Random(20261125)
    rows = []
    for p in photos:
        rows.append((p, "avatar"))
        rows.append((p, rng.choice(tasks[1:])))
    return rows, tasks, len(photos), nchild


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--max-rows", type=int, default=0)
    a = ap.parse_args()
    rows, tasks, nph, nchild = rows_all()
    if a.max_rows:
        rows = rows[:a.max_rows]
    OUT.mkdir(parents=True, exist_ok=True)
    if a.merge:
        P = [torch.load(OUT / f"part_{i}of{a.nparts}.pt") for i in range(a.nparts)]
        m = {k: (sum((p[k] for p in P), []) if isinstance(P[0][k], list) else torch.cat([p[k] for p in P])) for k in P[0]}
        assert len(m["key"]) == len(rows)
        m |= {"setting": {"teacher": "tB1", "steps": 8, "g_img": 3.0, "g_txt": 4.0, "noise_seed": 1234}, "tasks": tasks}
        torch.save(m, OUT / "teacher_tB1_real.pt")
        dl, si, ok = m["dL"].numpy(), m["skin_in"].numpy(), ~np.isnan(m["dL"].numpy())
        q = np.nanpercentile(si[ok], [25, 50, 75])
        summ = {"photos": nph, "children_excluded_captioned": nchild, "rows": len(rows), "skin_scored": int(ok.sum()),
                "dL_median": round(float(np.nanmedian(dl)), 1), "id_cos_mean": round(float(np.nanmean(m["id_cos"].numpy())), 3)}
        for name, sel in (("darkest25", si <= q[0]), ("q2", (si > q[0]) & (si <= q[1])), ("q3", (si > q[1]) & (si <= q[2])), ("lightest25", si > q[2])):
            d_ = dl[sel & ok]
            summ[name] = {"n": int(len(d_)), "dL_median": round(float(np.median(d_)), 1)} | {f"keep<= {t}": round(float((d_ <= t).mean()), 3) for t in (5, 10, 15)}
        (OUT / "teacher_tB1_real.json").write_text(json.dumps(summ, indent=1))
        print("[s5] merged", json.dumps(summ), flush=True)
        for i in range(a.nparts):
            (OUT / f"part_{i}of{a.nparts}.pt").unlink()
        return
    lo, hi = len(rows) * a.part // a.nparts, len(rows) * (a.part + 1) // a.nparts
    mine = rows[lo:hi]
    unet, T = TD.load_teacher("tB1")
    unet = unet.to(DEV).eval()
    assert T["tasks"] == tasks
    emb, mask = T["emb"].to(DEV, torch.float16), T["mask"].to(DEV)
    ne, nm = T["null_emb"].to(DEV, torch.float16), T["null_mask"].to(DEV)
    vae = DLA.load_vae(torch.float16).to(DEV).eval()
    noise = DLA.fixed_noise(DEV, torch.float32)
    fid = M.FaceID()
    skin_cache, feat_cache = {}, {}
    R = {k: [] for k in ("zi", "zi_f", "zt", "zt_f", "img256", "task", "skin_in", "dL", "id_cos")}
    ids, keys = [], []
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(mine), a.bs):
            ch = mine[i:i + a.bs]
            x = torch.stack([to_t(Image.open(p)) for p, _ in ch]).to(DEV, torch.float16)
            zi, zif = DLA.encode(vae, x), DLA.encode(vae, x.flip(-1))
            k = torch.tensor([tasks.index(t) for _, t in ch], device=DEV)
            zt = TD.sample(unet, zi, emb[k], mask[k], ne, nm, 3.0, 4.0, steps=8, noise=noise)
            ztf = TD.sample(unet, zif, emb[k], mask[k], ne, nm, 3.0, 4.0, steps=8, noise=noise)
            y = DLA.decode(vae, zt.to(torch.float16)).float().clamp(-1, 1)
            R["img256"].append(((F.interpolate(y, size=256, mode="area") + 1) * 127.5).round().to(torch.uint8).cpu())
            for yy, (p, t) in zip(y, ch):
                o = Image.fromarray(((yy + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy())
                if p not in skin_cache:
                    ph = Image.open(p)
                    skin_cache[p], feat_cache[p] = fid.skin_L(ph), fid.feat(ph)
                so, fo = fid.skin_L(o), fid.feat(o)
                si = skin_cache[p]
                R["skin_in"].append(float("nan") if si is None else si)
                R["dL"].append(float("nan") if (si is None or so is None) else so - si)
                R["id_cos"].append(float("nan") if (fo is None or feat_cache[p] is None) else fid.cos(feat_cache[p], fo))
                ids.append((f"real/{p.stem}", t)); keys.append((f"real/{p.stem}", t, "teacher_tB1_real"))
            for nme, v in (("zi", zi), ("zi_f", zif), ("zt", zt), ("zt_f", ztf)):
                R[nme].append(v.half().cpu())
            R["task"].append(k.cpu())
            if (i // a.bs) % 20 == 0:
                el = time.time() - t0
                print(f"[s5] part {a.part}: {i + len(ch)}/{len(mine)} {(i + len(ch)) / el:.1f} rows/s, ~{(len(mine) - i - len(ch)) / max((i + len(ch)) / el, 1e-9) / 3600:.1f} h left", flush=True)
    out = {k: torch.cat(v) if isinstance(v[0], torch.Tensor) else torch.tensor(v) for k, v in R.items()}
    out |= {"ids": ids, "key": keys}
    torch.save(out, OUT / f"part_{a.part}of{a.nparts}.pt")
    print(f"[s5] part {a.part} done {len(mine)} rows in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
