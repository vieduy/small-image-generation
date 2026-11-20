"""Step 1b: best-of-K teacher selection (raise the identity ceiling the students inherit from klein).
For the first N face avatars / face stickers of a pairs manifest, generate K-1 extra klein candidates (same per-person
prompt, seeds +1000*j), score each (SFace rank among 100 training inputs, skin dL*, sticker post-process), and pick the
eligible candidate (rank<=5, |dL*|<=max_dl, sticker pass) with the best rank (tie: higher SFace cosine).
Candidate 0 is the existing target. Output manifest rows get "target" (relative image path) and "cand" fields.
style_dino/refcopy of the chosen candidate are NOT recomputed (kept from candidate 0; metadata only).

  source env.sh; $VENV/bin/python scripts/best_of_k.py --style cartoon3d --k 3 --avatar-n 4000 --sticker-n 2000 \
      --manifest-in manifest_s5d.jsonl --manifest-out manifest_bestk.jsonl
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
os.environ.setdefault("MODEL", "klein"); os.environ.setdefault("KLEIN_REF", "0")
sys.path.insert(0, str(Path(__file__).parent))
import i2i_server as S, metrics as M  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--style", default="cartoon3d")
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--avatar-n", type=int, default=4000)
ap.add_argument("--sticker-n", type=int, default=2000)
ap.add_argument("--max-dl", type=float, default=12)
ap.add_argument("--manifest-in", default="manifest.jsonl")
ap.add_argument("--manifest-out", default="manifest_bestk.jsonl")
a = ap.parse_args()
out = M.ROOT / "data/pairs" / a.style
rows = [json.loads(l) for l in open(out / a.manifest_in)]
face_av = [r for r in rows if r["is_face"] and r["task"] == "avatar"][:a.avatar_n]
face_st = [r for r in rows if r["is_face"] and r["task"] == "sticker"][:a.sticker_n]
todo = [(r, j) for r in face_av + face_st for j in range(1, a.k)]
st = S.load_styles()[a.style]
t0 = time.time()
need = [(r, j) for r, j in todo if not (out / f"{r['task']}_c{j}" / f"{r['id']}.png").exists()]
if need:
    ad = S.Klein(); ad.load(); ad.p.set_progress_bar_config(disable=True)
    for i, (r, j) in enumerate(need):
        d = out / f"{r['task']}_c{j}"; d.mkdir(exist_ok=True)
        img = Image.open(out / "inputs" / f"{r['id']}.png").convert("RGB")
        with torch.inference_mode():
            ad.run([img], r["teacher_prompt"], st, [r["seed"] + (7 if r["task"] == "sticker" else 0) + 1000 * j])[0].save(d / f"{r['id']}.png")
        if i % 500 == 0:
            print(f"[bestk] teacher {i}/{len(need)} {time.time() - t0:.0f}s", flush=True)
    del ad; torch.cuda.empty_cache()
fc = M.FaceCache(M.FaceID(), out)
mat = M.Matter()
face_ids = [r["id"] for r in rows if r["is_face"] and r["task"] == "avatar"]  # same order/gallery as gen_pairs
infeat, inraw, inL = {}, {}, {}
for i_ in face_ids:
    f, inL[i_] = fc.get(out / "inputs" / f"{i_}.png")
    inraw[i_] = f
    infeat[i_] = None if f is None else f.ravel() / np.linalg.norm(f)
valid = [k for k in face_ids if infeat[k] is not None]
vidx = {k: i for i, k in enumerate(valid)}


def score(r, rel):
    fo, lo = fc.get(out / rel)
    s = {"target": rel, "rank": None, "cos": None, "dL": None, "pass_sticker": True}
    if fo is not None and infeat.get(r["id"]) is not None:
        i = vidx[r["id"]]
        gal = [r["id"]] + [valid[(i + 1 + j) % len(valid)] for j in range(99)]
        sims = np.stack([infeat[g] for g in gal]) @ (fo.ravel() / np.linalg.norm(fo))
        s["rank"] = int((sims > sims[0]).sum()) + 1
        s["cos"] = round(float(fc.fid.cos(inraw[r["id"]], fo)), 4)
    if lo is not None and inL[r["id"]] is not None:
        s["dL"] = round(lo - inL[r["id"]], 1)
    if r["task"] == "sticker":
        o = Image.open(out / rel).convert("RGB")
        s["pass_sticker"] = M.sticker_postprocess(o, mat(o))[1]["pass_sticker"]
    s["ok"] = bool(s["rank"] is not None and s["rank"] <= 5 and (s["dL"] is None or abs(s["dL"]) <= a.max_dl) and s["pass_sticker"])
    return s


sel = {}
for n, r in enumerate(face_av + face_st):
    cands = [score(r, f"{r['task']}/{r['id']}.png")] + [score(r, f"{r['task']}_c{j}/{r['id']}.png") for j in range(1, a.k)]
    ok = [(c["rank"], -(c["cos"] or 0), j) for j, c in enumerate(cands) if c["ok"]]
    if ok:
        j = min(ok)[2]
        c = cands[j]
        sel[(r["id"], r["task"])] = {"target": c["target"], "cand": j, "id_rank100": c["rank"], "id_cos": c["cos"], "dL": c["dL"],
                                     "id_pass": True, "keep": True, "cand_ranks": [x["rank"] for x in cands]}
    else:
        sel[(r["id"], r["task"])] = {"keep": False, "cand": None, "cand_ranks": [x["rank"] for x in cands]}
    if n % 1000 == 0:
        print(f"[bestk] scored {n}/{len(face_av) + len(face_st)} {time.time() - t0:.0f}s", flush=True)
fc.save()
with open(out / a.manifest_out, "w") as f:
    for r in rows:
        k = (r["id"], r["task"])
        if k in sel:
            r = r | sel[k]
        f.write(json.dumps(r) + "\n")
# summary vs candidate 0
for task, rs in (("avatar", face_av), ("sticker", face_st)):
    before = sum(r["keep"] for r in rs)
    after = sum(sel[(r["id"], task)]["keep"] for r in rs)
    r0 = [r["id_rank100"] for r in rs if r["keep"]]
    r1 = [sel[(r["id"], task)]["id_rank100"] for r in rs if sel[(r["id"], task)]["keep"]]
    print(f"[bestk] {task}: N={len(rs)} kept {before} -> {after}; rank==1 share among kept {np.mean([x == 1 for x in r0]):.3f} -> "
          f"{np.mean([x == 1 for x in r1]):.3f}; chosen cand counts {np.bincount([sel[(r['id'], task)]['cand'] for r in rs if sel[(r['id'], task)]['keep']], minlength=a.k).tolist()}", flush=True)
print(f"[bestk] done {time.time() - t0:.0f}s", flush=True)
