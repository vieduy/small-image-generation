"""Import real portrait photos from murali1729S/portrait_2_avatar (input_image only; the anime targets are not used).
Legal approval for product use was confirmed by the project owner on 2026-11-16 (LICENSES.md). Source: CelebA-derived.
Steps: exact-dedupe -> YuNet person gate -> framing.normalize_face_crop (same as user photos) -> SFace identity clusters
(greedy, cos >= 0.45; SFace same-person threshold is 0.363) -> identity-disjoint split:
  data/real_p2a/eval/   300 identities x 1 photo (prefer 1024px sources)  -> EVAL_SET=real
  data/real_p2a/train/  all photos of the remaining identities            -> gen_pairs --real-dir
  source env.sh; CUDA_VISIBLE_DEVICES= $VENV/bin/python scripts/import_real_p2a.py
"""
import glob, hashlib, io, json, random, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
import metrics as M  # noqa: E402
from framing import normalize_face_crop  # noqa: E402
ROOT = M.ROOT
SRC = "/mnt/project/kchat/thaitt/generative-cv/DreamLite/data/portrait_2_avatar/data"
out = ROOT / "data/real_p2a"
for d in ("eval", "train"):
    (out / d).mkdir(parents=True, exist_ok=True)
fid = M.FaceID()
seen, recs = set(), []
for f in sorted(glob.glob(SRC + "/*.parquet")):
    for b in pq.ParquetFile(f).iter_batches(batch_size=64, columns=["input_image", "edit_prompt"]):
        for r in b.to_pylist():
            by = r["input_image"]["bytes"]
            h = hashlib.sha1(by).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            im = Image.open(io.BytesIO(by)).convert("RGB")
            norm, info = normalize_face_crop(im, fid)
            if info is None or info["face_box_512"] is None:
                continue
            ft = fid.feat(norm)
            if ft is None:
                continue
            recs.append({"sha1": h, "src_size": im.size[0], "caption": r["edit_prompt"], "framing": {"reframed": info["reframed"], "padded": info["padded"]},
                         "skin_L": fid.skin_L(norm), "feat": (ft.ravel() / np.linalg.norm(ft)).astype(np.float32), "img": norm})
    print(f"[real] {Path(f).name}: kept {len(recs)} / seen {len(seen)}", flush=True)
# identity clusters (greedy leader clustering on SFace)
F = np.stack([r["feat"] for r in recs])
cl = -np.ones(len(recs), int)
leaders = []
for i in range(len(recs)):
    if leaders:
        s = F[leaders] @ F[i]
        j = int(np.argmax(s))
        if s[j] >= 0.45:
            cl[i] = cl[leaders[j]]
            continue
    cl[i] = len(leaders)
    leaders.append(i)
n_id = len(leaders)
rng = random.Random(2026)
ids = list(range(n_id)); rng.shuffle(ids)
by_id = {}
for i, c in enumerate(cl):
    by_id.setdefault(int(c), []).append(i)
# eval: 300 identities, one photo each, prefer 1024px sources (closest to a phone photo's resolution)
eval_ids = [c for c in ids if any(recs[i]["src_size"] >= 1024 for i in by_id[c])][:300]
ev = set(eval_ids)
with open(out / "eval/manifest.jsonl", "w") as me, open(out / "train/manifest.jsonl", "w") as mt:
    k_e = k_t = 0
    for c in eval_ids:
        i = next(i for i in by_id[c] if recs[i]["src_size"] >= 1024)
        r = recs[i]; name = f"rp_{k_e:03d}.png"; k_e += 1
        r["img"].save(out / "eval" / name)
        me.write(json.dumps({"file": name, "prompt": "real photo", "identity": c, **{k: r[k] for k in ("sha1", "src_size", "caption", "framing", "skin_L")}}) + "\n")
    for c in ids:
        if c in ev:
            continue
        for i in by_id[c]:
            r = recs[i]; name = f"rt_{k_t:05d}.png"; k_t += 1
            r["img"].save(out / "train" / name)
            mt.write(json.dumps({"file": name, "prompt": "real photo", "identity": c, **{k: r[k] for k in ("sha1", "src_size", "caption", "framing", "skin_L")}}) + "\n")
src = np.array([r["src_size"] for r in recs]); L = np.array([r["skin_L"] or np.nan for r in recs])
print(json.dumps({"kept": len(recs), "unique_sha": len(seen), "identities": n_id, "eval": k_e, "train": k_t,
                  "src_1024": int((src >= 1024).sum()), "src_256": int((src < 512).sum()),
                  "reframed": sum(r["framing"]["reframed"] for r in recs), "padded": sum(r["framing"]["padded"] for r in recs),
                  "skin_L_p5_50_95": [round(float(np.nanpercentile(L, q)), 1) for q in (5, 50, 95)],
                  "caption_girl_share": round(float(np.mean(["girl" in r["caption"] or "woman" in r["caption"] for r in recs])), 3)}), flush=True)
