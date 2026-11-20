"""Data v2 teacher pairs (2026-11-18): avatar prompt v4 + sticker/expression prompt v5, local VLM attributes, FFHQ adults,
fast klein (per-sample batching + torch.compile + background I/O). Same quality filter as gen_pairs.py round 4.

Stages (each resumable; outputs never overwritten):
  plan-test : the 500-image keep-rate test. Re-generates (id, task) pairs that ARE in manifest_round4 (old keep known) +
              FFHQ adults (no old baseline). Writes <out>/jobs.jsonl.
  caption   : batched Qwen3-VL attributes for every distinct input (one process), drops children (FFHQ) -> <out>/captions.jsonl
  gen       : klein, one process per GPU: --shard i --nshards n. Batches of --bs mixed tasks, compiled transformer,
              PNG writes in a thread pool. Prompts saved per job.
  score     : style DINO, sticker cut-out, SFace identity rank among 100 inputs of the same source, skin drift dL ->
              keep = id_pass & (avatar or pass_sticker) & extra_ok & |dL| <= 12 (round-4 rule, reproduced exactly from
              manifest_round4 fields) -> <out>/manifest.jsonl + keep-rate table (new vs old on the same pairs).
  $VENV/bin/python scripts/gen_pairs_v2.py plan-test --out data/pairs_v2/test500
  $VENV/bin/python scripts/gen_pairs_v2.py caption   --out data/pairs_v2/test500
  CUDA_VISIBLE_DEVICES=g $VENV/bin/python scripts/gen_pairs_v2.py gen --out ... --shard g --nshards 2
  $VENV/bin/python scripts/gen_pairs_v2.py score     --out data/pairs_v2/test500
Inputs: data/pairs/cartoon3d/inputs (synthetic f*/c*, CelebA-derived r*), data/ffhq (user-approved real-photo exception
2026-11-18; normalised like user photos into data/ffhq_norm, ids q*). Children are never prompted (VLM child flag).
"""
import argparse
import json
import os
import random
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("MODEL", "klein")  # as gen_pairs.py: text-only style, no style reference image
os.environ.setdefault("KLEIN_REF", "0")
os.environ.setdefault("PROMPTS_FAMILY", "klein_text")
sys.path.insert(0, str(Path(__file__).parent))
import teacher_prompt as TP  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/pairs/cartoon3d"
FFHQ, FNORM = ROOT / "data/ffhq", ROOT / "data/ffhq_norm"
MAX_DL = 12.0
# Skin-drift limit per input source (user decision 2026-11-18): the L* measurement reads real-photo lighting (shadows,
# exposure), so real photos measure +11 brighter than synthetic ones at the median (test500: FFHQ +18.6, CelebA +15.3 vs
# synthetic +7.5). Real photos get the synthetic limit + that offset; pairs kept only thanks to the looser limit are
# flagged provisional (to re-check once a lighting-robust skin measure exists).
DL_LIMIT = {"f": 12.0, "c": 12.0, "r": 23.0, "q": 23.0}
# User decision 2026-11-18 (after grids/skin_problems_examples.jpg): bypass the BRIGHTNESS check entirely and filter skin on
# COLOUR only (over-saturation / yellowing = the "Asian face turns South Asian" problem). dL is still computed and stored.
USE_DL = False


# Skin COLOUR limit (2026-11-18, user report "Asian selfies come out South Asian"): L* only sees brightness; klein and the
# students add saturation/yellowness to medium skin tones (results/skin_hue_by_tone_*.json). None = measured, not filtered
# yet (`skincolor` stage adds d_chroma/d_b/d_hue; the limit is set after a visual check, then `rekeep`).
# User decision 2026-11-18 ~08:10 UTC after grids/chroma_limit_sheet.jpg: keep d_chroma <= +17 (+18..+22 = reported orange/tan).
CHROMA_MAX = 17


def keep_rule(r):
    lim = DL_LIMIT.get(r["id"][0], MAX_DL)
    dl_ok = (not USE_DL) or r.get("dL") is None or abs(r["dL"]) <= lim
    r["dl_limit"] = lim
    col_ok = CHROMA_MAX is None or r.get("d_chroma") is None or r["d_chroma"] <= CHROMA_MAX
    r["color_ok"] = col_ok
    r["keep"] = bool(r["id_pass"] and (r["task"] == "avatar" or r.get("pass_sticker", False)) and r["extra_ok"] and dl_ok and col_ok)
    r["provisional"] = bool(USE_DL and r["keep"] and r.get("dL") is not None and abs(r["dL"]) > MAX_DL)
    return r


def expr_phrases():
    import yaml
    return {f"expr_{e['id']}": e["phrase"] for e in yaml.safe_load((ROOT / "configs/sticker_expressions.yaml").read_text())}


def seed_for(base, task):  # same offsets as gen_pairs.py
    return base + (7 if task == "sticker" else 0) + (zlib.crc32(task.encode()) % 997 + 11 if task.startswith("expr_") else 0)


def out_path(out, j):  # variant 1 of a best-of-2 job is <id>_b.png
    return Path(out) / j["task"] / f"{j['id']}{'_b' if j.get('var') else ''}.png"


def src_path(i):
    return FNORM / f"{i}.png" if i.startswith("q") else D / "inputs" / f"{i}.png"


def ffhq_input(i, fid):
    p = FNORM / f"{i}.png"
    if not p.exists():
        from framing import normalize_face_crop
        im, _ = normalize_face_crop(Image.open(FFHQ / f"{i[1:]}.png"), fid)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".tmp{os.getpid()}.png")   # atomic: a killed worker must not leave a truncated PNG behind
        im.save(tmp)
        tmp.replace(p)
    return p


# ------------------------------------------------------------------ plan
def plan_test(a):
    rnd = random.Random(20261118)
    rs = [json.loads(l) for l in open(D / "manifest_round4.jsonl")]
    face = [r for r in rs if r["is_face"] and r["id"][0] in "fcr"]
    jobs = []

    def take(task, n, prefixes):
        pool = [r for r in face if r["task"] == task and r["id"][0] in prefixes]
        for r in rnd.sample(pool, min(n, len(pool))):
            jobs.append({"id": r["id"], "task": task, "seed": seed_for(r["seed"], task), "gen_prompt": r["prompt"],
                         "old_keep": r["keep"], "old_extra_verified": r.get("extra_verified")})
    take("avatar", 80, "fcr")
    take("sticker", 80, "fcr")
    ex = [t for t in expr_phrases()]
    for t in ex:
        take(t, 12, "fc")
    ff = sorted(FFHQ.glob("*.png"))
    q = [f"q{p.stem}" for p in rnd.sample(ff, 160)]  # captioned later; children dropped; first adults used
    plan = {"jobs": jobs, "ffhq_pool": q, "ffhq_tasks": {"avatar": 20, "sticker": 20, "expr_each": 3}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    print(f"[v2] plan-test: {len(jobs)} old-baseline jobs + FFHQ pool {len(q)} (adult jobs added after captioning)", flush=True)


def plan_full(a):
    """Synthetic run (user-approved 2026-11-18): every synthetic photo (f*, c*): avatar best-of-2 seeds, 1 plain sticker,
    and --expr-per expression stickers per expression (<= 5 per photo), sized from the 500-pair keep rates."""
    rnd = random.Random(20261118)
    rs = [json.loads(l) for l in open(D / "manifest_round4.jsonl")]
    base = {r["id"]: r for r in rs if r["task"] == "avatar" and r["is_face"] and r["id"][0] in "fc"}
    ver = {}
    for r in rs:
        if r.get("extra_verified") is not None:
            ver[r["id"]] = r["extra_verified"]
    ids = sorted(base)
    J = lambda i, t, var=0: {"id": i, "task": t, "var": var, "seed": seed_for(base[i]["seed"], t) + 100003 * var,  # noqa: E731
                             "gen_prompt": base[i]["prompt"], "old_extra_verified": ver.get(i)}
    jobs = [J(i, "avatar", v) for i in ids for v in range(a.avatar_n)] + [J(i, "sticker") for i in ids]
    ex = list(expr_phrases())
    combos = [(i, t) for i in ids for t in ex]
    rnd.shuffle(combos)
    per_t, per_i = {t: 0 for t in ex}, {}
    for i, t in combos:
        if per_t[t] < a.expr_per and per_i.get(i, 0) < 5:
            jobs.append(J(i, t))
            per_t[t] += 1
            per_i[i] = per_i.get(i, 0) + 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "jobs.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    (out / "ffhq_adults.json").write_text("[]")
    print(f"[v2] plan-full: {len(ids)} photos, {len(jobs)} jobs (avatar x{a.avatar_n}, sticker x1, expressions "
          f"{min(per_t.values())}-{max(per_t.values())} each)", flush=True)


def _norm_one(i):
    global _FID
    if "_FID" not in globals():
        import metrics as M
        _FID = M.FaceID()
    try:
        ffhq_input(i, _FID)
        return True
    except Exception:
        return False


def _pool_init():
    """CPU-pool workers: 1 thread each. Without it every worker spawned ~319 OpenCV/torch threads (64 workers -> ~43k threads,
    load 1000+), which starved the klein GPU jobs on the same box (A/B gen fell from 5.9 to 0.26 img/s, 2026-11-18)."""
    import cv2
    cv2.setNumThreads(1)
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def plan_real(a):
    """Real-photo run (user request 2026-11-18): every CelebA-derived photo (r*, legal-approved) + --ffhq-pool random FFHQ
    photos (user-approved exception; reframed like user photos, CPU pool). 3 jobs per photo: avatar, plain sticker, one
    expression (spread evenly). Children are removed after captioning (`finalize-real`), FFHQ capped at --ffhq-adults."""
    from multiprocessing import Pool
    rnd = random.Random(20261119)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r_ids = sorted(p.stem for p in (D / "inputs").glob("r*.png"))
    q_ids = [f"q{p.stem}" for p in rnd.sample(sorted(FFHQ.glob("*.png")), a.ffhq_pool)]
    t0 = time.time()
    with Pool(a.workers, initializer=_pool_init) as pool:
        ok = pool.map(_norm_one, q_ids, chunksize=32)
    q_ids = [i for i, k in zip(q_ids, ok) if k]
    print(f"[v2] plan-real: {len(r_ids)} CelebA + {len(q_ids)} FFHQ reframed in {time.time() - t0:.0f}s", flush=True)
    ex = list(expr_phrases())
    jobs = []
    for i in r_ids + q_ids:
        base = (900000 if i[0] == "q" else 800000) + int(i[1:])
        t = ex[zlib.crc32(i.encode()) % len(ex)]
        jobs += [{"id": i, "task": k, "var": 0, "seed": seed_for(base, k)} for k in ("avatar", "sticker", t)]
    with open(out / "jobs.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"[v2] plan-real: {len(jobs)} candidate jobs (before child filter / FFHQ cap)", flush=True)


def finalize_real(a):
    out = Path(a.out)
    caps = load_caps(out)
    jobs = [json.loads(l) for l in open(out / "jobs.jsonl")]
    kids = {i for i, c in caps.items() if (c.get("child") or "").lower().startswith("y")}
    q_adults = sorted({j["id"] for j in jobs if j["id"][0] == "q" and j["id"] not in kids})[: a.ffhq_adults]
    keep_ids = {j["id"] for j in jobs if j["id"][0] == "r" and j["id"] not in kids} | set(q_adults)
    jobs = [j for j in jobs if j["id"] in keep_ids]
    with open(out / "jobs.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    (out / "ffhq_adults.json").write_text(json.dumps(q_adults))
    nr = len({j["id"] for j in jobs if j["id"][0] == "r"})
    print(f"[v2] finalize-real: children dropped {len(kids)} (CelebA {sum(k[0] == 'r' for k in kids)}, FFHQ "
          f"{sum(k[0] == 'q' for k in kids)}); photos: CelebA {nr}, FFHQ adults {len(q_adults)}; jobs {len(jobs)}", flush=True)


def caption_ids(a):
    """Captions for every photo in jobs.jsonl, split over --part/--nparts processes (merged by the next stage)."""
    import caption_photos as CP
    out = Path(a.out)
    ids = sorted({json.loads(l)["id"] for l in open(out / "jobs.jsonl")})[a.part:: a.nparts]
    cf = out / f"captions_part{a.part}.jsonl"
    have = {json.loads(l)["id"] for l in open(cf)} if cf.exists() else set()
    todo = [i for i in ids if i not in have]
    proc, model = CP.load()
    t0 = time.time()
    with open(cf, "a") as f:
        for k in range(0, len(todo), 32):
            ch = todo[k:k + 32]
            for i, c in zip(ch, CP.caption_batch(proc, model, [Image.open(src_path(i)).convert("RGB").resize((448, 448)) for i in ch])):
                f.write(json.dumps({"id": i} | c) + "\n")
            if (k // 32) % 50 == 0:
                print(f"[v2] caption part{a.part} {k + len(ch)}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print(f"[v2] CAPTION_DONE part{a.part} {time.time() - t0:.0f}s", flush=True)


def load_caps(out):
    caps = {}
    for p in sorted(Path(out).glob("captions*.jsonl")):
        for l in open(p):
            r = json.loads(l)
            caps[r["id"]] = r
    return caps


# ------------------------------------------------------------------ caption (+ finalise jobs)
def caption(a):
    import caption_photos as CP
    import metrics as M
    out = Path(a.out)
    plan = json.loads((out / "plan.json").read_text())
    fid = M.FaceID()
    ids = sorted({j["id"] for j in plan["jobs"]} | set(plan["ffhq_pool"]))
    paths = {i: (ffhq_input(i, fid) if i.startswith("q") else src_path(i)) for i in ids}
    cf = out / "captions.jsonl"
    have = {json.loads(l)["id"]: json.loads(l) for l in open(cf)} if cf.exists() else {}
    todo = [i for i in ids if i not in have]
    if todo:
        proc, model = CP.load()
        t0 = time.time()
        with open(cf, "a") as f:
            for k in range(0, len(todo), 32):
                ch = todo[k:k + 32]
                for i, c in zip(ch, CP.caption_batch(proc, model, [Image.open(paths[i]).convert("RGB").resize((448, 448)) for i in ch])):
                    have[i] = {"id": i} | c
                    f.write(json.dumps(have[i]) + "\n")
        print(f"[v2] captioned {len(todo)} photos in {time.time() - t0:.0f}s", flush=True)
    kids = {i for i in ids if (have[i].get("child") or "").lower().startswith("y")}
    adults = [i for i in plan["ffhq_pool"] if i not in kids]
    ft = plan["ffhq_tasks"]
    jobs = [j for j in plan["jobs"] if j["id"] not in kids]
    ex = list(expr_phrases())
    for n, i in enumerate(adults):
        base = 900000 + int(i[1:])
        if n < ft["avatar"]:
            jobs += [{"id": i, "task": "avatar", "seed": seed_for(base, "avatar")},
                     {"id": i, "task": "sticker", "seed": seed_for(base, "sticker")}]
    for n, t in enumerate(ex):  # 3 different adults per expression
        for i in adults[(3 * n) % len(adults):][:ft["expr_each"]]:
            jobs.append({"id": i, "task": t, "seed": seed_for(900000 + int(i[1:]), t)})
    with open(out / "jobs.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"[v2] children dropped: {len(kids)} ({sorted(kids)[:8]}...) | jobs: {len(jobs)} "
          f"({sum(j['id'][0] == 'q' for j in jobs)} FFHQ) | FFHQ adults available for galleries: {len(adults)}", flush=True)
    (out / "ffhq_adults.json").write_text(json.dumps(adults))


NATURAL_SKIN = ("soft vibrant colors", "vibrant clothing and hair colors, natural skin color exactly matching the photo, "
                "not tanned and not more saturated")


def style_text(st, natural):
    """cartoon3d style text; natural=1 (2026-11-18 test) swaps 'soft vibrant colors' for natural skin colour wording."""
    d = st["description"]
    if natural:
        assert NATURAL_SKIN[0] in d, d
        d = d.replace(*NATURAL_SKIN)
    return d


def build_prompt(j, cap, style_desc, phrases, fc_in):
    attrs = TP.parse(j["gen_prompt"]) if j.get("gen_prompt") else None
    if attrs and j.get("old_extra_verified") is False:  # as gen_pairs.py: never describe an attribute the photo does not show
        attrs = dict(attrs, extra=None)
    tone = TP.tone_word(fc_in.get(src_path(j["id"]))[1]) if attrs else None
    if j["task"] == "avatar":
        return TP.avatar_prompt_v4(style_desc, cap, tone, attrs)
    return TP.sticker_prompt_v5(style_desc, cap, tone, attrs, None if j["task"] == "sticker" else phrases[j["task"]])


# ------------------------------------------------------------------ gen
def gen(a):
    import torch
    import i2i_server as S
    import metrics as M
    from klein_batch import KleinBatch
    out = Path(a.out)
    jobs = [json.loads(l) for l in open(out / "jobs.jsonl")][a.shard:: a.nshards]
    caps = load_caps(out)
    jobs = [j for j in jobs if not out_path(out, j).exists()]
    st = S.load_styles()["cartoon3d"]
    phrases = expr_phrases()
    fc_in = M.FaceCache(M.FaceID(), D)  # read-only use of the round-4 input cache (never saved here)
    for j in jobs:
        j["teacher_prompt"] = build_prompt(j, caps[j["id"]], style_text(st, a.natural_skin), phrases, fc_in)
    kb = KleinBatch()
    if a.compile:
        if a.check_compile and jobs:  # compiled vs eager on the first batch
            ch = jobs[:a.bs]
            ims = [Image.open(src_path(j["id"])).convert("RGB") for j in ch]
            eager = kb.run(ims, [j["teacher_prompt"] for j in ch], [j["seed"] for j in ch])
        kb.p.transformer = torch.compile(kb.p.transformer, dynamic=False)
        if a.check_compile and jobs:
            comp = kb.run(ims, [j["teacher_prompt"] for j in ch], [j["seed"] for j in ch])
            dd = [float(np.abs(np.asarray(x, np.float32) - np.asarray(y, np.float32)).mean()) for x, y in zip(eager, comp)]
            print(f"[v2] compiled vs eager, mean abs pixel diff (0-255): mean {np.mean(dd):.2f} max {np.max(dd):.2f}", flush=True)
    io = ThreadPoolExecutor(8)
    load = lambda j: Image.open(src_path(j["id"])).convert("RGB")  # noqa: E731

    def save(j, im):
        p = out_path(out, j)
        p.parent.mkdir(parents=True, exist_ok=True)
        im.save(p)

    t0, pend = time.time(), []
    print(f"[v2] shard {a.shard}/{a.nshards}: {len(jobs)} images, bs {a.bs}, compile {a.compile}", flush=True)
    nxt = io.submit(lambda c: [load(j) for j in c], jobs[:a.bs])
    for k in range(0, len(jobs), a.bs):
        ch, ims = jobs[k:k + a.bs], nxt.result()
        nxt = io.submit(lambda c: [load(j) for j in c], jobs[k + a.bs:k + 2 * a.bs])  # prefetch the next batch
        outs = kb.run(ims, [j["teacher_prompt"] for j in ch], [j["seed"] for j in ch])
        pend += [io.submit(save, j, im) for j, im in zip(ch, outs)]
        n = min(k + a.bs, len(jobs))
        if n % (a.bs * 10) == 0 or n == len(jobs):
            print(f"[v2] PROGRESS shard{a.shard} {n}/{len(jobs)} {time.time() - t0:.0f}s {n / (time.time() - t0):.2f} img/s", flush=True)
    for f in pend:
        f.result()
    with open(out / f"prompts_shard{a.shard}.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps({"id": j["id"], "task": j["task"], "var": j.get("var", 0), "teacher_prompt": j["teacher_prompt"]}) + "\n")
    print(f"[v2] GEN_DONE shard{a.shard} {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ score
def score(a):
    import torch
    import metrics as M
    nt = int(os.environ.get("SCORE_THREADS", "0"))   # many small parts on a 768-core box: cap threads (CPU face models + WebP)
    if nt:
        import cv2
        torch.set_num_threads(nt)
        cv2.setNumThreads(nt)
    out = Path(a.out)
    jobs = [json.loads(l) for l in open(out / "jobs.jsonl")][a.part:: a.nparts]
    pr = {}
    for p in out.glob("prompts_shard*.jsonl"):
        for l in open(p):
            r = json.loads(l)
            pr[(r["id"], r["task"], r.get("var", 0))] = r["teacher_prompt"]
    fid = M.FaceID()
    fc_in = M.FaceCache(fid, D)          # synthetic / CelebA inputs (read-only)
    fc_q = M.FaceCache(fid, FNORM, name="_face_cache_tmp.pkl")
    fc_out = M.FaceCache(fid, out)
    emb, mat = M.Embedder(), M.Matter()
    refs = [Image.open(p) for p in sorted((ROOT / "styles/cartoon3d").glob("ref_*.png"))]
    rd, _ = emb(refs)
    rdm = torch.nn.functional.normalize(rd.mean(0), dim=0)
    # identity galleries: own + 99 neighbours of the same source, in sorted order (gen_pairs.py id_rank)
    pools = {p: sorted(x.stem for x in (D / "inputs").glob(f"{p}*.png")) for p in "fcr"}
    pools["q"] = sorted(json.loads((out / "ffhq_adults.json").read_text()))
    feat = {}

    def f_in(i):
        if i not in feat:
            f, _ = (fc_q if i.startswith("q") else fc_in).get(src_path(i))
            feat[i] = None if f is None else f.ravel() / np.linalg.norm(f)
        return feat[i]

    pools = {p: [i for i in v if f_in(i) is not None] for p, v in pools.items()}  # faces only, as gen_pairs.py vgroups
    print(f"[v2] identity galleries: { {p: len(v) for p, v in pools.items()} }", flush=True)

    def id_rank(own, fo):
        if fo is None or f_in(own) is None:
            return None
        pool = pools[own[0]]
        k = pool.index(own)
        gal = [own] + [pool[(k + 1 + m) % len(pool)] for m in range(min(99, len(pool) - 1))]
        sims = np.stack([f_in(g) for g in gal]) @ (fo.ravel() / np.linalg.norm(fo))
        return int((sims > sims[0]).sum()) + 1, len(gal)

    rows = []
    for j in jobs:
        p = out_path(out, j)
        if not p.exists():
            continue
        o = Image.open(p).convert("RGB")
        d, _ = emb([o])
        r = dict(j) | {"teacher_prompt": pr.get((j["id"], j["task"], j.get("var", 0))), "style_dino": round(float(d[0] @ rdm), 4)}
        if j["task"] != "avatar":
            _, info = M.sticker_postprocess(o, mat(o))
            r.update(info)
        fo, lo = fc_out.get(p)
        rk = id_rank(j["id"], fo)
        r["id_rank100"], r["gallery"] = (rk if rk else (None, None))
        r["id_pass"] = r["id_rank100"] is not None and r["id_rank100"] <= (10 if j["task"].startswith("expr_") else 5)
        _, li = (fc_q if j["id"].startswith("q") else fc_in).get(src_path(j["id"]))
        r["dL"] = None if li is None or lo is None else round(lo - li, 1)
        at = TP.parse(j["gen_prompt"]) if j.get("gen_prompt") else None
        r["extra_ok"] = not (at and at["extra"] and at["extra"] in (r["teacher_prompt"] or "") and j.get("old_extra_verified") is False)
        rows.append(keep_rule(r))
    if a.nparts > 1:
        with open(out / f"manifest_part{a.part}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[v2] SCORE_PART_DONE part{a.part} {len(rows)}", flush=True)
        return
    write_manifest(out, rows)


def _skin_lab(path):
    """Median CIELAB of the central face box (same box as metrics.FaceID.skin_L): [L, a, b, chroma, hue_deg] or None."""
    import cv2
    global _FID
    if "_FID" not in globals():
        import metrics as M
        _FID = M.FaceID()
    a = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = a.shape[:2]
    _FID.det.setInputSize((w, h))
    _, fs = _FID.det.detect(a)
    if fs is None or len(fs) == 0:
        return None
    x, y, fw, fh = max(fs, key=lambda q: q[2] * q[3])[:4]
    cx, cy = x + fw / 2, y + fh * 0.55
    x0, x1, y0, y1 = int(max(0, cx - .2 * fw)), int(min(w, cx + .2 * fw)), int(max(0, cy - .2 * fh)), int(min(h, cy + .2 * fh))
    if x1 <= x0 or y1 <= y0:
        return None
    v = cv2.cvtColor(a[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    L, aa, bb = float(np.median(v[:, 0])) * 100 / 255, float(np.median(v[:, 1])) - 128, float(np.median(v[:, 2])) - 128
    return [round(L, 1), round(aa, 1), round(bb, 1), round(float(np.hypot(aa, bb)), 1), round(float(np.degrees(np.arctan2(bb, aa))), 1)]


def _skin_pair(args):
    src, outp = args
    return _skin_lab(src), _skin_lab(outp)


def skincolor(a):
    """Adds skin_in/skin_out [L,a,b,C,h] and d_chroma / d_b / d_hue to every face row (CPU pool), then re-applies keep_rule."""
    from multiprocessing import Pool
    out = Path(a.out)
    rows = [json.loads(l) for l in open(out / "manifest.jsonl")]
    todo = [k for k, r in enumerate(rows) if "d_chroma" not in r]
    t0 = time.time()
    with Pool(a.workers, initializer=_pool_init) as pool:
        res = pool.map(_skin_pair, [(str(src_path(rows[k]["id"])), str(out_path(out, rows[k]))) for k in todo], chunksize=64)
    for k, (si, so) in zip(todo, res):
        r = rows[k]
        r["skin_in"], r["skin_out"] = si, so
        ok = si is not None and so is not None
        r["d_chroma"] = round(so[3] - si[3], 1) if ok else None
        r["d_b"] = round(so[2] - si[2], 1) if ok else None
        r["d_hue"] = round(so[4] - si[4], 1) if ok else None
    print(f"[v2] skincolor: {len(todo)} rows in {time.time() - t0:.0f}s ({a.workers} workers)", flush=True)
    write_manifest(out, [keep_rule(r) for r in rows])


def rekeep(a):
    """Re-apply keep_rule (e.g. new DL_LIMIT) to an existing manifest without re-scoring."""
    out = Path(a.out)
    write_manifest(out, [keep_rule(json.loads(l)) for l in open(out / "manifest.jsonl")])


def merge(a):
    out = Path(a.out)
    rows = [json.loads(l) for p in sorted(out.glob("manifest_part*.jsonl")) for l in open(p)]
    write_manifest(out, rows)


def write_manifest(out, rows):
    # best-of-N: one target per (photo, task): variant 0 if kept, else the first kept variant
    by = {}
    for r in rows:
        by.setdefault((r["id"], r["task"]), []).append(r)
    for k, v in by.items():
        v.sort(key=lambda r: r.get("var", 0))
        pick = next((r for r in v if r["keep"]), None)
        for r in v:
            r["selected"] = r is pick
    with open(out / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # keep-rate table
    grp = lambda t: "avatar" if t == "avatar" else "sticker" if t == "sticker" else "expressions"  # noqa: E731
    res = {}
    import numpy as np
    for src in ("synthetic+CelebA (same pairs as round 4)", "FFHQ adults (new)"):
        for g in ("avatar", "sticker", "expressions", "all"):
            sub = [r for r in rows if (r["id"][0] == "q") == src.startswith("FFHQ") and (g == "all" or grp(r["task"]) == g)]
            if not sub:
                continue
            e = {"n": len(sub), "new_keep": round(np.mean([r["keep"] for r in sub]), 3),
                 "kept_pairs": len({(r["id"], r["task"]) for r in sub if r["keep"]})}
            if not src.startswith("FFHQ") and all("old_keep" in r for r in sub):
                e["old_keep"] = round(np.mean([r["old_keep"] for r in sub]), 3)
            drop = [r for r in sub if not r["keep"]]
            e["new_drops"] = {"id": sum(not r["id_pass"] for r in drop),
                              "sticker": sum(r["id_pass"] and r["task"] != "avatar" and not r.get("pass_sticker") for r in drop),
                              "dL": sum(r["dL"] is not None and abs(r["dL"]) > r.get("dl_limit", MAX_DL) for r in drop)}
            e["provisional"] = sum(r.get("provisional", False) for r in sub)
            res[f"{src} | {g}"] = e
            print(f"{src:42s} {g:12s} {e}", flush=True)
    per = {}
    for r in rows:
        if r["task"].startswith("expr_"):
            per.setdefault(r["task"], []).append(r)
    res["per_expression_new_keep"] = {t: round(np.mean([r["keep"] for r in v]), 2) for t, v in sorted(per.items())}
    res["per_expression_kept_pairs"] = {t: sum(r["keep"] for r in v) for t, v in sorted(per.items())}
    if all("old_keep" in r for v in per.values() for r in v):
        res["per_expression_old_keep"] = {t: round(np.mean([r["old_keep"] for r in v]), 2) for t, v in sorted(per.items())}
    (out / "keep_rate.json").write_text(json.dumps(res, indent=1))
    print("[v2] SCORE_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["plan-test", "plan-full", "plan-real", "finalize-real", "caption", "caption-ids", "gen", "score", "merge",
                             "rekeep", "skincolor"])
    ap.add_argument("--ffhq-pool", type=int, default=27500)
    ap.add_argument("--ffhq-adults", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--natural-skin", type=int, default=0)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--avatar-n", type=int, default=2)
    ap.add_argument("--expr-per", type=int, default=2700)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--check-compile", type=int, default=1)
    a = ap.parse_args()
    a.out = str(ROOT / a.out) if not a.out.startswith("/") else a.out
    {"plan-test": plan_test, "plan-full": plan_full, "caption": caption, "caption-ids": caption_ids, "gen": gen, "score": score,
     "merge": merge, "rekeep": rekeep, "skincolor": skincolor, "plan-real": plan_real, "finalize-real": finalize_real}[a.stage](a)
