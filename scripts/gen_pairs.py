"""Step 3: teacher data engine. FLUX.2-klein-4B makes (photo -> avatar) and (photo -> sticker) pairs for ONE style.

Inputs are NEW synthetic photos (seeds 100000+, never the eval set in data/faces), generated with klein T2I from an
attribute grid (adults only, no names). Targets: klein photo-only edit with the style as text (PROMPTS_FAMILY=klein_text).
Every pair is scored with scripts/metrics.py and flagged keep=True only if it passes:
  avatar : face detected in output and the true input ranks top-5 of 100 by SFace similarity (absolute SFace cosine is
           not meaningful photo->cartoon: klein avatars score below impostor photo pairs)
  sticker: sticker post-process pass (+ identity pass for faces)

  source env.sh; $VENV/bin/python scripts/gen_pairs.py --style cartoon3d --faces 5000 --subjects 1200 --sticker-faces 2500
Out: data/pairs/<style>/{inputs,avatar,sticker}/*.png + manifest.jsonl (prompt, seed, metrics, keep)
"""
import argparse
import zlib
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image

os.environ.setdefault("MODEL", "klein")
os.environ.setdefault("KLEIN_REF", "0")
os.environ.setdefault("PROMPTS_FAMILY", "klein_text")
sys.path.insert(0, str(Path(__file__).parent))
import i2i_server as S  # noqa: E402
import metrics as M  # noqa: E402
import teacher_prompt as TP  # noqa: E402

ROOT = S.ROOT
AGES = ["in their early twenties"] * 3 + ["in their late twenties"] * 3 + ["in their thirties"] * 3 + \
       ["in their forties"] * 2 + ["in their fifties", "in their sixties", "in their seventies"]
GENDER = ["woman", "man", "person"]
SKIN = ["East Asian", "Southeast Asian", "South Asian", "Black", "white", "Latino", "Middle Eastern", "Pacific Islander", "mixed-race"]
HAIR = ["short black hair", "long straight black hair", "long wavy brown hair", "curly hair", "an afro", "a shaved head", "a bob cut",
        "grey hair in a bun", "red shoulder-length hair", "braided hair", "a buzz cut", "straight blonde hair", "a hijab",
        "dyed pink hair", "a ponytail", "messy short hair", "a man bun", "a turban", "dreadlocks", "a side part"]
EXTRA = ["", "", "", "wearing glasses", "wearing sunglasses on the head", "with a beard", "with a mustache", "with freckles",
         "wearing earrings", "wearing a cap", "wearing a beanie", "smiling broadly", "laughing", "neutral expression",
         "slight smile", "wearing headphones around the neck", "with a nose ring"]
FRAME = ["selfie taken at arm's length", "close-up portrait", "head and shoulders portrait", "half-body photo", "slightly low-angle selfie"]
POSE = ["facing the camera", "head turned slightly left", "head turned slightly right", "three-quarter view", "head tilted"]
LIGHT = ["soft window light", "outdoor daylight", "warm indoor light", "overcast light", "evening street light", "bright sunlight", "fluorescent office light"]
BG = ["plain wall", "blurred city street", "blurred park", "home interior", "office", "cafe", "car interior", "beach", "bedroom"]
SUBJ = ["cat", "kitten", "dog", "puppy", "golden retriever", "pug", "corgi", "shiba inu", "husky", "dachshund", "poodle", "rabbit",
        "hamster", "guinea pig", "parrot", "budgie", "goldfish", "turtle", "hedgehog", "ferret", "duckling", "chick",
        "cup of coffee", "bubble tea", "slice of pizza", "burger", "donut", "cupcake", "bowl of noodles", "sushi plate", "ice cream cone",
        "potted plant", "cactus", "sneaker", "backpack", "guitar", "camera", "skateboard", "bicycle", "teddy bear", "game controller",
        "headphones", "smartphone", "car toy", "basketball", "soccer ball", "sunflower", "rose", "apple", "avocado"]
SUBJ_ADJ = ["", "small ", "fluffy ", "colorful ", "old ", "shiny ", "cute "]


def face_prompt(rng):
    e = rng.choice(EXTRA)
    return (f"Realistic smartphone photo, {rng.choice(FRAME)} of a {rng.choice(SKIN)} {rng.choice(GENDER)} {rng.choice(AGES)} "
            f"with {rng.choice(HAIR)}{', ' + e if e else ''}, {rng.choice(POSE)}, {rng.choice(LIGHT)}, {rng.choice(BG)} background, "
            f"natural skin texture, sharp focus")


SELFIE_CLOSE = ["face filling most of the frame", "head and shoulders cut at the chest", "top of the head slightly cut off",
                "face slightly off-center", "tilted camera angle", "chin-level camera angle"]
SELFIE_CAM = ["phone camera noise", "slight motion blur", "harsh phone flash", "low light grain", "overexposed background",
              "auto-HDR look", "visible JPEG compression", "mixed indoor lighting color cast"]


def selfie_prompt(rng):
    """v3 inputs: real-looking arm's-length front-camera selfies (tight framing, phone artifacts). Keeps the
    '<gender> in their <age> with <hair>[, extra], <pose>' structure that teacher_prompt.parse() expects."""
    e = rng.choice(EXTRA)
    return (f"Unedited front-camera smartphone selfie taken at arm's length, {rng.choice(SELFIE_CLOSE)}, of a {rng.choice(SKIN)} "
            f"{rng.choice(GENDER)} {rng.choice(AGES)} with {rng.choice(HAIR)}{', ' + e if e else ''}, {rng.choice(POSE)}, "
            f"{rng.choice(LIGHT)}, {rng.choice(BG)} background, slight wide-angle lens distortion, {rng.choice(SELFIE_CAM)}, "
            f"natural skin texture with pores and small blemishes, realistic")


def subj_prompt(rng):
    return (f"Realistic smartphone photo of a {rng.choice(SUBJ_ADJ)}{rng.choice(SUBJ)}, full subject visible, "
            f"{rng.choice(LIGHT)}, {rng.choice(BG)} background")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="cartoon3d")
    ap.add_argument("--faces", type=int, default=1600)
    ap.add_argument("--subjects", type=int, default=400)
    ap.add_argument("--sticker-faces", type=int, default=800)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-dl", type=float, default=1e9, help="drop face pairs whose |skin L* shift| exceeds this")
    ap.add_argument("--selfies", type=int, default=0, help="v3: realistic close-up selfie inputs (ids c*, seeds 500000+)")
    ap.add_argument("--selfie-stickers", type=int, default=0)
    ap.add_argument("--verify-extra", action="store_true", help="CLIP-verify prompt extras on the input; drop unverified")
    ap.add_argument("--real-n", type=int, default=0, help="real photos from data/real_p2a/train (ids r*, legal-approved)")
    ap.add_argument("--expr-faces", type=int, default=0, help="sticker-set expressions for the first N f* faces")
    ap.add_argument("--expr-k", type=int, default=4, help="expressions per face (sampled per face id, deterministic)")
    ap.add_argument("--real-stickers", type=int, default=0)
    # round 4 (hands): a SECOND expression batch on f* faces [expr2-from, expr2-to) restricted to a subset (e.g. the hand gestures);
    # the first batch (--expr-faces/--expr-k over all expressions) is unchanged, so existing pairs/prompts are kept
    ap.add_argument("--expr2-from", type=int, default=0)
    ap.add_argument("--expr2-to", type=int, default=0)
    ap.add_argument("--expr2-set", default="", help="comma-separated expression ids, e.g. think,thumbs,wave,ok,facepalm,shrug,thanks")
    ap.add_argument("--expr2-k", type=int, default=3)
    ap.add_argument("--expr2-adult", action="store_true", help="expr2 batch uses TP.expr_sticker_prompt_adult (age decade, no child bodies)")
    a = ap.parse_args()
    rng = random.Random(4242)
    out = ROOT / "data/pairs" / a.style
    for d in ("inputs", "avatar", "sticker"):
        (out / d).mkdir(parents=True, exist_ok=True)
    # faces: one sequential rng -> raising --faces keeps every existing prompt. Subjects: own rng and new ids "o*" (v2).
    # The v1 subject pairs "s*" (seeds 200000+) had humanized targets (README deviation 8) and are no longer used.
    rng_s = random.Random(4343)
    items = [("f%05d" % i, face_prompt(rng), 100_000 + i, True) for i in range(a.faces)] + \
            [("o%05d" % i, subj_prompt(rng_s), 300_000 + i, False) for i in range(a.subjects)]
    rng_c = random.Random(4545)
    selfies = [("c%05d" % i, selfie_prompt(rng_c), 500_000 + i, True) for i in range(a.selfies)]
    items = items + selfies
    real_src = sorted((ROOT / "data/real_p2a/train").glob("rt_*.png"))[:a.real_n]
    reals = [("r%05d" % i, "real photo", 700_000 + i, True) for i in range(len(real_src))]
    for (rid, _, _, _), sp in zip(reals, real_src):  # inputs are already framing-normalised by import_real_p2a.py
        if not (out / "inputs" / f"{rid}.png").exists():
            import shutil
            shutil.copyfile(sp, out / "inputs" / f"{rid}.png")
    items = items + reals

    ad = S.Klein()
    ad.load()
    ad.p.set_progress_bar_config(disable=True)
    t0 = time.time()
    # 1) synthetic inputs (text-to-image)
    if selfies:  # raw phone-aspect selfie -> the SAME framing normaliser as user photos (scripts/framing.py) -> input
        from framing import normalize_face_crop
        (out / "inputs_raw").mkdir(exist_ok=True)
        fid_fr = M.FaceID()
        todo_c = [x for x in selfies if not (out / "inputs" / f"{x[0]}.png").exists()]
        for i in range(0, len(todo_c), a.bs):
            ch = todo_c[i:i + a.bs]
            with torch.inference_mode():
                ims = ad.p(prompt=[x[1] for x in ch], width=512, height=640, num_inference_steps=4, guidance_scale=1.0,
                           generator=[torch.Generator("cuda").manual_seed(x[2]) for x in ch]).images
            for x, im in zip(ch, ims):
                im.save(out / "inputs_raw" / f"{x[0]}.png")
                normalize_face_crop(im, fid_fr)[0].save(out / "inputs" / f"{x[0]}.png")
    todo = [x for x in items if x[0][0] not in "cr" and not (out / "inputs" / f"{x[0]}.png").exists()]
    for i in range(0, len(todo), a.bs):
        ch = todo[i:i + a.bs]
        with torch.inference_mode():
            ims = ad.p(prompt=[x[1] for x in ch], width=512, height=512, num_inference_steps=4, guidance_scale=1.0,
                       generator=[torch.Generator("cuda").manual_seed(x[2]) for x in ch]).images
        for x, im in zip(ch, ims):
            im.save(out / "inputs" / f"{x[0]}.png")
    print(f"[pairs] inputs done {time.time() - t0:.0f}s", flush=True)
    # 2) teacher targets
    st = S.load_styles()[a.style]
    fc = M.FaceCache(M.FaceID(), out)  # cached SFace feature + skin L* per image (scoring re-runs only touch new files)
    tprompt = {}
    idx = {x[0]: x for x in items}
    old_tp = {}
    if (out / "manifest.jsonl").exists():
        for l in open(out / "manifest.jsonl"):
            r = json.loads(l)
            old_tp[(r["id"], r["task"])] = r
    ac = M.AttrCheck() if a.verify_extra else None
    verified = {}  # id -> True/False/None (None = nothing to verify)
    for x in items:
        if not x[3]:
            continue
        o = old_tp.get((x[0], "avatar"))
        if o is not None and "extra_verified" in o:
            verified[x[0]] = o["extra_verified"]
        elif ac is not None:
            at = TP.parse(x[1])
            verified[x[0]] = ac(Image.open(out / "inputs" / f"{x[0]}.png"), at["gender"], at["extra"]) if at else None
    for x in items:  # per-person prompt (privileged attributes + measured skin tone), see scripts/teacher_prompt.py
        at = TP.parse(x[1]) if x[3] else None
        if at is not None and verified.get(x[0]) is False and not (out / "avatar" / f"{x[0]}.png").exists():
            at = dict(at, extra=None)  # new pair: never describe an attribute the photo does not show
        pdesc = TP.person_desc(at, fc.get(out / "inputs" / f"{x[0]}.png")[1]) if x[3] else None
        tprompt[(x[0], "avatar")] = TP.avatar_prompt(st["description"], pdesc)
        tprompt[(x[0], "sticker")] = (TP.sticker_prompt(st["description"], pdesc) if x[3] else
                                      TP.subject_sticker_prompt(st.get("subject_description", st["description"]), TP.parse_subject(x[1])))
    import yaml as _y
    EXPR = _y.safe_load((ROOT / "configs/sticker_expressions.yaml").read_text())
    for x in [x for x in items if x[3] and x[0][0] == "f"][:a.expr_faces]:
        at = TP.parse(x[1])
        if at is not None and verified.get(x[0]) is False:
            at = dict(at, extra=None)
        pd = TP.person_desc(at, fc.get(out / "inputs" / f"{x[0]}.png")[1])
        for e in random.Random(f"expr|{x[0]}").sample(EXPR, a.expr_k):
            tprompt[(x[0], f"expr_{e['id']}")] = TP.expr_sticker_prompt(st["description"], pd, e["phrase"])
    if a.expr2_to > a.expr2_from:
        sub = [e for e in EXPR if e["id"] in a.expr2_set.split(",")]
        assert len(sub) == len(a.expr2_set.split(",")), "unknown id in --expr2-set"
        for x in [x for x in items if x[3] and x[0][0] == "f"][a.expr2_from:a.expr2_to]:
            at = TP.parse(x[1])
            if at is not None and verified.get(x[0]) is False:
                at = dict(at, extra=None)
            pd = TP.person_desc(at, fc.get(out / "inputs" / f"{x[0]}.png")[1])
            for e in random.Random(f"expr2|{x[0]}").sample(sub, min(a.expr2_k, len(sub))):
                tprompt[(x[0], f"expr_{e['id']}")] = (TP.expr_sticker_prompt_adult(st["description"], pd, e["phrase"], at["age"])
                                                      if a.expr2_adult and at is not None else
                                                      TP.expr_sticker_prompt(st["description"], pd, e["phrase"]))
    expr_jobs = [(x, t) for (i_, t) in tprompt if t.startswith("expr_") for x in [idx[i_]]] if (a.expr_faces or a.expr2_to) else []
    jobs = [(x, "avatar") for x in items if x[3]] + [(x, "sticker") for x in items if x[3] and x[0][0] == "f"][:a.sticker_faces] + \
           [(x, "sticker") for x in items if not x[3]] + [(x, "sticker") for x in selfies][:a.selfie_stickers] + \
           [(x, "sticker") for x in reals][:a.real_stickers] + expr_jobs
    for k, (x, task) in enumerate(jobs):
        dst = out / task / f"{x[0]}.png"
        if dst.exists():
            continue
        dst.parent.mkdir(exist_ok=True)
        img = Image.open(out / "inputs" / f"{x[0]}.png").convert("RGB")
        with torch.inference_mode():
            sd = x[2] + (7 if task == "sticker" else 0) + (zlib.crc32(task.encode()) % 997 + 11 if task.startswith("expr_") else 0)
            ad.run([img], tprompt[(x[0], task)], st, [sd])[0].save(dst)
        if k % 200 == 0:
            print(f"[pairs] teacher {k}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    del ad
    torch.cuda.empty_cache()
    # 3) score + filter (style/sticker scores of pairs already in the manifest are reused; SFace ranks are recomputed)
    old = {}
    if (out / "manifest.jsonl").exists():
        for l in open(out / "manifest.jsonl"):
            r = json.loads(l)
            if r.get("teacher_prompt") == tprompt.get((r["id"], r["task"])) and "style_dino" in r:
                old[(r["id"], r["task"])] = r
    fc.save()
    fid = fc.fid
    new = [(x, t) for x, t in jobs if (x[0], t) not in old]
    print(f"[pairs] scoring: {len(jobs) - len(new)} reused, {len(new)} new", flush=True)
    pre = {}
    if new:  # GPU scorers only when there is something new to score
        emb, mat, sc = M.Embedder(), M.Matter(), M.SubjectCheck()
        refs = [Image.open(p) for p in sorted((ROOT / "styles" / a.style).glob("ref_*.png"))]
        rd, rc = emb(refs)
        rdm = torch.nn.functional.normalize(rd.mean(0), dim=0)
        if os.environ.get("FAST_SCORE", "1" if (ROOT / "configs/fast_score.on").exists() else "0") == "1":  # score_fast.py (verified: scripts/test_score_fast.py)
            import score_fast as SF
            pre = SF.precompute(out, [(x[0], t, x[3], x[1]) for x, t in new], fc, emb, mat, sc, rdm, rd,
                                subj_name=TP.parse_subject, workers=int(os.environ.get("SCORE_WORKERS", "32")),
                                log=lambda m: print(f"{m} {time.time() - t0:.0f}s", flush=True))
            fc.save()
    import numpy as np
    face_ids = [x[0] for x in items if x[3]]
    infeat, inraw, inL = {}, {}, {}
    for fid_ in face_ids:
        f, inL[fid_] = fc.get(out / "inputs" / f"{fid_}.png")
        inraw[fid_] = f
        infeat[fid_] = None if f is None else f.ravel() / np.linalg.norm(f)
    valid = [k for k in face_ids if infeat[k] is not None]
    vgroups = {}
    for k in valid:  # galleries per id prefix (f*, c*): adding selfies does not change the f* ranks
        vgroups.setdefault(k[0], []).append(k)

    def id_rank(own, fo):
        """rank of the true input among itself + 99 other training inputs (photo->cartoon robust identity check)"""
        if fo is None or infeat.get(own) is None:
            return None
        vg = vgroups[own[0]]
        i = vg.index(own)
        gal = [own] + [vg[(i + 1 + j) % len(vg)] for j in range(min(99, len(vg) - 1))]
        sims = np.stack([infeat[g] for g in gal]) @ (fo.ravel() / np.linalg.norm(fo))
        return int((sims > sims[0]).sum()) + 1

    rows = []
    for x, task in jobs:
            r = {"id": x[0], "task": task, "style": a.style, "prompt": x[1], "teacher_prompt": tprompt[(x[0], task)],
                 "seed": x[2], "is_face": x[3]}
            if (x[0], task) in old:
                r |= {k: v for k, v in old[(x[0], task)].items() if k not in r and k not in ("id_cos", "id_rank100", "id_pass", "keep")}
            elif (x[0], task) in pre:
                r |= pre[(x[0], task)]
            else:
                o = Image.open(out / task / f"{x[0]}.png").convert("RGB")
                d, _ = emb([o])
                r |= {"style_dino": round(float(d[0] @ rdm), 4), "refcopy_dino": round(float((d[0] @ rd.T).max()), 4)}
                if task != "avatar":  # plain and expression stickers
                    _, info = M.sticker_postprocess(o, mat(o))
                    r.update(info)
                if not x[3]:
                    r |= sc(o, TP.parse_subject(x[1]))
            if x[3]:
                fo, lo = fc.get(out / task / f"{x[0]}.png")
                cs = fid.cos(inraw[x[0]], fo)
                r["id_cos"] = None if cs is None else round(cs, 4)
                r["id_rank100"] = id_rank(x[0], fo)
                # expressions (closed eyes, hands near the face) are harder for SFace: top-10 instead of top-5
                r["id_pass"] = r["id_rank100"] is not None and r["id_rank100"] <= (10 if task.startswith("expr_") else 5)
                li = inL[x[0]]  # teacher skin-tone drift (students copy it), see README step 3c
                r["dL"] = None if li is None or lo is None else round(lo - li, 1)
            if x[3]:
                at_ = TP.parse(x[1])
                r["extra_verified"] = verified.get(x[0])
                # extra_ok: the teacher prompt only names attributes that are visible (or there was nothing to verify)
                r["extra_ok"] = not (at_ and at_["extra"] and at_["extra"] in r["teacher_prompt"] and verified.get(x[0]) is False)
            r["keep"] = bool(r.get("id_pass", True) and (task == "avatar" or r.get("pass_sticker", False))
                             and (not a.verify_extra or r.get("extra_ok", True))
                             and (r.get("dL") is None or abs(r["dL"]) <= a.max_dl)
                             and (x[3] or (r.get("clip_subject_ok", False)
                                           # YuNet fires on cartoon animal faces (false positives), so it only vetoes objects
                                           and (TP.is_animal(TP.parse_subject(x[1])) or not r.get("human_face", True)))))
            rows.append(r)
            if len(rows) % 500 == 0:
                print(f"[pairs] scored {len(rows)}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    fc.save()
    print(f"[pairs] face cache hits {fc.hits} misses {fc.miss}", flush=True)
    with open(out / "manifest.jsonl", "w") as man:
        for r in rows:
            man.write(json.dumps(r) + "\n")
    print(f"[pairs] scored, total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
