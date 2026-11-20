"""Teacher consistency check (2026-11-18, no training): how predictable is klein's target from the photo alone?
A student trained on one random target per photo learns the AVERAGE of what the teacher might draw; anything the teacher
varies seed to seed (clothes colour, framing, hair layout) becomes blur in the student. Compares teacher prompt settings:
  current : the manifest's teacher prompt (what the training data used)
  keep    : + keep framing/pose/head size/position and clothing colours from the photo, plain light-grey background
  simple  : keep + a student-friendly style (dropped after run 1: no gain, lower likeness)
  v2      : full rewrite: local Qwen3-VL attributes (scripts/caption_photos.py) for EVERY photo, incl. clothing; one explicit
            "keep exactly" list (hair colour, head pose, clothing colours, framing); light grey background
Photos: --n-train cartoon3d avatar photos from manifest_round4 (3/4 synthetic f*, 1/4 selfie-style c*) and --n-ffhq FFHQ
photos (data/ffhq, user-approved real-photo exception 2026-11-18), FFHQ passed through framing.normalize_face_crop like
user photos. Settings run with the same env as gen_pairs.py (KLEIN_REF=0, PROMPTS_FAMILY=klein_text).
  gen  : $VENV/bin/python scripts/teacher_consistency.py gen --shard 0 --nshards 2 [--tag T --settings v2 --n-train 8 ...]
  score: $VENV/bin/python scripts/teacher_consistency.py score [same --tag/--settings/--n-*]
Out: data/consistency[_T]/<setting>/<id>_s<k>.png, results/teacher_consistency[_T].json, grids/teacher_consistency[_T]_sheet.jpg
Metrics per setting (lower = more consistent, except id_cos):
  seed_lpips  mean pairwise LPIPS(VGG, 256 px) between the seeds of a photo
  frame_jit   across-seed std of the face-box centre (fraction of image width) and of log face size
  shirt_dE    across-seed std of the torso colour (CIELAB, bottom-centre region), and input-vs-target torso dE
  id_cos      SFace cosine input photo vs target (likeness proxy; higher = better)
"""
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("MODEL", "klein")  # exactly as gen_pairs.py: text-only style, no style reference image
os.environ.setdefault("KLEIN_REF", "0")
os.environ.setdefault("PROMPTS_FAMILY", "klein_text")
sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/pairs/cartoon3d"
FFHQ = ROOT / "data/ffhq"
KEEP = ("Keep the exact framing, pose, head size and head position of the photo, and the same clothing with the same "
        "colors. Plain light grey background.")
SIMPLE = " Simple clean stylized shapes: hair made of a few large smooth shapes, large clear eyes with a white highlight."


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gen", "score", "caption"])
    ap.add_argument("--task", default="avatar", help="avatar | sticker | expr_<id> (configs/sticker_expressions.yaml)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--settings", default="current,keep,simple")
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-ffhq", type=int, default=0)
    ap.add_argument("--nseed", type=int, default=3)
    return ap.parse_args()


def outdir(a):
    t = getattr(a, "task", "avatar")
    return ROOT / ("data/consistency" + (f"_{a.tag}" if a.tag else "") + (f"_{t}" if t != "avatar" else ""))


def rows(a):
    rs = [json.loads(l) for l in open(D / "manifest_round4.jsonl")]
    rs = [r for r in rs if r["keep"] and r["task"] == "avatar"]
    nf, nc = a.n_train - a.n_train // 4, a.n_train // 4
    f = sorted([r for r in rs if r["id"].startswith("f")], key=lambda r: r["id"])
    c = sorted([r for r in rs if r["id"].startswith("c")], key=lambda r: r["id"])
    out = [dict(r, src=str(D / "inputs" / f"{r['id']}.png")) for r in
           f[:: max(1, len(f) // max(nf, 1))][:nf] + c[:: max(1, len(c) // max(nc, 1))][:nc]]
    if a.n_ffhq:
        ff = sorted(p for p in FFHQ.rglob("*.png") if p.stem.isdigit())
        for p in ff[:: max(1, len(ff) // a.n_ffhq)][:a.n_ffhq]:
            out.append({"id": f"q{p.stem}", "src": str(p), "seed": 900000 + int(p.stem), "teacher_prompt": None})
    return out


def input_img(r, fid=None):
    """512 px teacher input. FFHQ: the same framing normaliser as user photos (cached next to the outputs)."""
    if not r["id"].startswith("q"):
        return Image.open(r["src"]).convert("RGB")
    cache = ROOT / "data/ffhq_norm" / f"{r['id']}.png"  # outside data/ffhq, so rows() never picks it up
    if cache.exists():
        return Image.open(cache).convert("RGB")
    import metrics as M
    from framing import normalize_face_crop
    im, _ = normalize_face_crop(Image.open(r["src"]), fid or M.FaceID())
    cache.parent.mkdir(parents=True, exist_ok=True)
    im.save(cache)
    return im


def v2_prompt(style_desc, cap, tone):
    g = cap.get("gender") if cap.get("gender") in ("man", "woman") else "person"
    age = cap.get("age") or ""
    d = f"a {g}" + (f" in their {age}" if age and "older" not in age else " in their seventies or older" if age else "")
    d += f" with {cap.get('hair') or 'their hairstyle'}"
    for k in ("facial_hair", "glasses", "headwear"):
        v = (cap.get(k) or "").strip()
        if v and v.lower() not in ("none", "no", "not visible", "n/a"):
            d += f", {v}"
    if tone:
        d += f", with {tone} skin"
    cl = cap.get("clothing") or ""
    wear = f", wearing {cl}" if cl and cl.lower() != "not visible" else ""
    ex = cap.get("expression") or ""
    exs = f", with a {ex} expression" if ex else ""
    return (f"Redraw the person in this photo as an avatar portrait, {style_desc}. The person is {d}{wear}{exs}. "
            f"Keep exactly from the photo: the face, skin tone, age, gender, hairstyle and hair colour, facial hair, glasses, "
            f"expression, head pose and head angle, and the same clothing with the same colors. Keep the photo's framing, "
            f"head size and head position. Head and shoulders on a plain light grey background, nothing else in the background.")


NONE = ("", "none", "no", "not visible", "n/a", "unknown")


def v3_prompt(style_desc, cap, tone, attrs):
    """v2 lessons (stage A, 8 photos): naming items in the keep list ("glasses", "facial hair") made klein ADD them to
    everyone; the 2B VLM is wrong on hair length/baldness/moustaches. v3: generator attributes (gender/age/hair) when we have
    them, VLM only for clothing (+ gender/age on real photos, + glasses/facial hair only when it reports them present),
    hair otherwise taken from the photo, and a keep list that never names an item that may be absent."""
    # gender: what the photo shows (VLM); stage A had a synthetic photo whose generator attribute said "woman" for a man
    g = cap.get("gender") if cap.get("gender") in ("man", "woman") else (attrs or {}).get("gender", "person")
    if attrs:
        d = f"a {g} in their {attrs['age']} with {attrs['hair']}" + (f", {attrs['extra']}" if attrs.get("extra") else "")
    else:
        age = cap.get("age") or ""
        d = f"a {g}" + (f" in their {age}" if age and "older" not in age else " in their seventies or older" if age else "")
    for k in ("glasses", "facial_hair"):
        v = (cap.get(k) or "").strip()
        if v.lower() not in NONE:
            d += f", with {v}"
    if tone:
        d += f", with {tone} skin"
    cl = (cap.get("clothing") or "").strip()
    wear = f", wearing a {cl}" if cl.lower() not in NONE else ""
    return (f"Redraw the person in this photo as an avatar portrait, {style_desc}. The person is {d}{wear}. "
            f"Keep the face, skin tone, age, gender, expression and head pose exactly as in the photo, the same hairstyle, "
            f"hair length and hair colour as in the photo, and the same clothing in the same colors. Keep the photo's framing, "
            f"head size and head position. Head and shoulders on a plain light grey background, nothing else in the background.")


def v4_desc(cap, tone, attrs):
    """Person description of v4 (same text as v4_prompt builds): returns (desc, ', wearing a ...' or '', age phrase or '')."""
    g = cap.get("gender") if cap.get("gender") in ("man", "woman") else (attrs or {}).get("gender", "person")
    if attrs:
        age = attrs["age"]
        d = f"a {g} in their {age} with {attrs['hair']}" + (f", {attrs['extra']}" if attrs.get("extra") else "")
    else:
        age = cap.get("age") or ""
        d = f"a {g}" + (f" in their {age}" if age else "")
        hair = "a bald or shaved head" if (cap.get("bald") or "").lower().startswith("y") else (cap.get("hair") or "")
        if hair and hair.lower() not in NONE:
            d += f" with {hair}"
    for k in ("glasses", "facial_hair"):
        v = (cap.get(k) or "").strip()
        if v.lower() not in NONE:
            d += f", with {v}"
    if not attrs:
        hw = (cap.get("headwear") or "").strip()
        if hw.lower() not in NONE:
            d += f", wearing a {hw} exactly where it is in the photo"
    sk = tone if attrs else (cap.get("skin_tone") or "").strip()
    if sk and sk.lower() not in NONE:
        d += f", with {sk} skin"
    cl = (cap.get("clothing") or "").strip()
    return d, (f", wearing a {cl}" if cl.lower() not in NONE else ""), age


def v4_sticker_prompt(style_desc, cap, tone, attrs, phrase=None):
    """v4 for the sticker tasks: v4 person description + one safe keep list (no absent items named); "head and upper body"
    for every person sticker (the old plain-sticker prompt said "Full subject"); the round-4 adult sentence (doubled SFace on
    older faces, teacher_prompt.expr_sticker_prompt_adult) whenever the age is known; expression stickers get the menu
    phrase as "Expression and pose", the plain sticker keeps the photo's expression and head pose."""
    d, wear, age = v4_desc(cap, tone, attrs)
    if phrase:  # synthetic attributes carry the photo's expression ("smiling broadly"): it would fight e.g. "crying"
        import re
        d = ", ".join(x for x in d.split(", ") if not re.search(r"smil|laugh|grin|frown|neutral|serious", x))
    keep = ("Keep the face, skin tone, age and gender, the same hairstyle, hair length and hair colour, and the same clothing in "
            "the same colors as in the photo.")
    act = f" Expression and pose: {phrase}." if phrase else " Keep the expression and head pose from the photo."
    adult = (f" This is an adult in their {age}: realistic adult head-to-body ratio, adult shoulders and adult hands, keep the "
             f"visible signs of age from the photo; never draw a child or a baby.") if age else ""
    return (f"Redraw the person in this photo as a die-cut chat sticker, {style_desc}. The person is {d}{wear}. {keep}{act}{adult} "
            f"Head and upper body, {'expressive, ' if phrase else ''}thick white outline, isolated on a plain flat white background.")


OLDER = ("forties", "fifties", "sixties", "seventies")


def v5_sticker_prompt(style_desc, cap, tone, attrs, phrase=None):
    """v5 (after the v4 sticker test, grids/sticker_prompt_test_expr_*.jpg): v4 made gesture stickers show more torso, so
    smaller, younger-looking people, and older faces lost wrinkles/grey hair; the old prompt aged young people (it named
    wrinkles/grey hair for everyone). v5 = v4 + a close crop with a large head + "the sticker shows exactly this person" +
    the age cues named only for people in their forties or older."""
    d, wear, age = v4_desc(cap, tone, attrs)
    if phrase:
        import re
        d = ", ".join(x for x in d.split(", ") if not re.search(r"smil|laugh|grin|frown|neutral|serious", x))
    keep = ("The sticker shows exactly this person: keep the face, skin tone, age and gender, the same hairstyle, hair length and "
            "hair colour, and the same clothing in the same colors as in the photo.")
    act = f" Expression and pose: {phrase}." if phrase else " Keep the expression and head pose from the photo."
    cues = " (wrinkles, grey hair)" if any(o in age for o in OLDER) else ""
    adult = (f" This is an adult in their {age}: realistic adult head-to-body ratio, adult shoulders and adult hands, keep the "
             f"visible signs of age from the photo{cues}; never draw a child or a baby.") if age else ""
    return (f"Redraw the person in this photo as a die-cut chat sticker, {style_desc}. The person is {d}{wear}. {keep}{act}{adult} "
            f"Head and upper body in a close crop with a large head, {'expressive, ' if phrase else ''}thick white outline, isolated "
            f"on a plain flat white background.")


def old_task_prompt(r, task, style_desc, tone):
    """The ORIGINAL data-pipeline prompt for a sticker task (baseline): same person text the manifest used for this photo
    (synthetic: generator attributes; real photos: 'a person with <tone> skin'), gen_pairs/teacher_prompt templates."""
    import re
    import teacher_prompt as TP
    m = re.search(r"The person is (.+?); the", r["teacher_prompt"] or "")
    pd = m.group(1) if m else (f"a person with {tone} skin" if tone else None)
    if task == "sticker":
        return TP.sticker_prompt(style_desc, pd)
    ph = EXPR[task[5:]]
    at = TP.parse(r["prompt"]) if r.get("prompt") else None
    return TP.expr_sticker_prompt_adult(style_desc, pd, ph, at["age"]) if at else TP.expr_sticker_prompt(style_desc, pd, ph)


def _expr():
    import yaml
    return {e["id"]: e["phrase"] for e in yaml.safe_load((ROOT / "configs/sticker_expressions.yaml").read_text())}


EXPR = _expr()


def v4_prompt(style_desc, cap, tone, attrs):
    """v4 (after stage A on FFHQ): v3 + for photos without generator attributes (real photos) the VLM's hair (or "a bald or
    shaved head"), headwear and skin-tone word replace the L*-based tone word, which mislabelled FFHQ lighting both ways.
    Children are dropped before prompting (gen)."""
    if attrs:
        return v3_prompt(style_desc, cap, tone, attrs)
    g = cap.get("gender") if cap.get("gender") in ("man", "woman") else "person"
    age = cap.get("age") or ""
    d = f"a {g}" + (f" in their {age}" if age and "older" not in age else " in their seventies or older" if age else "")
    hair = "a bald or shaved head" if (cap.get("bald") or "").lower().startswith("y") else (cap.get("hair") or "")
    if hair and hair.lower() not in NONE:
        d += f" with {hair}"
    for k in ("glasses", "facial_hair"):
        v = (cap.get(k) or "").strip()
        if v.lower() not in NONE:
            d += f", with {v}"
    hw = (cap.get("headwear") or "").strip()
    if hw.lower() not in NONE:
        d += f", wearing a {hw} exactly where it is in the photo"
    sk = (cap.get("skin_tone") or "").strip()
    if sk.lower() not in NONE:
        d += f", with {sk} skin"
    cl = (cap.get("clothing") or "").strip()
    wear = f", wearing a {cl}" if cl.lower() not in NONE else ""
    return (f"Redraw the person in this photo as an avatar portrait, {style_desc}. The person is {d}{wear}. "
            f"Keep the face, skin tone, age, gender, expression and head pose exactly as in the photo, the same hairstyle, "
            f"hair length and hair colour as in the photo, and the same clothing in the same colors. Keep the photo's framing, "
            f"head size and head position. Head and shoulders on a plain light grey background, nothing else in the background.")


def prompt(r, s, st=None, cap=None, tone=None, task="avatar"):
    if task != "avatar":
        import teacher_prompt as TP
        at = TP.parse(r["prompt"]) if r.get("prompt") else None
        if s in ("v4", "v5"):
            f = v4_sticker_prompt if s == "v4" else v5_sticker_prompt
            return f(st["description"], cap or {}, tone, at, None if task == "sticker" else EXPR[task[5:]])
        assert s == "current", f"{s} is avatar-only"
        return old_task_prompt(r, task, st["description"], tone)
    if s == "v4":
        import teacher_prompt as TP
        at = TP.parse(r["prompt"]) if r.get("prompt") else None
        return v4_prompt(st["description"], cap or {}, tone, at)
    if s == "v2":
        return v2_prompt(st["description"], cap or {}, tone)
    if s == "v3":
        import teacher_prompt as TP
        at = TP.parse(r["prompt"]) if r.get("prompt") else None
        return v3_prompt(st["description"], cap or {}, tone, at)
    p = r["teacher_prompt"]
    if s == "current":
        return p
    tail = "Head and shoulders, centered, plain background."
    p = p.replace(tail, "Head and shoulders. " + KEEP) if tail in p else p + " " + KEEP
    return p + (SIMPLE if s == "simple" else "")


def captions(a, rs, fid):
    """Qwen3-VL attributes for every photo of the run (v2 only), cached in <out>/captions.jsonl."""
    import teacher_prompt as TP
    cf = ROOT / "data/consistency_captions.jsonl"  # shared by every tag/task (run `caption` mode once before parallel gens)
    have = {json.loads(l)["path"]: json.loads(l) for l in open(cf)} if cf.exists() else {}
    tmp = ROOT / "data/consistency_inputs"
    tmp.mkdir(parents=True, exist_ok=True)
    need = []
    for r in rs:
        p = str(tmp / f"{r['id']}.png")
        if p not in have:
            input_img(r, fid).save(p)
            need.append(p)
    if need:
        import caption_photos as CP
        proc, model = CP.load()
        with open(cf, "a") as f:
            for i in range(0, len(need), 32):
                ch = need[i:i + 32]
                for p, c in zip(ch, CP.caption_batch(proc, model, [Image.open(p).convert("RGB").resize((448, 448)) for p in ch])):
                    f.write(json.dumps({"path": p} | c) + "\n")
                    have[p] = {"path": p} | c
        del model
        import torch
        torch.cuda.empty_cache()
    out = {}
    for r in rs:
        p = str(tmp / f"{r['id']}.png")
        out[r["id"]] = (have[p], TP.tone_word(fid.skin_L(Image.open(p))))
    return out


def gen(a):
    import torch
    import i2i_server as S
    import metrics as M
    fid = M.FaceID()
    settings = a.settings.split(",")
    rs = rows(a)[a.shard:: a.nshards]
    out = outdir(a)
    caps = captions(a, rs, fid) if any(x in settings for x in ("v2", "v3", "v4")) else {}
    kids = [r["id"] for r in rs if (caps.get(r["id"], ({}, 0))[0].get("child") or "").lower().startswith("y")]
    if kids:
        print(f"[cons] dropping {len(kids)} photos the VLM marks as children: {kids}", flush=True)
        rs = [r for r in rs if r["id"] not in kids]
    jobs = [(r, s, k) for r in rs for s in settings for k in range(a.nseed)
            if not (a.task == "avatar" and s in ("current", "keep", "simple") and r["teacher_prompt"] is None)]
    jobs = [j for j in jobs if not (out / j[1] / f"{j[0]['id']}_s{j[2]}.png").exists()]
    st = S.load_styles()["cartoon3d"]
    ad = S.Klein()
    ad.load()
    t0 = time.time()
    print(f"[cons] shard {a.shard}/{a.nshards}: {len(jobs)} images to make", flush=True)
    prompts = {}
    for n, (r, s, k) in enumerate(jobs, 1):
        dst = out / s / f"{r['id']}_s{k}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        img = input_img(r, fid)
        cap, tone = caps.get(r["id"], (None, None))
        p = prompt(r, s, st, cap, tone, a.task)
        prompts[f"{s}/{r['id']}"] = p
        with torch.inference_mode():
            ad.run([img], p, st, [int(r["seed"]) + 7919 * k])[0].save(dst)
        if n % 20 == 0 or n == len(jobs):
            print(f"[cons] PROGRESS shard{a.shard} {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    (out / f"prompts_shard{a.shard}.json").write_text(json.dumps(prompts, indent=1))
    print(f"[cons] GEN_DONE shard{a.shard} {time.time() - t0:.0f}s", flush=True)


def score(a):
    import cv2
    import lpips
    import torch
    import torch.nn.functional as F
    import metrics as M
    from train_student import to_t
    dev = "cuda"
    lp = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
    fid = M.FaceID()
    rs = rows(a)
    out = outdir(a)
    settings = a.settings.split(",")
    K = a.nseed

    def face(im):
        a_ = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        fid.det.setInputSize((a_.shape[1], a_.shape[0]))
        _, f = fid.det.detect(a_)
        return None if f is None or len(f) == 0 else max(f, key=lambda q: q[2] * q[3])

    def torso(im):  # mean CIELAB colour of the bottom-centre region (shirt in a head-and-shoulders crop)
        x = cv2.cvtColor(np.array(im.resize((512, 512))), cv2.COLOR_RGB2LAB).astype(np.float32)
        x = x[440:505, 170:342].reshape(-1, 3)
        return np.array([x[:, 0].mean() * 100 / 255, x[:, 1].mean() - 128, x[:, 2].mean() - 128])

    res = {}
    for s in settings:
        for grp in ("train", "ffhq"):
            sub = [r for r in rs if r["id"].startswith("q") == (grp == "ffhq")]
            lps, cj, sj, tsd, tde, idc = [], [], [], [], [], []
            for r in sub:
                ps = [out / s / f"{r['id']}_s{k}.png" for k in range(K)]
                if not all(p.exists() for p in ps):
                    continue
                ims = [Image.open(p).convert("RGB") for p in ps]
                inp = input_img(r, fid)
                if K > 1:
                    with torch.no_grad():
                        x = torch.stack([to_t(im) for im in ims]).to(dev)
                        x = F.interpolate(x, size=256, mode="bilinear", antialias=True)
                        lps.append(float(np.mean([lp(x[i:i + 1], x[j:j + 1]).item() for i, j in itertools.combinations(range(K), 2)])))
                    fs = [face(im) for im in ims]
                    if all(f is not None for f in fs):
                        c = np.array([[f[0] + f[2] / 2, f[1] + f[3] / 2] for f in fs]) / ims[0].width
                        cj.append(float(np.linalg.norm(c.std(0))))
                        sj.append(float(np.std([np.log(f[2]) for f in fs])))
                tc = np.array([torso(im) for im in ims])
                tsd.append(float(np.linalg.norm(tc.std(0))))
                ti = torso(inp)
                tde.append(float(np.mean([np.linalg.norm(t - ti) for t in tc])))
                fi = fid.feat(inp)
                if fi is not None:
                    for im in ims:
                        ft = fid.feat(im)
                        if ft is not None:
                            idc.append(float(fid.rec.match(fi, ft, cv2.FaceRecognizerSF_FR_COSINE)))
            if not tde:
                continue
            m = lambda v, d=4: round(float(np.mean(v)), d) if v else None  # noqa: E731
            res[f"{s}/{grp}"] = {"photos": len(tde), "seed_lpips": m(lps), "frame_centre_jit": m(cj), "frame_logsize_jit": m(sj),
                                 "shirt_seed_std_dE": m(tsd, 2), "shirt_vs_photo_dE": m(tde, 2), "id_cos": m(idc)}
            print(f"{s}/{grp}", res[f"{s}/{grp}"], flush=True)
    sfx = f"_{a.tag}" if a.tag else ""
    (ROOT / f"results/teacher_consistency{sfx}.json").write_text(json.dumps(res, indent=1))
    # sheet: up to 12 photos x (photo + settings x seeds)
    from PIL import ImageDraw
    pick = rs if len(rs) <= 12 else rs[:: len(rs) // 12][:12]
    c = 200
    g = Image.new("RGB", (c * (1 + len(settings) * K), c * len(pick) + 24), "white")
    d = ImageDraw.Draw(g)
    d.text((4, 4), "photo (teacher input)", fill="black")
    for j, s in enumerate(settings):
        d.text((c * (1 + j * K) + 4, 4), f"{s}: seeds 0..{K - 1}", fill="black")
    for i, r in enumerate(pick):
        g.paste(input_img(r, fid).resize((c, c)), (0, 24 + i * c))
        for j, s in enumerate(settings):
            for k in range(K):
                p = out / s / f"{r['id']}_s{k}.png"
                if p.exists():
                    g.paste(Image.open(p).convert("RGB").resize((c, c)), (c * (1 + j * K + k), 24 + i * c))
    g.save(ROOT / f"grids/teacher_consistency{sfx}_sheet.jpg", quality=90)
    print("[cons] SCORE_DONE", flush=True)


if __name__ == "__main__":
    a = args()
    if a.mode == "caption":
        import metrics as M
        captions(a, rows(a), M.FaceID())
        print("[cons] CAPTION_DONE", flush=True)
    else:
        gen(a) if a.mode == "gen" else score(a)
