"""Quality evaluation: (optionally) generate outputs with a model adapter, then score them with scripts/metrics.py.

Eval set (held out from any training): data/faces first 100 (avatar on all 100, sticker on the first 50) and
data/subjects (30, sticker only), for every style in styles/ (or --styles).
Layout: outputs/<tag>/<style>/<task>/<input_stem>.png  (+ .webp sticker after post-processing)

  # generate + score klein (photo-only, style as text):
  MODEL=klein KLEIN_REF=0 PROMPTS_FAMILY=klein_text $VENV/bin/python scripts/eval_quality.py --tag klein_text --generate
  # score outputs written by another program (e.g. the student):
  $VENV/bin/python scripts/eval_quality.py --tag student_v1 --styles cartoon3d

Writes results/quality_<tag>.csv (per image), appends results/quality_summary.jsonl (per style x task),
and grids/quality_<tag>_<style>.jpg.
"""
import argparse
import csv
import json
import os
import zlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import metrics as M  # noqa: E402
import teacher_prompt as TP  # noqa: E402

ROOT = M.ROOT


EVAL_SET = os.environ.get("EVAL_SET", "v1")


def eval_items():
    """v1: 100 faces (avatar) + 50 faces & 30 subjects (sticker).
    v2: v1 + data/faces_eval2 (300 more faces from the same attribute generator, seeds 400000+): 400 avatar,
        200 face stickers (+30 subjects). Identity retrieval always ranks among a gallery of 100 (4 groups), so v1 and v2
        numbers are on the same scale; v2 has 4x the samples (95% CI ~ +-4.5 pts on top-5 instead of +-9)."""
    faces = sorted((ROOT / "data/faces").glob("*.png"))[:100]
    subj = sorted((ROOT / "data/subjects").glob("*.png"))
    sf = faces[:50]
    if EVAL_SET == "v2":
        e2 = sorted((ROOT / "data/faces_eval2").glob("*.png"))[:300]
        faces, sf = faces + e2, sf + e2[:150]
    elif EVAL_SET == "real":  # real photos (legal-approved, identity-disjoint from training), framing-normalised
        faces = sorted((ROOT / "data/real_p2a/eval").glob("*.png"))[:300]
        sf = faces[:150]
    elif EVAL_SET == "selfie":  # realistic close-up selfies, framing-normalised like user photos (gen_eval_selfie.py)
        faces = sorted((ROOT / "data/faces_selfie").glob("*.png"))[:200]
        sf = faces[:100]
    return [(p, "avatar") for p in faces] + [(p, "sticker") for p in sf] + [(p, "sticker") for p in subj]


def is_face_dir(p):
    return p.parent.name.startswith("faces") or p.parent.parent.name == "real_p2a"


def generate(tag, styles, items, bs):
    import i2i_server as S  # MODEL / PROMPTS_FAMILY / KLEIN_REF env select adapter + prompts
    st_all = S.load_styles()
    ad = S.ADAPTERS[S.MODEL]()
    ad.load()
    ad.p.set_progress_bar_config(disable=True)
    if isinstance(ad, S.SanaSprint):
        for st in st_all.values():
            for t in ("avatar", "sticker"):
                ad.embed(st[f"{t}_text"])
        ad.free_text_encoder()
    bs = bs if ad.batch_ok else 1
    global MAN, FID
    if os.environ.get("PERSONAL") == "1":
        MAN = {json.loads(l)["file"]: json.loads(l)["prompt"] for d in ("faces", "faces_eval2", "faces_selfie", "subjects", "real_p2a/eval")
               if (ROOT / f"data/{d}/manifest.jsonl").exists() for l in open(ROOT / f"data/{d}/manifest.jsonl")}
        FID = M.FaceID()
    global ACHK
    ACHK = M.AttrCheck() if os.environ.get("VERIFY_EXTRA") == "1" else None
    t0, n = time.time(), 0
    for sname in styles:
        st = st_all[sname]
        for task in ("avatar", "sticker"):
            todo = [p for p, t in items if t == task and not (ROOT / f"outputs/{tag}/{sname}/{task}/{p.stem}.png").exists()]
            (ROOT / f"outputs/{tag}/{sname}/{task}").mkdir(parents=True, exist_ok=True)
            for i in range(0, len(todo), bs):
                chunk = todo[i:i + bs]
                imgs = [Image.open(p).convert("RGB").resize((S.SIZE, S.SIZE)) for p in chunk]
                seeds = [zlib.crc32(f"{p.stem}|{sname}|{task}".encode()) for p in chunk]
                if os.environ.get("PERSONAL") == "1":  # per-input teacher prompt (bs forced to 1 for klein anyway)
                    outs = []
                    for p_, im_, sd_ in zip(chunk, imgs, seeds):
                        at_ = TP.parse(MAN[p_.name]) if is_face_dir(p_) else None
                        if at_ is not None and ACHK is not None and ACHK(im_, at_["gender"], at_["extra"]) is False:
                            at_ = dict(at_, extra=None)  # VERIFY_EXTRA=1: never describe an attribute the photo does not show
                        pdesc = TP.person_desc(at_, FID.skin_L(im_)) if is_face_dir(p_) else None
                        if p_.parent.name == "subjects":  # v2: subject-safe style text + named subject (deviation 8)
                            pr = TP.subject_sticker_prompt(st.get("subject_description", st["description"]), TP.parse_subject(MAN[p_.name]))
                        else:
                            pr = (TP.avatar_prompt if task == "avatar" else TP.sticker_prompt)(st["description"], pdesc)
                        with torch.inference_mode():
                            outs += ad.run([im_], pr, st, [sd_])
                else:
                    with torch.inference_mode():
                        outs = ad.run(imgs, st[f"{task}_text"], st, seeds)
                for p, o in zip(chunk, outs):
                    o.save(ROOT / f"outputs/{tag}/{sname}/{task}/{p.stem}.png")
                n += len(chunk)
        print(f"[gen] {tag} {sname} done ({n} imgs, {time.time() - t0:.0f}s)", flush=True)
    del ad
    torch.cuda.empty_cache()


def checker(size=512, k=32):
    a = (np.indices((size, size)).sum(0) // k % 2) * 40 + 200
    return Image.fromarray(np.dstack([a, a, a]).astype(np.uint8)).convert("RGBA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--styles", nargs="+", default=None)
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    styles = a.styles or sorted(p.parent.name for p in (ROOT / "styles").glob("*/style.yaml"))
    items = eval_items()
    if a.generate:
        generate(a.tag, styles, items, a.bs)

    fid, emb, mat, sck = M.FaceID(), M.Embedder(), M.Matter(), M.SubjectCheck()
    SMAN = {json.loads(l)["file"]: json.loads(l)["prompt"] for l in open(ROOT / "data/subjects/manifest.jsonl")}
    in_feat = {p.stem: fid.feat(Image.open(p)) for p, _ in items}
    # calibration: impostor = two different synthetic faces; photo_style = how "stylized" an unedited photo scores
    stems = [p.stem for p, t in items if t == "avatar"]
    imp = [fid.cos(in_feat[stems[i]], in_feat[stems[(i + 7) % len(stems)]]) for i in range(len(stems))]
    imp = [x for x in imp if x is not None]
    rows, summ, out_feat = [], [], {}
    # galleries of 100 consecutive eval inputs (v1: one gallery = all 100; v2: 4 galleries)
    group = {st: i // 100 for i, st in enumerate(stems)}
    gal = {}
    for st in stems:
        if in_feat[st] is not None:
            gal.setdefault(group[st], []).append(st)
    G = {g: np.stack([in_feat[st].ravel() / np.linalg.norm(in_feat[st]) for st in v]) for g, v in gal.items()}
    gal_stems = [st for v in gal.values() for st in v]

    def retrieval(keys):
        """Identity by retrieval (robust to the photo->cartoon domain gap): rank of the true input among all 100 inputs.
        Undetected faces count as misses."""
        ranks = []
        for (sn, tk, stem) in keys:
            f = out_feat.get((sn, tk, stem))
            if f is None or stem not in gal_stems:
                ranks.append(None)
                continue
            g = group[stem]
            sims = G[g] @ (f.ravel() / np.linalg.norm(f))
            ranks.append(int((sims > sims[gal[g].index(stem)]).sum()) + 1)
        n = len(ranks)
        ok = [r for r in ranks if r is not None]
        return {"id_top1": round(sum(r == 1 for r in ok) / n, 3) if n else None,
                "id_top5": round(sum(r <= 5 for r in ok) / n, 3) if n else None,
                "id_mean_rank": round(float(np.mean(ok)), 1) if ok else None, "id_chance_top1": round(1 / max(len(v) for v in gal.values()), 3),
                "id_top5_ci95": round(1.96 * (max(1e-9, np.mean([r is not None and r <= 5 for r in ranks]) * (1 - np.mean([r is not None and r <= 5 for r in ranks]))) / n) ** 0.5, 3) if n else None}
    for sname in styles:
        refs = [Image.open(p) for p in sorted((ROOT / "styles" / sname).glob("ref_*.png"))]
        rd, rc = emb(refs)
        rdm, rcm = torch.nn.functional.normalize(rd.mean(0), dim=0), torch.nn.functional.normalize(rc.mean(0), dim=0)
        photo_d, photo_c = emb([Image.open(p) for p, t in items if t == "avatar"][:32])
        base = {"photo_style_dino": round(float((photo_d @ rdm).mean()), 4), "photo_style_clip": round(float((photo_c @ rcm).mean()), 4)}
        for task in ("avatar", "sticker"):
            its = [p for p, t in items if t == task and (ROOT / f"outputs/{a.tag}/{sname}/{task}/{p.stem}.png").exists()]
            if not its:
                continue
            outs = [Image.open(ROOT / f"outputs/{a.tag}/{sname}/{task}/{p.stem}.png").convert("RGB") for p in its]
            d_all, c_all = [], []
            for i in range(0, len(outs), 32):
                d, c = emb(outs[i:i + 32])
                d_all.append(d)
                c_all.append(c)
            d_all, c_all = torch.cat(d_all), torch.cat(c_all)
            sd, sc = (d_all @ rdm).cpu().numpy(), (c_all @ rcm).cpu().numpy()
            cp = (d_all @ rd.T).max(1).values.cpu().numpy()
            for k, (p, o) in enumerate(zip(its, outs)):
                is_face = is_face_dir(p)
                r = {"tag": a.tag, "style": sname, "task": task, "input": p.stem, "is_face": is_face,
                     "style_dino": round(float(sd[k]), 4), "style_clip": round(float(sc[k]), 4), "refcopy_dino": round(float(cp[k]), 4)}
                if is_face:
                    fo = fid.feat(o)
                    out_feat[(sname, task, p.stem)] = fo
                    r["face_detected"] = fo is not None
                    cs = fid.cos(in_feat[p.stem], fo)
                    r["id_cos"] = None if cs is None else round(cs, 4)
                    r["id_pass"] = cs is not None and cs >= M.SFACE_THR
                if not is_face:
                    r.update(sck(o, TP.parse_subject(SMAN[p.name])))
                if task == "sticker":
                    data, info = M.sticker_postprocess(o, mat(o))
                    r.update(info)
                    if data:
                        (ROOT / f"outputs/{a.tag}/{sname}/{task}/{p.stem}.webp").write_bytes(data)
                rows.append(r)
            rr = [r for r in rows if r["style"] == sname and r["task"] == task]
            fr = [r for r in rr if r["is_face"]]
            ids = [r["id_cos"] for r in fr if r["id_cos"] is not None]
            s = {"tag": a.tag, "eval_set": EVAL_SET, "style": sname, "task": task, "n": len(rr), "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "face_det_rate": round(np.mean([r["face_detected"] for r in fr]), 3) if fr else None,
                 "id_cos_mean": round(float(np.mean(ids)), 4) if ids else None,
                 "id_pass_rate": round(np.mean([r["id_pass"] for r in fr]), 3) if fr else None,
                 "style_dino": round(float(np.mean([r["style_dino"] for r in rr])), 4),
                 "style_clip": round(float(np.mean([r["style_clip"] for r in rr])), 4),
                 "refcopy_dino": round(float(np.mean([r["refcopy_dino"] for r in rr])), 4),
                 "impostor_id_cos": round(float(np.mean(imp)), 4), **base,
                 **(retrieval([(sname, task, r["input"]) for r in fr]) if fr else {})}
            if task == "sticker":
                s |= {"sticker_pass_rate": round(np.mean([r["pass_sticker"] for r in rr]), 3),
                      "raw_bg_white": round(float(np.nanmean([r.get("raw_bg_white", np.nan) for r in rr])), 3),
                      "webp_kb_mean": round(float(np.nanmean([r["webp_kb"] or np.nan for r in rr])), 1),
                      "fails": {f: sum(r.get("fail") == f for r in rr) for f in ("no_subject", "fg_frac", "fragmented", "size")}}
                sr = [r for r in rr if not r["is_face"]]
                if sr:
                    # YuNet also fires on cartoon animal faces, so the humanization rate is only meaningful for objects
                    so = [r for r in sr if not TP.is_animal(TP.parse_subject(SMAN[r["input"] + ".png"]))]
                    s |= {"subj_clip_ok": round(np.mean([r.get("clip_subject_ok", False) for r in sr]), 3),
                          "subj_obj_human_face_rate": round(np.mean([r["human_face"] for r in so]), 3) if so else None,
                          "subj_n_obj": len(so), "subj_animal_face_det_rate": round(np.mean([r["human_face"] for r in sr if r not in so]), 3)}
            summ.append(s)
            print(json.dumps(s), flush=True)
        # grid: 8 inputs | avatar | sticker (on checkerboard)
        sel = [p for p, t in items if t == "sticker" and is_face_dir(p)][:5] + [p for p, t in items if t == "sticker" and p.parent.name == "subjects"][:3]
        c = 192
        g = Image.new("RGB", (3 * c, len(sel) * c + 18), "white")
        dr = ImageDraw.Draw(g)
        dr.text((4, 3), f"{a.tag} / {sname}: input | avatar | sticker(post-processed)", fill="black")
        for i, p in enumerate(sel):
            g.paste(Image.open(p).convert("RGB").resize((c, c)), (0, 18 + i * c))
            av = ROOT / f"outputs/{a.tag}/{sname}/avatar/{p.stem}.png"
            if av.exists():
                g.paste(Image.open(av).convert("RGB").resize((c, c)), (c, 18 + i * c))
            wb = ROOT / f"outputs/{a.tag}/{sname}/sticker/{p.stem}.webp"
            if wb.exists():
                ck = checker()
                st = Image.open(wb).convert("RGBA")
                g.paste(Image.alpha_composite(ck, st).convert("RGB").resize((c, c)), (2 * c, 18 + i * c))
        g.save(ROOT / f"grids/quality_{a.tag}_{sname}{'' if EVAL_SET == 'v1' else '_' + EVAL_SET}.jpg", quality=88)
    with open(ROOT / f"results/quality_{a.tag}{'' if EVAL_SET == 'v1' else '_' + EVAL_SET}.csv", "w", newline="") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(ROOT / "results/quality_summary.jsonl", "a") as f:
        for s in summ:
            f.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    main()
