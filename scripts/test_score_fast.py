"""Equivalence + speed test: slow per-image scoring vs score_fast.precompute on 120 existing cartoon3d rows
(40 face avatars, 40 face stickers, 40 subject stickers). Uses a throw-away FaceCache (no file written)."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
import metrics as M, teacher_prompt as TP, score_fast as SF  # noqa: E402


def main():
    ROOT = M.ROOT; out = ROOT / "data/pairs/cartoon3d"
    rows = [json.loads(l) for l in open(out / "manifest_s5d.jsonl")]
    sel = [r for r in rows if r["is_face"] and r["task"] == "avatar"][:40] + [r for r in rows if r["is_face"] and r["task"] == "sticker"][:40] + \
          [r for r in rows if not r["is_face"]][:40]
    emb, mat, sc = M.Embedder(), M.Matter(), M.SubjectCheck()
    refs = [Image.open(p) for p in sorted((ROOT / "styles/cartoon3d").glob("ref_*.png"))]
    rd, _ = emb(refs); rdm = torch.nn.functional.normalize(rd.mean(0), dim=0)
    fid = M.FaceID()
    t0 = time.time(); slow = {}
    for r in sel:
        o = Image.open(out / r["task"] / f"{r['id']}.png").convert("RGB")
        d, _ = emb([o]); x = {"style_dino": round(float(d[0] @ rdm), 4)}
        if r["task"] != "avatar":
            x |= M.sticker_postprocess(o, mat(o))[1]
        if not r["is_face"]:
            x |= sc(o, TP.parse_subject(r["prompt"]))
        if r["is_face"]:
            x["feat"] = fid.feat(o); x["L"] = fid.skin_L(o)
        slow[(r["id"], r["task"])] = x
    ts = time.time() - t0
    fc = M.FaceCache.__new__(M.FaceCache); fc.fid, fc.root, fc.d, fc.hits, fc.miss = fid, out, {}, 0, 0
    t0 = time.time()
    pre = SF.precompute(out, [(r["id"], r["task"], r["is_face"], r["prompt"]) for r in sel], fc, emb, mat, sc, rdm, rd, subj_name=TP.parse_subject, workers=16)
    tf = time.time() - t0
    dd, pm, kb, hf, co, fl = [], 0, [], 0, [], []
    for r in sel:
        k = (r["id"], r["task"]); a, b = slow[k], pre[k]
        dd.append(abs(a["style_dino"] - b["style_dino"]))
        if "pass_sticker" in a:
            pm += a["pass_sticker"] != b["pass_sticker"]; kb.append(abs((a["webp_kb"] or 0) - (b["webp_kb"] or 0)))
        if "human_face" in a:
            hf += (a["human_face"] != b["human_face"]) + (a.get("clip_subject_ok") != b.get("clip_subject_ok"))
        if r["is_face"]:
            p = out / r["task"] / f"{r['id']}.png"; st = p.stat()
            f, L = fc.d[(f"{r['task']}/{r['id']}.png", st.st_mtime_ns, st.st_size)]
            if a["feat"] is not None and f is not None:
                co.append(float(np.dot(a["feat"].ravel(), f.ravel()) / np.linalg.norm(a["feat"]) / np.linalg.norm(f)))
            if a["L"] is not None and L is not None:
                fl.append(abs(a["L"] - L))
    print(json.dumps({"rows": len(sel), "slow_s": round(ts, 1), "fast_s": round(tf, 1), "speedup": round(ts / tf, 1),
                      "style_dino_maxdiff": round(max(dd), 5), "sticker_pass_mismatch": pm, "webp_kb_maxdiff": round(max(kb), 1),
                      "subject_check_mismatch": hf, "face_feat_cos_min": round(min(co), 5) if co else None,
                      "skin_L_maxdiff": round(max(fl), 3) if fl else None}))


if __name__ == "__main__":
    main()
