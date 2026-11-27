"""Stage 2 (2026-11-23): evaluate the multi-step DreamLite teacher with the SAME metrics/faces as eval_e1.py (imported from it), so
numbers compare directly with dl6g2 (results/e1/dl6g2.json) and the other students.
  --sweep : small CFG grid g_img x g_txt on TUNE faces (8 expression faces x 20 edits + 24 avatars, all from data/faces, disjoint
            from eval_e1's 24 expression faces where possible); pick = best expr-acc + likeness (sum), report all.
  default : full eval at --g-img/--g-txt (or the sweep's pick): expression acc all / face / outside (24 faces, bootstrap CI),
            per edit, avatar likeness (400 v2 faces), skin dL* on avatars; expression outputs are also written to
            outputs/expr_eval_<name>/cartoon3d/ so scripts/sharp_ratio.py can score sharpness vs klein.
Decoder: DreamLite's original TinyVAE (the teacher is trained on clean TinyVAE latents of klein images).
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/teacher_eval.py --tag tB1 [--ckpt step_02500] [--sweep]
Out: results/teacher/<name>.json (+ <name>_sweep.json), grids/teacher/<name>_expr.jpg
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402
import metrics as M  # noqa: E402
import teacher_dl as TD  # noqa: E402
from eval_e1 import FACE_ONLY, OUTSIDE, avatar_faces, boot_ci, expr_faces, to_t  # noqa: E402

ROOT, DEV = TD.ROOT, "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", default="final")
    ap.add_argument("--g-img", type=float, default=None)
    ap.add_argument("--g-txt", type=float, default=None)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    unet, T = TD.load_teacher(a.tag, a.ckpt)
    unet = unet.to(DEV).eval()
    tasks = T["tasks"]
    emb, mask = T["emb"].to(DEV, torch.float16), T["mask"].to(DEV)
    ne, nm = T["null_emb"].to(DEV, torch.float16), T["null_mask"].to(DEV)
    vae = DLA.load_vae(torch.float16).to(DEV).eval()
    EX = yaml.safe_load((ROOT / "configs/sticker_expressions.yaml").read_text())
    ids = [e["id"] for e in EX]
    fid = M.FaceID()
    from transformers import CLIPModel, CLIPProcessor
    cp = M._snap("openai/clip-vit-large-patch14")
    cm, pr = CLIPModel.from_pretrained(str(cp), dtype=torch.float16).to(DEV).eval(), CLIPProcessor.from_pretrained(str(cp))
    with torch.no_grad():
        ti = pr(text=[f"a cartoon sticker of a person, {e['phrase']}" for e in EX], return_tensors="pt", padding=True).to(DEV)
        tf = cm.text_projection(cm.text_model(**ti).pooler_output).float(); tf = tf / tf.norm(dim=-1, keepdim=True)

    def gen(paths, ks, gi, gt):
        outs = []
        with torch.inference_mode():
            for i in range(0, len(paths), a.bs):
                x = torch.stack([to_t(Image.open(p)) for p in paths[i:i + a.bs]]).to(DEV, torch.float16)
                k = torch.tensor(ks[i:i + a.bs], device=DEV)
                z = TD.sample(unet, DLA.encode(vae, x), emb[k], mask[k], ne, nm, gi, gt, steps=a.steps,
                              generator=torch.Generator().manual_seed(1000 + i))
                y = DLA.decode(vae, z.to(torch.float16))
                outs += [Image.fromarray(((yy.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()) for yy in y]
        return outs

    def score_expr(faces, imgs):
        hit = np.zeros((len(faces), len(ids)))
        with torch.inference_mode():
            for i in range(len(faces)):
                pv = pr.image_processor(images=imgs[i * len(ids):(i + 1) * len(ids)], return_tensors="pt")["pixel_values"].to(DEV, torch.float16)
                f = cm.visual_projection(cm.vision_model(pixel_values=pv).pooler_output).float(); f = f / f.norm(dim=-1, keepdim=True)
                hit[i] = ((f @ tf.T).argmax(1) == torch.arange(len(ids), device=DEV)).cpu().numpy()
        return hit

    def score_avatar(faces, imgs):
        cos, dl = [], []
        for p, o in zip(faces, imgs):
            im = Image.open(p)
            fa, fo = fid.feat(im), fid.feat(o)
            if fa is not None and fo is not None:
                cos.append(fid.cos(fa, fo))
            la, lo = fid.skin_L(im), fid.skin_L(o)
            if la is not None and lo is not None:
                dl.append(lo - la)
        return cos, dl

    xf_eval = expr_faces(fid)
    name = f"{a.tag}_{a.ckpt}"
    (ROOT / "results/teacher").mkdir(parents=True, exist_ok=True)
    if a.sweep:
        allf = sorted((ROOT / "data/faces").glob("*.png"))[:100]
        tune_x = [p for p in allf if p not in set(xf_eval)][::9][:8]
        tune_a = sorted((ROOT / "data/faces").glob("*.png"))[100:120]  # (fix 2026-11-23: faces_eval2 has only 300 -> [300:324] was empty); not in any eval set
        res = {}
        for gi in (1.5, 2.0, 3.0):
            for gt in (1.5, 2.5, 4.0):
                ox = gen([p for p in tune_x for _ in ids], [tasks.index(f"expr_{e}") for _ in tune_x for e in ids], gi, gt)
                oa = gen(tune_a, [tasks.index("avatar")] * len(tune_a), gi, gt)
                hit = score_expr(tune_x, ox)
                cos, dl = score_avatar(tune_a, oa)
                r = {"expr_all": round(float(hit.mean()), 4), "outside": round(float(hit[:, [ids.index(e) for e in OUTSIDE]].mean()), 4),
                     "likeness": round(float(np.mean(cos)), 4) if cos else None, "n_face": len(cos), "skin_dL": round(float(np.median(dl)), 1) if dl else None}
                res[f"img{gi}_txt{gt}"] = r
                print("[sweep]", gi, gt, r, flush=True)
        ok = {k: v for k, v in res.items() if v["likeness"] is not None and v["n_face"] >= 0.8 * len(tune_a)}
        pick = max(ok, key=lambda k: ok[k]["expr_all"] + ok[k]["likeness"])
        (ROOT / f"results/teacher/{name}_sweep.json").write_text(json.dumps({"grid": res, "pick": pick,
                                                                             "rule": "max expr_all + likeness, >=80% faces detected"}, indent=1))
        print("[sweep] pick", pick, res[pick], flush=True)
        a.g_img, a.g_txt = float(pick.split("_")[0][3:]), float(pick.split("_")[1][3:])
        name = f"{name}_{pick}"
    gi = a.g_img if a.g_img is not None else T.get("g_img", 2.0)
    gt = a.g_txt if a.g_txt is not None else T.get("g_txt", 2.0)
    if not a.sweep and a.g_img is not None:  # 2026-11-25: don't overwrite <tag>_<ckpt>.json with an explicit CFG / step setting
        name = f"{name}_img{gi}_txt{gt}_s{a.steps}"
    af = avatar_faces()
    ox = gen([p for p in xf_eval for _ in ids], [tasks.index(f"expr_{e}") for _ in xf_eval for e in ids], gi, gt)
    od = ROOT / f"outputs/expr_eval_teacher_{name}/cartoon3d"
    od.mkdir(parents=True, exist_ok=True)
    for j, im in enumerate(ox):
        im.save(od / f"{xf_eval[j // len(ids)].stem}_{ids[j % len(ids)]}.png")
    oa = gen(af, [tasks.index("avatar")] * len(af), gi, gt)
    hit = score_expr(xf_eval, ox)
    cos, dl = score_avatar(af, oa)
    fi, oi = [ids.index(e) for e in FACE_ONLY], [ids.index(e) for e in OUTSIDE]
    rng = np.random.default_rng(0)
    res = {"tag": a.tag, "ckpt": a.ckpt, "g_img": gi, "g_txt": gt, "steps": a.steps,
           "expr_acc": {"all": round(float(hit.mean()), 4), "face_only": round(float(hit[:, fi].mean()), 4), "outside": round(float(hit[:, oi].mean()), 4)},
           "expr_acc_ci95": {"all": boot_ci(hit, rng), "outside": boot_ci(hit[:, oi], rng)},
           "per_edit": {e: round(float(hit[:, j].mean()), 4) for j, e in enumerate(ids)},
           "avatar_likeness": {"id_cos_mean": round(float(np.mean(cos)), 4), "n": len(cos), "of": len(af)},
           "avatar_skin_dL_median": round(float(np.median(dl)), 1) if dl else None,
           "expr_outputs": str(od.relative_to(ROOT)), "metric_code": "eval_e1.py helpers", "minutes": round((time.time() - t0) / 60, 1)}
    (ROOT / f"results/teacher/{name}.json").write_text(json.dumps(res, indent=1))
    c, order, show = 112, FACE_ONLY + OUTSIDE, list(range(0, len(xf_eval), 3))[:8]
    g = Image.new("RGB", ((len(order) + 1) * c, 30 + len(show) * c), "white"); d = ImageDraw.Draw(g)
    d.text((4, 4), f"teacher {name} (img {gi}, txt {gt}, {a.steps} steps)", fill="black")
    for j, e in enumerate(order):
        d.text(((j + 1) * c + 3, 4), e, fill="black" if e in FACE_ONLY else "blue")
    for r_, i in enumerate(show):
        g.paste(Image.open(xf_eval[i]).convert("RGB").resize((c, c)), (0, 30 + r_ * c))
        for j, e in enumerate(order):
            g.paste(ox[i * len(ids) + ids.index(e)].resize((c, c)), ((j + 1) * c, 30 + r_ * c))
    (ROOT / "grids/teacher").mkdir(parents=True, exist_ok=True)
    g.save(ROOT / f"grids/teacher/{name}_expr.jpg", quality=85)
    print(f"[teacher-eval] {name}: {res['expr_acc']} likeness {res['avatar_likeness']['id_cos_mean']} skin dL {res['avatar_skin_dL_median']} "
          f"({res['minutes']} min)", flush=True)


if __name__ == "__main__":
    main()
