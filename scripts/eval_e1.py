"""E1 evaluation (2026-11-20): ONE code path for every E1 run, new and existing (dl6g2, dlBd, dlB, frA2, frUp1, ...).

For each run (weights/student/<run>, phone path: DreamLite TinyVAE encoder -> one-step UNet (fp16) -> decoder --decoder):
  1. Expression stickers: 24 synthetic faces x 20 expressions. Faces = the expr_eval.py set (data/faces[:100] ordered by
     skin lightness, np.linspace picks 24). Accuracy = CLIP ViT-L/14 20-way zero-shot, text "a cartoon sticker of a person,
     {phrase}" (configs/sticker_expressions.yaml), correct iff the own phrase scores highest. Chance = 1/20 = 0.05.
     Reported: all (20 edits), FACE_ONLY (8), OUTSIDE (12), per edit; 95% bootstrap CI over faces (2000 resamples).
  2. Avatar likeness on synthetic faces: the v2 avatar set (data/faces[:100] + data/faces_eval2[:300] = 400 faces),
     OpenCV SFace cosine(photo, avatar) as metrics.FaceID (same as eval_quality.py id_cos); mean over faces where both are detected.
Outputs: outputs/e1/<run>/{expr,avatar}/*.png, grids/e1/<run>_expr.jpg (8 faces x 20 edits, face-only edits first),
         results/e1/<run>.json (numbers + run config + option-reading modules from reports/e1_plan.json + git/eval-code hashes).
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/eval_e1.py --runs dlB frA2 [--decoder decg2]
"""
import argparse
import hashlib
import json
import subprocess
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

ROOT = Path(__file__).resolve().parent.parent
DEV = "cuda"

# ---- the split (pre-registered 2026-11-20 ~07:35 UTC in README "paper P2"; defined HERE as the single source of truth) ----
# FACE_ONLY: the edit changes the face itself (mouth, eyes, tears, blush, eyewear on the face).
FACE_ONLY = ["happy", "lol", "cry", "wow", "sad", "shy", "wink", "cool"]          # cool = sunglasses: on the face
# OUTSIDE: the edit adds new structure outside the face: hands/gestures, objects or symbols around or above the head.
OUTSIDE = ["love",      # small red hearts floating around the head
           "angry",     # puff of steam above the head
           "think",     # hand on the chin
           "sleepy",    # 'Zzz' above the head
           "thumbs", "wave", "ok", "facepalm", "shrug", "thanks",   # hand gestures
           "party",     # party hat + confetti
           "night"]     # pillow + crescent moon
assert len(set(FACE_ONLY) | set(OUTSIDE)) == 20 and not set(FACE_ONLY) & set(OUTSIDE)


def git_hash():
    try:
        h = subprocess.run(["git", "-c", f"safe.directory={ROOT}", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
        d = subprocess.run(["git", "-c", f"safe.directory={ROOT}", "status", "--porcelain", "--", "scripts"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
        return h + ("+dirty" if d else "")
    except Exception:
        return "?"


def expr_faces(fid):
    faces = sorted((ROOT / "data/faces").glob("*.png"))[:100]
    L = {p: fid.skin_L(Image.open(p)) for p in faces}
    order = sorted([p for p in faces if L[p] is not None], key=lambda p: L[p])
    return [order[int(i)] for i in np.linspace(0, len(order) - 1, 24)]


def avatar_faces():
    return sorted((ROOT / "data/faces").glob("*.png"))[:100] + sorted((ROOT / "data/faces_eval2").glob("*.png"))[:300]


def to_t(img):
    return torch.from_numpy(np.array(img.convert("RGB").resize((512, 512)))).permute(2, 0, 1).float() / 127.5 - 1


def boot_ci(hit, rng, n=2000):
    """hit [faces, edits] 0/1 -> 95% CI of the mean, resampling faces."""
    F = hit.shape[0]
    m = [hit[rng.integers(0, F, F)].mean() for _ in range(n)]
    return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--decoder", default="decg2")
    a = ap.parse_args()
    EX = yaml.safe_load((ROOT / "configs/sticker_expressions.yaml").read_text())
    ids = [e["id"] for e in EX]
    assert sorted(ids) == sorted(FACE_ONLY + OUTSIDE), "edit list changed"
    fid = M.FaceID()
    xf, af = expr_faces(fid), avatar_faces()
    from transformers import CLIPModel, CLIPProcessor
    cp = M._snap("openai/clip-vit-large-patch14")
    cm, pr = CLIPModel.from_pretrained(str(cp), dtype=torch.float16).to(DEV).eval(), CLIPProcessor.from_pretrained(str(cp))
    with torch.no_grad():
        ti = pr(text=[f"a cartoon sticker of a person, {e['phrase']}" for e in EX], return_tensors="pt", padding=True).to(DEV)
        tf = cm.text_projection(cm.text_model(**ti).pooler_output).float()
        tf = tf / tf.norm(dim=-1, keepdim=True)
    plan = {}
    pj = ROOT / "reports/e1_plan.json"
    if pj.exists():
        plan = json.loads(pj.read_text()).get("runs", {})
    eval_code = hashlib.sha1((ROOT / "scripts/eval_e1.py").read_bytes()).hexdigest()[:10]
    venc = DLA.load_vae(torch.float16).to(DEV).eval()
    vdec = DLA.load_vae(torch.float16, ROOT / "weights/decoder" / a.decoder).to(DEV).eval()
    in_feat = {p.stem: fid.feat(Image.open(p)) for p in af}

    for run in a.runs:
        t0 = time.time()
        w = ROOT / "weights/student" / run
        cfg = json.loads((w / "config.json").read_text())
        assert cfg.get("arch") == "dreamlite" and cfg.get("lora_rank", 0) == 0, f"{run}: full DreamLite students only"
        unet = DLA.load_unet(w / "unet").to(DEV, torch.float16).eval()
        te = torch.load(w / "task_emb.pt")
        tasks, emb, mask = te["tasks"], te["emb"].to(DEV, torch.float16), te["mask"].to(DEV)
        od = ROOT / "outputs/e1" / run
        (od / "expr").mkdir(parents=True, exist_ok=True)
        (od / "avatar").mkdir(parents=True, exist_ok=True)

        def gen(paths, ks, sub, names):
            with torch.inference_mode():
                for i in range(0, len(paths), 16):
                    x = torch.stack([to_t(Image.open(p)) for p in paths[i:i + 16]]).to(DEV, torch.float16)
                    k = torch.tensor(ks[i:i + 16], device=DEV)
                    y = DLA.decode(vdec, DLA.forward(unet, DLA.encode(venc, x), emb[k], mask[k], sigmas=cfg.get("ms_sigmas")))
                    for yy, nm in zip(y, names[i:i + 16]):
                        Image.fromarray(((yy.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()).save(od / sub / f"{nm}.png")
        xp = [p for p in xf for _ in ids]
        gen(xp, [tasks.index(f"expr_{e}") for _ in xf for e in ids], "expr", [f"{p.stem}_{e}" for p in xf for e in ids])
        gen(af, [tasks.index("avatar")] * len(af), "avatar", [p.stem for p in af])
        del unet
        torch.cuda.empty_cache()

        hit = np.zeros((len(xf), len(ids)))
        with torch.inference_mode():
            for i, p in enumerate(xf):
                ims = [Image.open(od / "expr" / f"{p.stem}_{e}.png").convert("RGB") for e in ids]
                pv = pr.image_processor(images=ims, return_tensors="pt")["pixel_values"].to(DEV, torch.float16)
                f = cm.visual_projection(cm.vision_model(pixel_values=pv).pooler_output).float()
                f = f / f.norm(dim=-1, keepdim=True)
                hit[i] = ((f @ tf.T).argmax(1) == torch.arange(len(ids), device=DEV)).cpu().numpy()
        fi, oi = [ids.index(e) for e in FACE_ONLY], [ids.index(e) for e in OUTSIDE]
        rng = np.random.default_rng(0)
        cos = []
        for p in af:
            fo = fid.feat(Image.open(od / "avatar" / f"{p.stem}.png"))
            c = fid.cos(in_feat[p.stem], fo) if (fo is not None and in_feat[p.stem] is not None) else None
            if c is not None:
                cos.append(c)
        res = {"run": run, "weights": str(w.relative_to(ROOT)), "decoder": a.decoder, "eval_code_sha1": eval_code, "git": git_hash(),
               "config": {k: cfg.get(k) for k in ("init_from", "steps", "lr", "emb_lr", "bs", "seed", "manifest", "shared_trunk",
                                                  "shared_scope", "gan_w", "lp512_w", "eye_w", "eye_lpips_w", "proper_flip", "tgt_cache")},
               "option_modules": plan.get(run, {}).get("reads"), "n_option_inputs": plan.get(run, {}).get("n_reads"),
               "self_attn_after_entry": plan.get(run, {}).get("self_attn_after"), "entry": plan.get(run, {}).get("entry"),
               "expr_acc": {"all": round(float(hit.mean()), 4), "face_only": round(float(hit[:, fi].mean()), 4),
                            "outside": round(float(hit[:, oi].mean()), 4)},
               "expr_acc_ci95": {"all": boot_ci(hit, rng), "face_only": boot_ci(hit[:, fi], rng), "outside": boot_ci(hit[:, oi], rng)},
               "per_edit": {e: round(float(hit[:, j].mean()), 4) for j, e in enumerate(ids)},
               "avatar_likeness": {"id_cos_mean": round(float(np.mean(cos)), 4), "n": len(cos), "of": len(af),
                                   "set": "v2 synthetic: data/faces[:100] + data/faces_eval2[:300]"},
               "n_faces_expr": len(xf), "split": {"face_only": FACE_ONLY, "outside": OUTSIDE}, "chance": 0.05,
               "outputs": str(od.relative_to(ROOT)), "grid": f"grids/e1/{run}_expr.jpg", "minutes": round((time.time() - t0) / 60, 1)}
        (ROOT / "results/e1").mkdir(parents=True, exist_ok=True)
        (ROOT / f"results/e1/{run}.json").write_text(json.dumps(res, indent=1))
        # grid: 8 faces x 20 edits, face-only first, with a per-cell hit marker
        c = 112
        order = FACE_ONLY + OUTSIDE
        show = list(range(0, len(xf), 3))[:8]
        g = Image.new("RGB", ((len(order) + 1) * c, 30 + len(show) * c), "white")
        d = ImageDraw.Draw(g)
        d.text((4, 4), run, fill="black")
        for j, e in enumerate(order):
            d.text(((j + 1) * c + 3, 4), e, fill="black" if e in FACE_ONLY else "blue")
        for r_, i in enumerate(show):
            g.paste(Image.open(xf[i]).convert("RGB").resize((c, c)), (0, 30 + r_ * c))
            for j, e in enumerate(order):
                g.paste(Image.open(od / "expr" / f"{xf[i].stem}_{e}.png").convert("RGB").resize((c, c)), ((j + 1) * c, 30 + r_ * c))
                ok = hit[i, ids.index(e)] > 0
                d.rectangle([(j + 1) * c + 2, 32 + r_ * c, (j + 1) * c + 10, 40 + r_ * c], fill="green" if ok else "red")
        (ROOT / "grids/e1").mkdir(parents=True, exist_ok=True)
        g.save(ROOT / f"grids/e1/{run}_expr.jpg", quality=85)
        print(f"[e1] {run}: {res['expr_acc']} CI {res['expr_acc_ci95']} likeness {res['avatar_likeness']['id_cos_mean']} "
              f"({res['minutes']} min)", flush=True)


if __name__ == "__main__":
    main()
