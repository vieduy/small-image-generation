"""Two-stage idea, step 1 (2026-11-21, user: "from avatar with the style, gen sticker from this instead of real image?").
Zero-training check with the production model (dl6g2 + decg2, cartoon3d), on the 24 synthetic expr_eval faces:
  A. photo  -> 20 expression stickers               (today's flow)
  B. photo  -> avatar (same model, avatar option) -> the avatar image re-encoded -> 20 expression stickers (two-stage)
Same code path, decoder, faces and CLIP metric as scripts/eval_e1.py (imported). Metrics per condition:
  * expression accuracy all / face-only / outside (+ per edit), CLIP-L 20-way;
  * likeness to the PHOTO: SFace cosine(photo, sticker), mean over detected faces;
  * set consistency: mean pairwise SFace cosine between the 20 stickers of the same person (higher = same character);
  * cos(avatar, sticker): how close each sticker stays to the avatar.
  source env.sh; CUDA_VISIBLE_DEVICES=2 $VENV/bin/python scripts/avatar2sticker_test.py
Out: results/avatar2sticker_test.json, outputs/avatar2sticker/{photo,avatar}/..., grids/avatar2sticker_test.jpg
"""
import itertools
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
from eval_e1 import FACE_ONLY, OUTSIDE, boot_ci, expr_faces, to_t  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEV = "cuda"


def main():
    t0 = time.time()
    EX = yaml.safe_load((ROOT / "configs/sticker_expressions.yaml").read_text())
    ids = [e["id"] for e in EX]
    fid = M.FaceID()
    xf = expr_faces(fid)
    w = ROOT / "weights/student/dl6g2"
    unet = DLA.load_unet(w / "unet").to(DEV, torch.float16).eval()
    te = torch.load(w / "task_emb.pt")
    tasks, emb, mask = te["tasks"], te["emb"].to(DEV, torch.float16), te["mask"].to(DEV)
    venc = DLA.load_vae(torch.float16).to(DEV).eval()
    vdec = DLA.load_vae(torch.float16, ROOT / "weights/decoder/decg2").to(DEV).eval()
    od = ROOT / "outputs/avatar2sticker"

    def run(imgs, k):
        with torch.inference_mode():
            x = torch.stack([to_t(i) for i in imgs]).to(DEV, torch.float16)
            kk = torch.tensor(k, device=DEV)
            y = DLA.decode(vdec, DLA.forward(unet, DLA.encode(venc, x), emb[kk], mask[kk]))
        return [Image.fromarray(((v.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()) for v in y]

    ek = [tasks.index(f"expr_{e}") for e in ids]
    avatars = {}
    for cond in ("photo", "avatar"):
        (od / cond).mkdir(parents=True, exist_ok=True)
    for p in xf:
        photo = Image.open(p).convert("RGB")
        av = run([photo], [tasks.index("avatar")])[0]
        av.save(od / "avatar" / f"{p.stem}__avatar.png")
        avatars[p.stem] = av
        for cond, src in (("photo", photo), ("avatar", av)):
            for e, s in zip(ids, run([src] * len(ids), ek)):
                s.save(od / cond / f"{p.stem}_{e}.png")
    del unet
    torch.cuda.empty_cache()

    from transformers import CLIPModel, CLIPProcessor
    cp = M._snap("openai/clip-vit-large-patch14")
    cm, pr = CLIPModel.from_pretrained(str(cp), dtype=torch.float16).to(DEV).eval(), CLIPProcessor.from_pretrained(str(cp))
    with torch.no_grad():
        ti = pr(text=[f"a cartoon sticker of a person, {e['phrase']}" for e in EX], return_tensors="pt", padding=True).to(DEV)
        tf = cm.text_projection(cm.text_model(**ti).pooler_output).float()
        tf = tf / tf.norm(dim=-1, keepdim=True)
    fi, oi = [ids.index(e) for e in FACE_ONLY], [ids.index(e) for e in OUTSIDE]
    photo_f = {p.stem: fid.feat(Image.open(p)) for p in xf}
    av_f = {k: fid.feat(v) for k, v in avatars.items()}
    av_like = [fid.cos(photo_f[k], av_f[k]) for k in av_f if photo_f[k] is not None and av_f[k] is not None]
    res = {"model": "dl6g2 + decg2 (cartoon3d), no training", "n_faces": len(xf),
           "avatar_likeness_to_photo": round(float(np.mean(av_like)), 4), "conditions": {}}
    for cond in ("photo", "avatar"):
        hit = np.zeros((len(xf), len(ids)))
        like, cons, to_av = [], [], []
        for i, p in enumerate(xf):
            ims = [Image.open(od / cond / f"{p.stem}_{e}.png").convert("RGB") for e in ids]
            with torch.inference_mode():
                pv = pr.image_processor(images=ims, return_tensors="pt")["pixel_values"].to(DEV, torch.float16)
                f = cm.visual_projection(cm.vision_model(pixel_values=pv).pooler_output).float()
                f = f / f.norm(dim=-1, keepdim=True)
                hit[i] = ((f @ tf.T).argmax(1) == torch.arange(len(ids), device=DEV)).cpu().numpy()
            feats = [fid.feat(im) for im in ims]
            ok = [x for x in feats if x is not None]
            like += [fid.cos(photo_f[p.stem], x) for x in ok if photo_f[p.stem] is not None]
            to_av += [fid.cos(av_f[p.stem], x) for x in ok if av_f[p.stem] is not None]
            cons += [float(np.mean([fid.cos(a, b) for a, b in itertools.combinations(ok, 2)]))] if len(ok) > 1 else []
        rng = np.random.default_rng(0)
        res["conditions"][cond] = {
            "expr_acc": {"all": round(float(hit.mean()), 4), "face_only": round(float(hit[:, fi].mean()), 4),
                         "outside": round(float(hit[:, oi].mean()), 4)},
            "expr_acc_ci95": {"all": boot_ci(hit, rng), "outside": boot_ci(hit[:, oi], rng)},
            "per_edit": {e: round(float(hit[:, j].mean()), 3) for j, e in enumerate(ids)},
            "sticker_likeness_to_photo": round(float(np.mean(like)), 4),
            "set_consistency_pairwise": round(float(np.mean(cons)), 4),
            "sticker_to_avatar": round(float(np.mean(to_av)), 4),
            "face_detect_rate": round(len(like) / (len(xf) * len(ids)), 3)}
        print(cond, json.dumps(res["conditions"][cond]["expr_acc"]), {k: v for k, v in res["conditions"][cond].items()
                                                                        if k in ("sticker_likeness_to_photo", "set_consistency_pairwise", "sticker_to_avatar")}, flush=True)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    (ROOT / "results/avatar2sticker_test.json").write_text(json.dumps(res, indent=1))

    show = ["happy", "cry", "love", "angry", "think", "thumbs", "wave", "shrug", "party", "thanks"]
    faces = [xf[i] for i in (2, 9, 15, 21)]
    c, L = 140, 150
    g = Image.new("RGB", (L + (2 + len(show)) * c, 22 + len(faces) * (2 * c + 10)), "white")
    d = ImageDraw.Draw(g)
    for j, t in enumerate(["photo / avatar", ""] + show):
        d.text((L + j * c + 4, 5), t, fill="black")
    y = 22
    for p in faces:
        for cond, lab in (("photo", "A. from photo\n(today)"), ("avatar", "B. from avatar\n(two-stage)")):
            d.text((6, y + c // 2 - 10), lab, fill="black")
            g.paste((Image.open(p) if cond == "photo" else avatars[p.stem]).convert("RGB").resize((c, c)), (L, y))
            for j, e in enumerate(show):
                g.paste(Image.open(od / cond / f"{p.stem}_{e}.png").convert("RGB").resize((c, c)), (L + (j + 2) * c, y))
            y += c
        y += 10
    g.save(ROOT / "grids/avatar2sticker_test.jpg", quality=86)
    print(f"done {res['minutes']} min", flush=True)


if __name__ == "__main__":
    main()
