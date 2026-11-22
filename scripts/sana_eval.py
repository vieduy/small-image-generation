"""Plan B3 eval (2026-11-20): Sana-Sprint students with the SAME metrics and faces as scripts/eval_e1.py (imported from it):
expression accuracy all / face-only / outside (CLIP-L 20-way, 24 synthetic faces, bootstrap CI), per edit, and avatar likeness
(SFace cosine) on the 400 v2 synthetic faces. Decoder: the tiny DC-AE decoder (phone). Photo encoder: --enc dcae (full DC-AE, as
in training) or tiny (tiny DC-AE encoder = the phone path). dl6g2 reference numbers come from eval_e1.py --runs dl6g2.
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/sana_eval.py --tag sD1 --tiny-dec tdec1r --tiny-enc tenc1r --enc tiny
Out: results/sana/<tag>_<enc>.json, outputs/sana/<tag>_<enc>/{expr,avatar}/, grids/sana/<tag>_<enc>_expr.jpg
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
import metrics as M  # noqa: E402
import sana_common as SC  # noqa: E402
from eval_e1 import FACE_ONLY, OUTSIDE, avatar_faces, boot_ci, expr_faces, to_t  # noqa: E402

ROOT, DEV = SC.ROOT, "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", default="final")
    ap.add_argument("--tiny-dec", default="tdec1r")
    ap.add_argument("--tiny-enc", default="tenc1r")
    ap.add_argument("--enc", default="tiny", choices=["tiny", "dcae"])
    ap.add_argument("--dec", default="tiny", choices=["tiny", "dcae"], help="dcae = Sana's full DC-AE decoder (server quality; "
                    "separates the tiny decoder's share of the quality gap)")
    a = ap.parse_args()
    t0 = time.time()
    w = ROOT / "weights/sana_student" / a.tag
    meta = json.loads((w / "sana_student.json").read_text()) if (w / "sana_student.json").exists() else {}
    cond = meta.get("cond") or ("concat" if "C" in a.tag else "diptych")
    tr = SC.load_student(w / a.ckpt / "transformer").to(DEV).eval()
    te = torch.load(w / a.ckpt / "task_emb.pt")
    tasks, emb, mask = te["tasks"], te["emb"].to(DEV, torch.float16), te["mask"].to(DEV)
    dc = None
    if a.dec == "tiny":
        tdec = SC.load_tiny(a.tiny_dec)[0].to(DEV, torch.float16).eval()
        decode = lambda z: tdec.decode(z).sample
    else:
        dc = SC.load_dcae().to(DEV).eval()
        decode = lambda z: dc.decode((z.float() / dc.config.scaling_factor).to(torch.bfloat16)).sample.float()
    if a.enc == "tiny":
        tenc = SC.load_tiny(a.tiny_enc)[0].to(DEV, torch.float16).eval()
        encode = lambda x: tenc.encode(x.half()).latents
    else:
        dc = dc or SC.load_dcae().to(DEV).eval()
        encode = lambda x: (dc.encode(x.to(torch.bfloat16)).latent.float() * dc.config.scaling_factor).half()
    noise = SC.fixed_noise(DEV, torch.float16)
    name = f"{a.tag}_{a.enc}" + ("" if a.dec == "tiny" else f"_dec{a.dec}") + ("" if a.ckpt == "final" else f"_{a.ckpt}")
    od = ROOT / "outputs/sana" / name
    for s in ("expr", "avatar"):
        (od / s).mkdir(parents=True, exist_ok=True)
    EX = yaml.safe_load((ROOT / "configs/sticker_expressions.yaml").read_text())
    ids = [e["id"] for e in EX]
    fid = M.FaceID()
    xf, af = expr_faces(fid), avatar_faces()

    def gen(paths, ks, sub, names):
        with torch.inference_mode():
            for i in range(0, len(paths), 16):
                x = torch.stack([to_t(Image.open(p)) for p in paths[i:i + 16]]).to(DEV, torch.float16)
                k = torch.tensor(ks[i:i + 16], device=DEV)
                y = decode(SC.forward(tr, encode(x), emb[k], mask[k], cond, noise))
                for yy, nm in zip(y, names[i:i + 16]):
                    Image.fromarray(((yy.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()).save(od / sub / f"{nm}.png")
    gen([p for p in xf for _ in ids], [tasks.index(f"expr_{e}") for _ in xf for e in ids], "expr", [f"{p.stem}_{e}" for p in xf for e in ids])
    gen(af, [tasks.index("avatar")] * len(af), "avatar", [p.stem for p in af])
    del tr
    torch.cuda.empty_cache()

    from transformers import CLIPModel, CLIPProcessor
    cp = M._snap("openai/clip-vit-large-patch14")
    cm, pr = CLIPModel.from_pretrained(str(cp), dtype=torch.float16).to(DEV).eval(), CLIPProcessor.from_pretrained(str(cp))
    with torch.no_grad():
        ti = pr(text=[f"a cartoon sticker of a person, {e['phrase']}" for e in EX], return_tensors="pt", padding=True).to(DEV)
        tf = cm.text_projection(cm.text_model(**ti).pooler_output).float()
        tf = tf / tf.norm(dim=-1, keepdim=True)
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
        fa, fo = fid.feat(Image.open(p)), fid.feat(Image.open(od / "avatar" / f"{p.stem}.png"))
        if fa is not None and fo is not None:
            cos.append(fid.cos(fa, fo))
    res = {"tag": a.tag, "ckpt": a.ckpt, "cond": cond, "encoder": a.enc, "decoder": a.dec, "tiny_dec": a.tiny_dec if a.dec == "tiny" else None, "tiny_enc": a.tiny_enc if a.enc == "tiny" else None,
           "expr_acc": {"all": round(float(hit.mean()), 4), "face_only": round(float(hit[:, fi].mean()), 4), "outside": round(float(hit[:, oi].mean()), 4)},
           "expr_acc_ci95": {"all": boot_ci(hit, rng), "face_only": boot_ci(hit[:, fi], rng), "outside": boot_ci(hit[:, oi], rng)},
           "per_edit": {e: round(float(hit[:, j].mean()), 4) for j, e in enumerate(ids)},
           "avatar_likeness": {"id_cos_mean": round(float(np.mean(cos)), 4) if cos else None, "n": len(cos), "of": len(af)},
           "metric_code": "scripts/eval_e1.py (FACE_ONLY/OUTSIDE, faces, CLIP prompt, SFace)", "minutes": round((time.time() - t0) / 60, 1)}
    (ROOT / "results/sana").mkdir(parents=True, exist_ok=True)
    (ROOT / f"results/sana/{name}.json").write_text(json.dumps(res, indent=1))
    c, order, show = 112, FACE_ONLY + OUTSIDE, list(range(0, len(xf), 3))[:8]
    g = Image.new("RGB", ((len(order) + 1) * c, 30 + len(show) * c), "white")
    d = ImageDraw.Draw(g)
    d.text((4, 4), name, fill="black")
    for j, e in enumerate(order):
        d.text(((j + 1) * c + 3, 4), e, fill="black" if e in FACE_ONLY else "blue")
    for r_, i in enumerate(show):
        g.paste(Image.open(xf[i]).convert("RGB").resize((c, c)), (0, 30 + r_ * c))
        for j, e in enumerate(order):
            g.paste(Image.open(od / "expr" / f"{xf[i].stem}_{e}.png").convert("RGB").resize((c, c)), ((j + 1) * c, 30 + r_ * c))
    (ROOT / "grids/sana").mkdir(parents=True, exist_ok=True)
    g.save(ROOT / f"grids/sana/{name}_expr.jpg", quality=85)
    print(f"[sana-eval] {name}: {res['expr_acc']} likeness {res['avatar_likeness']['id_cos_mean']} ({res['minutes']} min)", flush=True)


if __name__ == "__main__":
    main()
