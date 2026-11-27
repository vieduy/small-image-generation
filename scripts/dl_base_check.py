"""2026-11-23: sanity check of DreamLite-base (28-step, undistilled; carlofkl/DreamLite-base@diffusers, CC BY-NC 4.0) vs the
4-step DreamLite-mobile we start from. Same photos, same edit prompt (dl_sanity.py's cartoon3d wording), official pipelines, 1024 px
(DreamLite-native) and 512 px. Also: UNet weight difference base vs mobile. -> grids/dl_base_vs_mobile.jpg, results/dl_base_check.json
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/dl_base_check.py
"""
import json, time
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from diffusers import DreamLiteMobilePipeline, DreamLitePipeline

ROOT = Path(__file__).resolve().parent.parent
DEV = "cuda"
short = "been redrawn as a 3D cartoon avatar portrait with a stylized rounded face, large expressive eyes and smooth soft shading"
photos = [ROOT / "data/faces_eval2" / n for n in ("fe2_020.png", "fe2_077.png", "fe2_140.png")]
base = DreamLitePipeline.from_pretrained("carlofkl/DreamLite-base", revision="diffusers", torch_dtype=torch.bfloat16).to(DEV)
mob = DreamLiteMobilePipeline.from_pretrained("carlofkl/DreamLite-mobile", revision="diffusers", torch_dtype=torch.bfloat16,
                                              text_encoder=base.text_encoder, processor=base.processor, tokenizer=base.tokenizer).to(DEV)
sd_b, sd_m = base.unet.state_dict(), mob.unet.state_dict()
rel = [((sd_b[k].float() - sd_m[k].float()).norm() / sd_m[k].float().norm().clamp_min(1e-8)).item() for k in sd_m if sd_m[k].dtype.is_floating_point]
res = {"unet_rel_weight_diff_mean": round(sum(rel) / len(rel), 4), "unet_rel_weight_diff_max": round(max(rel), 4), "runs": []}
rows = []
for ph in photos:
    img = Image.open(ph).convert("RGB")
    row = [img.resize((384, 384))]
    for name, pipe, kw in (("mobile_4", mob, dict(num_inference_steps=4)),
                           ("base_28", base, dict(num_inference_steps=28))):
        for hw in (1024, 512):
            t = time.time()
            out = pipe(prompt=short, image=img.resize((hw, hw)), height=hw, width=hw, generator=torch.Generator(DEV).manual_seed(0), **kw).images[0]
            torch.cuda.synchronize()
            res["runs"].append({"photo": ph.name, "model": name, "res": hw, "sec": round(time.time() - t, 2)})
            row.append(out.resize((384, 384)))
    rows.append(row)
labels = ["photo", "mobile 4st 1024", "mobile 4st 512", "base 28st 1024", "base 28st 512"]
g = Image.new("RGB", (384 * 5, 24 + 384 * len(rows)), "white"); d = ImageDraw.Draw(g)
for j, l in enumerate(labels): d.text((j * 384 + 6, 5), l, fill="black")
for i, r in enumerate(rows):
    for j, im in enumerate(r): g.paste(im, (j * 384, 24 + i * 384))
g.save(ROOT / "grids/dl_base_vs_mobile.jpg", quality=88)
(ROOT / "results/dl_base_check.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "runs"}), [(r["model"], r["res"], r["sec"]) for r in res["runs"][:4]], flush=True)
