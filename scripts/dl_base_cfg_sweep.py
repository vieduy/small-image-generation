"""2026-11-23 (user: "run the sweep"): fair base-vs-mobile comparison. DreamLite-base (28 steps) has two knobs, mobile none.
Tune base on 8 faces over image CFG {1.5, 2.5, 4} x text CFG {3.5, 2}; pick the best avatar likeness among settings whose outputs
still read as a cartoon (CLIP-L: '3D cartoon character' vs 'real photograph', cartoon rate >= 0.9); compare that setting with
mobile (4 steps) on 8 OTHER faces. 1024 px (both 1024-native), same prompt as dl_sanity.py, seed 0.
Metrics: SFace id cosine photo vs output (metrics.FaceID, = eval_quality id_cos), skin dL* (output - photo), cartoon rate.
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/dl_base_cfg_sweep.py
-> results/dl_base_cfg_sweep.json, grids/dl_base_cfg_sweep_test.jpg
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).parent))
import metrics as M  # noqa: E402
from diffusers import DreamLiteMobilePipeline, DreamLitePipeline

ROOT, DEV = Path(__file__).resolve().parent.parent, "cuda"
PROMPT = "been redrawn as a 3D cartoon avatar portrait with a stylized rounded face, large expressive eyes and smooth soft shading"
faces = sorted((ROOT / "data/faces_eval2").glob("*.png"))
pick = [faces[int(i)] for i in np.linspace(0, len(faces) - 1, 16)]
tune, test = pick[0::2], pick[1::2]
base = DreamLitePipeline.from_pretrained("carlofkl/DreamLite-base", revision="diffusers", torch_dtype=torch.bfloat16).to(DEV)
mob = DreamLiteMobilePipeline.from_pretrained("carlofkl/DreamLite-mobile", revision="diffusers", torch_dtype=torch.bfloat16,
                                              text_encoder=base.text_encoder, processor=base.processor, tokenizer=base.tokenizer).to(DEV)
for p in (base, mob):
    p.set_progress_bar_config(disable=True)
fid = M.FaceID()
from transformers import CLIPModel, CLIPProcessor
cp = M._snap("openai/clip-vit-large-patch14")
cm, pr = CLIPModel.from_pretrained(str(cp), dtype=torch.float16).to(DEV).eval(), CLIPProcessor.from_pretrained(str(cp))
with torch.no_grad():
    ti = pr(text=["a 3D cartoon character", "a real photograph of a person"], return_tensors="pt", padding=True).to(DEV)
    tf = cm.text_projection(cm.text_model(**ti).pooler_output).float(); tf = tf / tf.norm(dim=-1, keepdim=True)


def cartoon(img):
    with torch.no_grad():
        pv = pr.image_processor(images=[img], return_tensors="pt")["pixel_values"].to(DEV, torch.float16)
        f = cm.visual_projection(cm.vision_model(pixel_values=pv).pooler_output).float(); f = f / f.norm(dim=-1, keepdim=True)
    return float((f @ tf.T).argmax() == 0)


def run(pipe, photos, **kw):
    outs, ids, dls, car = [], [], [], []
    for ph in photos:
        img = Image.open(ph).convert("RGB").resize((1024, 1024))
        o = pipe(prompt=PROMPT, image=img, height=1024, width=1024, generator=torch.Generator(DEV).manual_seed(0), **kw).images[0]
        outs.append(o)
        a, b = fid.feat(img), fid.feat(o)
        if a is not None and b is not None:
            ids.append(fid.cos(a, b))
        la, lb = fid.skin_L(img), fid.skin_L(o)
        if la is not None and lb is not None:
            dls.append(lb - la)
        car.append(cartoon(o))
    return outs, {"id_cos": round(float(np.mean(ids)), 4) if ids else None, "id_n": len(ids),
                  "skin_dL": round(float(np.median(dls)), 1) if dls else None, "cartoon_rate": round(float(np.mean(car)), 3)}


t0 = time.time()
res = {"tune_faces": [p.name for p in tune], "test_faces": [p.name for p in test], "tune": {}}
for ic in (1.5, 2.5, 4.0):
    for tc in (3.5, 2.0):
        _, m = run(base, tune, num_inference_steps=28, guidance_scale=tc, image_guidance_scale=ic)
        res["tune"][f"img{ic}_txt{tc}"] = m
        print("tune", ic, tc, m, flush=True)
_, res["tune"]["mobile"] = run(mob, tune, num_inference_steps=4)
print("tune mobile", res["tune"]["mobile"], flush=True)
ok = {k: v for k, v in res["tune"].items() if k != "mobile" and v["cartoon_rate"] >= 0.9 and v["id_cos"] is not None}
best = max(ok, key=lambda k: ok[k]["id_cos"]) if ok else max((k for k in res["tune"] if k != "mobile"), key=lambda k: res["tune"][k]["id_cos"] or -1)
ic, tc = float(best.split("_")[0][3:]), float(best.split("_")[1][3:])
res["best_base_setting"] = best
ob, res["test_base_best"] = run(base, test, num_inference_steps=28, guidance_scale=tc, image_guidance_scale=ic)
om, res["test_mobile"] = run(mob, test, num_inference_steps=4)
res["minutes"] = round((time.time() - t0) / 60, 1)
print("TEST base", best, res["test_base_best"], "| mobile", res["test_mobile"], flush=True)
(ROOT / "results/dl_base_cfg_sweep.json").write_text(json.dumps(res, indent=1))
c = 256
g = Image.new("RGB", (len(test) * c, 3 * c + 24), "white"); d = ImageDraw.Draw(g)
d.text((4, 4), f"rows: photo / mobile 4 steps / base 28 steps ({best}) - test faces", fill="black")
for j, (ph, a, b) in enumerate(zip(test, om, ob)):
    for r, im in enumerate((Image.open(ph).convert("RGB"), a, b)):
        g.paste(im.resize((c, c)), (j * c, 24 + r * c))
g.save(ROOT / "grids/dl_base_cfg_sweep_test.jpg", quality=88)
