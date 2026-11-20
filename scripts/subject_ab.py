"""A/B of non-person sticker teacher prompts (humanization bug). 40 training subject inputs, cartoon3d.
Metrics: human_face = YuNet detects a face in the output (humanized proxy; cartoon animals can also trigger it),
clip_ok = CLIP zero-shot prefers 'a cartoon sticker of a {subj}' over 'a cartoon sticker of a person's face'.
  source env.sh; $VENV/bin/python scripts/subject_ab.py
"""
import json, os, sys
from pathlib import Path
import torch
from PIL import Image, ImageDraw
os.environ["MODEL"] = "klein"; os.environ["KLEIN_REF"] = "0"
sys.path.insert(0, str(Path(__file__).parent))
import metrics as M, teacher_prompt as TP, i2i_server as S  # noqa
ROOT = M.ROOT
R = ROOT / "data/pairs/cartoon3d"
rows = [json.loads(l) for l in open(R / "manifest.jsonl")]
sub = [r for r in rows if not r["is_face"]][::10][:40]
st = S.load_styles()["cartoon3d"]
V = {"old": lambda s: TP.sticker_prompt(st["description"], None),
     "subjdesc_generic": lambda s: TP.subject_sticker_prompt(st["subject_description"], None),
     "subjdesc_named": lambda s: TP.subject_sticker_prompt(st["subject_description"], s)}
ad = S.Klein(); ad.load(); ad.p.set_progress_bar_config(disable=True)
for name, fn in V.items():
    od = ROOT / "outputs/subject_ab" / name; od.mkdir(parents=True, exist_ok=True)
    for r in sub:
        if name == "old":
            continue
        if (od / f"{r['id']}.png").exists():
            continue
        with torch.inference_mode():
            ad.run([Image.open(R / "inputs" / f"{r['id']}.png").convert("RGB")], fn(TP.parse_subject(r["prompt"])), st, [r["seed"] + 7])[0].save(od / f"{r['id']}.png")
del ad; torch.cuda.empty_cache()
from transformers import CLIPModel, CLIPProcessor
cp = M._snap("openai/clip-vit-large-patch14")
cm, pr = CLIPModel.from_pretrained(str(cp), dtype=torch.float16).cuda().eval(), CLIPProcessor.from_pretrained(str(cp))
fid = M.FaceID()
res = {}
for name in V:
    hf, ok = 0, 0
    for r in sub:
        f = (R / "sticker" if name == "old" else ROOT / "outputs/subject_ab" / name) / f"{r['id']}.png"
        im = Image.open(f).convert("RGB")
        hf += fid.feat(im) is not None
        s = TP.parse_subject(r["prompt"])
        x = pr(text=[f"a cartoon sticker of a {s}", "a cartoon sticker of a person's face"], images=im, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            lg = cm(**{k: (v.half() if v.dtype == torch.float32 else v) for k, v in x.items()}).logits_per_image[0]
        ok += int(lg[0] > lg[1])
    res[name] = {"n": len(sub), "human_face_rate": round(hf / len(sub), 3), "clip_subject_ok": round(ok / len(sub), 3)}
    print(name, res[name], flush=True)
(ROOT / "results/subject_ab.json").write_text(json.dumps(res, indent=1))
c = 128
g = Image.new("RGB", (4 * c, 16 * c + 14), "white")
ImageDraw.Draw(g).text((3, 1), "input | old | subjdesc_generic | subjdesc_named", fill="black")
for i, r in enumerate(sub[:16]):
    ims = [R / "inputs" / f"{r['id']}.png", R / "sticker" / f"{r['id']}.png"] + [ROOT / "outputs/subject_ab" / n / f"{r['id']}.png" for n in list(V)[1:]]
    for j, p in enumerate(ims):
        g.paste(Image.open(p).convert("RGB").resize((c, c)), (j * c, 14 + i * c))
g.save(ROOT / "grids/subject_ab.jpg", quality=86)
