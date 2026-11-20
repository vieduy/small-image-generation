"""Sticker variety pilot (priority #1): can klein produce the 20 sticker-set expressions while keeping identity and style?
8 eval faces (dark->light skin spread) x 20 expressions, cartoon3d, per-person prompt + expression phrase.
Metrics per expression: identity rank among the 100 v1 eval inputs (top-5 rate), CLIP zero-shot expression accuracy
(own phrase vs the 19 others), sticker post-process pass.
  source env.sh; $VENV/bin/python scripts/expr_pilot.py
Out: grids/expr_pilot_cartoon3d.jpg, results/expr_pilot.json
"""
import json, os, sys, zlib
from pathlib import Path
import numpy as np, torch, yaml
from PIL import Image, ImageDraw
os.environ.setdefault("MODEL", "klein"); os.environ.setdefault("KLEIN_REF", "0")
sys.path.insert(0, str(Path(__file__).parent))
import metrics as M, i2i_server as S, teacher_prompt as TP  # noqa: E402
ROOT = M.ROOT
EX = yaml.safe_load((ROOT / "configs/sticker_expressions.yaml").read_text())
st = S.load_styles()["cartoon3d"]
fid = M.FaceID()
faces = sorted((ROOT / "data/faces").glob("*.png"))[:100]
man = {json.loads(l)["file"]: json.loads(l)["prompt"] for l in open(ROOT / "data/faces/manifest.jsonl")}
L = {p: fid.skin_L(Image.open(p)) for p in faces}
order = sorted([p for p in faces if L[p] is not None], key=lambda p: L[p])
sel = [order[int(i)] for i in np.linspace(0, len(order) - 1, 8)]
od = ROOT / "outputs/expr_pilot/cartoon3d"; od.mkdir(parents=True, exist_ok=True)


def expr_prompt(pdesc, phrase):
    return (f"Redraw the person in this photo as a die-cut chat sticker, {st['description']}. The person is {pdesc}; the sticker "
            f"shows exactly this person with the same skin tone, age, gender and hairstyle, with {phrase}. Expressive pose, "
            f"full subject, thick white outline, isolated on a plain flat white background.")


need = [(p, e) for p in sel for e in EX if not (od / f"{p.stem}_{e['id']}.png").exists()]
ad = None
if need:
    ad = S.Klein(); ad.load(); ad.p.set_progress_bar_config(disable=True)
for p in sel:
    pdesc = TP.person_desc(TP.parse(man[p.name]), L[p])
    for e in EX:
        f = od / f"{p.stem}_{e['id']}.png"
        if not f.exists():
            with torch.inference_mode():
                ad.run([Image.open(p).convert("RGB")], expr_prompt(pdesc, e["phrase"]), st, [zlib.crc32(f"{p.stem}|{e['id']}".encode())])[0].save(f)
del ad; torch.cuda.empty_cache()
# metrics
from transformers import CLIPModel, CLIPProcessor
cp = M._snap("openai/clip-vit-large-patch14")
cm, pr = CLIPModel.from_pretrained(str(cp), dtype=torch.float16).cuda().eval(), CLIPProcessor.from_pretrained(str(cp))
TEXTS = [f"a cartoon sticker of a person, {e['phrase']}" for e in EX]
F = {p.stem: fid.feat(Image.open(p)) for p in faces}
gal = [s for s in F if F[s] is not None]
G = np.stack([F[s].ravel() / np.linalg.norm(F[s]) for s in gal])
mat = M.Matter()
res = {}
for k, e in enumerate(EX):
    t5, acc, ok = [], [], []
    for p in sel:
        o = Image.open(od / f"{p.stem}_{e['id']}.png").convert("RGB")
        fo = fid.feat(o)
        if fo is None:
            t5.append(False)
        else:
            sims = G @ (fo.ravel() / np.linalg.norm(fo)); t5.append(int((sims > sims[gal.index(p.stem)]).sum()) + 1 <= 5)
        with torch.inference_mode():
            x = pr(text=TEXTS, images=o, return_tensors="pt", padding=True).to("cuda")
            lg = cm(**{kk: (v.half() if v.dtype == torch.float32 else v) for kk, v in x.items()}).logits_per_image[0]
        acc.append(int(lg.argmax()) == k)
        ok.append(M.sticker_postprocess(o, mat(o))[1]["pass_sticker"])
    res[e["id"]] = {"id_top5": round(float(np.mean(t5)), 2), "clip_expr_acc": round(float(np.mean(acc)), 2), "sticker_pass": round(float(np.mean(ok)), 2)}
    print(e["id"], res[e["id"]], flush=True)
res["_mean"] = {k: round(float(np.mean([v[k] for kk, v in res.items() if not kk.startswith("_")])), 3) for k in ("id_top5", "clip_expr_acc", "sticker_pass")}
print("MEAN", res["_mean"], flush=True)
(ROOT / "results/expr_pilot.json").write_text(json.dumps(res, indent=1))
c = 128
g = Image.new("RGB", ((len(EX) + 1) * c, 18 + len(sel) * c), "white"); d = ImageDraw.Draw(g)
for j, e in enumerate(EX):
    d.text(((j + 1) * c + 3, 3), e["label"], fill="black")
for i, p in enumerate(sel):
    g.paste(Image.open(p).convert("RGB").resize((c, c)), (0, 18 + i * c))
    for j, e in enumerate(EX):
        g.paste(Image.open(od / f"{p.stem}_{e['id']}.png").convert("RGB").resize((c, c)), ((j + 1) * c, 18 + i * c))
g.save(ROOT / "grids/expr_pilot_cartoon3d.jpg", quality=85)
