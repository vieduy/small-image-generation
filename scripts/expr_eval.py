"""Sticker-set eval for a student with expression tokens (priority #1): 24 v1 eval faces (dark->light spread) x 20 expressions,
phone path (TAESD enc -> 1-step UNet -> TAESD dec). Same metrics as scripts/expr_pilot.py (klein teacher):
identity top-5 among the 100 v1 eval inputs, CLIP expression accuracy (own phrase vs the other 19), sticker post-process pass.
  source env.sh; $VENV/bin/python scripts/expr_eval.py --tag s5e
Out: results/expr_eval_<tag>.json, grids/expr_eval_<tag>.jpg (8 faces shown)
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, yaml
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).parent))
import metrics as M  # noqa: E402
from train_student import alphas_cumprod, forward, to_t, DEV, TASKS  # noqa: E402
from diffusers import AutoencoderTiny, UNet2DConditionModel  # noqa: E402
ROOT = M.ROOT
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--n", type=int, default=24)
ap.add_argument("--decoder", default="", help="DreamLite only: fine-tuned phone decoder weights/decoder/<name> (train_decoder.py); "
                "results are tagged <tag>_<name>")
a = ap.parse_args()
OTAG = a.tag + (f"_{a.decoder}" if a.decoder else "")
EX = yaml.safe_load((ROOT / "configs/sticker_expressions.yaml").read_text())
w = ROOT / "weights/student" / a.tag
cfg = json.loads((w / "config.json").read_text())
te = torch.load(w / "task_emb.pt")
emb, tasks = te["emb"].to(DEV, torch.float16), te["tasks"]
DLM = None
if cfg.get("arch") == "dreamlite":  # DreamLite student: unet/pack + TinyVAE; forward() is swapped below
    import dl_arch as DLA
    from peft import PeftModel
    unet = (PeftModel.from_pretrained(DLA.load_unet(ROOT / "weights/student" / cfg["init_from"] / "unet"), w / "unet_lora").merge_and_unload()
            if cfg.get("lora_rank", 0) > 0 else DLA.load_unet(w / "unet")).to(DEV, torch.float16).eval()
    DLM = te["mask"].to(DEV)
    if len(tasks) < emb.shape[0]:
        tasks = torch.load(ROOT / "weights/student" / cfg["init_from"] / "task_emb.pt")["tasks"]
elif cfg.get("lora_rank", 0) > 0:  # style pack (e.g. c3_dmd, an_b1_dmd): merge onto its base; token names come from the base
    from peft import PeftModel
    wb = ROOT / "weights/student" / cfg["init_from"]
    unet = PeftModel.from_pretrained(UNet2DConditionModel.from_pretrained(wb / "unet"), w / "unet_lora").merge_and_unload().to(DEV, torch.float16).eval()
    if len(tasks) < emb.shape[0]:
        tasks = torch.load(wb / "task_emb.pt")["tasks"]
else:
    unet = UNet2DConditionModel.from_pretrained(w / "unet", torch_dtype=torch.float16).to(DEV).eval()
assert len(tasks) == emb.shape[0], (len(tasks), emb.shape)
tae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).to(DEV).eval()
ac = alphas_cumprod().to(DEV)
fid = M.FaceID()
faces = sorted((ROOT / "data/faces").glob("*.png"))[:100]
L = {p: fid.skin_L(Image.open(p)) for p in faces}
order = sorted([p for p in faces if L[p] is not None], key=lambda p: L[p])
sel = [order[int(i)] for i in np.linspace(0, len(order) - 1, a.n)]
od = ROOT / f"outputs/expr_eval_{OTAG}/{cfg['style']}"; od.mkdir(parents=True, exist_ok=True)
for p in sel:
    x = to_t(Image.open(p))[None].to(DEV, torch.float16)
    with torch.inference_mode():
        if DLM is not None:
            dlv = globals().get("_dlv") or globals().setdefault("_dlv", DLA.load_vae(torch.float16).to(DEV).eval())
            dlvd = globals().get("_dlvd") or globals().setdefault("_dlvd", (DLA.load_vae(torch.float16, ROOT / "weights/decoder" / a.decoder)
                                                                            if a.decoder else dlv).to(DEV).eval())
            z = DLA.encode(dlv, x)
        else:
            z = tae.encode(x).latents
        for e in EX:
            k = tasks.index(f"expr_{e['id']}")
            if DLM is not None:
                y = DLA.decode(dlvd, DLA.forward(unet, z, emb[k:k + 1], DLM[k:k + 1], sigmas=cfg.get("ms_sigmas")))[0]
            else:
                y = tae.decode(forward(unet, z, emb[k:k + 1], cfg["param"], cfg["T"], ac)).sample[0]
            Image.fromarray(((y.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()).save(od / f"{p.stem}_{e['id']}.png")
del unet; torch.cuda.empty_cache()
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
        t5.append(fo is not None and int(((G @ (fo.ravel() / np.linalg.norm(fo))) > (G @ (fo.ravel() / np.linalg.norm(fo)))[gal.index(p.stem)]).sum()) + 1 <= 5)
        with torch.inference_mode():
            xx = pr(text=TEXTS, images=o, return_tensors="pt", padding=True).to("cuda")
            acc.append(int(cm(**{kk: (v.half() if v.dtype == torch.float32 else v) for kk, v in xx.items()}).logits_per_image[0].argmax()) == k)
        ok.append(M.sticker_postprocess(o, mat(o))[1]["pass_sticker"])
    res[e["id"]] = {"id_top5": round(float(np.mean(t5)), 3), "clip_expr_acc": round(float(np.mean(acc)), 3), "sticker_pass": round(float(np.mean(ok)), 3)}
res["_mean"] = {k: round(float(np.mean([v[k] for kk, v in res.items() if not kk.startswith("_")])), 3) for k in ("id_top5", "clip_expr_acc", "sticker_pass")}
res["_n_faces"] = len(sel)
print("MEAN", OTAG, res["_mean"], flush=True)
(ROOT / f"results/expr_eval_{OTAG}.json").write_text(json.dumps(res, indent=1))
c = 128; show = sel[:: max(1, len(sel) // 8)][:8]
g = Image.new("RGB", ((len(EX) + 1) * c, 18 + len(show) * c), "white"); d = ImageDraw.Draw(g)
for j, e in enumerate(EX):
    d.text(((j + 1) * c + 3, 3), e["label"], fill="black")
for i, p in enumerate(show):
    g.paste(Image.open(p).convert("RGB").resize((c, c)), (0, 18 + i * c))
    for j, e in enumerate(EX):
        g.paste(Image.open(od / f"{p.stem}_{e['id']}.png").convert("RGB").resize((c, c)), ((j + 1) * c, 18 + i * c))
g.save(ROOT / f"grids/expr_eval_{OTAG}.jpg", quality=85)
