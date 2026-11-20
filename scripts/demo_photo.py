"""One-off demo on a user-supplied photo (owner's own selfie, used with consent; NOT added to any dataset).
Product-realistic: auto face crop -> 512x512; klein uses the GENERIC style prompt (no privileged attributes exist for
real users); small students run the phone path (TAESD enc -> 1-step UNet -> TAESD dec). Stickers get the real
post-process (matte -> 8px white outline -> WebP <= 100 KB).
  source env.sh; $VENV/bin/python scripts/demo_photo.py private_demo/selfie_owner.jpg cartoon3d:s5d flat_vector:fv_lora
Out: <photo dir>/demo_<stem>/ (pngs, webps, sheet.jpg)
"""
import json, os, sys
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
os.environ.setdefault("MODEL", "klein"); os.environ.setdefault("KLEIN_REF", "0"); os.environ.setdefault("PROMPTS_FAMILY", "klein_text")
sys.path.insert(0, str(Path(__file__).parent))
import metrics as M, i2i_server as S  # noqa: E402
from train_student import alphas_cumprod, forward, to_t, DEV  # noqa: E402
from eval_quality import checker  # noqa: E402
from diffusers import AutoencoderTiny, UNet2DConditionModel  # noqa: E402
ROOT = M.ROOT
src = Path(sys.argv[1])
jobs = [a.split(":") for a in sys.argv[2:]]
od = src.parent / f"demo_{src.stem}"; od.mkdir(exist_ok=True)

# 1) shared framing normaliser (same framing as the training inputs), see scripts/framing.py
from framing import normalize_face_crop  # noqa: E402
fid = M.FaceID()
inp, finfo = normalize_face_crop(Image.open(src), fid)
print("framing", finfo, flush=True)
inp.save(od / "input_512.png")
print("skin L*", round(fid.skin_L(inp), 1), flush=True)

styles = S.load_styles()
seed = 12345
res = {}
ad = S.Klein(); ad.load(); ad.p.set_progress_bar_config(disable=True)
for sname, _ in jobs:
    st = styles[sname]
    for task in ("avatar", "sticker"):
        with torch.inference_mode():
            ad.run([inp], st[f"{task}_text"], st, [seed])[0].save(od / f"klein_{sname}_{task}.png")
del ad; torch.cuda.empty_cache()

tae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).to(DEV).eval()
ac = alphas_cumprod().to(DEV)
for sname, tag in jobs:
    w_ = ROOT / "weights/student" / tag
    if not (w_ / "config.json").exists():
        print("student not trained yet:", tag, flush=True); continue
    cfg = json.loads((w_ / "config.json").read_text())
    if cfg.get("lora_rank", 0) > 0:
        from peft import PeftModel
        unet = PeftModel.from_pretrained(UNet2DConditionModel.from_pretrained(ROOT / "weights/student" / cfg["init_from"] / "unet"),
                                         w_ / "unet_lora").merge_and_unload().to(DEV, torch.float16).eval()
    else:
        unet = UNet2DConditionModel.from_pretrained(w_ / "unet", torch_dtype=torch.float16).to(DEV).eval()
    emb = torch.load(w_ / "task_emb.pt")["emb"].to(DEV, torch.float16)
    x = to_t(inp)[None].to(DEV, torch.float16)
    for k, task in enumerate(("avatar", "sticker")):
        with torch.inference_mode():
            yv = tae.decode(forward(unet, tae.encode(x).latents, emb[k:k + 1], cfg["param"], cfg["T"], ac)).sample[0]
        Image.fromarray(((yv.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()).save(od / f"small_{sname}_{task}.png")
    del unet; torch.cuda.empty_cache()

mat = M.Matter()
for p in sorted(od.glob("*_sticker.png")):
    o = Image.open(p).convert("RGB")
    data, info = M.sticker_postprocess(o, mat(o))
    if data:
        (od / f"{p.stem}.webp").write_bytes(data)
    res[p.stem] = info
(od / "sticker_info.json").write_text(json.dumps(res, indent=1))

c = 256
cols = ["your photo (auto crop)", "avatar: klein-4B (server)", "avatar: small model (phone path)", "sticker: klein-4B", "sticker: small model"]
g = Image.new("RGB", (len(cols) * c, 40 + len(jobs) * c), "white"); d = ImageDraw.Draw(g)
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 14)
except OSError:
    font = ImageFont.load_default()
for j, s in enumerate(cols):
    d.text((j * c + 4, 20), s, fill="black", font=font)
ck = checker()
for i, (sname, tag) in enumerate(jobs):
    yy = 40 + i * c
    d.text((4, 4), "rows: " + " / ".join(f"{s} (small={t})" for s, t in jobs), fill="black", font=font)
    g.paste(inp.resize((c, c)), (0, yy))
    for j, f in enumerate([f"klein_{sname}_avatar.png", f"small_{sname}_avatar.png", f"klein_{sname}_sticker.webp", f"small_{sname}_sticker.webp"]):
        p = od / f
        if p.exists():
            im = Image.open(p)
            im = Image.alpha_composite(ck, im.convert("RGBA")).convert("RGB") if p.suffix == ".webp" else im.convert("RGB")
            g.paste(im.resize((c, c)), ((j + 1) * c, yy))
        else:
            d.text(((j + 1) * c + 10, yy + c // 2), "not trained yet", fill="grey", font=font)
g.save(od / "sheet.jpg", quality=92)
print("wrote", od / "sheet.jpg", json.dumps({k: (v.get("pass_sticker"), v.get("webp_kb")) for k, v in res.items()}), flush=True)
