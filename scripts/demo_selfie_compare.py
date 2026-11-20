"""Owner's own selfie (used with consent; NOT in any dataset; outputs stay in private_demo/, mode 700): SDXS c3_dmd3 vs
DreamLite dl1_dmd on the phone path, all 22 menu options, with the real sticker post-process (matte -> 8 px outline -> WebP).
Nothing is sent to external services.
  source env.sh; $VENV/bin/python scripts/demo_selfie_compare.py private_demo/selfie_owner.jpg c3_dmd3 dl1_dmd
Out: private_demo/demo_<stem>_v2/{input_512.png, <tag>_<task>.png|.webp, sheet_avatar_sticker.jpg, sheet_expressions.jpg}
"""
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402
import metrics as M  # noqa: E402
from eval_quality import checker  # noqa: E402
from framing import normalize_face_crop  # noqa: E402
from train_student import DEV, alphas_cumprod, forward, to_t  # noqa: E402
from diffusers import AutoencoderTiny, UNet2DConditionModel  # noqa: E402
from peft import PeftModel  # noqa: E402

ROOT = M.ROOT
src = Path(sys.argv[1])
tags = sys.argv[2:]
od = src.parent / f"demo_{src.stem}{os.environ.get('OUTSUFFIX', '_v2')}"  # OUTSUFFIX: keep earlier demos untouched
od.mkdir(exist_ok=True)
os.chmod(od, 0o700)
inp, finfo = normalize_face_crop(Image.open(src), M.FaceID())
print("framing", finfo, flush=True)
inp.save(od / "input_512.png")
x01 = to_t(inp)[None]  # [-1, 1]
tasks = None
for tag in tags:
    stu, _, dtag = tag.partition("+")  # "student+decoder": DreamLite student with a fine-tuned phone decoder (train_decoder.py)
    w = ROOT / "weights/student" / stu
    cfg = json.loads((w / "config.json").read_text())
    base = ROOT / "weights/student" / cfg["init_from"] if cfg.get("lora_rank", 0) > 0 else None
    te = torch.load(w / "task_emb.pt")
    tasks = te.get("tasks") or torch.load(base / "task_emb.pt")["tasks"]
    dl = cfg.get("arch") == "dreamlite"
    with torch.inference_mode():
        if dl:
            unet = DLA.load_unet(base / "unet" if base else w / "unet")
            if base:
                unet = PeftModel.from_pretrained(unet, w / "unet_lora").merge_and_unload()
            unet = unet.to(DEV, torch.float16).eval()
            vae = DLA.load_vae(torch.float16).to(DEV).eval()
            dvae = DLA.load_vae(torch.float16, ROOT / "weights/decoder" / dtag).to(DEV).eval() if dtag else vae
            emb, mask = te["emb"].to(DEV, torch.float16), te["mask"].to(DEV)
            z = DLA.encode(vae, x01.to(DEV, torch.float16))
            for k, t in enumerate(tasks):
                y = DLA.decode(dvae, DLA.forward(unet, z, emb[k:k + 1], mask[k:k + 1]))[0]
                Image.fromarray(((y.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()).save(od / f"{tag}_{t}.png")
        else:
            unet = UNet2DConditionModel.from_pretrained(base / "unet" if base else w / "unet")
            if base:
                unet = PeftModel.from_pretrained(unet, w / "unet_lora").merge_and_unload()
            unet = unet.to(DEV, torch.float16).eval()
            tae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).to(DEV).eval()
            ac = alphas_cumprod().to(DEV)
            emb = te["emb"].to(DEV, torch.float16)
            z = tae.encode(x01.to(DEV, torch.float16)).latents
            for k, t in enumerate(tasks):
                y = tae.decode(forward(unet, z, emb[k:k + 1], cfg["param"], cfg["T"], ac)).sample[0]
                Image.fromarray(((y.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()).save(od / f"{tag}_{t}.png")
    del unet
    torch.cuda.empty_cache()
    print("done", tag, flush=True)

mat = M.Matter()
info = {}
for tag in tags:
    for t in tasks[1:]:  # every option except the avatar is a sticker
        o = Image.open(od / f"{tag}_{t}.png").convert("RGB")
        data, inf = M.sticker_postprocess(o, mat(o))
        if data:
            (od / f"{tag}_{t}.webp").write_bytes(data)
        info[f"{tag}_{t}"] = {k: inf.get(k) for k in ("pass_sticker", "webp_kb")}
(od / "sticker_info.json").write_text(json.dumps(info, indent=1))

ck = checker()
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 14)
except OSError:
    font = ImageFont.load_default()


def cell(p, c):
    if not p.exists():
        return None
    im = Image.open(p)
    if p.suffix == ".webp":
        im = Image.alpha_composite(ck.resize(im.size), im.convert("RGBA"))
    return im.convert("RGB").resize((c, c), Image.LANCZOS)


# sheet 1: avatar + sticker, larger
c = 320
klein = src.parent / f"demo_{src.stem}" / "klein_cartoon3d_avatar.png"
cols = [("your photo (auto crop)", od / "input_512.png")] + ([("avatar: klein-4B (server, Nov 16)", klein)] if klein.exists() else [])
for tag in tags:
    cols += [(f"avatar: {tag}", od / f"{tag}_avatar.png"), (f"sticker: {tag}", od / f"{tag}_sticker.webp")]
g = Image.new("RGB", (len(cols) * c, c + 30), "white")
d = ImageDraw.Draw(g)
for j, (lab, p) in enumerate(cols):
    d.text((j * c + 4, 6), lab, fill="black", font=font)
    im = cell(p, c)
    if im:
        g.paste(im, (j * c, 30))
g.save(od / "sheet_avatar_sticker.jpg", quality=92)

# sheet 2: the 20 expressions, rows = model, two blocks of 10
c, lw = 200, 130
ex = tasks[2:]
blocks = [ex[:10], ex[10:]]
g = Image.new("RGB", (lw + 10 * c, len(blocks) * len(tags) * (c + 22)), "white")
d = ImageDraw.Draw(g)
y = 0
for blk in blocks:
    for tag in tags:
        d.text((4, y + 22 + c // 2), tag, fill="black", font=font)
        for j, t in enumerate(blk):
            d.text((lw + j * c + 4, y + 4), t.replace("expr_", ""), fill="black", font=font)
            im = cell(od / f"{tag}_{t}.webp", c) or cell(od / f"{tag}_{t}.png", c)
            if im:
                g.paste(im, (lw + j * c, y + 22))
        y += c + 22
g.save(od / "sheet_expressions.jpg", quality=92)
for p in od.iterdir():
    os.chmod(p, 0o600)
print("wrote", od / "sheet_avatar_sticker.jpg", od / "sheet_expressions.jpg", flush=True)
