"""Diagnostic (owner's selfie, consented demo): is the small model's weak result a framing domain gap?
Re-frame the photo to the training distribution (face height ~0.56 of the square, face top ~0.18) by padding with a
blurred copy of the image when the photo is too tight, then run the student (TAESD phone path and SD-VAE path).
Out: private_demo/demo_selfie_owner/reframe_sheet.jpg
"""
import json, sys
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps
sys.path.insert(0, str(Path(__file__).parent))
import metrics as M  # noqa: E402
from train_student import alphas_cumprod, forward, to_t, DEV, SDXS  # noqa: E402
from diffusers import AutoencoderTiny, AutoencoderKL, UNet2DConditionModel  # noqa: E402
ROOT = M.ROOT
src, tag = Path(sys.argv[1]), sys.argv[2]
od = src.parent / f"demo_{src.stem}"
img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
fid = M.FaceID()
a = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR); h, w = a.shape[:2]
fid.det.setInputSize((w, h)); _, fs = fid.det.detect(a)
x, y, fw, fh = max(fs, key=lambda r: r[2] * r[3])[:4]


def crop_at(side, cx, top_y):
    side = int(side)
    x0, y0 = int(cx - side / 2), int(top_y)
    pad = max(0, -x0, -y0, x0 + side - w, y0 + side - h)
    if pad:
        big = img.resize((w + 2 * pad, h + 2 * pad)).filter(ImageFilter.GaussianBlur(40))  # soft fill, no hard border
        big.paste(img, (pad, pad))
        return big.crop((x0 + pad, y0 + pad, x0 + pad + int(side), y0 + pad + int(side))).resize((512, 512), Image.LANCZOS)
    return img.crop((x0, y0, x0 + int(side), y0 + int(side))).resize((512, 512), Image.LANCZOS)


def box512(im):
    aa = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    fid.det.setInputSize((512, 512)); _, ff = fid.det.detect(aa)
    f = max(ff, key=lambda r: r[2] * r[3])
    return f[1] / 512, f[3] / 512, (f[0] + f[2] / 2) / 512


def reframe(face_frac, top):
    """Iterate so that the face box MEASURED ON THE 512 CROP (same way as for training inputs) hits the target."""
    side, cx, ty = float(min(w, h)), x + fw / 2, y - 0.3 * fh
    for _ in range(6):
        im = crop_at(side, cx, ty)
        t, fr, xc = box512(im)
        side *= fr / face_frac                        # zoom
        cx += (xc - 0.5) * side                       # recentre horizontally
        ty += (t - top) * side                        # put face top at `top`
    im = crop_at(side, cx, ty)
    print("reframe", face_frac, "->", [round(float(v), 3) for v in box512(im)[:2]], flush=True)
    return im


ins = {"tight (as before)": Image.open(od / "input_512.png").convert("RGB"),
       "reframed 0.56 (train median)": reframe(0.56, 0.18), "reframed 0.48 (looser)": reframe(0.48, 0.20)}
w_ = ROOT / "weights/student" / tag
cfg = json.loads((w_ / "config.json").read_text())
unet = UNet2DConditionModel.from_pretrained(w_ / "unet", torch_dtype=torch.float16).to(DEV).eval()
emb = torch.load(w_ / "task_emb.pt")["emb"].to(DEV, torch.float16)
tae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).to(DEV).eval()
vae = AutoencoderKL.from_pretrained(SDXS, subfolder="vae_large", torch_dtype=torch.float16).to(DEV).eval()
ac, sf = alphas_cumprod().to(DEV), vae.config.scaling_factor
c = 240
g = Image.new("RGB", (5 * c, 20 + len(ins) * c), "white"); d = ImageDraw.Draw(g)
for j, s in enumerate(["input framing", "avatar TAESD (phone)", "sticker TAESD (phone)", "avatar SD-VAE", "sticker SD-VAE"]):
    d.text((j * c + 4, 4), s, fill="black")
for i, (name, im) in enumerate(ins.items()):
    im.save(od / f"reframe_{i}.png")
    xx = to_t(im)[None].to(DEV, torch.float16)
    outs = []
    with torch.inference_mode():
        for enc in ("taesd", "sdvae"):
            z = tae.encode(xx).latents if enc == "taesd" else vae.encode(xx).latent_dist.mean * sf
            for k in (0, 1):
                zo = forward(unet, z, emb[k:k + 1], cfg["param"], cfg["T"], ac)
                yv = (tae.decode(zo).sample if enc == "taesd" else vae.decode(zo / sf).sample)[0]
                outs.append(Image.fromarray(((yv.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()))
    g.paste(im.resize((c, c)), (0, 20 + i * c))
    for j, o in enumerate(outs):
        g.paste(o.resize((c, c)), ((j + 1) * c, 20 + i * c))
    d.text((4, 22 + i * c), name, fill="yellow")
g.save(od / "reframe_sheet.jpg", quality=92)
print("wrote", od / "reframe_sheet.jpg")
