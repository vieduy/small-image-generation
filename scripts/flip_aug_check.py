"""Is horizontal-flip augmentation in LATENT space safe for the DreamLite TinyVAE? (2026-11-19)
train_dl.py / train_dmd.py flip zi/zt latents with p=0.5. If the VAE is not flip-equivariant, decode(flip(z)) is not the
flipped target image, so half of all training targets would be distorted. Compares, on 64 v2 targets:
  round trip       decode(z)               vs the target PNG
  latent flip      decode(flip(z))         vs flip(PNG)       (what training uses now)
  image flip       decode(enc(flip(PNG)))  vs flip(PNG)       (the proper augmentation)
Out: printed PSNRs + grids/flip_aug_check.jpg (flip(PNG) | decode(flip z) | decode(enc(flip PNG)) | |error| x4)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402
from train_student import to_t  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
d = ROOT / "data/pairs/cartoon3d"
SD = len(sys.argv) > 1 and sys.argv[1] == "sd"   # "sd": SDXS student (SD VAE encoder + TAESD decoder, as train_student.py)
c = torch.load(d / ("latents_sdvae_manifest_v2.pt" if SD else "latents_dl_manifest_v2.pt"))
rows = {(r["id"], r["task"]): r for r in map(json.loads, open(d / "manifest_v2.jsonl"))}
N = 64
sel = list(range(0, len(c["ids"]), len(c["ids"]) // N))[:N]


def tpath(i):
    k = c["ids"][i]
    return d / rows[k].get("target", f"{k[1]}/{k[0]}.png")


x = torch.stack([to_t(Image.open(tpath(i))) for i in sel]).cuda().float()
zt = c["zt"][sel].cuda()
if SD:
    from diffusers import AutoencoderKL, AutoencoderTiny
    from train_student import SDXS
    enc_vae = AutoencoderKL.from_pretrained(SDXS, subfolder="vae_large", torch_dtype=torch.bfloat16).cuda().eval()
    taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.bfloat16).cuda().eval()
    sf = enc_vae.config.scaling_factor
    enc = lambda x: enc_vae.encode(x).latent_dist.mean * sf
    dec = lambda z: taesd.decode(z.to(torch.bfloat16)).sample.float().clamp(-1, 1)
else:
    vae = DLA.load_vae(torch.bfloat16).cuda().eval()
    enc = lambda x: DLA.encode(vae, x)

    def dec(z):
        return DLA.decode(vae, z.to(torch.bfloat16)).float().clamp(-1, 1)


def psnr(a, b):
    return round((10 * torch.log10(4 / ((a - b) ** 2).mean())).item(), 1)


with torch.no_grad():
    a = dec(zt)
    b = dec(zt.flip(-1))
    zf = enc(x.flip(-1).to(torch.bfloat16)).float()
    e = dec(zf)
xf = x.flip(-1)
print("round trip      decode(z) vs PNG                  PSNR", psnr(a, x))
print("latent flip     decode(flip z) vs flip(PNG)       PSNR", psnr(b, xf), " <- current training augmentation")
print("image flip      decode(enc(flip PNG)) vs flip(PNG) PSNR", psnr(e, xf))
print("latent mse flip(z) vs enc(flip PNG)", round(((zt.flip(-1) - zf) ** 2).mean().item(), 4), " latent var", round(zt.var().item(), 3))
err = ((b - xf) ** 2).mean(1)
print("latent-flip error: whole", round(err.mean().item(), 4), " inner (32 px border removed)",
      round(err[:, 32:-32, 32:-32].mean().item(), 4), " round-trip error", round(((a - x) ** 2).mean().item(), 4))
t8 = lambda z: ((z.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
out = []
for i in [0, 7, 21, 40]:
    em = ((b[i] - xf[i]).abs().mean(0, keepdim=True) * 4 - 1).expand(3, -1, -1)
    out.append(np.concatenate([t8(xf[i]), t8(b[i]), t8(e[i]), t8(em)], 1))
o = "grids/flip_aug_check" + ("_sd" if SD else "") + ".jpg"
Image.fromarray(np.concatenate(out, 0)).save(ROOT / o, quality=90)
print("->", o)
