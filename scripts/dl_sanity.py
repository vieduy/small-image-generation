"""Sanity check of scripts/dl_arch.py conventions against the official DreamLiteMobilePipeline (512 px).
Out: grids/dl_sanity.jpg  rows = photos; cols = photo | official 4-step | ours 1-step (real prompt embeds) | ours 1-step (avg tokens)"""
import sys
from pathlib import Path
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402
import i2i_server as S  # noqa: E402
import teacher_prompt as TP  # noqa: E402
from train_student import to_t  # noqa: E402
from diffusers import DreamLiteMobilePipeline  # noqa: E402
ROOT = Path(__file__).resolve().parent.parent
DEV = "cuda"
pipe = DreamLiteMobilePipeline.from_pretrained(DLA.DL_REPO, revision=DLA.DL_REV, torch_dtype=torch.bfloat16).to(DEV)
st = S.load_styles()["cartoon3d"]
inst = TP.avatar_prompt(st["description"], None)
short = "been redrawn as a 3D cartoon avatar portrait with a stylized rounded face, large expressive eyes and smooth soft shading"
photos = [ROOT / "data/faces_eval2" / n for n in ("fe2_020.png", "fe2_077.png")]
tok = []
for ph in photos:
    img = Image.open(ph).convert("RGB").resize((512, 512))
    off = pipe(prompt=short, image=img, num_inference_steps=4, height=512, width=512,
               generator=torch.Generator(DEV).manual_seed(0)).images[0]
    p = ("[Edit]: A diptych with two side-by-side images of the same scene. Compared to the right side, the left one has " + short)
    e, m = pipe.encode_prompt(mode="edit", prompts=[p], image=img, device=DEV, dtype=torch.bfloat16)
    with torch.inference_mode():
        z = DLA.encode(pipe.vae, to_t(img)[None].to(DEV, torch.bfloat16))
        one = DLA.decode(pipe.vae, DLA.forward(pipe.unet, z, e, m))[0]
    tok.append((img, off, one))
    print("[sanity]", ph.name, "embeds", tuple(e.shape), flush=True)
tf = ROOT / "weights/student/_dl_tokens_cartoon3d_22.pt"
if tf.exists():
    t = torch.load(tf)
    with torch.inference_mode():
        for k, (img, off, one) in enumerate(tok):
            z = DLA.encode(pipe.vae, to_t(img)[None].to(DEV, torch.bfloat16))
            y = DLA.decode(pipe.vae, DLA.forward(pipe.unet, z, t["emb"][0:1].to(DEV, torch.bfloat16), t["mask"][0:1].to(DEV)))[0]
            tok[k] = (img, off, one, y)
pil = lambda y: Image.fromarray(((y.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy())
S2 = 384
sheet = Image.new("RGB", (S2 * 4, S2 * len(tok)), "white")
for r, row in enumerate(tok):
    for j, im in enumerate(row):
        im = im if isinstance(im, Image.Image) else pil(im)
        sheet.paste(im.resize((S2, S2)), (j * S2, r * S2))
sheet.save(ROOT / "grids/dl_sanity.jpg", quality=90)
print("SANITY_DONE", flush=True)
