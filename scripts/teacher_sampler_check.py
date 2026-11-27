"""2026-11-23: teacher_dl.sample (28 steps, 3-pass edit CFG, shifted sigmas) must reproduce the official DreamLitePipeline (edit,
512 px) with the same noise and the same Qwen3-VL encodings. Prints max |diff| of the decoded images (0..255) + latent diff."""
import sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA, teacher_dl as TD  # noqa: E402
from diffusers import DreamLitePipeline
DEV = "cuda"
ROOT = TD.ROOT
pipe = DreamLitePipeline.from_pretrained(TD.BASE_REPO, revision=TD.BASE_REV, torch_dtype=torch.bfloat16).to(DEV)
pipe.set_progress_bar_config(disable=True)
img = Image.open(sorted((ROOT / "data/faces_eval2").glob("*.png"))[20]).convert("RGB").resize((512, 512))
prompt = "been redrawn as a 3D cartoon avatar portrait with a stylized rounded face, large expressive eyes and smooth soft shading"
noise = torch.randn(1, 4, 64, 64, generator=torch.Generator().manual_seed(7)).to(DEV, torch.bfloat16)
pipe.prepare_latents = lambda *a, **k: noise.clone()
cap = {}
dec0 = pipe.vae.decode
def grab(z, *a, **k):
    cap["z"] = z.clone(); return dec0(z, *a, **k)
pipe.vae.decode = grab
ref = np.asarray(pipe(prompt=prompt, image=img, height=512, width=512, num_inference_steps=28, guidance_scale=2.0, image_guidance_scale=2.0).images[0]).astype(np.float32)
ps = (f"[Edit]: A diptych with two side-by-side images of the same scene. Compared to the right side, the left one has {prompt}")
e, m = pipe.encode_prompt(mode="edit", prompts=["", "", ps], image=img, device=DEV, dtype=torch.bfloat16)
zp = DLA.encode(pipe.vae, (torch.from_numpy(np.asarray(img)).permute(2, 0, 1)[None].float() / 127.5 - 1).to(DEV, torch.bfloat16))
zp_pipe = pipe.prepare_image_latents(pipe.image_processor.preprocess(img, height=512, width=512), dtype=torch.bfloat16, device=DEV)
print("photo latent: ours vs pipeline max diff", float((zp - zp_pipe).abs().max()))
with torch.inference_mode():
    z = TD.sample(pipe.unet, zp_pipe, e[2:3], m[2:3], e[0:1], m[0:1], g_img=2.0, g_txt=2.0, steps=28, noise=noise.float())
    out = ((DLA.decode(pipe.vae, z.to(torch.bfloat16))[0].float().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).cpu().numpy()
print("final latent: max diff", float((z.float() - cap["z"].float()).abs().max()), "| latent std", float(cap["z"].float().std()))
print("decoded image 0..255: mean |diff| %.2f, max %.1f" % (np.abs(out - ref).mean(), np.abs(out - ref).max()))
