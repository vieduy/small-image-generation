"""Multi-step DreamLite TEACHER (2026-11-23, user: fine-tune DreamLite-base on klein, then distil to the 1-step student).

Conventions copied from diffusers DreamLitePipeline (edit mode) at 512 px:
  * input  = [x_sigma | photo latent] along width (64 x 128 latents), timestep = 1000 * sigma, time_ids = [512, 512]
  * flow matching: x_sigma = (1 - sigma) * x0 + sigma * eps, velocity v = eps - x0 (left panel of the UNet output)
  * sampling: sigmas = linspace(1, 1/N, N), exponential time shift mu = calculate_shift(image_seq_len = 64*64/4 = 1024,
    256, 4096, 0.5, 1.15) ~ 0.630  ->  s' = e^mu / (e^mu + 1/s - 1), Euler to 0
  * edit CFG, 3 UNet passes per step:  uncond = [zero photo latent, null tokens], image = [photo, null], text = [photo, option]
    v = v_u + g_txt * (v_t - v_i) + g_img * (v_i - v_u)
Option tokens replace Qwen3-VL (as in the student): [n_tasks, L, 2048] + mask; null tokens = Qwen3-VL edit-mode encoding of
the empty prompt (the pipeline's negative prompt) averaged over 16 photos, learned further.
"""
import math
from pathlib import Path

import torch

import dl_arch as DLA

ROOT = Path(__file__).resolve().parent.parent
BASE_REPO, BASE_REV = "carlofkl/DreamLite-base", "diffusers"
MU = None


def mu():
    global MU
    if MU is None:
        m = (1.15 - 0.5) / (4096 - 256)
        MU = 1024 * m + (0.5 - m * 256)
    return MU


def shift(s):
    """Exponential time shift used by DreamLite at 512 px (s in (0, 1])."""
    e = math.exp(mu())
    return e / (e + (1.0 / s - 1.0)) if s > 0 else 0.0


def train_sigmas(n, device):
    """Training noise levels: logit-normal(0, 1) (SD3 / flow-matching default), then the same shift as sampling."""
    u = torch.sigmoid(torch.randn(n, device=device))
    e = math.exp(mu())
    return e / (e + (1.0 / u.clamp(1e-4, 1.0) - 1.0))


def velocity(unet, x, z_photo, emb, mask, sigma):
    """sigma: tensor [B] in (0, 1]. Returns v for the left panel."""
    B, dt = z_photo.shape[0], z_photo.dtype
    inp = torch.cat([x.to(dt), z_photo], dim=3)
    t = (1000.0 * sigma).to(dt)
    tid = torch.tensor([[DLA.RES, DLA.RES]], device=z_photo.device, dtype=dt).expand(B, 2)
    return unet(inp, timestep=t, encoder_hidden_states=emb.to(dt), encoder_attention_mask=mask,
                added_cond_kwargs={"time_ids": tid}, return_dict=False)[0][..., :z_photo.shape[-1]]


@torch.no_grad()
def sample(unet, z_photo, emb, mask, null_emb, null_mask, g_img=2.0, g_txt=2.0, steps=28, noise=None, generator=None):
    """28-step Euler with DreamLite's 3-pass edit CFG. emb/mask: [B, L, 2048]/[B, L]; null_*: [1, Ln, 2048]/[1, Ln]."""
    B, dev, dt = z_photo.shape[0], z_photo.device, z_photo.dtype
    x = noise.expand(B, -1, -1, -1).to(dev, torch.float32) if noise is not None else \
        torch.randn(B, 4, DLA.LAT, DLA.LAT, generator=generator, device="cpu").to(dev)
    L = max(emb.shape[1], null_emb.shape[1])
    pad = lambda e, m: (torch.nn.functional.pad(e, (0, 0, 0, L - e.shape[1])), torch.nn.functional.pad(m, (0, L - m.shape[1])))
    e_t, m_t = pad(emb, mask)
    e_n, m_n = pad(null_emb.expand(B, -1, -1), null_mask.expand(B, -1))
    zp3 = torch.cat([torch.zeros_like(z_photo), z_photo, z_photo])
    e3, m3 = torch.cat([e_n, e_n, e_t]), torch.cat([m_n, m_n, m_t])
    S = [shift(s) for s in torch.linspace(1.0, 1.0 / steps, steps).tolist()] + [0.0]
    for i in range(steps):
        s, s_next = S[i], S[i + 1]
        v_u, v_i, v_t = velocity(unet, torch.cat([x] * 3), zp3, e3, m3, torch.full((3 * B,), s, device=dev)).float().chunk(3)
        v = v_u + g_txt * (v_t - v_i) + g_img * (v_i - v_u)
        x = x + (s_next - s) * v
    return x.to(dt)


@torch.no_grad()
def null_tokens(photos, device="cuda"):
    """Edit-mode Qwen3-VL encoding of the empty prompt (= the pipeline's negative prompt), averaged over photos."""
    from diffusers import DreamLitePipeline
    from PIL import Image
    pipe = DreamLitePipeline.from_pretrained(BASE_REPO, revision=BASE_REV, unet=None, vae=None, torch_dtype=torch.bfloat16).to(device)
    acc = None
    for ph in photos:
        e, m = pipe.encode_prompt(mode="edit", prompts=[""], image=Image.open(ph).convert("RGB").resize((512, 512)),
                                  device=device, dtype=torch.bfloat16)
        acc = e.float() if acc is None else acc + e.float()
    out = (acc / len(photos)).cpu(), m.cpu()
    del pipe
    torch.cuda.empty_cache()
    return out


def load_teacher(tag, ckpt="final", dtype=torch.float16):
    """weights/teacher/<tag>/<ckpt>/{unet (EMA), tokens.pt}."""
    from diffusers import DreamLiteUNetModel
    d = ROOT / "weights/teacher" / tag / ckpt
    unet = DreamLiteUNetModel.from_pretrained(d / "unet", torch_dtype=dtype)
    return unet, torch.load(d / "tokens.pt")
