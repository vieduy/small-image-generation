"""DreamLite-mobile as a one-step, text-free phone student (bake-off vs SDXS; user request 2026-11-17).

DreamLite (ByteDance, 2026; arXiv 2603.28713; weights carlofkl/DreamLite-mobile, branch "diffusers", CC BY-NC 4.0):
390M mobile UNet (SnapGen-style compressed SDXL), TinyVAE (AutoencoderTiny, 4ch, f8), flow matching, 4-step DMD2.
Native edit = in-context diptych along WIDTH: unet(cat([x_t (target), z_cond (source)], dim=3)), velocity output,
left half kept; time_ids = [[W, H]]; text = Qwen3-VL-2B hidden states (dim 2048, projected inside the UNet).

Our student (same contract as the SDXS student):
  * text-free: learned task tokens [n_tasks, L, 2048] (+ fixed attention mask) replace Qwen3-VL. They are initialised from
    Qwen3-VL edit-mode embeddings of each task's instruction (mean over photos), see init_tokens.py.
    DEVIATION vs the paper: the VL encoder also sees the source photo; on the phone we only keep the latent diptych.
  * one step at sigma=1 from a FIXED noise latent (seed 1234, baked into the phone model as a constant):
        v = unet([noise | z_photo], t=1000);  x0 = noise - v      (flow matching: x_t=(1-s)x0+s*eps, v=eps-x0)
  * decode with DreamLite's TinyVAE (the phone decoder).
"""
import torch

DL_REPO = "carlofkl/DreamLite-mobile"
DL_REV = "diffusers"
RES = 512
LAT = RES // 8  # 64


def load_unet(path=None, dtype=torch.float32):
    from diffusers import DreamLiteUNetModel
    if path is None:
        return DreamLiteUNetModel.from_pretrained(DL_REPO, subfolder="unet", revision=DL_REV, torch_dtype=dtype)
    return DreamLiteUNetModel.from_pretrained(path, torch_dtype=dtype)


def load_vae(dtype=torch.float32):
    from diffusers import AutoencoderTiny
    return AutoencoderTiny.from_pretrained(DL_REPO, subfolder="vae", revision=DL_REV, torch_dtype=dtype)


def encode(vae, x):  # x in [-1, 1] -> latent (AutoencoderTiny: no scaling factor)
    return vae.encode(x).latents


def decode(vae, z):  # -> [-1, 1]
    return vae.decode(z).sample


_NOISE = {}


def fixed_noise(device, dtype=torch.float32):
    k = (str(device), dtype)
    if k not in _NOISE:
        g = torch.Generator("cpu").manual_seed(1234)
        _NOISE[k] = torch.randn(1, 4, LAT, LAT, generator=g).to(device, dtype)
    return _NOISE[k]


def forward(unet, z_photo, emb, mask=None, noise=None):
    """One-step student: [fixed noise | photo latent] -> x0 (left panel). emb: [B, L, 2048]; mask: [B, L] or None."""
    B = z_photo.shape[0]
    dt = z_photo.dtype
    x = (noise if noise is not None else fixed_noise(z_photo.device, dt)).expand(B, -1, -1, -1).to(dt)
    inp = torch.cat([x, z_photo], dim=3)
    t = torch.full((B,), 1000.0, device=z_photo.device, dtype=dt)
    tid = torch.tensor([[RES, RES]], device=z_photo.device, dtype=dt).expand(B, 2)
    v = unet(inp, timestep=t, encoder_hidden_states=emb.to(dt), encoder_attention_mask=mask,
             added_cond_kwargs={"time_ids": tid}, return_dict=False)[0][..., :z_photo.shape[-1]]
    return x - v
