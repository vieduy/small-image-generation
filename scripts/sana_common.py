"""Sana-Sprint phone student, shared pieces (Plan B2/B3, 2026-11-20).

One step from a FIXED noise, matching Sana-Sprint's own one-step math (diffusers SanaSprintPipeline + SCMScheduler, trigflow,
sigma_data 0.5, max_timesteps 1.5708): at t = pi/2 the model input is eps ~ N(0,1), scm_timestep = 1, guidance = 4.5 * 0.1 and
   x0 (scaled latent) = eps - F(eps)        [pipeline: latents = eps - F, decoded as vae.decode(latents / scaling_factor)]
The student adds the photo latent (DC-AE latent x scaling_factor, the space the transformer works in):
   cond = "diptych": F([eps | z_photo] along width)[..., :W]          (DreamLite-style in-context pair, 2x tokens)
   cond = "concat" : F(cat([eps, z_photo], channels)), patch-embed widened 32 -> 64 input channels (new weights zero)
Option tokens: learned [n_tasks, 300, 2304] + mask (replace Gemma-2-2B; initialised from Gemma encodings of each option's prompt).
"""
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SNAP = Path("/mnt/project/kchat/duyhv/sana/hf_cache/hub/models--Efficient-Large-Model--Sana_Sprint_0.6B_1024px_diffusers/snapshots/aa76e7f4f4928f378716b6716a2130fba3caf5b1")
LAT = 16  # 512 px / 32
GUIDANCE = 4.5


def load_transformer(path=None, cond="diptych", dtype=torch.float32):
    from diffusers import SanaTransformer2DModel
    tr = SanaTransformer2DModel.from_pretrained(path or (SNAP / "transformer"), torch_dtype=dtype)
    if cond == "concat" and tr.config.in_channels == 32:  # widen the patch embedding: photo channels start at zero
        pe = tr.patch_embed.proj
        new = torch.nn.Conv2d(64, pe.out_channels, pe.kernel_size, pe.stride, pe.padding, bias=pe.bias is not None).to(pe.weight.dtype)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :32] = pe.weight
            if pe.bias is not None:
                new.bias.copy_(pe.bias)
        tr.patch_embed.proj = new
        tr.register_to_config(in_channels=64)
    return tr


def fixed_noise(device, dtype=torch.float32, seed=1234):
    g = torch.Generator("cpu").manual_seed(seed)
    return torch.randn(1, 32, LAT, LAT, generator=g).to(device, dtype)


def forward(tr, z_photo, emb, mask, cond, noise=None):
    """z_photo [B,32,16,16] (scaled latent) -> x0 [B,32,16,16] (scaled latent)."""
    B, dt = z_photo.shape[0], z_photo.dtype
    eps = (noise if noise is not None else fixed_noise(z_photo.device, dt)).expand(B, -1, -1, -1).to(dt)
    x = torch.cat([eps, z_photo], 3 if cond == "diptych" else 1)
    t = torch.ones(B, device=z_photo.device, dtype=dt)
    g = torch.full((B,), GUIDANCE * tr.config.guidance_embeds_scale, device=z_photo.device, dtype=dt)
    F = tr(x, encoder_hidden_states=emb.to(dt), encoder_attention_mask=mask, timestep=t, guidance=g, return_dict=False)[0]
    if cond == "diptych":
        F = F[..., :LAT]
    return eps - F


def load_dcae(dtype=torch.bfloat16):
    from diffusers import AutoencoderDC
    return AutoencoderDC.from_pretrained(SNAP / "vae", torch_dtype=dtype)


def load_tiny(tag, step="final"):
    from diffusers import AutoencoderTiny
    d = ROOT / "weights/tiny_dcae" / tag
    p = d / step if (d / step).exists() else sorted(d.glob("step_*"))[-1]
    return AutoencoderTiny.from_pretrained(p), p.name


def option_prompts(style, tasks):
    """Text per option for the token init (configs/prompts.yaml 'default' family + styles/<style>/style.yaml + expression phrase)."""
    import yaml
    P = yaml.safe_load((ROOT / "configs/prompts.yaml").read_text())["default"]
    st = yaml.safe_load((ROOT / "styles" / style / "style.yaml").read_text())
    desc = st.get("description") or st.get("desc")
    EX = {e["id"]: e["phrase"] for e in yaml.safe_load((ROOT / "configs/sticker_expressions.yaml").read_text())}
    out = []
    for t in tasks:
        if t in ("avatar", "sticker"):
            out.append(P[t].format(desc=desc, short=st.get("short", "")))
        else:
            out.append(P["sticker"].format(desc=desc, short=st.get("short", "")) + " Expression: " + EX[t[len("expr_"):]] + ".")
    return out


@torch.no_grad()
def init_tokens(style, tasks, device="cuda"):
    """Gemma-2-2B encodings of each option's prompt through Sana-Sprint's own encode_prompt (complex human instruction, 300 tokens)."""
    from diffusers import SanaSprintPipeline
    pipe = SanaSprintPipeline.from_pretrained(SNAP, transformer=None, vae=None, torch_dtype=torch.bfloat16).to(device)
    embs, masks = [], []
    for p in option_prompts(style, tasks):
        e, m = pipe.encode_prompt(p, device=device)[:2]
        embs.append(e.float().cpu()); masks.append(m.cpu())
    del pipe
    torch.cuda.empty_cache()
    return torch.cat(embs), torch.cat(masks)


def save_student(tr, path):
    """fp16 copy of the weights (master weights untouched) + config."""
    from safetensors.torch import save_file
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    tr.save_config(path)
    save_file({k: v.detach().to(torch.float16).contiguous().cpu() for k, v in tr.state_dict().items()}, str(path / "model_fp16.safetensors"))


def load_student(path, dtype=torch.float16):
    from diffusers import SanaTransformer2DModel
    from safetensors.torch import load_file
    path = Path(path)
    tr = SanaTransformer2DModel.from_config(SanaTransformer2DModel.load_config(path))
    tr.load_state_dict(load_file(str(path / "model_fp16.safetensors")))
    return tr.to(dtype)


def save_meta(out, meta):
    (Path(out) / "sana_student.json").write_text(json.dumps(meta, indent=1))
