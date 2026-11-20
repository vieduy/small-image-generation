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
    unet = DreamLiteUNetModel.from_pretrained(path, torch_dtype=dtype)
    from pathlib import Path
    sc = Path(path).parent / "shared_ctx.pt"
    if sc.exists():  # shared-trunk student (idea B): down/mid cross-attention reads one learned context, not the task token
        c = torch.load(sc)
        install_shared_trunk(unet, c["emb"].to(dtype), c["mask"], scope=c.get("scope", "down_mid"))
    pr = Path(path).parent / "prune.json"
    if pr.exists():  # Plan A smaller student: these residual units are removed
        import json
        install_prune(unet, json.loads(pr.read_text())["units"])
    da = Path(path).parent / "drop_attn2.json"
    if da.exists():  # #5: cross-attention modules removed where the option has no effect
        import json
        install_drop_attn2(unet, json.loads(da.read_text())["groups"])
    hh = Path(path).parent / "half_head.json"
    if hh.exists():  # half-head student: up_blocks.1 onward computes only the output half of the diptych (+ margin)
        import json
        install_half_head(unet, json.loads(hh.read_text())["margin"])
    return unet


class _SharedCtxProc:
    """Wraps a cross-attention processor: the given context/mask are replaced by the UNet's shared context (idea B, 2026-11-19).
    With it, every down/mid block is independent of the menu option, so on the phone the down path + mid block (42% of UNet
    FLOPs) run once per photo and only the up path runs per sticker."""

    def __init__(self, inner, unet):
        self.inner, self.unet = inner, [unet]  # list: keep the UNet out of nn.Module attribute registration

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kw):
        if encoder_hidden_states is not None:
            u = self.unet[0]
            B = hidden_states.shape[0]
            encoder_hidden_states = u._shared_proj.expand(B, -1, -1)
            attention_mask = u._shared_bias.expand(B, -1, -1)
        return self.inner(attn, hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, **kw)


def install_shared_trunk(unet, emb, mask, trainable=False, scope="down_mid"):
    """emb [1, L, 2048] (raw token space, projected by the UNet's own encoder_hid_proj), mask [1, L] (1 = token).
    Registered on the UNet (moves with .to(); a Parameter if trainable). Same additive mask convention as the UNet (-10000)."""
    if trainable:
        unet.register_parameter("shared_emb", torch.nn.Parameter(emb.detach().clone().float()))
    else:
        unet.register_buffer("shared_emb", emb.detach().clone(), persistent=False)
    unet.register_buffer("shared_mask", mask.detach().clone().float(), persistent=False)
    n = 0
    for name, mod in unet.named_modules():
        if name.endswith("attn2") and (name.startswith("down_blocks") or (scope == "down_mid" and name.startswith("mid_block"))):
            mod.set_processor(_SharedCtxProc(mod.processor, unet))
            n += 1

    def pre(module, args, kwargs):
        ehs = kwargs["encoder_hidden_states"]
        e = module.shared_emb.to(ehs.dtype)
        module._shared_proj = module.process_encoder_hidden_states(e, kwargs.get("added_cond_kwargs"))
        module._shared_bias = ((1 - module.shared_mask.to(module._shared_proj.dtype)) * -10000.0).unsqueeze(1)

    unet.register_forward_pre_hook(pre, with_kwargs=True)
    unet._shared_trunk_modules = n
    unet._shared_scope = scope
    return n


class _SkipLayer(torch.nn.Module):
    """A removed transformer layer (residual unit): returns its input."""

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


class _SkipAttention(torch.nn.Module):
    """A removed attention module (Transformer2D, residual): returns (input,), as called with return_dict=False."""

    def forward(self, hidden_states, *args, **kwargs):
        return (hidden_states,)


def install_prune(unet, units):
    """Plan A (2026-11-20): replace residual units (names from scripts/block_scan.py) by identity modules; their weights are
    dropped (smaller model). units: '...attentions.i' (whole attention module) or '...attentions.i.transformer_blocks.j'."""
    for n in sorted(units, key=lambda x: -x.count(".")):  # layers before whole modules
        parent, idx = n.rsplit(".", 1)
        lst = unet.get_submodule(parent)
        lst[int(idx)] = _SkipLayer() if ".transformer_blocks." in n else _SkipAttention()
    unet._prune = sorted(units)
    return len(units)


class _ZeroAttn:
    """#5 (2026-11-20): a removed cross-attention. The transformer block adds attn2's output to the residual stream, so a zero
    output = the module is gone (an export drops it). Used where the influence map shows the option has no effect."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kw):
        return torch.zeros_like(hidden_states)


def install_drop_attn2(unet, groups):
    """groups: list of 'down0', 'down1', 'down2', 'mid', 'up0', 'up1' (or 'down' = all down blocks)."""
    def group(name):
        p = name.split(".")
        return {"down_blocks": f"down{p[1]}", "up_blocks": f"up{p[1]}", "mid_block": "mid"}[p[0]]
    want = set(groups) | ({"down0", "down1", "down2"} if "down" in groups else set())
    n = 0
    for name, mod in unet.named_modules():
        if name.endswith("attn2") and group(name) in want:
            mod.set_processor(_ZeroAttn())
            n += 1
    unet._drop_attn2 = sorted(groups)
    return n


def install_half_head(unet, margin=0, start=1):
    """Idea B, part 2 (2026-11-19): drop the photo half of the diptych in the per-option head.
    up_blocks.<start> (no self-attention) and later blocks only mix the width axis through 3x3 convs and GroupNorm statistics,
    so from there on we compute only the left (output) half plus `margin` columns of the photo half (at that block's input
    resolution). Inputs and skip features are cropped by forward pre-hooks; the UNet output is then LAT + 2*margin wide and
    _step keeps the first LAT columns. Not exact (GroupNorm statistics and conv padding at the cut): see scripts/halfhead_check.py;
    a student is fine-tuned with it on (train_dl.py --half-head M)."""
    ups = list(unet.up_blocks)
    for b in ups[start:start + 1]:
        assert not any(n.endswith("attn1") for n, _ in b.named_modules()), "half head: block has self-attention"

    def crop(t, wk):
        return t[..., :wk].contiguous()

    def first(module, args, kwargs):
        h = kwargs["hidden_states"]
        wk = h.shape[-1] // 2 + margin
        kwargs["hidden_states"] = crop(h, wk)
        kwargs["res_hidden_states_tuple"] = tuple(crop(r, wk) for r in kwargs["res_hidden_states_tuple"])
        return args, kwargs

    def later(module, args, kwargs):
        wk = kwargs["hidden_states"].shape[-1]
        kwargs["res_hidden_states_tuple"] = tuple(crop(r, wk) for r in kwargs["res_hidden_states_tuple"])
        return args, kwargs

    hs = [ups[start].register_forward_pre_hook(first, with_kwargs=True)]
    hs += [b.register_forward_pre_hook(later, with_kwargs=True) for b in ups[start + 1:]]
    unet._half_head = {"margin": margin, "start": start, "hooks": hs}
    return unet


def remove_half_head(unet):
    for h in unet._half_head["hooks"]:
        h.remove()
    del unet._half_head


def load_vae(dtype=torch.float32, path=None):  # path: a fine-tuned copy (scripts/train_decoder.py), default = DreamLite's TinyVAE
    from diffusers import AutoencoderTiny
    if path is not None:
        return AutoencoderTiny.from_pretrained(path, torch_dtype=dtype)
    return AutoencoderTiny.from_pretrained(DL_REPO, subfolder="vae", revision=DL_REV, torch_dtype=dtype)


def encode(vae, x):  # x in [-1, 1] -> latent (AutoencoderTiny: no scaling factor)
    return vae.encode(x).latents


def decode(vae, z):  # -> [-1, 1]
    return vae.decode(z).sample


_NOISE = {}


def fixed_noise(device, dtype=torch.float32, seed=1234):
    k = (str(device), dtype, seed)
    if k not in _NOISE:
        g = torch.Generator("cpu").manual_seed(seed)
        _NOISE[k] = torch.randn(1, 4, LAT, LAT, generator=g).to(device, dtype)
    return _NOISE[k]


def _step(unet, x, z_photo, emb, mask, sigma):
    """One denoising step at noise level sigma: [x_sigma | photo latent] -> x0 estimate (left panel)."""
    B = z_photo.shape[0]
    dt = z_photo.dtype
    inp = torch.cat([x, z_photo], dim=3)
    t = torch.full((B,), 1000.0 * sigma, device=z_photo.device, dtype=dt)
    tid = torch.tensor([[RES, RES]], device=z_photo.device, dtype=dt).expand(B, 2)
    v = unet(inp, timestep=t, encoder_hidden_states=emb.to(dt), encoder_attention_mask=mask,
             added_cond_kwargs={"time_ids": tid}, return_dict=False)[0][..., :z_photo.shape[-1]]
    if v.shape[1] > x.shape[1]:  # wide-latent student (idea A): channels 4.. are extra decoder inputs, not flow-matching velocity
        assert sigma == 1.0, "wide latent: one-step only"
        return torch.cat([x - v[:, :x.shape[1]], v[:, x.shape[1]:]], 1)
    return x - v if sigma == 1.0 else x - sigma * v


def refine_forward(unet, peft_model, z_photo, emb, mask, sigma, first=None):
    """3a two-step avatar (2026-11-20): step 1 = the one-step student with the LoRA adapter OFF (exactly the shipped model),
    re-noise x = (1-s)*x0 + s*eps (eps = fixed_noise seed 1235, a phone constant), step 2 = UNet WITH the adapter at sigma s.
    peft_model: the PeftModel whose adapter is toggled (unet may be the same object). first: precomputed step-1 x0 (optional)."""
    dt = z_photo.dtype
    if first is None:
        with torch.no_grad(), peft_model.disable_adapter():
            first = forward(unet, z_photo, emb, mask)
    x = ((1 - sigma) * first + sigma * fixed_noise(z_photo.device, dt, 1235).expand(z_photo.shape[0], -1, -1, -1)).to(dt)
    return _step(unet, x, z_photo, emb, mask, sigma)


def forward(unet, z_photo, emb, mask=None, noise=None, sigmas=None, k=None):
    """Student: [fixed noise | photo latent] -> x0 (left panel). emb: [B, L, 2048]; mask: [B, L] or None.
    sigmas None (or [1.0]) = the one-step student, unchanged. Multi-step (2026-11-18, e.g. sigmas=[1.0, 0.5]): the x0 of
    step j is re-noised with a FIXED per-step noise (seed 1234+j, a phone constant) to sigma_j, x = (1-s)*x0 + s*eps_j,
    and denoised again with the photo latent beside it (the photo conditions every step). k = run only the first k steps
    (training: DMD2 backward simulation; earlier steps run without gradient when gradients are enabled)."""
    B = z_photo.shape[0]
    dt = z_photo.dtype
    S = list(sigmas) if sigmas else [1.0]
    k = len(S) if k is None else k
    x = (noise if noise is not None else fixed_noise(z_photo.device, dt)).expand(B, -1, -1, -1).to(dt)
    x0 = None
    for j in range(k):
        if j:
            e = fixed_noise(z_photo.device, dt, 1234 + j).expand(B, -1, -1, -1)
            x = ((1 - S[j]) * x0 + S[j] * e).to(dt)
        if j < k - 1 and torch.is_grad_enabled():
            with torch.no_grad():
                x0 = _step(unet, x, z_photo, emb, mask, S[j]).detach()
        else:
            x0 = _step(unet, x, z_photo, emb, mask, S[j])
    return x0
