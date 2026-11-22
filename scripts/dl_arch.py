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


def load_unet(path=None, dtype=torch.float32, style_film=True):
    from diffusers import DreamLiteUNetModel
    if path is None:
        return DreamLiteUNetModel.from_pretrained(DL_REPO, subfolder="unet", revision=DL_REV, torch_dtype=dtype)
    unet = DreamLiteUNetModel.from_pretrained(path, torch_dtype=dtype)
    from pathlib import Path
    sc = Path(path).parent / "shared_ctx.pt"
    if sc.exists():  # shared-trunk student (idea B): down/mid cross-attention reads one learned context, not the task token
        c = torch.load(sc)
        install_shared_trunk(unet, c["emb"].to(dtype), c["mask"], scope=c.get("scope", "down_mid"))
    sf = Path(path).parent / "style_film.pt"
    if sf.exists() and style_film:  # one-model multi-style (dl8): per-style FiLM; a view dir fixes the style (film_style.json, default 0)
        import json
        d = torch.load(sf)
        fs = Path(path).parent / "film_style.json"
        install_style_film(unet, d["n_styles"], film=d["film"], fixed_style=json.loads(fs.read_text())["style"] if fs.exists() else 0)
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


SHARED_SCOPES = {  # which cross-attention modules read the shared context instead of the option token
    "down": lambda n: n.startswith("down_blocks"),
    "down_mid": lambda n: n.startswith(("down_blocks", "mid_block")),
    # paper P2 (2026-11-20): option enters only after the last self-attention (up_blocks.1: 6 modules)
    "down_mid_up0": lambda n: n.startswith(("down_blocks", "mid_block", "up_blocks.0.")),
    # paper P2c control: option only in up_blocks.0.attentions.2 (4 modules, self-attention still follows inside it)
    "only_up0a2": lambda n: not n.startswith("up_blocks.0.attentions.2."),
    # E1 (2026-11-20, reports/e1_status.md section 5; user run names in brackets). True = reads the shared (neutral) context.
    # (a) 4 option inputs, entry moved:
    "only_mid": lambda n: not n.startswith("mid_block."),                                    # [mid4]
    "only_up0a0": lambda n: not n.startswith("up_blocks.0.attentions.0."),                   # [up0b1_4 = up0start_4]
    "only_up0a1": lambda n: not n.startswith("up_blocks.0.attentions.1."),                   # [up0b2_4]
    "only_up1a01": lambda n: not n.startswith(("up_blocks.1.attentions.0.", "up_blocks.1.attentions.1.")),  # [up1_4]
    # (b) entry fixed, number of option inputs varied:
    "only_up0a2_tb0": lambda n: n != "up_blocks.0.attentions.2.transformer_blocks.0.attn2",  # [last1]
    "only_up0a2_tb01": lambda n: not n.startswith(("up_blocks.0.attentions.2.transformer_blocks.0.",
                                                   "up_blocks.0.attentions.2.transformer_blocks.1.")),  # [last2]
    "up0a2_plus_up1": lambda n: not n.startswith(("up_blocks.0.attentions.2.", "up_blocks.1.")),  # [last10]
    "only_up0": lambda n: not n.startswith("up_blocks.0."),                                  # [up0start_12]
}


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
        if name.endswith("attn2") and SHARED_SCOPES[scope](name):
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


def _film_hook(owner, key, fixed):
    def hook(mod, inp, out):
        s = owner._film_style if fixed is None else fixed
        g, b = owner.style_film[key + "__g"][s], owner.style_film[key + "__b"][s]
        shp = (-1, out.shape[1], 1, 1) if g.dim() == 2 else (1, -1, 1, 1)
        return out * (1 + g.to(out.dtype).view(shp)) + b.to(out.dtype).view(shp)
    return hook


def install_style_film(unet, n_styles, film=None, fixed_style=None, trainable=False):
    """One-model multi-style (2026-11-21, dl8): per-style FiLM (conditional normalisation, Dumoulin et al. 2017) after every
    GroupNorm of mid_block, up_blocks and conv_norm_out: y = GN(x) * (1 + g[style]) + b[style]; g = b = 0 at init (= the base
    model). Training sets unet._film_style = LongTensor [B]; a saved view fixes one style. ~KB per style."""
    mods = [(n, m) for n, m in unet.named_modules() if isinstance(m, torch.nn.GroupNorm)
            and (n.startswith(("mid_block", "up_blocks")) or n == "conv_norm_out")]
    P = torch.nn.ParameterDict()
    for n, m in mods:
        k = n.replace(".", "__")
        g = film[n][0].clone() if film else torch.zeros(n_styles, m.num_channels)
        b = film[n][1].clone() if film else torch.zeros(n_styles, m.num_channels)
        P[k + "__g"] = torch.nn.Parameter(g.float(), requires_grad=trainable)
        P[k + "__b"] = torch.nn.Parameter(b.float(), requires_grad=trainable)
    unet.style_film = P
    unet._film_style = None
    unet._film_names = [n for n, _ in mods]
    unet._film_hooks = [m.register_forward_hook(_film_hook(unet, n.replace(".", "__"), fixed_style)) for n, m in mods]
    return len(mods), sum(m.num_channels for _, m in mods) * 2 * n_styles


def remove_style_film(unet):
    for h in unet._film_hooks:
        h.remove()
    del unet.style_film
    for k in ("_film_hooks", "_film_names", "_film_style"):
        if hasattr(unet, k):
            delattr(unet, k)


def film_state(unet):
    return {n: (unet.style_film[n.replace(".", "__") + "__g"].detach().cpu(), unet.style_film[n.replace(".", "__") + "__b"].detach().cpu())
            for n in unet._film_names}


def install_decoder_film(vae, n_styles, film=None, fixed_style=None, trainable=False):
    """Same FiLM for the TinyVAE decoder (no norms there): after every AutoencoderTinyBlock of the decoder."""
    from diffusers.models.autoencoders.vae import AutoencoderTinyBlock
    mods = [(f"layers.{i}", m) for i, m in enumerate(vae.decoder.layers) if isinstance(m, AutoencoderTinyBlock)]
    P = torch.nn.ParameterDict()
    for n, m in mods:
        k = n.replace(".", "__")
        C = m.conv[-1].out_channels
        P[k + "__g"] = torch.nn.Parameter((film[n][0].clone() if film else torch.zeros(n_styles, C)).float(), requires_grad=trainable)
        P[k + "__b"] = torch.nn.Parameter((film[n][1].clone() if film else torch.zeros(n_styles, C)).float(), requires_grad=trainable)
    vae.decoder.style_film = P
    vae.decoder._film_style = None
    vae.decoder._film_names = [n for n, _ in mods]
    for n, m in mods:
        m.register_forward_hook(_film_hook(vae.decoder, n.replace(".", "__"), fixed_style))
    return len(mods)


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
        v = AutoencoderTiny.from_pretrained(path, torch_dtype=dtype)
        from pathlib import Path
        sf = Path(path) / "style_film_dec.pt"
        if sf.exists():  # one multi-style decoder (dec8): per-style FiLM, the view fixes the style
            import json
            d = torch.load(sf)
            fs = Path(path) / "film_style.json"
            install_decoder_film(v, d["n_styles"], film=d["film"], fixed_style=json.loads(fs.read_text())["style"] if fs.exists() else 0)
            v.decoder.style_film.to(dtype)
        return v
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
