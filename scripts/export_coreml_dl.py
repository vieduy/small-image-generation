"""Export a DreamLite one-step student (scripts/dl_arch.py; dl1, dl1_dmd, ...) to Core ML (.mlpackage, ML Program).

Same app contract as the SDXS export (scripts/export_coreml.py):
  inputs  image (512x512 RGB) + task (Int32[1], menu option 0..21)      output  stylized (512x512 RGB)
Graph: [0,255] -> [-1,1] -> TinyVAE encoder -> UNet once on the diptych [fixed noise | photo latent] at t=1000
       (task tokens + their attention mask baked in, selected by a one-hot; no text encoder) -> x0 = noise - v (left half)
       -> TinyVAE decoder -> [0,255].
Core ML changes, numerically equivalent (checked against dl_arch.forward below, fp32):
  * attention = plain matmul/softmax; grouped-query K/V heads repeated with expand+reshape instead of repeat_interleave,
    and the cross-attention mask applied as an additive bias (the UNet's own -10000 convention).
  source env.sh; CUDA_VISIBLE_DEVICES= $VENV/bin/python scripts/export_coreml_dl.py --tag dl1 --int8
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class PlainAttn:
    """DreamLiteAttnProcessor2_0 without SDPA dispatch / repeat_interleave (same math)."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kw):
        residual = hidden_states
        nd = hidden_states.ndim
        if nd == 4:
            b0, ch, hh, ww = hidden_states.shape
            hidden_states = hidden_states.view(b0, ch, hh * ww).transpose(1, 2)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        B = hidden_states.shape[0]
        q = attn.to_q(hidden_states)
        ctx = hidden_states if encoder_hidden_states is None else (
            attn.norm_encoder_hidden_states(encoder_hidden_states) if attn.norm_cross else encoder_hidden_states)
        k, v = attn.to_k(ctx), attn.to_v(ctx)
        H = attn.heads
        d = q.shape[-1] // H
        kvh = k.shape[-1] // d
        q, k, v = q.view(B, -1, H, d), k.view(B, -1, kvh, d), v.view(B, -1, kvh, d)
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)
        if kvh != H:  # == repeat_interleave(H // kvh, dim=2)
            r, L = H // kvh, k.shape[1]
            k = k.unsqueeze(3).expand(B, L, kvh, r, d).reshape(B, L, H, d)
            v = v.unsqueeze(3).expand(B, L, kvh, r, d).reshape(B, L, H, d)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        s = torch.matmul(q, k.transpose(-1, -2)) * (d ** -0.5)
        if attention_mask is not None:  # additive [B, 1, L] from the UNet
            s = s + attention_mask.reshape(B, 1, 1, -1)
        o = torch.matmul(s.softmax(-1), v).transpose(1, 2).reshape(B, -1, H * d)
        o = attn.to_out[1](attn.to_out[0](o))
        if nd == 4:
            o = o.transpose(-1, -2).reshape(b0, ch, hh, ww)
        if attn.residual_connection:
            o = o + residual
        return o / attn.rescale_output_factor


class DLStudent(torch.nn.Module):
    def __init__(self, unet, vae, emb, mask):
        super().__init__()
        self.unet, self.enc, self.dec = unet, vae.encoder, vae.decoder
        self.register_buffer("emb", emb)                                  # [n_tasks, L, 2048]
        self.register_buffer("mask", mask.float())                        # [n_tasks, L] 1 = token, 0 = pad
        self.register_buffer("ids", torch.arange(emb.shape[0]).float())
        self.register_buffer("noise", DLA.fixed_noise("cpu", torch.float32))  # [1, 4, 64, 64], seed 1234
        self.register_buffer("t", torch.tensor([1000.0]))
        self.register_buffer("tid", torch.tensor([[float(DLA.RES), float(DLA.RES)]]))

    def forward(self, image, task):                                       # image in [-1, 1]
        sel = (task.float().reshape(1) == self.ids).float()
        e = (sel.reshape(-1, 1, 1) * self.emb).sum(0, keepdim=True)
        m = (sel.reshape(-1, 1) * self.mask).sum(0, keepdim=True)
        z = self.enc(image)
        x = torch.cat([self.noise, z], dim=3)
        v = self.unet(x, timestep=self.t, encoder_hidden_states=e, encoder_attention_mask=m,
                      added_cond_kwargs={"time_ids": self.tid}, return_dict=False)[0][..., :DLA.LAT]
        return (self.dec(self.noise - v).clamp(-1, 1) + 1) * 127.5          # [0, 255]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--decoder", default=None, help="fine-tuned TinyVAE in weights/decoder/<tag> (scripts/train_decoder.py)")
    ap.add_argument("--skip-convert", action="store_true", help="only the PyTorch equivalence check")
    a = ap.parse_args()
    w = ROOT / "weights/student" / a.tag
    cfg = json.loads((w / "config.json").read_text())
    assert cfg.get("arch") == "dreamlite", "not a DreamLite student"
    if cfg.get("lora_rank", 0) > 0:  # DMD pack on a DreamLite base: merge
        from peft import PeftModel
        unet = PeftModel.from_pretrained(DLA.load_unet(ROOT / "weights/student" / cfg["init_from"] / "unet"), w / "unet_lora").merge_and_unload()
        base_te = torch.load(ROOT / "weights/student" / cfg["init_from"] / "task_emb.pt")
    else:
        unet = DLA.load_unet(w / "unet")
        base_te = None
    te = torch.load(w / "task_emb.pt")
    emb = te["emb"].float()
    mask = te.get("mask", base_te["mask"] if base_te else None)
    tasks = te.get("tasks") or base_te["tasks"]
    task_doc = ", ".join(f"{i}={t.replace('expr_', '')}" for i, t in enumerate(tasks))
    vae = DLA.load_vae(torch.float32, path=(ROOT / "weights/decoder" / a.decoder) if a.decoder else None).eval()
    name = a.tag + (f"_{a.decoder}" if a.decoder else "")
    unet = unet.float().eval()

    # ---- equivalence: Core ML-friendly graph vs the training/eval path (dl_arch.forward, original processor) ----
    photos = sorted((ROOT / "data/faces").glob("*.png"))[:3]
    xs = torch.stack([torch.from_numpy(np.array(Image.open(p).convert("RGB").resize((512, 512)))).permute(2, 0, 1).float() / 127.5 - 1
                      for p in photos])
    with torch.no_grad():
        ref = []
        for i, k in enumerate((0, 1, 2)):
            zp = DLA.forward(unet, DLA.encode(vae, xs[i:i + 1]), emb[k:k + 1], mask[k:k + 1].long())
            ref.append((DLA.decode(vae, zp).clamp(-1, 1) + 1) * 127.5)
        unet.set_attn_processor(PlainAttn())
        m = DLStudent(unet, vae, emb, mask).eval()
        got = [m(xs[i:i + 1], torch.tensor([k], dtype=torch.int32)) for i, k in enumerate((0, 1, 2))]
    diff = torch.cat([(g - r).abs() for g, r in zip(got, ref)])
    print(f"[coreml-dl] equivalence vs dl_arch.forward: mean {diff.mean():.4f}/255, max {diff.max():.3f}/255", flush=True)
    assert diff.mean() < 0.5, "export graph does not match the trained student"
    if a.skip_convert:
        return

    import coremltools as ct
    od = w / ("coreml" + (f"_{a.decoder}" if a.decoder else ""))
    od.mkdir(exist_ok=True)
    x = torch.rand(1, 3, 512, 512) * 2 - 1
    tk = torch.tensor([0], dtype=torch.int32)
    with torch.no_grad():
        tr = torch.export.export(m, (x, tk)).run_decompositions({})
    print("[coreml-dl] exported", flush=True)
    ml = ct.convert(tr, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS16, compute_precision=ct.precision.FLOAT16,
                    inputs=[ct.ImageType(name="image", shape=(1, 3, 512, 512), scale=1 / 127.5, bias=[-1, -1, -1],
                                         color_layout=ct.colorlayout.RGB),
                            ct.TensorType(name="task", shape=(1,), dtype=np.int32)],
                    outputs=[ct.ImageType(name="stylized", color_layout=ct.colorlayout.RGB)])
    ml.short_description = (f"1-step text-free DreamLite-mobile student '{name}' (style {cfg['style']}): task {task_doc}. "
                            "Input: face-framed 512x512 RGB photo. Base: DreamLite-mobile UNet (390M) + TinyVAE" + (f" (decoder fine-tuned: {a.decoder})." if a.decoder else "."))
    ml.author = "i2i-sticker-avatar study"
    ml.license = "Derivative of carlofkl/DreamLite-mobile (weights CC BY-NC 4.0); see LICENSES.md"
    ml.input_description["image"] = "512x512 RGB, face framed like scripts/framing.py"
    ml.input_description["task"] = f"Int32 menu option: {task_doc}"
    ml.output_description["stylized"] = "512x512 RGB stylized image (sticker: cut out with Vision + add 8px white outline in-app)"
    p = od / f"Student_{name}_fp16.mlpackage"
    if p.exists():
        shutil.rmtree(p)
    ml.save(str(p))
    size = lambda d: round(sum(f.stat().st_size for f in Path(d).rglob("*") if f.is_file()) / 2**20, 1)
    info = {"tag": a.tag, "decoder": a.decoder, "arch": "dreamlite", "style": cfg["style"], "fp16_mlpackage": str(p.relative_to(ROOT)), "fp16_MB": size(p),
            "min_ios": 16, "inputs": {"image": "RGB 512x512", "task": f"Int32[1] {task_doc}"}, "output": "stylized RGB 512x512",
            "tasks": {i: t for i, t in enumerate(tasks)}, "torch_equivalence_mean_255": round(float(diff.mean()), 4)}
    if a.int8:
        import coremltools.optimize.coreml as cto
        q = cto.linear_quantize_weights(ml, config=cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric")))
        p8 = od / f"Student_{name}_w8.mlpackage"
        if p8.exists():
            shutil.rmtree(p8)
        q.save(str(p8))
        info |= {"w8_mlpackage": str(p8.relative_to(ROOT)), "w8_MB": size(p8), "w8_note": "8-bit weights; quality NOT verified (Core ML cannot run on Linux)"}
    rd = od / "reference"
    rd.mkdir(exist_ok=True)
    ref_tasks = [t for t in ("avatar", "sticker", "expr_thumbs", "expr_wave", "expr_love") if t in tasks]
    for f in sorted((ROOT / "data/faces").glob("*.png"))[:3] + sorted((ROOT / "data/subjects").glob("*.png"))[:1]:
        im = Image.open(f).convert("RGB").resize((512, 512))
        im.save(rd / f"in_{f.stem}.png")
        xi = torch.from_numpy(np.array(im)).permute(2, 0, 1)[None].float() / 127.5 - 1
        for task in ref_tasks if f.stem.startswith("face") else ("avatar", "sticker"):
            with torch.no_grad():
                y = m(xi, torch.tensor([tasks.index(task)], dtype=torch.int32))[0]
            Image.fromarray(y.round().byte().permute(1, 2, 0).numpy()).save(rd / f"out_{f.stem}_{task}.png")
    (od / "export_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
