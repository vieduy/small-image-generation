"""Plan B0 (2026-11-20, user "let try new base: Sana sprint"): is Sana-Sprint 0.6B a feasible phone base? No training.

Measures (fp16 on GPU, batch 1, 512 px):
  * params + GFLOPs: transformer on one image (16x16 latent tokens, DC-AE f32c32) and on our diptych [noise | photo] (16x32),
    DC-AE encoder / decoder at 512; same for the current phone path (DreamLite UNet diptych, TinyVAE enc/dec) for reference;
  * H200 latency (indicative only, not phone);
  * autoencoder fidelity at 512 on klein cartoon3d targets (what the decoder must reproduce): DC-AE vs DreamLite TinyVAE
    (original) and decg2 (our phone decoder; decoding TinyVAE latents of the target) -> PSNR, LPIPS(alex), Laplacian-variance ratio.
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/sana_b0.py
Out: results/sana_b0.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SNAP = Path("/mnt/project/kchat/duyhv/sana/hf_cache/hub/models--Efficient-Large-Model--Sana_Sprint_0.6B_1024px_diffusers/snapshots/aa76e7f4f4928f378716b6716a2130fba3caf5b1")
DEV, DT = "cuda", torch.float16


def gflops(fn):
    from torch.utils.flop_counter import FlopCounterMode
    fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc:
        fn()
    return round(fc.get_total_flops() / 1e9, 1)


def ms(fn, n=20):
    with torch.no_grad():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    return round((time.time() - t) / n * 1000, 2)


def npar(m):
    return round(sum(p.numel() for p in m.parameters()) / 1e6, 1)


def main():
    from diffusers import AutoencoderDC, SanaTransformer2DModel
    tr = SanaTransformer2DModel.from_pretrained(SNAP / "transformer", torch_dtype=DT).to(DEV).eval()
    dc = AutoencoderDC.from_pretrained(SNAP / "vae", torch_dtype=DT).to(DEV).eval()
    res = {"snapshot": str(SNAP.name), "licence": "Apache-2.0 (LICENSE in snapshot); README adds Gemma terms for the text encoder"}
    cap = torch.randn(1, 300, tr.config.caption_channels, device=DEV, dtype=DT)  # stands in for learned option tokens

    def tf(w):
        x = torch.randn(1, tr.config.in_channels, 16, w, device=DEV, dtype=DT)
        t = torch.tensor([1.0], device=DEV, dtype=DT)
        kw = {"guidance": torch.tensor([4.5], device=DEV, dtype=DT)} if tr.config.guidance_embeds else {}
        return lambda: tr(x, encoder_hidden_states=cap, timestep=t, return_dict=False, **kw)

    img = torch.randn(1, 3, 512, 512, device=DEV, dtype=DT)
    lat = torch.randn(1, 32, 16, 16, device=DEV, dtype=DT)
    res["sana"] = {"transformer_params_M": npar(tr), "dcae_encoder_params_M": npar(dc.encoder), "dcae_decoder_params_M": npar(dc.decoder),
                   "gflops": {"transformer_single_16x16": gflops(tf(16)), "transformer_diptych_16x32": gflops(tf(32)),
                              "dcae_encode_512": gflops(lambda: dc.encode(img)), "dcae_decode_512": gflops(lambda: dc.decode(lat))},
                   "h200_ms_fp16": {"transformer_diptych": ms(tf(32)), "dcae_encode": ms(lambda: dc.encode(img)),
                                    "dcae_decode": ms(lambda: dc.decode(lat))}}
    print("[b0] sana", res["sana"], flush=True)

    u = DLA.load_unet(ROOT / "weights/student/dl6g2/unet").to(DEV, DT).eval()
    tv = DLA.load_vae(DT).to(DEV).eval()
    te = torch.load(ROOT / "weights/student/dl6g2/task_emb.pt")
    e, m = te["emb"][:1].to(DEV, DT), te["mask"][:1].to(DEV)
    z = torch.randn(1, 4, 64, 64, device=DEV, dtype=DT)
    res["dreamlite"] = {"unet_params_M": npar(u), "tinyvae_encoder_params_M": npar(tv.encoder), "tinyvae_decoder_params_M": npar(tv.decoder),
                        "gflops": {"unet_diptych": gflops(lambda: DLA.forward(u, z, e, m)), "tinyvae_encode_512": gflops(lambda: tv.encode(img)),
                                   "tinyvae_decode_512": gflops(lambda: tv.decode(z))},
                        "h200_ms_fp16": {"unet_diptych": ms(lambda: DLA.forward(u, z, e, m)), "tinyvae_decode": ms(lambda: tv.decode(z))}}
    print("[b0] dreamlite", res["dreamlite"], flush=True)
    del u

    # autoencoder fidelity on klein cartoon3d targets (expressions + avatars), 512 px
    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).to(DEV).eval()
    d2 = DLA.load_vae(DT, ROOT / "weights/decoder/decg2").to(DEV).eval()
    rows = [json.loads(l) for l in open(ROOT / "data/pairs/cartoon3d/manifest_v2.jsonl")]
    rows = [r for r in rows if r.get("keep")]
    rng = np.random.default_rng(0)
    pick = [rows[i] for i in rng.choice(len(rows), 64, replace=False)]
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=DEV).view(1, 1, 3, 3)
    lapv = lambda y: float(torch.nn.functional.conv2d(((y.float() + 1) * 127.5).mean(1, keepdim=True), k).var())
    acc = {n: {"psnr": [], "lpips": [], "lap_ratio": []} for n in ("dcae", "tinyvae_orig", "tinyvae_decg2")}
    with torch.no_grad():
        for r in pick:
            p = ROOT / "data/pairs/cartoon3d" / r.get("target", f"{r['task']}/{r['id']}.png")
            x = torch.from_numpy(np.array(Image.open(p).convert("RGB").resize((512, 512)))).permute(2, 0, 1)[None].to(DEV, DT) / 127.5 - 1
            zt = tv.encode(x).latents
            outs = {"dcae": dc.decode(dc.encode(x).latent).sample, "tinyvae_orig": tv.decode(zt).sample, "tinyvae_decg2": d2.decode(zt).sample}
            for n, y in outs.items():
                y = y.float().clamp(-1, 1)
                mse = float(((y - x.float()) ** 2).mean()) * (127.5 ** 2)
                acc[n]["psnr"].append(10 * np.log10(255 ** 2 / max(mse, 1e-10)))
                acc[n]["lpips"].append(float(lp(y, x.float())))
                acc[n]["lap_ratio"].append(lapv(y) / max(lapv(x), 1e-6))
    res["autoencoder_fidelity_512"] = {"n": len(pick), "data": "random kept klein cartoon3d targets (manifest_v2)"} | {
        n: {kk: round(float(np.mean(v)), 4) for kk, v in d.items()} for n, d in acc.items()}
    print("[b0] fidelity", res["autoencoder_fidelity_512"], flush=True)
    (ROOT / "results/sana_b0.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
