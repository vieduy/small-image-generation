"""Run a trained student on the eval set + measure efficiency.

Outputs: outputs/student_<tag>/<style>/<task>/*.png          (TAESD encoder + TAESD decoder: the phone path)
         outputs/student_<tag>_sdvae/<style>/<task>/*.png    (SD-VAE encoder + SD-VAE decoder: quality reference)
Perf   : results/student_<tag>_perf.json  - params, GFLOPs (torch flop counter, batch 1), fp16/int8 sizes, CUDA-event latency
         (warm-up 10, 100 runs, median/p95) per component at batch 1/4/16 on THIS GPU (server numbers, not phone numbers).

  source env.sh; $VENV/bin/python scripts/student_infer.py --tag s1 --style cartoon3d
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, AutoencoderTiny, UNet2DConditionModel
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from eval_quality import eval_items  # noqa: E402
from train_student import DEV, SDXS, TASKS, alphas_cumprod, forward, sample_chain, to_t  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def cuda_time(fn, warm=10, n=100):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return round(float(np.median(ts)), 3), round(float(np.percentile(ts, 95)), 3)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--style", default="cartoon3d")
    ap.add_argument("--no-perf", action="store_true")
    ap.add_argument("--fake-w8", action="store_true", help="simulate Core ML linear_symmetric per-channel int8 WEIGHTS "
                    "(the Student_*_w8.mlpackage) on the UNet; outputs go to student_<tag>_w8")
    ap.add_argument("--dl-k", type=int, default=0, help="DreamLite multi-step student: run only the first k steps "
                    "(outputs go to student_<tag>_k<k>); 0 = all of its trained steps")
    ap.add_argument("--decoder", default="", help="DreamLite only: fine-tuned phone decoder weights/decoder/<name> "
                    "(train_decoder.py); outputs go to student_<tag>_<name>")
    ap.add_argument("--ts", default="", help="override the student's timesteps, comma-separated (default: its trained msteps_ts)")
    a = ap.parse_args()
    w = ROOT / "weights/student" / a.tag
    cfg = json.loads((w / "config.json").read_text())
    dt = torch.float16
    if cfg.get("arch") == "dreamlite":  # DreamLite one-step student (scripts/dl_arch.py): phone path only, no SD-VAE variant
        return infer_dreamlite(a, w, cfg)
    if cfg.get("lora_rank", 0) > 0:  # style pack: base student UNet + LoRA add-on, merged for inference
        from peft import PeftModel
        base = UNet2DConditionModel.from_pretrained(ROOT / "weights/student" / cfg["init_from"] / "unet")
        unet = PeftModel.from_pretrained(base, w / "unet_lora").merge_and_unload().to(DEV, dt).eval()
    else:
        unet = UNet2DConditionModel.from_pretrained(w / "unet", torch_dtype=dt).to(DEV).eval()
    if a.fake_w8:  # per-output-channel symmetric int8 fake-quant of conv/linear weights (activations stay fp16)
        with torch.no_grad():
            for mod in unet.modules():
                if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
                    wt = mod.weight.float()
                    sc = wt.abs().flatten(1).amax(1).clamp(min=1e-8) / 127
                    sc = sc.view(-1, *([1] * (wt.dim() - 1)))
                    mod.weight.copy_(((wt / sc).round().clamp(-127, 127) * sc).to(mod.weight.dtype))
        a.tag_out = a.tag + "_w8"
    else:
        a.tag_out = a.tag
    emb = torch.load(w / "task_emb.pt")["emb"].to(DEV, dt)
    vae = AutoencoderKL.from_pretrained(SDXS, subfolder="vae_large", torch_dtype=dt).to(DEV).eval()
    tae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=dt).to(DEV).eval()
    ac = alphas_cumprod().to(DEV)
    sf = vae.config.scaling_factor

    ts = cfg.get("msteps_ts", [cfg["T"]])  # 1-step students: [T]
    if a.ts:  # e.g. run a 4-step student at 2 steps: --ts 999,499 (outputs tagged <tag>_k2)
        ts = [int(t) for t in a.ts.split(",")]
        a.tag_out += f"_k{len(ts)}"
    print(f"[infer] {a.tag}: {len(ts)} UNet step(s) at t={ts}", flush=True)

    def run(x, task_ids, enc="taesd"):
        z = tae.encode(x).latents if enc == "taesd" else vae.encode(x).latent_dist.mean * sf
        return sample_chain(unet, z, emb[task_ids], cfg["param"], ts, ac, torch.Generator(DEV).manual_seed(0))

    def save(xs):
        for x, p in xs:
            Image.fromarray(((x.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()).save(p)

    items = eval_items()
    for task in TASKS:
        its = [p for p, t in items if t == task]
        for variant, enc in ((f"student_{a.tag_out}", "taesd"), (f"student_{a.tag_out}_sdvae", "sdvae")):
            od = ROOT / f"outputs/{variant}/{a.style}/{task}"
            od.mkdir(parents=True, exist_ok=True)
            for i in range(0, len(its), 16):
                ch = its[i:i + 16]
                x = torch.stack([to_t(Image.open(p)) for p in ch]).to(DEV, dt)
                tid = torch.full((len(ch),), TASKS.index(task), device=DEV, dtype=torch.long)
                z = run(x, tid, enc)
                y = tae.decode(z).sample if enc == "taesd" else vae.decode(z / sf).sample
                save(list(zip(y, [od / f"{p.stem}.png" for p in ch])))
    print(f"[infer] wrote outputs/student_{a.tag}[_sdvae]/{a.style}", flush=True)
    if a.no_perf:
        return

    perf = {"tag": a.tag, "gpu": torch.cuda.get_device_name(0), "dtype": "fp16", "note": "server GPU numbers, NOT phone numbers",
            "params_M": {"unet": round(nparams(unet) / 1e6, 1), "taesd_enc": round(nparams(tae.encoder) / 1e6, 2),
                         "taesd_dec": round(nparams(tae.decoder) / 1e6, 2), "sdvae_enc": round(nparams(vae.encoder) / 1e6, 1),
                         "sdvae_dec": round(nparams(vae.decoder) / 1e6, 1), "text_encoder": 0},
            "size_MB": {"unet_fp16": round(nparams(unet) * 2 / 2**20, 1), "unet_int8": round(nparams(unet) / 2**20, 1)}}
    from torch.utils.flop_counter import FlopCounterMode
    x1 = torch.randn(1, 3, 512, 512, device=DEV, dtype=dt)
    z1 = tae.encode(x1).latents
    t1 = torch.zeros(1, dtype=torch.long, device=DEV)
    fl = {}
    for name, fn in [("taesd_enc", lambda: tae.encode(x1)), ("unet", lambda: forward(unet, z1, emb[t1], cfg["param"], cfg["T"], ac)),
                     ("taesd_dec", lambda: tae.decode(z1))]:
        with FlopCounterMode(display=False) as fc:
            fn()
        fl[name] = round(fc.get_total_flops() / 1e9, 1)
    perf["GFLOPs_b1"] = fl
    lat = {}
    for bs in (1, 4, 16):
        x = torch.randn(bs, 3, 512, 512, device=DEV, dtype=dt)
        z = tae.encode(x).latents
        tid = torch.zeros(bs, dtype=torch.long, device=DEV)
        lat[f"b{bs}"] = {"taesd_enc": cuda_time(lambda: tae.encode(x)),
                         "unet": cuda_time(lambda: forward(unet, z, emb[tid], cfg["param"], cfg["T"], ac)),
                         "taesd_dec": cuda_time(lambda: tae.decode(z)),
                         "full_phone_path": cuda_time(lambda: tae.decode(run(x, tid, "taesd")))}
        lat[f"b{bs}"]["gpu_ms_per_img"] = round(lat[f"b{bs}"]["full_phone_path"][0] / bs, 3)
    perf["latency_ms_median_p95"] = lat
    perf["peak_vram_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    (ROOT / f"results/student_{a.tag}_perf.json").write_text(json.dumps(perf, indent=1))
    print(json.dumps(perf), flush=True)


@torch.inference_mode()
def infer_dreamlite(a, w, cfg):
    import dl_arch as DLA
    dt = torch.float16
    RS = cfg.get("refine_sigma") or 0.0  # 3a refiner: step-2-only LoRA, adapter kept separate (not merged)
    if cfg.get("lora_rank", 0) > 0 and RS > 0:
        from peft import PeftModel
        unet = PeftModel.from_pretrained(DLA.load_unet(ROOT / "weights/student" / cfg["init_from"] / "unet"), w / "unet_lora")
    elif cfg.get("lora_rank", 0) > 0:
        from peft import PeftModel
        unet = PeftModel.from_pretrained(DLA.load_unet(ROOT / "weights/student" / cfg["init_from"] / "unet"), w / "unet_lora").merge_and_unload()
    else:
        unet = DLA.load_unet(w / "unet")
    unet = unet.to(DEV, dt).eval()
    te = torch.load(w / "task_emb.pt")
    emb, mask = te["emb"].to(DEV, dt), te["mask"].to(DEV)
    vae = DLA.load_vae(dt).to(DEV).eval()
    vdec = DLA.load_vae(dt, ROOT / "weights/decoder" / a.decoder).to(DEV).eval() if a.decoder else vae  # encoder stays the original
    items = eval_items()
    MS = cfg.get("ms_sigmas")
    otag = a.tag + (f"_k{a.dl_k}" if a.dl_k else "") + (f"_{a.decoder}" if a.decoder else "")
    for task in TASKS:
        its = [p for p, t in items if t == task]
        od = ROOT / f"outputs/student_{otag}/{a.style}/{task}"
        od.mkdir(parents=True, exist_ok=True)
        for i in range(0, len(its), 16):
            ch = its[i:i + 16]
            x = torch.stack([to_t(Image.open(p)) for p in ch]).to(DEV, dt)
            k = TASKS.index(task)
            zx, ek, mk = DLA.encode(vae, x), emb[[k] * len(ch)], mask[[k] * len(ch)]
            if RS > 0 and task in (cfg.get("only_tasks") or "avatar").split(","):  # 2 steps: init student, then the refiner
                lat = DLA.refine_forward(unet, unet, zx, ek, mk, RS)
            elif RS > 0:  # other options: the unchanged one-step init student (adapter off)
                with unet.disable_adapter():
                    lat = DLA.forward(unet, zx, ek, mk)
            else:
                lat = DLA.forward(unet, zx, ek, mk, sigmas=MS, k=a.dl_k or None)
            y = DLA.decode(vdec, lat)
            for yy, p in zip(y, ch):
                Image.fromarray(((yy.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()).save(od / f"{p.stem}.png")
    print(f"[infer] dreamlite {a.tag} (sigmas {MS}, k {a.dl_k or 'all'}): wrote outputs/student_{otag}/{a.style}", flush=True)


if __name__ == "__main__":
    main()
