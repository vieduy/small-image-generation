"""Image-to-image serving harness (one model per process, single GPU) for avatar/sticker candidates.

  source env.sh
  MODEL=sdturbo MAX_BATCH=8 MAX_WAIT_MS=30 $VENV/bin/python scripts/i2i_server.py --port 8200

POST /v1/i2i  {"image": b64 (input photo), "task": "avatar"|"sticker", "style": "flat_vector"|..., "seed": int,
               "response_format": "b64_json"|"none"}
Requests with the same (task, style) are dynamically batched up to MAX_BATCH (models with batch_ok=False run 1 at a time).
The batcher drains everything already queued before checking the wait deadline (fixes the bench/server_sprint.py bug
where batches collapsed to ~1 under load).
Styles are config: styles/<name>/style.yaml (prompt templates + reference image). Nothing style-specific in code.
Every response carries timing: queue_ms, infer_ms, encode_ms, batch_size.
"""
import argparse
import asyncio
import base64
import gc
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
MODEL = os.environ.get("MODEL", "sdturbo")
MAX_BATCH = int(os.environ.get("MAX_BATCH", "1"))
MAX_WAIT_MS = float(os.environ.get("MAX_WAIT_MS", "30"))
SIZE = int(os.environ.get("SIZE", "512"))
DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}[os.environ.get("DTYPE", "fp16")]
DEV = "cuda"


_P = yaml.safe_load((ROOT / "configs/prompts.yaml").read_text())
PROMPTS = _P.get(os.environ.get("PROMPTS_FAMILY", MODEL), _P["default"])


def load_styles():
    out = {}
    for y in sorted((ROOT / "styles").glob("*/style.yaml")):
        c = yaml.safe_load(y.read_text())
        c["ref_img"] = Image.open(y.parent / c["default_ref"]).convert("RGB").resize((SIZE, SIZE))
        for t in ("avatar", "sticker"):
            c[f"{t}_text"] = PROMPTS[t].format(desc=c["description"], short=c.get("short", c["name"]))
        out[c["name"]] = c
    return out


# ---------------------------------------------------------------- model adapters
class Adapter:
    batch_ok = True
    steps = 4
    notes = ""

    def load(self): ...

    def run(self, imgs, prompt, style, seeds):  # same prompt/style for the whole batch
        raise NotImplementedError

    def gens(self, seeds):
        return [torch.Generator(DEV).manual_seed(s) for s in seeds]


class Klein(Adapter):
    """FLUX.2-klein-4B multi-reference edit: [input photo, style ref] -> output. Teacher/baseline."""
    batch_ok = False  # per-request reference images; pipeline batches refs across prompts, not per-sample
    steps = 4

    def load(self):
        from diffusers import Flux2KleinPipeline
        self.p = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16,
                                                    cache_dir=os.environ["KLEIN_CACHE"] + "/hub").to(DEV)

    def run(self, imgs, prompt, style, seeds):
        refs = [imgs[0], style["ref_img"]] if os.environ.get("KLEIN_REF", "1") == "1" else [imgs[0]]
        return self.p(prompt=prompt, image=refs, width=SIZE, height=SIZE,
                      num_inference_steps=self.steps, guidance_scale=1.0, generator=self.gens(seeds)).images


class SanaSprint(Adapter):
    """Sana-Sprint 0.6B img2img (SDEdit). Prompt embeddings cached per (task, style); text encoder freed from GPU."""
    steps = int(os.environ.get("STEPS", "2"))
    strength = float(os.environ.get("STRENGTH", "0.5"))

    def load(self):
        from diffusers import SanaSprintImg2ImgPipeline
        self.p = SanaSprintImg2ImgPipeline.from_pretrained("Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
                                                           torch_dtype=torch.bfloat16,
                                                           cache_dir=os.environ["SANA_CACHE"] + "/hub").to(DEV)
        self.cache = {}

    def embed(self, prompt):
        if prompt not in self.cache:
            with torch.inference_mode():
                pe, pm = self.p.encode_prompt(prompt, device=DEV)[:2]
            self.cache[prompt] = (pe, pm)
        return self.cache[prompt]

    def free_text_encoder(self):
        self.p.text_encoder.to("cpu")
        torch.cuda.empty_cache()

    def run(self, imgs, prompt, style, seeds):
        pe, pm = self.embed(prompt)
        n = len(imgs)
        return self.p(prompt_embeds=pe.expand(n, -1, -1), prompt_attention_mask=pm.expand(n, -1), image=imgs,
                      strength=self.strength, num_inference_steps=self.steps, width=SIZE, height=SIZE,
                      generator=self.gens(seeds)).images


class IP2P(Adapter):
    """InstructPix2Pix (SD1.5 UNet with input-image channel concat). 3 UNet passes/step (text+image CFG)."""
    steps = int(os.environ.get("STEPS", "10"))

    def load(self):
        from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline
        self.p = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=DTYPE,
                                                                        variant="fp16", safety_checker=None).to(DEV)
        self.p.scheduler = EulerAncestralDiscreteScheduler.from_config(self.p.scheduler.config)

    def run(self, imgs, prompt, style, seeds):
        return self.p(prompt=[prompt] * len(imgs), image=imgs, num_inference_steps=self.steps, guidance_scale=7.5,
                      image_guidance_scale=1.5, generator=self.gens(seeds)).images


class SD15IPA(Adapter):
    """SD1.5 img2img + IP-Adapter Plus (style reference image, swappable) + LCM-LoRA for 4 steps."""
    steps = int(os.environ.get("STEPS", "4"))
    strength = float(os.environ.get("STRENGTH", "0.6"))

    def load(self):
        from diffusers import LCMScheduler, StableDiffusionImg2ImgPipeline
        from transformers import CLIPVisionModelWithProjection
        enc = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=DTYPE)
        self.p = StableDiffusionImg2ImgPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=DTYPE,
                                                                variant="fp16", image_encoder=enc, safety_checker=None).to(DEV)
        self.p.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.safetensors")
        self.p.set_ip_adapter_scale(float(os.environ.get("IPA_SCALE", "0.6")))
        self.p.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        self.p.fuse_lora()
        self.p.scheduler = LCMScheduler.from_config(self.p.scheduler.config)

    def run(self, imgs, prompt, style, seeds):
        n = len(imgs)
        return self.p(prompt=[prompt] * n, image=imgs, ip_adapter_image=[style["ref_img"]] * n, strength=self.strength,
                      num_inference_steps=self.steps, guidance_scale=1.0, generator=self.gens(seeds)).images


class SDTurbo(Adapter):
    """SD-Turbo img2img: steps*strength = effective denoising steps (2 x 0.5 = 1 UNet pass), no CFG."""
    steps = int(os.environ.get("STEPS", "2"))
    strength = float(os.environ.get("STRENGTH", "0.75"))

    def load(self):
        from diffusers import AutoPipelineForImage2Image
        self.p = AutoPipelineForImage2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=DTYPE, variant="fp16").to(DEV)

    def run(self, imgs, prompt, style, seeds):
        return self.p(prompt=[prompt] * len(imgs), image=imgs, strength=self.strength, num_inference_steps=self.steps,
                      guidance_scale=0.0, generator=self.gens(seeds)).images


class SDXS(SDTurbo):
    """SDXS-512-0.9 (0.32B pruned UNet, 1-step) used as img2img; same steps*strength convention as SD-Turbo."""

    def load(self):
        from diffusers import AutoencoderKL, StableDiffusionImg2ImgPipeline
        vae = AutoencoderKL.from_pretrained("IDKiro/sdxs-512-0.9", subfolder="vae_large", torch_dtype=DTYPE)
        self.p = StableDiffusionImg2ImgPipeline.from_pretrained("IDKiro/sdxs-512-0.9", vae=vae, torch_dtype=DTYPE,
                                                                safety_checker=None).to(DEV)


ADAPTERS = {"klein": Klein, "sana_sprint": SanaSprint, "ip2p": IP2P, "sd15_ipa_lcm": SD15IPA, "sdturbo": SDTurbo, "sdxs": SDXS}


# ---------------------------------------------------------------- serving
class Req(BaseModel):
    image: str
    task: str = "avatar"
    style: str = "flat_vector"
    seed: int | None = None
    response_format: str = "b64_json"


@dataclass
class Job:
    req: Req
    img: Image.Image
    fut: asyncio.Future
    t_arrive: float = field(default_factory=time.perf_counter)

    @property
    def key(self):
        return (self.req.task, self.req.style)


app = FastAPI()
S = {"ad": None, "styles": None, "queue": None, "stats": {"batches": 0, "images": 0}}
gpu = ThreadPoolExecutor(max_workers=1)


def _infer(jobs):
    ad, st = S["ad"], S["styles"][jobs[0].req.style]
    prompt = st[f"{jobs[0].req.task}_text"]
    seeds = [j.req.seed if j.req.seed is not None else time.time_ns() % 2**31 for j in jobs]
    t = time.perf_counter()
    with torch.inference_mode():
        out = ad.run([j.img for j in jobs], prompt, st, seeds)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t


async def batcher():
    q, loop, pending = S["queue"], asyncio.get_running_loop(), []
    mb = MAX_BATCH if S["ad"].batch_ok else 1
    while True:
        if not pending:
            pending.append(await q.get())
        while not q.empty():  # drain everything already waiting
            pending.append(q.get_nowait())
        key = pending[0].key
        deadline = pending[0].t_arrive + MAX_WAIT_MS / 1000
        while sum(j.key == key for j in pending) < mb:
            tmo = deadline - time.perf_counter()
            if tmo <= 0:
                break
            try:
                pending.append(await asyncio.wait_for(q.get(), tmo))
            except asyncio.TimeoutError:
                break
        batch = [j for j in pending if j.key == key][:mb]
        for j in batch:
            pending.remove(j)
        t0 = time.perf_counter()
        try:
            imgs, inf = await loop.run_in_executor(gpu, _infer, batch)
        except Exception as e:  # noqa: BLE001
            for j in batch:
                j.fut.done() or j.fut.set_exception(e)
            continue
        if inf > 1.0:
            print(f"[slow] batch={len(batch)} key={batch[0].key} infer={inf:.2f}s", flush=True)
        S["stats"]["batches"] += 1
        S["stats"]["images"] += len(batch)
        for j, im in zip(batch, imgs):
            j.fut.done() or j.fut.set_result((im, t0 - j.t_arrive, inf, len(batch)))


@app.on_event("startup")
async def startup():
    torch.backends.cuda.matmul.allow_tf32 = True
    S["styles"] = load_styles()
    ad = ADAPTERS[MODEL]()
    t = time.perf_counter()
    ad.load()
    getattr(ad, "p", None) and ad.p.set_progress_bar_config(disable=True)
    if isinstance(ad, SanaSprint):  # pre-compute all prompt embeddings, then move text encoder off the GPU
        for st in S["styles"].values():
            for task in ("avatar", "sticker"):
                ad.embed(st[f"{task}_text"])
        ad.free_text_encoder()
    S["ad"] = ad
    print(f"[server] {MODEL} loaded {time.perf_counter() - t:.1f}s", flush=True)
    st = next(iter(S["styles"].values()))
    dummy = st["ref_img"]
    mb = MAX_BATCH if ad.batch_ok else 1
    t = time.perf_counter()

    def warm():  # runs on the SAME executor thread that serves requests (per-thread cuBLAS/cuDNN state)
        for bs in range(1, mb + 1):  # every batch size the batcher can form x every (style, task) prompt
            for s_ in S["styles"].values():
                for task in ("avatar", "sticker"):
                    with torch.inference_mode():
                        ad.run([dummy] * bs, s_[f"{task}_text"], s_, list(range(bs)))
        for _ in range(2):
            with torch.inference_mode():
                ad.run([dummy] * mb, st["avatar_text"], st, list(range(mb)))
        torch.cuda.synchronize()

    await asyncio.get_running_loop().run_in_executor(gpu, warm)
    torch.cuda.synchronize()
    print(f"[server] warmup {time.perf_counter() - t:.1f}s MAX_BATCH={mb} MAX_WAIT_MS={MAX_WAIT_MS} "
          f"peak_vram_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}", flush=True)
    if os.environ.get("GC_FREEZE", "1") == "1":  # move warm-up objects out of GC tracking (same trick as vLLM/SGLang)
        gc.collect()
        gc.freeze()
    S["queue"] = asyncio.Queue()
    asyncio.create_task(batcher())


@app.get("/health")
async def health():
    return {"ok": S["ad"] is not None and S["queue"] is not None, "model": MODEL, "max_batch": MAX_BATCH,
            "batch_ok": S["ad"].batch_ok if S["ad"] else None, "steps": getattr(S["ad"], "steps", None),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), **S["stats"]}


@app.post("/v1/i2i")
async def i2i(req: Req):
    if req.style not in S["styles"] or req.task not in ("avatar", "sticker"):
        raise HTTPException(400, f"bad style/task; styles={list(S['styles'])}")
    img = Image.open(io.BytesIO(base64.b64decode(req.image))).convert("RGB").resize((SIZE, SIZE))
    fut = asyncio.get_running_loop().create_future()
    await S["queue"].put(Job(req, img, fut))
    out, qs, inf, bs = await fut
    t = time.perf_counter()

    def enc():
        if req.response_format == "none":
            return None
        b = io.BytesIO()
        out.save(b, format="PNG")
        return base64.b64encode(b.getvalue()).decode()

    data = await asyncio.to_thread(enc)
    return {"data": [{"b64_json": data}], "timing": {"queue_ms": round(qs * 1000, 1), "infer_ms": round(inf * 1000, 1),
                                                      "encode_ms": round((time.perf_counter() - t) * 1000, 1), "batch_size": bs}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8200)
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
