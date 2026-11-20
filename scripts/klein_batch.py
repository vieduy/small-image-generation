"""Batched FLUX.2-klein-4B photo edit with ONE reference photo PER SAMPLE (2026-11-18, data-generation speed-up).

Stock Flux2KleinPipeline treats `image=[a, b, ...]` as several references for EVERY sample (concatenated, repeated over the
batch), which is why i2i_server.Klein runs batch 1. Here prepare_image_latents is replaced on the pipeline instance: the
B reference photos are VAE-encoded in one pass and sample i attends only to photo i (same position ids as a single
reference). Prompts, seeds (one generator per sample) and 4 steps / guidance 1.0 / 512 px are as in i2i_server.Klein with
KLEIN_REF=0 (gen_pairs.py setting: text-only style, no style reference image).
  from klein_batch import KleinBatch; kb = KleinBatch(); ims = kb.run(photos, prompts, seeds)
Equivalence vs batch 1 and throughput: scripts/bench_klein_batch.py.
"""
import os
import types

import torch

SIZE = int(os.environ.get("SIZE", "512"))


def _per_sample_prepare_image_latents(self, images, batch_size, generator, device, dtype):
    assert len(images) == batch_size, f"per-sample references: {len(images)} images for batch {batch_size}"
    x = torch.cat(list(images)).to(device=self.vae.device, dtype=dtype)
    lat = self._encode_vae_image(image=x, generator=generator)          # (B, 128, h, w)
    ids = self._prepare_image_ids([lat[:1]]).repeat(batch_size, 1, 1)   # one reference's coordinates, for every sample
    return self._pack_latents(lat).to(device), ids.to(device)


class KleinBatch:
    def __init__(self, device="cuda"):
        from diffusers import Flux2KleinPipeline
        self.p = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16,
                                                    cache_dir=os.environ["KLEIN_CACHE"] + "/hub").to(device)
        self.p.prepare_image_latents = types.MethodType(_per_sample_prepare_image_latents, self.p)
        self.p.set_progress_bar_config(disable=True)
        self.device = device

    @torch.inference_mode()
    def run(self, imgs, prompts, seeds, steps=4):
        gens = [torch.Generator(self.device).manual_seed(int(s)) for s in seeds]
        return self.p(prompt=list(prompts), image=list(imgs), width=SIZE, height=SIZE, num_inference_steps=steps,
                      guidance_scale=1.0, generator=gens).images
