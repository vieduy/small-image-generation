"""Eval set "selfie" (v3): 200 realistic close-up synthetic selfies (gen_pairs.selfie_prompt, own rng 7777, seeds 600000+,
never trained on), generated at phone aspect 512x640 and normalised with scripts/framing.py exactly like a user photo.
Measures the real-photo gap (tight framing, phone artifacts) the head-and-shoulders eval sets cannot.
  source env.sh; $VENV/bin/python scripts/gen_eval_selfie.py
"""
import json, os, random, sys, time
from pathlib import Path
import torch
os.environ.setdefault("MODEL", "klein")
sys.path.insert(0, str(Path(__file__).parent))
import gen_pairs as GP, metrics as M  # noqa: E402
from framing import normalize_face_crop  # noqa: E402
from diffusers import Flux2KleinPipeline  # noqa: E402
ROOT = M.ROOT
raw, out = ROOT / "data/faces_selfie_raw", ROOT / "data/faces_selfie"
raw.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
t = time.time()
pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16,
                                          cache_dir=os.environ["KLEIN_CACHE"] + "/hub").to("cuda")
pipe.set_progress_bar_config(disable=True)
rng = random.Random(7777)
prompts = [GP.selfie_prompt(rng) for _ in range(200)]
fid = M.FaceID()
with open(out / "manifest.jsonl", "w") as man:
    for b in range(0, 200, 8):
        seeds = [600_000 + b + k for k in range(8)]
        ims = pipe(prompt=prompts[b:b + 8], width=512, height=640, num_inference_steps=4, guidance_scale=1.0,
                   generator=[torch.Generator("cuda").manual_seed(s) for s in seeds]).images
        for k, (im, s) in enumerate(zip(ims, seeds)):
            name = f"fs_{b + k:03d}.png"
            im.save(raw / name)
            norm, info = normalize_face_crop(im, fid)
            norm.save(out / name)
            man.write(json.dumps({"file": name, "prompt": prompts[b + k], "seed": s, "size": [512, 640],
                                  "framing": None if info is None else {"reframed": info["reframed"], "padded": info["padded"]}}) + "\n")
print(f"[eval_selfie] 200 selfies {time.time() - t:.0f}s", flush=True)
