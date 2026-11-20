"""Create (or recreate) one placeholder style: 8 self-generated reference images + style.yaml (swappable config).
Generated with FLUX.2-klein-4B (Apache-2.0) text-to-image. No brand names / no real people in prompts.

  source env.sh; $VENV/bin/python scripts/gen_style.py --name cartoon3d \
      --desc "3D cartoon avatar, ..." --short "3D cartoon avatar" --seed 31000
"""
import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from diffusers import Flux2KleinPipeline

ROOT = Path(__file__).resolve().parent.parent
SUBJECTS = ["a young woman", "an older man with grey hair", "a man with a beard", "a woman with curly hair",
            "a woman wearing a headscarf", "a bald man with glasses", "a teenage-looking young man with a cap", "a woman with long braids"]

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--desc", required=True)
ap.add_argument("--short", required=True)
ap.add_argument("--seed", type=int, default=31000)
a = ap.parse_args()

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16,
                                          cache_dir=os.environ["KLEIN_CACHE"] + "/hub").to("cuda")
pipe.set_progress_bar_config(disable=True)
d = ROOT / "styles" / a.name
d.mkdir(parents=True, exist_ok=True)
prompts = [f"Avatar portrait of {s}, {a.desc}, head and shoulders, centered, plain light background" for s in SUBJECTS]
gens = [torch.Generator("cuda").manual_seed(a.seed + i) for i in range(len(prompts))]
imgs = pipe(prompt=prompts, width=512, height=512, num_inference_steps=4, guidance_scale=1.0, generator=gens).images
with open(d / "manifest.jsonl", "w") as m:
    for i, (im, p) in enumerate(zip(imgs, prompts)):
        im.save(d / f"ref_{i:03d}.png")
        m.write(json.dumps({"file": f"ref_{i:03d}.png", "prompt": p, "seed": a.seed + i, "model": "FLUX.2-klein-4B", "steps": 4}) + "\n")
cfg = {"name": a.name, "description": a.desc, "short": a.short, "refs": [f"ref_{i:03d}.png" for i in range(len(imgs))],
       "default_ref": "ref_000.png", "lora": None,
       "license": "self-generated with FLUX.2-klein-4B (Apache-2.0); prompts/seeds in manifest.jsonl",
       "avatar_prompt": "(see configs/prompts.yaml)", "sticker_prompt": "(see configs/prompts.yaml)"}
(d / "style.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
print("wrote", d)
