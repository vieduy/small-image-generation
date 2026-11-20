"""Generate SYNTHETIC inputs and style references with FLUX.2-klein-4B (Apache-2.0) text-to-image.

No real people: prompts are attribute grids only, never names. Adults only.
Outputs (512x512 PNG) + manifest.jsonl with prompt/seed for every image:
  data/faces/      120 portraits   (avatar + sticker inputs)
  data/subjects/    30 pets/objects (sticker inputs)
  styles/<name>/    8 refs per placeholder style + style.yaml (prompt templates; swappable config)

  source env.sh; $VENV/bin/python scripts/gen_data.py
"""
import itertools
import json
import os
import random
import time
from pathlib import Path

import torch
import yaml
from diffusers import Flux2KleinPipeline

ROOT = Path(__file__).resolve().parent.parent
MODEL = "black-forest-labs/FLUX.2-klein-4B"
STEPS, SIZE, BS = 4, 512, 8

AGES = ["in their twenties", "in their thirties", "in their forties", "in their fifties", "in their sixties", "in their seventies"]
GENDER = ["woman", "man", "person"]
SKIN = ["East Asian", "Southeast Asian", "South Asian", "Black", "white", "Latino", "Middle Eastern", "Pacific Islander"]
HAIR = ["short black hair", "long wavy brown hair", "curly hair", "a shaved head", "grey hair in a bun", "red shoulder-length hair",
        "braided hair", "a buzz cut", "straight blonde hair", "a hijab"]
EXTRA = ["", "wearing glasses", "with a beard", "with freckles", "wearing earrings", "wearing a cap", "smiling broadly", "neutral expression"]
POSE = ["facing the camera", "head turned slightly left", "head turned slightly right", "three-quarter view"]
LIGHT = ["soft window light", "outdoor daylight", "warm indoor light", "overcast light", "studio lighting"]
BG = ["plain grey background", "blurred city street", "blurred park", "home interior", "office background"]

SUBJECTS = ["an orange tabby cat", "a golden retriever", "a black pug", "a white rabbit", "a hamster", "a parrot",
            "a shiba inu", "a tuxedo cat", "a corgi", "a goldfish in a bowl", "a hedgehog", "a turtle",
            "a cup of coffee", "a slice of pizza", "a potted cactus", "a sneaker", "a bicycle", "a guitar",
            "a red apple", "a teddy bear", "a camera", "a skateboard", "a cupcake", "a backpack",
            "a dachshund", "a husky", "a siamese cat", "a duckling", "a donut", "a houseplant in a pot"]

STYLES = {
    "flat_vector": {
        "desc": "flat vector illustration, bold clean outlines, flat colors, no gradients, minimal shading, geometric simplified shapes",
        "ref_subjects": ["a young woman portrait", "an old man portrait", "a man with a beard portrait", "a woman with curly hair portrait",
                         "a cat", "a dog", "a person waving", "a person with glasses portrait"],
    },
    "chibi": {
        "desc": "chibi anime style, big head small body, large shiny eyes, cute proportions, cel shading, pastel colors",
        "ref_subjects": ["a girl character", "a boy character", "an old woman character", "a man with glasses character",
                         "a cat character", "a puppy character", "a character waving", "a character laughing"],
    },
    "toy3d": {
        "desc": "3D toy figure render, glossy vinyl material, soft studio lighting, rounded shapes, collectible designer toy look",
        "ref_subjects": ["a woman figure", "a man figure", "an elderly man figure", "a woman with a bun figure",
                         "a cat figure", "a dog figure", "a figure waving", "a figure with headphones"],
    },
}


def face_prompts(n, rng):
    grid = list(itertools.product(AGES, GENDER, SKIN))
    rng.shuffle(grid)
    out = []
    for i in range(n):
        age, g, skin = grid[i % len(grid)]
        p = (f"Realistic smartphone photo portrait of a {skin} {g} {age} with {rng.choice(HAIR)}"
             f"{', ' + e if (e := rng.choice(EXTRA)) else ''}, {rng.choice(POSE)}, head and shoulders, "
             f"{rng.choice(LIGHT)}, {rng.choice(BG)}, natural skin texture, sharp focus")
        out.append(p)
    return out


def run(pipe, prompts, outdir, seed0, tag):
    outdir.mkdir(parents=True, exist_ok=True)
    man = open(outdir / "manifest.jsonl", "w")
    for b in range(0, len(prompts), BS):
        chunk = prompts[b:b + BS]
        seeds = [seed0 + b + k for k in range(len(chunk))]
        gens = [torch.Generator("cuda").manual_seed(s) for s in seeds]
        imgs = pipe(prompt=chunk, width=SIZE, height=SIZE, num_inference_steps=STEPS, guidance_scale=1.0, generator=gens).images
        for k, (im, p, s) in enumerate(zip(imgs, chunk, seeds)):
            name = f"{tag}_{b + k:03d}.png"
            im.save(outdir / name)
            man.write(json.dumps({"file": name, "prompt": p, "seed": s, "model": MODEL, "steps": STEPS, "size": SIZE}) + "\n")
    man.close()


def main():
    rng = random.Random(1234)
    t = time.time()
    pipe = Flux2KleinPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, cache_dir=os.environ["KLEIN_CACHE"] + "/hub").to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"loaded {time.time() - t:.0f}s", flush=True)

    run(pipe, face_prompts(120, rng), ROOT / "data/faces", 10_000, "face")
    run(pipe, [f"Realistic smartphone photo of {s}, centered, full subject visible, natural light, casual background" for s in SUBJECTS],
        ROOT / "data/subjects", 20_000, "subj")
    for si, (name, st) in enumerate(STYLES.items()):
        d = ROOT / "styles" / name
        run(pipe, [f"{s}, {st['desc']}, centered, plain light background" for s in st["ref_subjects"]], d, 30_000 + 100 * si, "ref")
        cfg = {
            "name": name,
            "description": st["desc"],
            "refs": sorted(p.name for p in d.glob("ref_*.png")),
            "default_ref": "ref_000.png",
            # task prompts; {desc} is filled with description. Used by text-conditioned candidates.
            "avatar_prompt": "Turn the person in the photo into an avatar portrait in this style: {desc}. Keep the same face, hairstyle, "
                             "skin tone, accessories and expression. Head and shoulders, centered, plain background.",
            "sticker_prompt": "Turn the subject in the photo into a die-cut chat sticker in this style: {desc}. Keep identity and key features, "
                              "full subject, thick white outline around the subject, isolated on a plain flat white background.",
            "lora": None,  # Phase 3: path to a style LoRA (optional)
            "license": "self-generated with FLUX.2-klein-4B (Apache-2.0); prompts/seeds in manifest.jsonl",
        }
        (d / "style.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"done {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
