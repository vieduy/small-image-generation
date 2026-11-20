"""Eval set v2: 300 more synthetic eval faces with the SAME attribute generator as data/faces (gen_data.face_prompts),
new rng (5678) and seeds 400000+. Never used for training (training inputs use seeds 100000+/300000+).
  source env.sh; $VENV/bin/python scripts/gen_eval2.py
"""
import os, random, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
import gen_data as G  # noqa: E402
from diffusers import Flux2KleinPipeline  # noqa: E402

t = time.time()
pipe = Flux2KleinPipeline.from_pretrained(G.MODEL, torch_dtype=torch.bfloat16, cache_dir=os.environ["KLEIN_CACHE"] + "/hub").to("cuda")
pipe.set_progress_bar_config(disable=True)
G.run(pipe, G.face_prompts(300, random.Random(5678)), G.ROOT / "data/faces_eval2", 400_000, "fe2")
print(f"[eval2] 300 faces {time.time() - t:.0f}s", flush=True)
