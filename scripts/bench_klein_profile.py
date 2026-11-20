"""klein time split + torch.compile check (2026-11-18): text-encoder vs full call at batch 8, then compiled transformer."""
import json, os, sys, time
from pathlib import Path
import torch
from PIL import Image
os.environ.setdefault("KLEIN_REF", "0")
sys.path.insert(0, str(Path(__file__).parent))
from klein_batch import KleinBatch
ROOT = Path(__file__).resolve().parent.parent
pr = json.load(open(ROOT / "data/consistency/prompts_shard0.json"))
jobs = [(Image.open(ROOT / "data/pairs/cartoon3d/inputs" / f"{k.split('/')[1]}.png").convert("RGB"), p, 7 + n)
        for n, (k, p) in enumerate(pr.items()) if k.startswith("v4/f")][:64]
kb = KleinBatch(); B = 8
def bench(tag):
    kb.run([j[0] for j in jobs[:B]], [j[1] for j in jobs[:B]], [j[2] for j in jobs[:B]])
    torch.cuda.synchronize(); t = time.time()
    for i in range(0, len(jobs), B):
        ch = jobs[i:i + B]; kb.run([j[0] for j in ch], [j[1] for j in ch], [j[2] for j in ch])
    torch.cuda.synchronize(); print(f"[prof] {tag}: {len(jobs) / (time.time() - t):.2f} img/s", flush=True)
bench("eager bs8")
torch.cuda.synchronize(); t = time.time()
with torch.inference_mode():
    for i in range(0, len(jobs), B):
        kb.p.encode_prompt(prompt=[j[1] for j in jobs[i:i + B]], device="cuda")
torch.cuda.synchronize(); te = time.time() - t
print(f"[prof] text encoder alone: {te / len(jobs) * 1000:.0f} ms/img", flush=True)
kb.p.transformer = torch.compile(kb.p.transformer, dynamic=False)
t = time.time(); bench("compiled bs8 (first call includes compile)"); print(f"[prof] compile+run wall {time.time() - t:.0f}s", flush=True)
bench("compiled bs8 (warm)")
print("[prof] DONE", flush=True)
