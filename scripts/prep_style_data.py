"""Training data for a new chat style from the v2 generator output (2026-11-20, chibi / comic_line).

data/pairs_v2/<style>/manifest.jsonl (gen_pairs_v2 merge: keep + selected = one target per photo/option)
  -> data/pairs/<style>/manifest_v2.jsonl  (same row format as data/pairs/cartoon3d/manifest_v2.jsonl: targets point into
     data/pairs_v2/<style>, inputs into data/pairs/cartoon3d/inputs via the per-row `input` field)
  -> latent cache (train_dl.build_cache_dl) + proper-flip latents (build_flip_cache.py --no-tgt512; the 512 px target cache is
     skipped: 40 GB per style, NFS is 97% full; train without --tgt-cache).
Used by: train_dl.py --style <style> (LoRA pack) and train_dl.py --joint-styles <style>,... (joint model).
  source env.sh; CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/prep_style_data.py comic_line
"""
import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent


def main():
    s = sys.argv[1]
    src = ROOT / "data/pairs_v2" / s
    d = ROOT / "data/pairs" / s
    d.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(src / "manifest.jsonl")]
    keep = [r for r in rows if r["keep"] and r.get("selected", True)]
    with open(d / "manifest_v2.jsonl", "w") as f:
        for r in keep:
            assert r["id"][0] in "fc", r["id"]  # synthetic + CelebA photos of data/pairs/cartoon3d/inputs
            o = {k: r.get(k) for k in ("id", "task", "var", "keep", "teacher_prompt", "d_chroma", "id_rank100")}
            o["target"] = f"../../pairs_v2/{s}/{r['task']}/{r['id']}{'_b' if r.get('var') else ''}.png"
            o["input"] = f"../cartoon3d/inputs/{r['id']}.png"
            o["src"] = "pairs_v2"
            f.write(json.dumps(o) + "\n")
    miss = [r for r in keep if not (src / r["task"] / f"{r['id']}{'_b' if r.get('var') else ''}.png").exists()]
    assert not miss, f"{len(miss)} targets missing, e.g. {miss[0]}"
    print(f"[prep] {s}: {len(keep)} pairs -> {d / 'manifest_v2.jsonl'}", flush=True)
    import dl_arch as DLA
    import train_dl as T
    from train_student import TASKS, expr_tasks
    vae = DLA.load_vae(torch.bfloat16).to("cuda").eval()
    c = T.build_cache_dl(s, vae, "manifest_v2.jsonl", TASKS + expr_tasks())
    print(f"[prep] latent cache: {len(c['task'])} rows", flush=True)
    del c, vae
    torch.cuda.empty_cache()
    subprocess.run([sys.executable, str(ROOT / "scripts/build_flip_cache.py"), "--style", s, "--manifest", "manifest_v2.jsonl",
                    "--expr", "--no-tgt512"], check=True)
    print(f"PREP_DONE {s}", flush=True)


if __name__ == "__main__":
    main()
