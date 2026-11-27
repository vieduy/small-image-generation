"""Merge a LoRA style pack into its base student so the full-student evals (eval_e1, expr_eval, student_infer) can score it
(2026-11-27, comic pack pkC1). The phone still ships base + pack; this is an eval copy.
  python scripts/s16_merge_pack.py pkC1 [pk_comic ...]  ->  weights/student/<pack>_m/{unet, task_emb.pt, config.json}"""
import json
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))
import dl_arch as DLA  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
for tag in sys.argv[1:]:
    w, dst = ROOT / "weights/student" / tag, ROOT / "weights/student" / f"{tag}_m"
    if (dst / "unet").exists():
        print(f"[merge] {dst} exists, skipped", flush=True)
        continue
    cfg = json.loads((w / "config.json").read_text())
    base = cfg["init_from"]
    unet = PeftModel.from_pretrained(DLA.load_unet(ROOT / "weights/student" / base / "unet", style_film=False), w / "unet_lora").merge_and_unload()
    dst.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(dst / "unet")
    te = w / "task_emb.pt" if (w / "task_emb.pt").exists() else ROOT / "weights/student" / base / "task_emb.pt"
    shutil.copy(te, dst / "task_emb.pt")
    (dst / "config.json").write_text(json.dumps(cfg | {"lora_rank": 0, "merged_from": f"{base} + {tag}/unet_lora", "tag": f"{tag}_m"}, indent=1))
    print(f"[merge] {tag} -> {dst}", flush=True)
