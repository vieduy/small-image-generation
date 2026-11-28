"""Eye-landmark cache for train_dl_pose.py (2026-11-27): the per-run YuNet pass over all 54k targets took hours on the busy local CPU
(tS4o: 3.9 h setup). Same code as train_dl.py's eye block, run once (CPU only) -> data/pairs/<style>/_eyes_<manifest>.pt {key, eyes, eye_sig}.
  CUDA_VISIBLE_DEVICES= $VENV/bin/python scripts/s24_eye_cache.py cartoon3d manifest_v2.jsonl"""
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent))
from train_dl import build_cache_dl  # noqa: E402
from train_student import TASKS, expr_tasks  # noqa: E402
from metrics import FaceID  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
style, manifest = sys.argv[1], sys.argv[2]
t0 = time.time()
c = build_cache_dl(style, None, manifest, TASKS + expr_tasks())
n = len(c["task"])
eyes, eye_sig = torch.zeros(n, 2, 2), torch.zeros(n)
det = FaceID().det
det.setInputSize((256, 256))
for i in range(n):
    arr = cv2.cvtColor(np.ascontiguousarray(c["img256"][i].permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR)
    _, f = det.detect(arr)
    if f is None or len(f) == 0:
        continue
    f = max(f, key=lambda r: r[2] * r[3])
    eyes[i] = torch.tensor([[float(f[4]), float(f[5])], [float(f[6]), float(f[7])]])
    eye_sig[i] = 0.07 * float(f[2])
    if i % 10000 == 0:
        print(f"[eye] {i}/{n} {time.time() - t0:.0f}s", flush=True)
out = ROOT / "data/pairs" / style / f"_eyes_{Path(manifest).stem}.pt"
torch.save({"key": [tuple(k) for k in c["key"]], "eyes": eyes, "eye_sig": eye_sig}, out)
print(f"[eye] done {(eye_sig > 0).sum().item()}/{n} in {time.time() - t0:.0f}s -> {out}", flush=True)
