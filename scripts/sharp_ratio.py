"""Sharpness vs the teacher (2026-11-20, comic blur check): median Laplacian variance of student outputs / of the kept klein
targets of the same style, per expression (24 each; different photos -> distribution-level, approximate).
  python3 scripts/sharp_ratio.py <style> <expr_eval out tag> [<expr_eval out tag> ...]   e.g. comic_line pk_comic_decg2 pk_comic_decCm
"""
import collections, glob, json, sys
import numpy as np
from PIL import Image


def lv(p):
    im = Image.open(p)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255)); bg.paste(im, mask=im.split()[3]); im = bg
    g = np.asarray(im.convert("L").resize((512, 512)), np.float32)
    l = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(l.var())


st, tags = sys.argv[1], sys.argv[2:]
rows = [json.loads(l) for l in open(f"data/pairs/{st}/manifest_v2.jsonl")]
by = collections.defaultdict(list)
for r in rows:
    by[r["task"]].append(r)
E = ["thanks", "shrug", "party", "ok", "wave", "facepalm", "thumbs", "happy", "wink", "cool"]
T = {e: np.median([lv(f"data/pairs/{st}/" + r["target"]) for r in by[f"expr_{e}"][:24]]) for e in E}
print(f"{'tag':22s} " + " ".join(f"{e:>8s}" for e in E) + "     mean")
for t in tags:
    rr = [np.median([lv(p) for p in sorted(glob.glob(f"outputs/expr_eval_{t}/{st}/*_{e}.png"))[:24]]) / T[e] for e in E]
    print(f"{t:22s} " + " ".join(f"{x:8.2f}" for x in rr) + f" {np.mean(rr):8.2f}")
