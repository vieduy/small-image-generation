"""Collect the base-probe results (scripts/h200_probe_base.sh) into one table.

  $VENV/bin/python scripts/probe_summary.py sdxs turbo sd15
Out: results/probe_base.csv, results/probe_base.md (rows = base x test A/B/C; missing pieces are left blank).
"""
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GEST = ["think", "thumbs", "wave", "ok", "facepalm", "shrug", "thanks"]
TESTS = {"A": "regression sprint (3k steps)", "B": "frozen-base LoRA (2k steps)", "C": "DMD on A (500 steps)"}


def last_quality(tag):
    out = {}
    f = ROOT / "results/quality_summary.jsonl"
    for r in map(json.loads, open(f)) if f.exists() else []:
        if r.get("tag") == f"student_{tag}" and r.get("eval_set") == "v2" and r.get("style") == "cartoon3d":
            out[r["task"]] = r  # later rows win
    return out


def val_curve(tag):
    f = ROOT / "weights/student" / tag / "train_log.jsonl"
    if not f.exists():
        return {}
    return {r["step"]: r["val_lpips"] for r in map(json.loads, open(f)) if "val_lpips" in r and "step" in r}


def sharpness(tag):
    d = ROOT / "outputs" / f"student_{tag}" / "cartoon3d" / "avatar"
    if not d.exists():
        return None
    v = [cv2.Laplacian(cv2.imread(str(p), 0), cv2.CV_64F).var() for p in sorted(d.glob("*.png"))]
    return round(float(np.median(v))) if v else None


def main():
    rows = []
    for n in sys.argv[1:]:
        skin = {}
        f = ROOT / f"results/skin_compare_probe_{n}.json"
        if f.exists():
            skin = json.loads(f.read_text())
        for t in TESTS:
            tag = f"pr_{n}_{t}"
            cfgf = ROOT / "weights/student" / tag / "config.json"
            cfg = json.loads(cfgf.read_text()) if cfgf.exists() else {}
            q = last_quality(tag)
            ex = {}
            ef = ROOT / f"results/expr_eval_{tag}.json"
            if ef.exists():
                ex = json.loads(ef.read_text())
            vc = val_curve(tag)
            s = skin.get(f"student_{tag}", {})
            rows.append({
                "base": n, "test": t, "what": TESTS[t], "base_repo": cfg.get("base", ""),
                "val_lpips_1k": vc.get(1000, ""), "val_lpips_2k": vc.get(2000, ""), "val_lpips_end": vc[max(vc)] if vc else "",
                "avatar_top5": q.get("avatar", {}).get("id_top5", ""), "sticker_top5": q.get("sticker", {}).get("id_top5", ""),
                "style_dino": q.get("avatar", {}).get("style_dino", ""),
                "skin_dL": s.get("dL_median", ""), "skin_dL_dark25": s.get("dL_dark25_median", ""),
                "sharpness": sharpness(tag) or "",
                "expr_acc": ex.get("_mean", {}).get("clip_expr_acc", ""),
                "gesture_acc": round(float(np.mean([ex[g]["clip_expr_acc"] for g in GEST])), 3) if ex else "",
                "minutes": cfg.get("minutes", ""),
            })
    if not rows:
        sys.exit("usage: probe_summary.py <name> [<name> ...]")
    with open(ROOT / "results/probe_base.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cols = ["base", "test", "val_lpips_1k", "val_lpips_2k", "val_lpips_end", "avatar_top5", "sticker_top5", "style_dino",
            "skin_dL", "skin_dL_dark25", "sharpness", "expr_acc", "gesture_acc", "minutes"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (ROOT / "results/probe_base.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
