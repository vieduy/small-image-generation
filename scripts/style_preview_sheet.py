"""Contact sheet for the new-style previews (2026-11-20): per style, 6 synthetic photos x [input, avatar, sticker, happy,
thumbs, love], plus the keep-rate numbers from score (likeness, sticker cut-out, style DINO, skin dL / d_chroma if present).
  $VENV/bin/python scripts/style_preview_sheet.py chibi toy3d comic_line anime flat_vector
Out: grids/style_preview_<style>.jpg, results/style_preview.json
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parent.parent
COLS = ["avatar", "sticker", "expr_happy", "expr_thumbs", "expr_love"]
c, summ = 192, {}
for st in sys.argv[1:]:
    d = ROOT / f"data/pairs_v2/{st}_preview"
    rows = [json.loads(l) for l in open(d / "manifest.jsonl")]
    ids = sorted({r["id"] for r in rows})
    by = lambda k: [r for r in rows if (k == "all" or r["task"] == k or (k == "expr" and r["task"].startswith("expr_")))]
    summ[st] = {k: {"n": len(by(k)), "keep": round(float(np.mean([r["keep"] for r in by(k)])), 2),
                    "id_pass": round(float(np.mean([r["id_pass"] for r in by(k)])), 2),
                    "style_dino": round(float(np.mean([r["style_dino"] for r in by(k)])), 3),
                    "dL_median": float(np.median([r["dL"] for r in by(k) if r.get("dL") is not None] or [np.nan])),
                    "sticker_pass": round(float(np.mean([bool(r.get("pass_sticker")) for r in by(k) if r["task"] != "avatar"] or [np.nan])), 2)}
                for k in ("avatar", "sticker", "expr", "all")}
    show = ids[:6]
    g = Image.new("RGB", ((len(COLS) + 1) * c, 20 + len(show) * c), "white")
    dr = ImageDraw.Draw(g)
    for j, t in enumerate(["input"] + COLS):
        dr.text((j * c + 4, 4), t, fill="black")
    for i, pid in enumerate(show):
        g.paste(Image.open(ROOT / f"data/pairs/cartoon3d/inputs/{pid}.png").convert("RGB").resize((c, c)), (0, 20 + i * c))
        for j, t in enumerate(COLS):
            p = d / t / f"{pid}.png"
            if p.exists():
                g.paste(Image.open(p).convert("RGB").resize((c, c)), ((j + 1) * c, 20 + i * c))
    g.save(ROOT / f"grids/style_preview_{st}.jpg", quality=88)
    print(st, json.dumps(summ[st]["all"]), flush=True)
(ROOT / "results/style_preview.json").write_text(json.dumps(summ, indent=1))
