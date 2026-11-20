"""List the training pairs whose klein target lightened darker skin a lot (skin-tone fix "C", 2026-11-19).
Root cause: on real photos the student lightens the darkest quartile by +32 L* (klein +6.7) while on synthetic sets it matches
klein; the real-photo training pairs (CelebA/FFHQ, legal-approved) carry klein targets +31..+41 L* lighter for inputs with
skin L* < 50 (brightness filter bypassed on 2026-11-18). Rule: drop rows with input skin L* < LMAX and target dL > DMAX
(stored by gen_pairs_v2 in data/pairs_v2/{real,cartoon3d}/manifest.jsonl). Writes data/pairs/cartoon3d/drop_skin_lighten.json.
"""
import json, sys, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
LMAX, DMAX = float(sys.argv[1]) if len(sys.argv) > 1 else 55, float(sys.argv[2]) if len(sys.argv) > 2 else 15
man = [json.loads(l) for l in open(ROOT / "data/pairs/cartoon3d/manifest_v2.jsonl")]
meta = {}
for d in ("real", "cartoon3d"):
    for l in open(ROOT / f"data/pairs_v2/{d}/manifest.jsonl"):
        r = json.loads(l); meta[(r["id"], r["task"], r.get("var", 0))] = r
drop, cnt = [], collections.Counter()
for r in man:
    if not r.get("keep"):
        continue
    m = meta[(r["id"], r["task"], r.get("var", 0))]
    if m.get("skin_in") and m["skin_in"][0] < LMAX and m.get("dL") is not None and m["dL"] > DMAX:
        drop.append([r["id"], r["task"]]); cnt["real" if r["id"][0] in "rq" else "syn"] += 1
out = ROOT / "data/pairs/cartoon3d/drop_skin_lighten.json"
out.write_text(json.dumps(drop))
print(f"L_in<{LMAX} & dL>{DMAX}: {len(drop)} pairs listed {dict(cnt)} -> {out}")
