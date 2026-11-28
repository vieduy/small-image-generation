"""Ghost-hand metric, step 2 (2026-11-27): rates from results/s18_owl_detections.json with thresholds fixed after the eye check of
grids/s18_owl_check.jpg and BEFORE computing any rate: hand = OWLv2 box score >= 0.12, boxes merged when >= 60% of the smaller lies inside
the larger; one-handed options thumbs / wave / ok (think excluded: chin hand). Per model (base inputs, 300 stickers):
  extra-hand rate = >= 2 hands (main ghost metric; the teacher's rate = legitimate two-hand drawings), no-hand rate = 0 hands,
  faint-hand rate = any hand with opacity < 0.15 (noisy). Detector limits (eye check): faint white smears are missed -> lower bound.
Hypothesis re-test (rule of s15): teacher UNSTABLE = set of hand sides (box centre x < 256 -> L) on any of 4 small input variants differs
from base; CONFIRMED if extra-hand rate (tS4 + tS4o pooled) on unstable cases >= 2x stable and Fisher p < 0.05; REJECTED if ratio < 1.5 or
p >= 0.2; else INCONCLUSIVE.  python scripts/s18_ghost_rate.py  -> results/s18_ghost_rate.json"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact
ROOT = Path(__file__).resolve().parent.parent
D = {}
for f_ in sorted((ROOT / "results").glob("s18_owl_detections*.json")):  # 2026-11-27: + per-run files of filtered runs
    D.update(json.load(open(f_)))
OPTS, SMALL = ["thumbs", "wave", "ok"], ["v1", "v2", "v3", "v4"]
MODELS = ["teacher", "dl6g2", "tS4", "tS4o"] + sorted({k.split("/")[0] for k in D} - {"teacher", "dl6g2", "tS4", "tS4o"})  # 2026-11-27: + new students (tS6)


def hands(k):
    bs = sorted([b for b in D.get(k, []) if b["score"] >= 0.12], key=lambda b: -b["score"])
    out = []
    for b in bs:
        x0, y0, x1, y1 = b["box"]
        a = (x1 - x0) * (y1 - y0)
        dup = False
        for o in out:
            X0, Y0, X1, Y1 = o["box"]
            iw, ih = max(0, min(x1, X1) - max(x0, X0)), max(0, min(y1, Y1) - max(y0, Y0))
            if iw * ih >= 0.6 * min(a, (X1 - X0) * (Y1 - Y0)):
                dup = True
                break
        if not dup:
            out.append(b)
    return out


def sides(k):
    return "".join(sorted({"L" if (h["box"][0] + h["box"][2]) / 2 < 256 else "R" for h in hands(k)})) or "-"


faces = sorted({k.split("/")[2].rsplit("_", 1)[0] for k in D if k.startswith("teacher/base/")})
res = {"rule": __doc__.split("Hypothesis")[0].strip(), "models": {}}
for m in MODELS:
    H = [hands(f"{m}/base/{f}_{o}.png") for f in faces for o in OPTS]
    n = len(H)
    per = {o: round(float(np.mean([len(hands(f"{m}/base/{f}_{o}.png")) >= 2 for f in faces])), 3) for o in OPTS}
    res["models"][m] = {"n": n, "extra_hand_rate": round(float(np.mean([len(h) >= 2 for h in H])), 3),
                        "extra_hand_by_option": per,
                        "no_hand_rate": round(float(np.mean([len(h) == 0 for h in H])), 3),
                        "faint_hand_rate": round(float(np.mean([any(x["opacity"] < 0.15 for x in h) for h in H])), 3),
                        "side_unstable_rate": round(float(np.mean([any(sides(f"{m}/{v}/{f}_{o}.png") != sides(f"{m}/base/{f}_{o}.png") for v in SMALL)
                                                                    for f in faces for o in OPTS])), 3)}
tab = np.zeros((2, 2), int)
for f in faces:
    for o in OPTS:
        u = any(sides(f"teacher/{v}/{f}_{o}.png") != sides(f"teacher/base/{f}_{o}.png") for v in SMALL)
        for m in ("tS4", "tS4o"):
            tab[0 if u else 1, 0 if len(hands(f"{m}/base/{f}_{o}.png")) >= 2 else 1] += 1
gu, gs = tab[0, 0] / max(tab[0].sum(), 1), tab[1, 0] / max(tab[1].sum(), 1)
ratio = gu / gs if gs else float("inf")
p = fisher_exact(tab)[1]
res["hypothesis"] = {"table_unstable_stable_x_extra_noextra": tab.tolist(), "extra_unstable": round(float(gu), 3), "extra_stable": round(float(gs), 3),
                     "ratio": round(float(ratio), 2), "fisher_p": round(float(p), 4),
                     "verdict": "CONFIRMED" if ratio >= 2 and p < 0.05 else ("REJECTED" if ratio < 1.5 or p >= 0.2 else "INCONCLUSIVE")}
(ROOT / "results/s18_ghost_rate.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "rule"}, indent=1))
