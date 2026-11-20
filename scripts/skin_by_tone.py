"""Skin-tone shift split by the INPUT's skin tone (2026-11-18, user report: Asian selfies come out darker / "Indian").
A single median dL hides regression to the mean (light skin darkened, dark skin lightened). Per eval set: input L* quartiles
(Q1 darkest .. Q4 lightest), median dL = L*(out) - L*(in) per quartile (+ = lighter output, - = darker output), and the slope
of L*(out) on L*(in) (1.0 = tones kept apart; < 1 = tones pulled toward the middle).
  EVAL_SET=real TAGS="klein_personal student_c3_dmd3 student_dl1_dmd" $VENV/bin/python scripts/skin_by_tone.py
"""
import json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
import metrics as M
from eval_quality import eval_items, EVAL_SET
ROOT = M.ROOT
TAGS = os.environ.get("TAGS", "klein_personal student_c3_dmd3 student_dl1_dmd").split()
fid = M.FaceID()
faces = [p for p, t in eval_items() if t == "avatar"]
Lin = {p.stem: fid.skin_L(Image.open(p)) for p in faces}
ok = sorted([p for p in faces if Lin[p.stem] is not None], key=lambda p: Lin[p.stem])
q = np.array_split(np.array([p.stem for p in ok]), 4)
qL = [round(float(np.median([Lin[s] for s in g])), 1) for g in q]
res = {"eval_set": EVAL_SET, "input_L_median_per_quartile": qL}
print(f"{EVAL_SET}: n={len(ok)}, input L* median per quartile Q1(dark)..Q4(light): {qL}")
for t in TAGS:
    li, lo, per = [], [], {}
    for p in ok:
        o = ROOT / "outputs" / t / "cartoon3d/avatar" / p.name
        if not o.exists():
            continue
        L = fid.skin_L(Image.open(o))
        if L is None:
            continue
        li.append(Lin[p.stem]); lo.append(L); per[p.stem] = L - Lin[p.stem]
    if len(li) < 20:
        continue
    slope = float(np.polyfit(li, lo, 1)[0])
    qd = [round(float(np.median([per[s] for s in g if s in per])), 1) for g in q]
    res[t] = {"n": len(li), "dL_median_Q1_to_Q4": qd, "slope_out_vs_in": round(slope, 2)}
    print(f"  {t:24s} dL per quartile Q1(dark)..Q4(light): {qd} | slope {slope:.2f} (n={len(li)})")
(ROOT / f"results/skin_by_tone_{EVAL_SET}.json").write_text(json.dumps(res, indent=1))
