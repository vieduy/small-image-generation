#!/bin/bash
# Student retrain on DATA V2 (user request 2026-11-18 ~10:50 UTC): the root-cause fix (consistent teacher targets: prompt
# v4/v5, likeness + sticker filter, colour <= +17, one target per photo/option). manifest_v2.jsonl = 53,930 pairs
# (synthetic 30,471 + CelebA 5,803 + FFHQ adults 17,656). EXACT dl1 / dl1_dmd recipes, steps and evals, so the only change
# vs dl1 / dl1_dmd is the data:
#   dl2      = train_dl.py, 16k steps, bs 16 (as dl1)
#   dl2_dmd  = train_dmd.py from dl2, configs/dmd_recipe.txt, 5000-pair subset, 60% expressions, 2000 steps (as dl1_dmd)
# Evals: expr_eval + v2 / real / selfie (student_infer, eval_quality, skin_compare) + iris colour, vs dl1 / dl1_dmd / klein.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
M=manifest_v2.jsonl
evals() {  # $1 = student tag, $2 = comparison tags
  $P scripts/expr_eval.py --tag $1 || echo "EXPR_FAIL $1"
  for ES in v2 real selfie; do
    EVAL_SET=$ES $P scripts/student_infer.py --tag $1 --style cartoon3d --no-perf && \
      EVAL_SET=$ES $P scripts/eval_quality.py --tag student_$1 --styles cartoon3d || echo "EVAL_FAIL $1 $ES"
    EVAL_SET=$ES STYLE=cartoon3d TAGS="klein_personal $2 student_$1" OUT=_$1_$ES $P scripts/skin_compare.py || echo "SKIN_FAIL $1 $ES"
  done
}
# ---- stage 1: regression (as dl1)
$P scripts/train_dl.py --style cartoon3d --tag dl2 --manifest $M --expr --steps 16000 || { echo "DL2_FAIL train"; exit 1; }
echo "DL2_TRAIN_DONE $(date -Is)"
evals dl2 "student_dl1_dmd"   # dl1 has v2 outputs only; dl1_dmd has v2 + real + selfie
echo "DL2_DONE $(date -Is)"
# ---- stage 2: DMD pack (as dl1_dmd)
R=$(cat configs/dmd_recipe.txt); echo "RECIPE $R"
D="--style cartoon3d --manifest $M --expr --max-pairs 5000 --expr-frac 0.6 --init-from dl2 --fake-warmup 200 --lr 1e-4"
$P scripts/train_dmd.py $D --tag smoke_dl2_dmd --steps 3 --fake-warmup 2 --val-every 1000 $R || { echo "DL2DMD_SMOKE_FAIL"; exit 1; }
rm -rf weights/student/smoke_dl2_dmd
$P scripts/train_dmd.py $D --tag dl2_dmd --steps 2000 --ckpt-every 250 $R || { echo "DL2DMD_FAIL train"; exit 1; }
echo "DL2DMD_TRAIN_DONE $(date -Is)"
evals dl2_dmd "student_c3_dmd3 student_dl1_dmd"
$P scripts/iris_colour.py student_c3_dmd3 student_dl1 student_dl1_dmd student_dl2 student_dl2_dmd || echo "IRIS_FAIL"
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "DL2DMD_DONE $(date -Is)"
