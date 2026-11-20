#!/bin/bash
# SDXS student retrain on DATA V2 (user request 2026-11-18 ~13:50 UTC), GPU 1 of dev-duyhv, in parallel with dl2 (GPU 0).
# manifest_v2.jsonl (53,930 pairs). Exact SDXS recipes, so data is the only change:
#   sx2      = train_student.py --param direct, 16k steps (as b1)
#   sx2_dmd  = train_dmd.py from sx2, configs/dmd_recipe.txt, 5000-pair subset, 60% expressions, 2000 steps (as c3_dmd3)
# Evals: expr_eval + v2 / real / selfie (student_infer, eval_quality, skin_compare) vs c3_dmd3 / dl2 / klein.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=1
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
# ---- stage 1: regression (as b1)
$P scripts/train_student.py --style cartoon3d --tag sx2 --param direct --steps 16000 --expr --manifest $M || { echo "SX2_FAIL train"; exit 1; }
echo "SX2_TRAIN_DONE $(date -Is)"
evals sx2 "student_c3_dmd3"
echo "SX2_DONE $(date -Is)"
# ---- stage 2: DMD pack (as c3_dmd3)
R=$(cat configs/dmd_recipe.txt); echo "RECIPE $R"
D="--style cartoon3d --manifest $M --expr --max-pairs 5000 --expr-frac 0.6 --init-from sx2 --fake-warmup 200 --lr 1e-4"
$P scripts/train_dmd.py $D --tag smoke_sx2_dmd --steps 3 --fake-warmup 2 --val-every 1000 $R || { echo "SX2DMD_SMOKE_FAIL"; exit 1; }
rm -rf weights/student/smoke_sx2_dmd
$P scripts/train_dmd.py $D --tag sx2_dmd --steps 2000 --ckpt-every 250 $R || { echo "SX2DMD_FAIL train"; exit 1; }
echo "SX2DMD_TRAIN_DONE $(date -Is)"
evals sx2_dmd "student_c3_dmd3 student_dl2_dmd"
$P scripts/iris_colour.py student_c3_dmd3 student_sx2 student_sx2_dmd student_dl2_dmd || echo "IRIS_FAIL"
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "SX2DMD_DONE $(date -Is)"
