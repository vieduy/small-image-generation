#!/bin/bash
# DreamLite eye fix v3 (user-approved 2026-11-17): v1/v2 eye-loss weights could not counter DMD pushing eyes to solid black.
# v3: --eye-no-dmd blocks the DMD gradient inside the eye masks; eyes follow regression + eye chroma + mild 8x8 luminance.
# Continued from dl1_dmd, 750 DMD steps, otherwise the c3_dmd3 recipe. 3-step smoke run first (new code path).
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
R=$(cat configs/dmd_recipe.txt); X="--eye-no-dmd --eye-luma-w 2 --eye-luma-pool 8"; echo "RECIPE $R $X"
D="--style cartoon3d --manifest manifest_round4.jsonl --expr --max-pairs 5000 --expr-frac 0.6 --init-from dl1_dmd --lr 1e-4"
$P scripts/train_dmd.py $D --tag smoke_eye3 --steps 3 --fake-warmup 2 --val-every 1000 $R $X || { echo "EYE3_SMOKE_FAIL"; exit 1; }
rm -rf weights/student/smoke_eye3
$P scripts/train_dmd.py $D --tag dl1_dmd_eye3 --steps 750 --fake-warmup 200 --ckpt-every 250 $R $X || { echo "EYE3_FAIL"; exit 1; }
$P scripts/expr_eval.py --tag dl1_dmd_eye3 || echo "EXPR_FAIL dl1_dmd_eye3"
for ES in v2 real; do
  EVAL_SET=$ES $P scripts/student_infer.py --tag dl1_dmd_eye3 --style cartoon3d --no-perf && \
    EVAL_SET=$ES $P scripts/eval_quality.py --tag student_dl1_dmd_eye3 --styles cartoon3d || echo "EVAL_FAIL $ES"
  EVAL_SET=$ES STYLE=cartoon3d TAGS="klein_personal student_c3_dmd3 student_dl1_dmd student_dl1_dmd_eye2 student_dl1_dmd_eye3" OUT=_dl1eye3_$ES \
    $P scripts/skin_compare.py || echo "SKIN_FAIL $ES"
done
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "DL1EYE3_DONE $(date -Is)"
