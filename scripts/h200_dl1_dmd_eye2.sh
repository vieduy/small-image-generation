#!/bin/bash
# DreamLite eye fix v2 (user-approved 2026-11-17): v1 (--eye-luma-w 4, 4x4) brought the iris back but greyish and blurred the
# highlights. v2: weaker (2) and coarser (8x8) luminance term, continued from dl1_dmd (not from v1), 750 DMD steps.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
R=$(cat configs/dmd_recipe.txt); echo "RECIPE $R --eye-luma-w 2 --eye-luma-pool 8"
$P scripts/train_dmd.py --style cartoon3d --manifest manifest_round4.jsonl --expr --max-pairs 5000 --expr-frac 0.6 \
   --init-from dl1_dmd --tag dl1_dmd_eye2 --steps 750 --fake-warmup 200 --lr 1e-4 --ckpt-every 250 $R --eye-luma-w 2 --eye-luma-pool 8 \
   || { echo "EYE2_FAIL"; exit 1; }
$P scripts/expr_eval.py --tag dl1_dmd_eye2 || echo "EXPR_FAIL dl1_dmd_eye2"
for ES in v2 real; do
  EVAL_SET=$ES $P scripts/student_infer.py --tag dl1_dmd_eye2 --style cartoon3d --no-perf && \
    EVAL_SET=$ES $P scripts/eval_quality.py --tag student_dl1_dmd_eye2 --styles cartoon3d || echo "EVAL_FAIL $ES"
  EVAL_SET=$ES STYLE=cartoon3d TAGS="klein_personal student_c3_dmd3 student_dl1_dmd student_dl1_dmd_eye student_dl1_dmd_eye2" OUT=_dl1eye2_$ES \
    $P scripts/skin_compare.py || echo "SKIN_FAIL $ES"
done
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "DL1EYE2_DONE $(date -Is)"
