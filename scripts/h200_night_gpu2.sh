#!/bin/bash
# Overnight chain GPU2, after h200_base_v1.sh: an_b1_dmd = DMD on the anime pack rebuilt on b1 (recipe file, see gpu1 script)
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
until grep -qaE "^B1_ALL_DONE" logs/h200_base_v1.log; do sleep 60; done
until [ -f configs/dmd_recipe.txt ]; do sleep 60; done
R=$(cat configs/dmd_recipe.txt); echo "RECIPE $R"
export EVAL_SET=v2
$P scripts/train_dmd.py --style anime --init-from an_b1 --tag an_b1_dmd --steps 2000 --fake-warmup 200 --lr 4e-5 $R
$P scripts/student_infer.py --tag an_b1_dmd --style anime --no-perf
$P scripts/eval_quality.py --tag student_an_b1_dmd --styles anime
STYLE=anime TAGS="klein_personal student_an_lora student_an_dmd2 student_an_b1 student_an_b1_dmd" OUT=_an_b1_dmd $P scripts/skin_compare.py
chmod -R a+rwX outputs results weights logs 2>/dev/null
echo "N2_ALL_DONE $(date -Is)"
