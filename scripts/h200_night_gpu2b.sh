#!/bin/bash
# GPU2 after h200_night_gpu2.sh (an_b1_dmd): fv_b1_dmd
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
until grep -qaE "^N2_ALL_DONE" logs/h200_night_gpu2.log; do sleep 60; done
R=$(cat configs/dmd_recipe.txt); echo "RECIPE $R"
export EVAL_SET=v2
$P scripts/train_dmd.py --style flat_vector --init-from fv_b1 --tag fv_b1_dmd --steps 2000 --fake-warmup 200 --lr 4e-5 $R
$P scripts/student_infer.py --tag fv_b1_dmd --style flat_vector --no-perf
$P scripts/eval_quality.py --tag student_fv_b1_dmd --styles flat_vector
STYLE=flat_vector TAGS="klein_personal student_fv_lora student_fv_b1 student_fv_b1_dmd" OUT=_fv_b1_dmd $P scripts/skin_compare.py
chmod -R a+rwX outputs results weights logs 2>/dev/null
echo "N2B_ALL_DONE $(date -Is)"
