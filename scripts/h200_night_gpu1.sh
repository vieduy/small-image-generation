#!/bin/bash
# Overnight chain GPU1 (user-approved 2026-11-16, deliver 09:00 GMT+7):
#  1) an_dmd4  : DMD colour fix on the old anime pack (run-2 settings + --color-w 4 --eye-w 8)
#  2) c3_dmd   : DMD pack (fresh LoRA r32) on base b1, cartoon3d, recipe from configs/dmd_recipe.txt
#  3) fv_b1_dmd: DMD on the flat_vector pack rebuilt on b1
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
echo "N1_START $(date -Is)"
$P scripts/train_dmd.py --style anime --init-from an_lora --tag an_dmd4 --steps 2000 --fake-warmup 200 \
   --dmd-w 4 --reg-w 0.25 --cfg 4.5 --lr 4e-5 --color-w 4 --eye-w 8
export EVAL_SET=v2
$P scripts/student_infer.py --tag an_dmd4 --style anime --no-perf
$P scripts/eval_quality.py --tag student_an_dmd4 --styles anime
STYLE=anime TAGS="klein_personal student_an_lora student_an_dmd2 student_an_dmd4" OUT=_anime_dmd4 $P scripts/skin_compare.py
chmod -R a+rwX outputs results weights logs 2>/dev/null
echo "DMD4_DONE $(date -Is)"
# recipe: written by the operator after judging an_dmd4 (fallback after 45 min: run-2 settings + colour terms)
for i in $(seq 45); do [ -f configs/dmd_recipe.txt ] && break; sleep 60; done
[ -f configs/dmd_recipe.txt ] || echo "--dmd-w 4 --reg-w 0.25 --cfg 4.5 --color-w 4 --eye-w 8" > configs/dmd_recipe.txt
R=$(cat configs/dmd_recipe.txt); echo "RECIPE $R"
until grep -qaE "^B1_TRAIN_DONE" logs/h200_base_v1.log; do sleep 60; done
$P scripts/train_dmd.py --style cartoon3d --manifest manifest_round3b.jsonl --max-pairs 3000 --init-from b1 --tag c3_dmd \
   --steps 2000 --fake-warmup 200 --lr 1e-4 $R
for ES in v2 selfie real; do
  export EVAL_SET=$ES
  $P scripts/student_infer.py --tag c3_dmd --style cartoon3d --no-perf
  $P scripts/eval_quality.py --tag student_c3_dmd --styles cartoon3d
  TAGS="klein_personal student_s5d student_b1 student_c3_dmd" OUT=_c3dmd_$ES STYLE=cartoon3d $P scripts/skin_compare.py
done
chmod -R a+rwX outputs results weights logs 2>/dev/null
echo "C3DMD_DONE $(date -Is)"
until grep -qaE "^B1_ALL_DONE" logs/h200_base_v1.log; do sleep 60; done
export EVAL_SET=v2
$P scripts/train_dmd.py --style flat_vector --init-from fv_b1 --tag fv_b1_dmd --steps 2000 --fake-warmup 200 --lr 4e-5 $R
$P scripts/student_infer.py --tag fv_b1_dmd --style flat_vector --no-perf
$P scripts/eval_quality.py --tag student_fv_b1_dmd --styles flat_vector
STYLE=flat_vector TAGS="klein_personal student_fv_lora student_fv_b1 student_fv_b1_dmd" OUT=_fv_b1_dmd $P scripts/skin_compare.py
chmod -R a+rwX outputs results weights logs 2>/dev/null
echo "N1_ALL_DONE $(date -Is)"
