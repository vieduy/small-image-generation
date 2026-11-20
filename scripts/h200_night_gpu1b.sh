#!/bin/bash
# GPU1 (replaces the stopped h200_night_gpu1.sh after run 5): c3_dmd = DMD pack on base b1 (cartoon3d), recipe = run 5
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
R=$(cat configs/dmd_recipe.txt); echo "RECIPE $R"
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
