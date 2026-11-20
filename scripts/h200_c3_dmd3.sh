#!/bin/bash
# c3_dmd3: combined hand fix = DMD pack on b1 with expressions, on manifest_round4 (5x gesture pairs, adult-age prompt),
# 60% expression rows; gesture progress sheets + loadable checkpoints every 250 steps. Doubles as the SDXS bake-off baseline.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
R=$(cat configs/dmd_recipe.txt); echo "RECIPE $R"
$P scripts/train_dmd.py --style cartoon3d --manifest manifest_round4.jsonl --expr --max-pairs 5000 --expr-frac 0.6 \
   --init-from b1 --tag c3_dmd3 --steps 2000 --fake-warmup 200 --lr 1e-4 --ckpt-every 250 $R
$P scripts/expr_eval.py --tag c3_dmd3 || echo "EXPR_FAIL c3_dmd3"
export EVAL_SET=v2
$P scripts/student_infer.py --tag c3_dmd3 --style cartoon3d --no-perf
$P scripts/eval_quality.py --tag student_c3_dmd3 --styles cartoon3d
TAGS="klein_personal student_b1 student_c3_dmd student_c3_dmd2 student_c3_dmd3" OUT=_c3dmd3_v2 STYLE=cartoon3d $P scripts/skin_compare.py
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "C3DMD3_DONE $(date -Is)"
