#!/bin/bash
# c3_dmd2: cartoon3d DMD pack on b1 WITH the 20 expression tasks (hands/gestures were smeared; c3_dmd saw avatars+stickers only)
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
until grep -qaE "^CHECK_EXPR_DONE" logs/h200_check_expr.log; do sleep 30; done
R=$(cat configs/dmd_recipe.txt); echo "RECIPE $R"
$P scripts/train_dmd.py --style cartoon3d --manifest manifest_round3b.jsonl --expr --max-pairs 4000 --expr-frac 0.5 \
   --init-from b1 --tag c3_dmd2 --steps 2000 --fake-warmup 200 --lr 1e-4 $R
$P scripts/expr_eval.py --tag c3_dmd2 || echo "EXPR_FAIL c3_dmd2"
export EVAL_SET=v2
$P scripts/student_infer.py --tag c3_dmd2 --style cartoon3d --no-perf
$P scripts/eval_quality.py --tag student_c3_dmd2 --styles cartoon3d
TAGS="klein_personal student_b1 student_c3_dmd student_c3_dmd2" OUT=_c3dmd2_v2 STYLE=cartoon3d $P scripts/skin_compare.py
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "C3DMD2_DONE $(date -Is)"
