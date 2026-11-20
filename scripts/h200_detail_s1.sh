#!/bin/bash
# Detail experiment step 1 (user-approved 2026-11-17): less averaging. Fresh DMD pack on dl1 like dl1_dmd, but regression
# anchor 0.25 -> 0.1 and fake critic 2 -> 5 updates per generator step (DMD2 TTUR). 1000 generator steps, compared with the
# dl1_dmd checkpoint at step 1000 (dl1_dmd_s1000): same data, start and step count, only these two settings differ.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
R=$(cat configs/dmd_recipe.txt); X="--reg-w 0.1 --fake-updates 5"; echo "RECIPE $R $X (later flags win)"
$P scripts/train_dmd.py --style cartoon3d --manifest manifest_round4.jsonl --expr --max-pairs 5000 --expr-frac 0.6 \
   --init-from dl1 --tag dl1_det1 --steps 1000 --fake-warmup 200 --lr 1e-4 --ckpt-every 250 $R $X || { echo "DET1_FAIL"; exit 1; }
for T in dl1_dmd_s1000 dl1_det1; do
  $P scripts/expr_eval.py --tag $T || echo "EXPR_FAIL $T"
  for ES in v2 real; do
    EVAL_SET=$ES $P scripts/student_infer.py --tag $T --style cartoon3d --no-perf && \
      EVAL_SET=$ES $P scripts/eval_quality.py --tag student_$T --styles cartoon3d || echo "EVAL_FAIL $T $ES"
  done
done
for ES in v2 real; do
  EVAL_SET=$ES STYLE=cartoon3d TAGS="klein_personal student_dl1_dmd student_dl1_dmd_s1000 student_dl1_det1" OUT=_det1_$ES $P scripts/skin_compare.py || echo "SKIN_FAIL $ES"
done
$P scripts/detail_bands.py student_c3_dmd3 student_dl1_dmd student_dl1_dmd_s1000 student_dl1_det1 || echo "BANDS_FAIL"
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "DET1_DONE $(date -Is)"
