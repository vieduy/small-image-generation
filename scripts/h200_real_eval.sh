#!/bin/bash
# Real-photo (300 legal-approved, identity-disjoint) + selfie (200 synthetic close-ups) evals: does DreamLite's lead hold
# outside the synthetic v2 set? klein_personal / b1 / c3_dmd baselines on these sets already exist.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
for ES in selfie real; do
  export EVAL_SET=$ES
  for T in c3_dmd3 dl1_dmd; do
    $P scripts/student_infer.py --tag $T --style cartoon3d --no-perf && $P scripts/eval_quality.py --tag student_$T --styles cartoon3d || echo "EVAL_FAIL $T $ES"
  done
  STYLE=cartoon3d TAGS="klein_personal student_b1 student_c3_dmd student_c3_dmd3 student_dl1_dmd" OUT=_dl1dmd_$ES $P scripts/skin_compare.py || echo "SKIN_FAIL $ES"
done
chmod -R a+rwX outputs results grids logs 2>/dev/null
echo "REAL_EVAL_DONE $(date -Is)"
