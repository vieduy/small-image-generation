#!/bin/bash
# anime / flat_vector finals on the selfie + real sets, with klein baselines (klein real/selfie for these styles not generated before)
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
for ES in selfie real; do
  export EVAL_SET=$ES
  for x in anime:an_b1_dmd flat_vector:fv_b1; do S=${x%%:*}; T=${x##*:}
    [ "$S:$ES" = "flat_vector:selfie" ] || VERIFY_EXTRA=1 PERSONAL=1 MODEL=klein KLEIN_REF=0 PROMPTS_FAMILY=klein_text $P scripts/eval_quality.py --tag klein_personal --styles $S --generate
    $P scripts/student_infer.py --tag $T --style $S --no-perf
    $P scripts/eval_quality.py --tag student_$T --styles $S
    STYLE=$S TAGS="klein_personal student_$T" OUT=_${T}_$ES $P scripts/skin_compare.py
  done
done
chmod -R a+rwX outputs results grids logs 2>/dev/null
echo "CHECK_REAL_DONE $(date -Is)"
