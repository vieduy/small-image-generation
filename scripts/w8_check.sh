#!/bin/bash
# int8-weight simulation (student_infer --fake-w8) for final models, v2 set; usage: w8_check.sh TAG:STYLE ...
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0 EVAL_SET=v2
for x in "$@"; do T=${x%%:*}; S=${x##*:}
  $VENV/bin/python scripts/student_infer.py --tag $T --style $S --no-perf --fake-w8
done
chmod -R a+rwX outputs logs 2>/dev/null
echo "W8_DONE $(date -Is)"
