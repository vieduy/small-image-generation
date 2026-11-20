#!/bin/bash
# DMD run 5: run 4 restored colour/skin/likeness but re-blurred (sharpness 931). Eye term chroma-only 4x4, colour weight halved.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
$P scripts/train_dmd.py --style anime --init-from an_lora --tag an_dmd5 --steps 2000 --fake-warmup 200 \
   --dmd-w 4 --reg-w 0.25 --cfg 4.5 --lr 4e-5 --color-w 2 --eye-w 4 --eye-chroma
export EVAL_SET=v2
$P scripts/student_infer.py --tag an_dmd5 --style anime --no-perf
$P scripts/eval_quality.py --tag student_an_dmd5 --styles anime
STYLE=anime TAGS="klein_personal student_an_lora student_an_dmd2 student_an_dmd4 student_an_dmd5" OUT=_anime_dmd5 $P scripts/skin_compare.py
chmod -R a+rwX outputs results weights logs 2>/dev/null
echo "DMD5_DONE $(date -Is)"
