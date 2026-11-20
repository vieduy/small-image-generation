#!/bin/bash
# DMD fine-tune of the anime style pack (deviation 13): an_lora -> an_dmd, then the standard v2 eval.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
$P scripts/train_dmd.py --style anime --init-from an_lora --tag an_dmd --steps 2000 --fake-warmup 200
export EVAL_SET=v2
$P scripts/student_infer.py --tag an_dmd --style anime --no-perf
$P scripts/eval_quality.py --tag student_an_dmd --styles anime
STYLE=anime TAGS="klein_personal student_an_lora student_an_dmd" OUT=_anime_dmd $P scripts/skin_compare.py
echo DMD_DONE $(date -Is)
