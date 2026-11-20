#!/bin/bash
# DMD variant 2: an_dmd plateaued by step ~750 (crisper lines only). Stronger distribution signal, weaker regression anchor.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
until grep -qaE "^DMD_DONE|^Traceback" logs/h200_dmd_anime.log; do sleep 60; done
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
$P scripts/train_dmd.py --style anime --init-from an_lora --tag an_dmd2 --steps 2000 --fake-warmup 200 --dmd-w 4 --reg-w 0.25 --cfg 4.5 --lr 4e-5
export EVAL_SET=v2
$P scripts/student_infer.py --tag an_dmd2 --style anime --no-perf
$P scripts/eval_quality.py --tag student_an_dmd2 --styles anime
STYLE=anime TAGS="klein_personal student_an_lora student_an_dmd student_an_dmd2" OUT=_anime_dmd2 $P scripts/skin_compare.py
echo DMD2_DONE $(date -Is)
