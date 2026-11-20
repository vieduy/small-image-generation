#!/bin/bash
# DMD variant 3: an_dmd2 crisp but lost iris colour + lighter skin. Same DMD strength, 2x regression anchor, CFG 3.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
:
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
$P scripts/train_dmd.py --style anime --init-from an_lora --tag an_dmd3 --steps 2000 --fake-warmup 200 --dmd-w 4 --reg-w 0.5 --cfg 3 --lr 4e-5
export EVAL_SET=v2
$P scripts/student_infer.py --tag an_dmd3 --style anime --no-perf
$P scripts/eval_quality.py --tag student_an_dmd3 --styles anime
STYLE=anime TAGS="klein_personal student_an_lora student_an_dmd student_an_dmd2 student_an_dmd3" OUT=_anime_dmd3 $P scripts/skin_compare.py
echo DMD3_DONE $(date -Is)
