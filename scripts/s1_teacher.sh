#!/bin/bash
# Stage 1 (user 2026-11-23 "okay go with the plan you do"): DreamLite-base -> klein teacher tB1 (cartoon3d), dev-duyhv 1x H200.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
echo "S1_START $(date -Is)"
CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/train_teacher_dl.py --tag tB1 --styles cartoon3d --steps 20000 --bs 32 > logs/teacher_tB1.log 2>&1 \
  && echo "S1_TRAIN_DONE $(date -Is)" || echo "S1_FAIL $(date -Is)"
chmod -R a+rwX weights/teacher logs/teacher_tB1.log 2>/dev/null
