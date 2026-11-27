#!/bin/bash
# Waits for stage 4 (tS1 evals) to free dev-duyhv's GPU, then measures teacher tB1 on real photos (likeness + skin).
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
until grep -q "S4_ALL_DONE\|S4_FAIL train\|S4_FAIL smoke" logs/s4_student.out; do sleep 60; done
CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/teacher_real.py --tag tB1 > logs/s4/teacher_real.log 2>&1 && echo "TREAL_DONE $(date -Is)" || echo "TREAL_FAIL $(date -Is)"
chmod -R a+rwX outputs/teacher_real results/teacher logs/s4 2>/dev/null
