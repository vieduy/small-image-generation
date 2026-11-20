#!/bin/bash
# v4 sticker/expression prompt test (user request 2026-11-18): old pipeline prompt vs v4 on 8 train + 6 FFHQ photos, 2 seeds,
# plain sticker + 4 expressions (cry, thumbs, love, shrug). Captions once (batched), then tasks split over the 2 GPUs.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
B="--tag stickA --settings current,v4,v5 --n-train 8 --n-ffhq 6 --nseed 2"
CUDA_VISIBLE_DEVICES=0 $P scripts/teacher_consistency.py caption $B || { echo CAPTION_FAIL; exit 1; }
run() { for t in $2; do CUDA_VISIBLE_DEVICES=$1 $P scripts/teacher_consistency.py gen $B --task $t && \
        CUDA_VISIBLE_DEVICES=$1 $P scripts/teacher_consistency.py score $B --task $t || echo "TASK_FAIL $t"; done; }
run 0 "sticker expr_cry expr_love" > logs/sticker_test_gpu0.log 2>&1 &
run 1 "expr_thumbs expr_shrug" > logs/sticker_test_gpu1.log 2>&1
wait
chmod -R a+rwX data/consistency_* data/ffhq_norm results grids 2>/dev/null
echo STICKER_TEST_DONE
