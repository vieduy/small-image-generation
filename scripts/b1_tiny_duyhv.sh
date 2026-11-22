#!/bin/bash
# Plan B1 restarted on dev-duyhv (user 2026-11-20 ~14:05 UTC: "stop dev-duyhv-2, only use 2 gpu h200"). Same recipe as
# scripts/b1_tiny.sh (tdec1/tenc1 stopped at step ~1900/3200 on dev-duyhv-2); fresh start, new tags.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-b1 TRITON_CACHE_DIR=/tmp/triton-cache-b1
CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/train_tiny_dcae.py --part dec --tag tdec1r > logs/b1_tdec1r.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $VENV/bin/python scripts/train_tiny_dcae.py --part enc --tag tenc1r > logs/b1_tenc1r.log 2>&1 &
wait
chmod -R a+rwX weights/tiny_dcae logs/b1_* 2>/dev/null
echo "B1_DONE $(date -Is)"
