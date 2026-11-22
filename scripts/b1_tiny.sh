#!/bin/bash
# Plan B1 (user 2026-11-20 "let training sana sprint with decoder tiny"): tiny DC-AE decoder (GPU 0) + encoder (GPU 1), dev-duyhv-2
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-b1 TRITON_CACHE_DIR=/tmp/triton-cache-b1
CUDA_VISIBLE_DEVICES=0 $VENV/bin/python scripts/train_tiny_dcae.py --part dec --tag tdec1 > logs/b1_tdec1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $VENV/bin/python scripts/train_tiny_dcae.py --part enc --tag tenc1 > logs/b1_tenc1.log 2>&1 &
wait
chmod -R a+rwX weights/tiny_dcae logs/b1_* 2>/dev/null
echo "B1_DONE $(date -Is)"
