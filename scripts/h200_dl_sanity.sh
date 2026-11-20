#!/bin/bash
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
$VENV/bin/python scripts/dl_sanity.py
chmod -R a+rwX grids logs 2>/dev/null
