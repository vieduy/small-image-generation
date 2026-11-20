#!/bin/bash
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
$VENV/bin/python scripts/train_dmd.py --style cartoon3d --manifest manifest_s5d.jsonl --init-from s5d --tag smoke_dmd \
  --steps 20 --fake-warmup 5 --max-pairs 64 --color-w 4 --eye-w 8 --val-every 10 --dmd-w 4 --reg-w 0.25 --cfg 4.5
echo SMOKE_DONE $(date -Is)
