#!/bin/bash
# Smoke for the new train_dl.py options (--ffl-w, --gan-w), run beside dl3 on GPU 0 at bs 8: code path, memory, loss scale.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
B="--style cartoon3d --manifest manifest_v2.jsonl --expr --init-from dl2 --lr 1e-5 --bs 8 --steps 100 --val-every 1000 --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25"
$P scripts/train_dl.py $B --tag smoke_ffl --ffl-w 1 || echo "FFLGAN_SMOKE_FAIL ffl"
$P scripts/train_dl.py $B --tag smoke_gan --gan-w 0.1 --gan-start 50 || echo "FFLGAN_SMOKE_FAIL gan"
echo "FFLGAN_SMOKE_DONE $(date -Is)"
