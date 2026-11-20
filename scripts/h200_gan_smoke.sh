#!/bin/bash
# Smoke test of train_dmd.py --gan-w (tiny subset, 6 generator steps); runs next to dl1_det1 on GPU2. Output deleted afterwards.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
R=$(cat configs/dmd_recipe.txt)
$VENV/bin/python scripts/train_dmd.py --style cartoon3d --manifest manifest_round4.jsonl --expr --max-pairs 64 --expr-frac 0.6 \
   --init-from dl1 --tag smoke_gan --steps 6 --fake-warmup 3 --lr 1e-4 --val-every 1000 $R --reg-w 0.1 --fake-updates 5 --gan-w 5e-3 \
   && echo GAN_SMOKE_OK || echo GAN_SMOKE_FAIL
rm -rf weights/student/smoke_gan
