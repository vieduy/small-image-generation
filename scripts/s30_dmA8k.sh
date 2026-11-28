#!/bin/bash
# dmA -> 8k steps (2026-11-27, user "yes start dmA 8k on dev-duyhv"). dmA (DMD2 + reg_w 0.25, 4k steps) ended at extra-hand 2.3%, still falling
# (5.0 / 5.3 / 3.7 / 2.3% at 1k-4k). Continuation: init dmA, +4000 steps (= 8k total), same recipe (scripts/train_dl_dmd.py, lr 5e-6, fake lr 5e-6,
# 2 fake updates / step); the fake score was not saved by dmA -> re-initialised from tB1 with the 200-step warm-up; new cosine schedule.
# Snapshot +2000 (= 6k total). Fast evals (scripts/s21_eval_fast.sh). Pass rule (unchanged): extra-hand <= 2.0% and no-hand <= 2%;
# expr_all >= 0.65; likeness within 0.01 of tS4; sharpness >= 0.69.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-dmA8 TRITON_CACHE_DIR=/tmp/triton-cache-dmA8
P=$VENV/bin/python
L=logs/s21_dmA8; mkdir -p $L
export CUDA_VISIBLE_DEVICES=0
echo "S30_START $(date -Is)"
NAME=teacher_tB1_s8_img3.0_k1
D=data/pairs/cartoon3d
$P scripts/train_dl_dmd.py --style cartoon3d --manifest manifest_v2.jsonl --expr --init-from dmA --lr 5e-6 --val-every 1000 \
  --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --gan-w 0.2 --gan-start 200 \
  --teacher-latents $D/$NAME.pt --drop-list $D/${NAME}_consistency_drop.json --reg-w 0.25 --dmd-w 1 --fake-warmup 200 \
  --steps 4000 --save-at 2000 --tag dmA8 > $L/train.log 2>&1 || { echo "S30_FAIL train"; exit 1; }
echo "S30_TRAIN_DONE $(date -Is)"
TAG=dmA8 GPU=0 RUNS="dmA8_s2000 dmA8" bash scripts/s21_eval_fast.sh > logs/s21f_dmA8.out 2>&1 || echo "S30_FAIL eval"
echo "S30_ALL_DONE $(date -Is)"
