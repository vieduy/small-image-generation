#!/bin/bash
# tSP1 (2026-11-27, user "can you use [dev-duyhv] to accelerate the tSP0 plan"): full pose-conditioned student, started speculatively on
# dev-duyhv while tSP0's swap test (local) is pending; to be STOPPED if tSP0's swap test says STOP (< 70% follow rate).
# = tSP0 recipe (scripts/train_dl_pose.py: tS4 init, lr 2e-5, pose conv zero-init lr 1e-4, tS4 losses / pairs / drop list), 12000 steps
# (tS4 curve: 12k ~ 18k), snapshot 6000. Evals (pose-aware, templates) are a separate script.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-s25 TRITON_CACHE_DIR=/tmp/triton-cache-s25
P=$VENV/bin/python
L=logs/s25; mkdir -p $L
export CUDA_VISIBLE_DEVICES=0
echo "S25_START $(date -Is)"
NAME=teacher_tB1_s8_img3.0_k1
D=data/pairs/cartoon3d
$P scripts/train_dl_pose.py --style cartoon3d --manifest manifest_v2.jsonl --expr --init-from tS4 --lr 2e-5 --val-every 1000 \
  --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --gan-w 0.2 --gan-start 200 \
  --teacher-latents $D/$NAME.pt --drop-list $D/${NAME}_consistency_drop.json --pose-maps $D/pose_maps_$NAME.pt --pose-lr 1e-4 \
  --steps 12000 --save-at 6000 --tag tSP1 > $L/train_tSP1.log 2>&1 || { echo "S25_FAIL train"; exit 1; }
chmod -R a+rwX weights/student/tSP1* $L 2>/dev/null
echo "S25_TRAIN_DONE $(date -Is)"
