#!/bin/bash
# Pose-conditioning 2k-step go / no-go (2026-11-27, user "yes go with the 2k step check"), LOCAL H100 GPU 0.
# tSP0 = tS4 + pose input (scripts/train_dl_pose.py: zero-init pose conv on conv_in, pose lr 1e-4; rest = tS4 recipe, lr 2e-5), 2000 steps;
# then scripts/s24_swap_test.py (template vs mirrored template vs no map; CONTINUE if hand side follows the map >= 70%).
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-s24 TRITON_CACHE_DIR=/tmp/triton-cache-s24
P=$VENV/bin/python
L=logs/s24; mkdir -p $L
export CUDA_VISIBLE_DEVICES=0
echo "S24_START $(date -Is)"
until grep -q "\[s23\] done" logs/s23/pose_maps.log; do grep -q Traceback logs/s23/pose_maps.log && { echo "S24_FAIL pose maps"; exit 1; }; sleep 30; done
echo "S24_MAPS_READY $(date -Is)"
NAME=teacher_tB1_s8_img3.0_k1
D=data/pairs/cartoon3d
ARGS="--style cartoon3d --manifest manifest_v2.jsonl --expr --init-from tS4 --lr 2e-5 --val-every 1000
  --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --gan-w 0.2 --gan-start 200
  --teacher-latents $D/$NAME.pt --drop-list $D/${NAME}_consistency_drop.json --pose-maps $D/pose_maps_$NAME.pt --pose-lr 1e-4"
$P scripts/train_dl_pose.py $ARGS --steps 30 --save-at 20 --tag _smk_tSP0 > $L/smoke.log 2>&1 || { echo "S24_FAIL smoke"; exit 1; }
[ -f weights/student/_smk_tSP0_s20/pose_conv.pt ] && [ -f weights/student/_smk_tSP0/pose_conv.pt ] || { echo "S24_FAIL smoke files"; exit 1; }
rm -rf weights/student/_smk_tSP0 weights/student/_smk_tSP0_s20; echo "S24_SMOKE_OK $(date -Is)"
$P scripts/train_dl_pose.py $ARGS --steps 2000 --tag tSP0 > $L/train_tSP0.log 2>&1 || { echo "S24_FAIL train"; exit 1; }
echo "S24_TRAIN_DONE $(date -Is)"
$P scripts/s24_swap_test.py tSP0 > $L/swap.log 2>&1 || { echo "S24_FAIL swap"; exit 1; }
echo "S24_ALL_DONE $(tail -1 $L/swap.log | cut -c1-300) $(date -Is)"
