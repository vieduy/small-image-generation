#!/bin/bash
# tSP0c (2026-11-27, user "yes run tSP0c"): = s27_tSP0b.sh with scripts/train_dl_pose3.py (+ map-active flag channel, adapter in-ch 4).
# Rule: CONTINUE (full 12k tSP1) if follow rate >= 0.70 AND extra-hand with the map <= extra-hand without (verdict_with_extra_hand_rule).
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-s28 TRITON_CACHE_DIR=/tmp/triton-cache-s28
P=$VENV/bin/python
L=logs/s28; mkdir -p $L
export CUDA_VISIBLE_DEVICES=0
echo "S28_START $(date -Is)"
NAME=teacher_tB1_s8_img3.0_k1
D=data/pairs/cartoon3d
ARGS="--style cartoon3d --manifest manifest_v2.jsonl --expr --init-from tS4 --lr 2e-5 --val-every 1000
  --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --gan-w 0.2 --gan-start 200
  --teacher-latents $D/$NAME.pt --drop-list $D/${NAME}_consistency_drop.json --pose-maps $D/pose_maps_dense_$NAME.pt
  --pose-lr 1e-3 --pose-scale 4 --gesture-frac 0.5"
$P scripts/train_dl_pose3.py $ARGS --steps 30 --save-at 20 --tag _smk_tSP0c > $L/smoke.log 2>&1 || { echo "S28_FAIL smoke"; exit 1; }
[ -f weights/student/_smk_tSP0c_s20/pose_adapter.pt ] && [ -f weights/student/_smk_tSP0c/pose_adapter.pt ] || { echo "S28_FAIL smoke files"; exit 1; }
rm -rf weights/student/_smk_tSP0c weights/student/_smk_tSP0c_s20; echo "S28_SMOKE_OK $(date -Is)"
$P scripts/train_dl_pose3.py $ARGS --steps 2000 --tag tSP0c > $L/train.log 2>&1 || { echo "S28_FAIL train"; exit 1; }
echo "S28_TRAIN_DONE $(date -Is)"
$P scripts/s24_swap_test.py tSP0c pose_templates_dense_$NAME.pt 4 > $L/swap.log 2>&1 || { echo "S28_FAIL swap"; exit 1; }
chmod -R a+rwX weights/student/tSP0c $L results grids 2>/dev/null
echo "S28_ALL_DONE $(grep -o "verdict_with_extra_hand_rule[^,]*" $L/swap.log) $(date -Is)"
