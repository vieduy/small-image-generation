#!/bin/bash
# tSP0b (2026-11-27, user "yes run tSP0b on dev-duyhv"): stronger pose injection after tSP0's STOP. scripts/train_dl_pose2.py: multi-level
# adapter (down_blocks 0/1/2, zero-init outputs, lr 1e-3), DENSE maps (s26) x 4, half of each batch from gesture rows; rest = tSP0 / tS4 recipe.
# 2000 steps, then scripts/s24_swap_test.py with the dense templates x 4. Rule (same as tSP0): CONTINUE (full run) if follow rate >= 0.70,
# else drop pose conditioning (focus: DMD2-anchored + best-of-N teacher pairs).
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-s27 TRITON_CACHE_DIR=/tmp/triton-cache-s27
P=$VENV/bin/python
L=logs/s27; mkdir -p $L
export CUDA_VISIBLE_DEVICES=0
echo "S27_START $(date -Is)"
NAME=teacher_tB1_s8_img3.0_k1
D=data/pairs/cartoon3d
ARGS="--style cartoon3d --manifest manifest_v2.jsonl --expr --init-from tS4 --lr 2e-5 --val-every 1000
  --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --gan-w 0.2 --gan-start 200
  --teacher-latents $D/$NAME.pt --drop-list $D/${NAME}_consistency_drop.json --pose-maps $D/pose_maps_dense_$NAME.pt
  --pose-lr 1e-3 --pose-scale 4 --gesture-frac 0.5"
$P scripts/train_dl_pose2.py $ARGS --steps 30 --save-at 20 --tag _smk_tSP0b > $L/smoke.log 2>&1 || { echo "S27_FAIL smoke"; exit 1; }
[ -f weights/student/_smk_tSP0b_s20/pose_adapter.pt ] && [ -f weights/student/_smk_tSP0b/pose_adapter.pt ] || { echo "S27_FAIL smoke files"; exit 1; }
rm -rf weights/student/_smk_tSP0b weights/student/_smk_tSP0b_s20; echo "S27_SMOKE_OK $(date -Is)"
$P scripts/train_dl_pose2.py $ARGS --steps 2000 --tag tSP0b > $L/train.log 2>&1 || { echo "S27_FAIL train"; exit 1; }
echo "S27_TRAIN_DONE $(date -Is)"
$P scripts/s24_swap_test.py tSP0b pose_templates_dense_$NAME.pt 4 > $L/swap.log 2>&1 || { echo "S27_FAIL swap"; exit 1; }
chmod -R a+rwX weights/student/tSP0b $L results grids 2>/dev/null
echo "S27_ALL_DONE $(grep '^\[s24\]' $L/swap.log | cut -c1-200) $(date -Is)"
