#!/bin/bash
# Best-of-N gestures + retrain + DMD2 (2026-11-27, user "go best-of-N clean teacher pairs for gestures + retrain with DMD2"), dev-duyhv-gpu.
# 1) scripts/s31_best_of_n.py, 2 shards on GPU 0 / 1 in parallel; 2) scripts/s31_merge.py (gate: <= 25% of these rows dropped);
# 3) tS7 = tS4 + regression fine-tune on the cleaned pairs (tS4 recipe, lr 2e-5, 6000 steps - copied from tS1/tS3, unverified) + fast evals;
# 4) dmC = DMD2 (reg_w 0.25, = dmA recipe) from tS7, 4000 steps, snapshot 2000 + fast evals.
# Pass rule for dmC (= DMD rule): extra-hand <= 2.0% and no-hand <= 2%; expr_all >= 0.65; likeness within 0.01 of tS4; sharpness >= 0.69.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-s31 TRITON_CACHE_DIR=/tmp/triton-cache-s31
P=$VENV/bin/python
L=logs/s31; mkdir -p $L
echo "S31_START $(date -Is)"
mkdir -p data/pairs/cartoon3d/bon_tB1 && chmod a+rwX data/pairs/cartoon3d/bon_tB1
( CUDA_VISIBLE_DEVICES=0 $P scripts/s31_best_of_n.py --shard 0 --nshards 2 > $L/bon_0.log 2>&1 || echo "S31_FAIL bon 0" ) &
( CUDA_VISIBLE_DEVICES=1 $P scripts/s31_best_of_n.py --shard 1 --nshards 2 > $L/bon_1.log 2>&1 || echo "S31_FAIL bon 1" ) &
wait
[ -f data/pairs/cartoon3d/bon_tB1/shard_0.pt ] && [ -f data/pairs/cartoon3d/bon_tB1/shard_1.pt ] || { echo "S31_FAIL shards"; exit 1; }
export CUDA_VISIBLE_DEVICES=0
$P scripts/s31_merge.py > $L/merge.log 2>&1 || { echo "S31_GATE_FAIL $(tail -1 $L/merge.log | cut -c1-300)"; exit 1; }
echo "S31_MERGE_DONE $(grep '^\[s31m\]' $L/merge.log | cut -c1-300) $(date -Is)"
NAME=teacher_tB1_s8_img3.0_k1
D=data/pairs/cartoon3d
BASE="--style cartoon3d --manifest manifest_v2.jsonl --expr --val-every 1000 --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip
  --gan-w 0.2 --gan-start 200 --teacher-latents $D/${NAME}_bon.pt --drop-list $D/${NAME}_bon_drop.json"
$P scripts/train_dl.py $BASE --init-from tS4 --lr 2e-5 --steps 6000 --tag tS7 > $L/train_tS7.log 2>&1 || { echo "S31_FAIL train tS7"; exit 1; }
echo "S31_TS7_DONE $(date -Is)"
TAG=tS7 GPU=0 RUNS="tS7" bash scripts/s21_eval_fast.sh > logs/s21f_tS7.out 2>&1 || echo "S31_FAIL eval tS7"
echo "S31_TS7_EVAL_DONE $(date -Is)"
$P scripts/train_dl_dmd.py $BASE --init-from tS7 --lr 5e-6 --reg-w 0.25 --dmd-w 1 --fake-warmup 200 --steps 4000 --save-at 2000 --tag dmC \
  > $L/train_dmC.log 2>&1 || { echo "S31_FAIL train dmC"; exit 1; }
echo "S31_DMC_DONE $(date -Is)"
TAG=dmC GPU=0 RUNS="dmC_s2000 dmC" bash scripts/s21_eval_fast.sh > logs/s21f_dmC.out 2>&1 || echo "S31_FAIL eval dmC"
echo "S31_ALL_DONE $(date -Is)"
