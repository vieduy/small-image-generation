#!/bin/bash
# #2 (user 2026-11-25 "do 2 first, use 2 idle gpu"), dev-duyhv-2 (2x H200):
#  smoke (pairs on 96 rows + 20-step train with them) -> real-photo teacher pairs on 2 GPUs -> merge (skin stats)
#  -> tS2 = tS1 recipe + real pairs with skin dL* in [-10, 10], DDP on 2 GPUs (global bs 16), 6000 steps (= tS1, for a one-variable
#     comparison; not curve-based) -> evals split over the 2 GPUs (same set as tS1).
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-s5 TRITON_CACHE_DIR=/tmp/triton-cache-s5
P=$VENV/bin/python
L=logs/s5; mkdir -p $L
ARGS="--style cartoon3d --manifest manifest_v2.jsonl --expr --init-from dl6g2 --lr 2e-5 --val-every 1000 --bs 16
  --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --gan-w 0.2 --gan-start 200
  --teacher-latents data/pairs/cartoon3d/teacher_tB1_s8.pt"
echo "S5_START $(date -Is)"
CUDA_VISIBLE_DEVICES=0 $P scripts/s5_real_pairs.py --max-rows 96 --nparts 1 > $L/smoke_pairs.log 2>&1 && \
CUDA_VISIBLE_DEVICES=0 $P scripts/s5_real_pairs.py --max-rows 96 --nparts 1 --merge >> $L/smoke_pairs.log 2>&1 || { echo "S5_FAIL smoke pairs"; exit 1; }
mv data/pairs/real_teacher/teacher_tB1_real.pt data/pairs/real_teacher/_smoke.pt
CUDA_VISIBLE_DEVICES=0 $P scripts/train_dl.py $ARGS --extra-pairs data/pairs/real_teacher/_smoke.pt --steps 20 --tag _smk_tS2 > $L/smoke_train.log 2>&1 \
  || { echo "S5_FAIL smoke train"; exit 1; }
rm -rf weights/student/_smk_tS2 data/pairs/real_teacher/_smoke.pt data/pairs/real_teacher/teacher_tB1_real.json
echo "S5_SMOKE_OK $(date -Is)"
CUDA_VISIBLE_DEVICES=0 $P scripts/s5_real_pairs.py --part 0 --nparts 2 > $L/pairs0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $P scripts/s5_real_pairs.py --part 1 --nparts 2 > $L/pairs1.log 2>&1 &
wait
$P scripts/s5_real_pairs.py --nparts 2 --merge > $L/merge.log 2>&1 || { echo "S5_FAIL pairs/merge"; exit 1; }
echo "S5_PAIRS_DONE $(date -Is)"
CUDA_VISIBLE_DEVICES=0,1 $VENV/bin/torchrun --nproc_per_node 2 --master_port 29551 scripts/train_dl.py $ARGS \
  --extra-pairs data/pairs/real_teacher/teacher_tB1_real.pt --steps 6000 --tag tS2 > $L/train_tS2.log 2>&1 || { echo "S5_FAIL train"; exit 1; }
echo "S5_TRAIN_DONE $(date -Is)"
( CUDA_VISIBLE_DEVICES=0 $P scripts/eval_e1.py --runs tS2 || echo "S5_FAIL eval_e1"
  CUDA_VISIBLE_DEVICES=0 $P scripts/expr_eval.py --tag tS2 --decoder decg2 || echo "S5_FAIL expr"
  $P scripts/sharp_ratio.py cartoon3d dl6g2_decg2 tS1_decg2 tS2_decg2 > results/sharp_tS2_decg2.txt 2>&1
  CUDA_VISIBLE_DEVICES=0 EVAL_SET=v2 $P scripts/student_infer.py --tag tS2 --decoder decg2 --style cartoon3d --no-perf && \
  CUDA_VISIBLE_DEVICES=0 EVAL_SET=v2 $P scripts/eval_quality.py --tag student_tS2_decg2 --styles cartoon3d || echo "S5_FAIL q v2" ) > $L/eval_a.log 2>&1 &
( for E in real selfie; do
    CUDA_VISIBLE_DEVICES=1 EVAL_SET=$E $P scripts/student_infer.py --tag tS2 --decoder decg2 --style cartoon3d --no-perf && \
    CUDA_VISIBLE_DEVICES=1 EVAL_SET=$E $P scripts/eval_quality.py --tag student_tS2_decg2 --styles cartoon3d || echo "S5_FAIL q $E"
    [ $E = real ] && { EVAL_SET=real STYLE=cartoon3d TAGS="klein_personal student_dl6g2_decg2 student_tS1_decg2 student_tS2_decg2" OUT=_tS2_decg2_real \
       $P scripts/skin_compare.py || echo "S5_FAIL skin"; }
  done ) > $L/eval_b.log 2>&1 &
wait
chmod -R a+rwX weights/student/tS2 data/pairs/real_teacher outputs results grids $L 2>/dev/null
echo "S5_ALL_DONE $(date -Is)"
