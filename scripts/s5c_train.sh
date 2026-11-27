#!/bin/bash
# tS2 (user 2026-11-25 "remove the skin filter pls"): waits for the merged real-photo teacher pairs (s5b_merge.sh), then
# tS2 = tS1 recipe + ALL 58,964 real-photo teacher pairs (no skin filter), DDP on dev-duyhv-2's 2 GPUs (global bs 16), 6000 steps
# (= tS1, one-variable comparison; not curve-based) -> evals on both GPUs (same set as tS1).
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-s5 TRITON_CACHE_DIR=/tmp/triton-cache-s5
P=$VENV/bin/python
L=logs/s5
until grep -q "S5B_MERGED\|S5B_FAIL" logs/s5b_merge.out 2>/dev/null; do sleep 60; done
grep -q S5B_MERGED logs/s5b_merge.out || { echo "S5C_FAIL no merged pairs"; exit 1; }
echo "S5C_START $(date -Is)"
CUDA_VISIBLE_DEVICES=0,1 $VENV/bin/torchrun --nproc_per_node 2 --master_port 29552 scripts/train_dl.py \
  --style cartoon3d --manifest manifest_v2.jsonl --expr --init-from dl6g2 --lr 2e-5 --val-every 1000 --bs 16 \
  --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --gan-w 0.2 --gan-start 200 \
  --teacher-latents data/pairs/cartoon3d/teacher_tB1_s8.pt --extra-pairs data/pairs/real_teacher/teacher_tB1_real.pt --extra-no-filter \
  --steps 6000 --tag tS2 > $L/train_tS2.log 2>&1 || { echo "S5C_FAIL train"; exit 1; }
echo "S5C_TRAIN_DONE $(date -Is)"
( CUDA_VISIBLE_DEVICES=0 $P scripts/eval_e1.py --runs tS2 || echo "S5C_FAIL eval_e1"
  CUDA_VISIBLE_DEVICES=0 $P scripts/expr_eval.py --tag tS2 --decoder decg2 || echo "S5C_FAIL expr"
  $P scripts/sharp_ratio.py cartoon3d dl6g2_decg2 tS1_decg2 tS2_decg2 > results/sharp_tS2_decg2.txt 2>&1
  CUDA_VISIBLE_DEVICES=0 EVAL_SET=v2 $P scripts/student_infer.py --tag tS2 --decoder decg2 --style cartoon3d --no-perf && \
  CUDA_VISIBLE_DEVICES=0 EVAL_SET=v2 $P scripts/eval_quality.py --tag student_tS2_decg2 --styles cartoon3d || echo "S5C_FAIL q v2" ) > $L/eval_a.log 2>&1 &
( for E in real selfie; do
    CUDA_VISIBLE_DEVICES=1 EVAL_SET=$E $P scripts/student_infer.py --tag tS2 --decoder decg2 --style cartoon3d --no-perf && \
    CUDA_VISIBLE_DEVICES=1 EVAL_SET=$E $P scripts/eval_quality.py --tag student_tS2_decg2 --styles cartoon3d || echo "S5C_FAIL q $E"
    if [ $E = real ]; then EVAL_SET=real STYLE=cartoon3d TAGS="klein_personal student_dl6g2_decg2 student_tS1_decg2 student_tS2_decg2" OUT=_tS2_decg2_real \
       $P scripts/skin_compare.py || echo "S5C_FAIL skin"; fi
  done ) > $L/eval_b.log 2>&1 &
wait
chmod -R a+rwX weights/student/tS2 outputs results grids $L 2>/dev/null
echo "S5C_ALL_DONE $(date -Is)"
