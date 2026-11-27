#!/bin/bash
# Stage 4 (user 2026-11-23 "do it"): one-step student on the 8-step teacher tB1's fixed-noise pairs (cartoon3d), dev-duyhv 1x H200.
# tS1 = dl6g2 fine-tuned with ONLY the target swapped (klein -> teacher; latent MSE + LPIPS256 + lp512 + eye + critic, same as dlB).
# lr 2e-5, 6000 steps: copied from dlB (dl6g2 -> 6000 @ 2e-5), not curve-based; val every 1000 decides.
# Teacher feature matching is a later run (one variable at a time).
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-s4 TRITON_CACHE_DIR=/tmp/triton-cache-s4
P=$VENV/bin/python
L=logs/s4; mkdir -p $L
ARGS="--style cartoon3d --manifest manifest_v2.jsonl --expr --init-from dl6g2 --lr 2e-5 --val-every 1000
  --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --gan-w 0.2 --gan-start 200
  --teacher-latents data/pairs/cartoon3d/teacher_tB1_s8.pt"
echo "S4_START $(date -Is)"
CUDA_VISIBLE_DEVICES=0 $P scripts/train_dl.py $ARGS --steps 20 --tag _smk_tS1 > $L/smoke.log 2>&1 || { echo "S4_FAIL smoke"; exit 1; }
rm -rf weights/student/_smk_tS1; echo "S4_SMOKE_OK $(date -Is)"
CUDA_VISIBLE_DEVICES=0 $P scripts/train_dl.py $ARGS --steps 6000 --tag tS1 > $L/train_tS1.log 2>&1 || { echo "S4_FAIL train"; exit 1; }
echo "S4_TRAIN_DONE $(date -Is)"
CUDA_VISIBLE_DEVICES=0 $P scripts/eval_e1.py --runs tS1 > $L/eval_e1.log 2>&1 || echo "S4_FAIL eval_e1"
CUDA_VISIBLE_DEVICES=0 $P scripts/expr_eval.py --tag tS1 --decoder decg2 > $L/expr.log 2>&1 || echo "S4_FAIL expr"
$P scripts/sharp_ratio.py cartoon3d dl6g2_decg2 tS1_decg2 > results/sharp_tS1_decg2.txt 2>&1 || echo "S4_FAIL sharp"
for E in v2 real selfie; do
  CUDA_VISIBLE_DEVICES=0 EVAL_SET=$E $P scripts/student_infer.py --tag tS1 --decoder decg2 --style cartoon3d --no-perf >> $L/quality.log 2>&1 && \
  CUDA_VISIBLE_DEVICES=0 EVAL_SET=$E $P scripts/eval_quality.py --tag student_tS1_decg2 --styles cartoon3d >> $L/quality.log 2>&1 || echo "S4_FAIL quality $E"
done
EVAL_SET=real STYLE=cartoon3d TAGS="klein_personal student_dl6g2_decg2 student_tS1_decg2" OUT=_tS1_decg2_real $P scripts/skin_compare.py > $L/skin.log 2>&1 || echo "S4_FAIL skin"
chmod -R a+rwX weights/student/tS1 outputs results grids $L 2>/dev/null
echo "S4_ALL_DONE $(date -Is)"
