#!/bin/bash
# Fast eval for a DMD run (2026-11-27, user "yes relaunch dmB evals the fast way"): same metrics as s21_dmd.sh's eval part, but
# checkpoints evaluated IN PARALLEL on the GPU, ghost renders on base inputs only (s21_render_base.py), expr_eval (sticker id) on the final
# model only, quality sets inferred sequentially then scored in parallel. SKIP = space-separated steps already done (e.g. "e1:dmB_s1000").
# Usage: TAG=dmB GPU=0 SKIP="e1:dmB_s1000 r:dmB_s1000 e1:dmB_s2000" bash scripts/s21_eval_fast.sh
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
L=logs/s21_$TAG; mkdir -p $L
export CUDA_VISIBLE_DEVICES=${GPU:-0}
echo "S21F_START $TAG $(date -Is)"
RUNS="${RUNS:-${TAG}_s1000 ${TAG}_s2000 ${TAG}_s3000 $TAG}"  # 2026-11-27: overridable
for R in $RUNS; do
  ( [[ " $SKIP " == *" e1:$R "* ]] || $P scripts/eval_e1.py --runs $R > $L/eval_e1_$R.log 2>&1 || echo "S21F_FAIL eval_e1 $R"
    [[ " $SKIP " == *" r:$R "* ]] || $P scripts/s21_render_base.py $R > $L/render_$R.log 2>&1 || echo "S21F_FAIL render $R" ) &
done
( $P scripts/expr_eval.py --tag $TAG --decoder decg2 > $L/expr_$TAG.log 2>&1 || echo "S21F_FAIL expr $TAG" ) &
( for E in v2 real selfie; do EVAL_SET=$E $P scripts/student_infer.py --tag $TAG --decoder decg2 --style cartoon3d --no-perf >> $L/infer.log 2>&1 || echo "S21F_FAIL infer $E"; done
  for E in v2 real selfie; do ( EVAL_SET=$E $P scripts/eval_quality.py --tag student_${TAG}_decg2 --styles cartoon3d > $L/quality_$E.log 2>&1 || echo "S21F_FAIL quality $E" ) & done; wait ) &
wait
echo "S21F_PARALLEL_DONE $(date -Is)"
$P scripts/s18_owl_hands.py $RUNS > $L/owl.log 2>&1 || echo "S21F_FAIL owl"
$P scripts/s18_ghost_rate.py > $L/ghost_rate.log 2>&1 || echo "S21F_FAIL ghost_rate"
$P scripts/sharp_ratio.py cartoon3d dl6g2_decg2 tS4_decg2 ${TAG}_decg2 > results/sharp_${TAG}_decg2.txt 2>&1 || echo "S21F_FAIL sharp"
$P scripts/s8_split_eval.py --tags dl6g2 tS4 $TAG --out $TAG > $L/split.log 2>&1 || echo "S21F_FAIL split"
$P scripts/s5_visual_tS2.py dl6g2 tS4 $TAG > $L/visual.log 2>&1 || echo "S21F_FAIL visual"
chmod -R a+rwX weights/student/${TAG}* outputs results grids $L 2>/dev/null
echo "S21F_ALL_DONE $(date -Is)"
