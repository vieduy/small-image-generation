#!/bin/bash
# tS5 (2026-11-26, user picked "tS5: hand/prop loss"): = tS4 recipe (dl6g2 init, teacher pairs img3-k1, consistency drop list, same losses)
# + scripts/train_dl_hc.py --hard-crop-w 1 --hard-crop 160 (LPIPS + L1 on the hardest 160 px crop OUTSIDE the face: hands, props, ghost
# hands; weight 1 = lp512's weight, crop 160 ~ a hand at 512 px; both unverified). 12000 steps (tS4's curve: 12k ~ 18k), snapshot 6000.
# Then eval_e1 + expr_eval (tS5_s6000, tS5), sharpness, quality v2/real/selfie, split evals (dl6g2 / tS4 / tS5), visual vs tS4, report.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-s14 TRITON_CACHE_DIR=/tmp/triton-cache-s14
P=$VENV/bin/python
L=logs/s14; mkdir -p $L
export CUDA_VISIBLE_DEVICES=0
echo "S14_START $(date -Is)"
NAME=teacher_tB1_s8_img3.0_k1
ARGS="--style cartoon3d --manifest manifest_v2.jsonl --expr --init-from dl6g2 --lr 2e-5 --val-every 1000
  --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --gan-w 0.2 --gan-start 200
  --teacher-latents data/pairs/cartoon3d/$NAME.pt --drop-list data/pairs/cartoon3d/${NAME}_consistency_drop.json --hard-crop-w 1 --hard-crop 160"
$P scripts/train_dl_hc.py $ARGS --steps 12000 --save-at 6000 --tag tS5 > $L/train_tS5.log 2>&1 || { echo "S14_FAIL train"; exit 1; }
echo "S14_TRAIN_DONE $(date -Is)"
for R in tS5_s6000 tS5; do
  $P scripts/eval_e1.py --runs $R > $L/eval_e1_$R.log 2>&1 || echo "S14_FAIL eval_e1 $R"
  $P scripts/expr_eval.py --tag $R --decoder decg2 > $L/expr_$R.log 2>&1 || echo "S14_FAIL expr $R"
done
$P scripts/s14_report.py > $L/report.log 2>&1 || echo "S14_FAIL report"
$P scripts/sharp_ratio.py cartoon3d dl6g2_decg2 tS4_decg2 tS5_decg2 > results/sharp_tS5_decg2.txt 2>&1 || echo "S14_FAIL sharp"
for E in v2 real selfie; do
  EVAL_SET=$E $P scripts/student_infer.py --tag tS5 --decoder decg2 --style cartoon3d --no-perf >> $L/quality.log 2>&1 && \
  EVAL_SET=$E $P scripts/eval_quality.py --tag student_tS5_decg2 --styles cartoon3d >> $L/quality.log 2>&1 || echo "S14_FAIL quality $E"
done
$P scripts/s8_split_eval.py --tags dl6g2 tS4 tS5 --out tS5 > $L/split_tS5.log 2>&1 || echo "S14_FAIL split"
$P scripts/s5_visual_tS2.py dl6g2 tS4 tS5 > $L/visual.log 2>&1 || echo "S14_FAIL visual"
chmod -R a+rwX weights/student/tS5* outputs results grids $L 2>/dev/null
echo "S14_ALL_DONE $(date -Is)"
