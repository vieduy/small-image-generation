#!/bin/bash
# tS4 long run with checkpoint evals (2026-11-26, user "yes do #1, train tS4 longer with checkpoint evals").
# Replaces s10_chain.sh's training part (s10 controller stopped at ~04:01 UTC; its pair generation, pid 217131, kept running).
#  1) wait for the pairs (data/pairs/cartoon3d/teacher_tB1_s8_img3.0_k1.json is written last)
#  2) per-row consistency filter (s8_consistency.py, drop if > 1.5x klein detail)
#  3) tS4 = tS3 recipe + new pairs + drop list, 18000 steps (3x the copied 6000; to measure the curve), snapshots at 2000/6000/12000
#     (train_dl --save-at; mid-schedule, cosine lr not annealed) -> weights/student/tS4_s{2000,6000,12000}, final weights/student/tS4
#  4) per checkpoint: eval_e1 (expressions incl. per-option hands / props, likeness) + expr_eval (sticker id, outputs for sheets)
#  5) final tS4: sharp, quality v2/real/selfie, skin, split evals (dl6g2/tS1/tS3/tS4), visual sheets vs dl6g2, checkpoint report.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-s11 TRITON_CACHE_DIR=/tmp/triton-cache-s11
P=$VENV/bin/python
L=logs/s11; mkdir -p $L
export CUDA_VISIBLE_DEVICES=0
echo "S11_START $(date -Is)"
NAME=teacher_tB1_s8_img3.0_k1
while [ ! -f data/pairs/cartoon3d/$NAME.json ]; do
  pgrep -f "s8_teacher_pairs.py" > /dev/null || { sleep 30; [ -f data/pairs/cartoon3d/$NAME.json ] || { echo "S11_FAIL pairs process gone, no output"; exit 1; }; }
  sleep 60
done
echo "S11_PAIRS_DONE $(date -Is)"
$P scripts/s8_consistency.py --pairs $NAME > $L/consistency.log 2>&1 || { echo "S11_FAIL consistency"; exit 1; }
echo "S11_CONSISTENCY_DONE $(tail -1 $L/consistency.log | cut -c1-400) $(date -Is)"
ARGS="--style cartoon3d --manifest manifest_v2.jsonl --expr --init-from dl6g2 --lr 2e-5 --val-every 1000
  --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --gan-w 0.2 --gan-start 200
  --teacher-latents data/pairs/cartoon3d/$NAME.pt --drop-list data/pairs/cartoon3d/${NAME}_consistency_drop.json"
$P scripts/train_dl.py $ARGS --steps 30 --save-at 20 --tag _smk_tS4 > $L/smoke.log 2>&1 || { echo "S11_FAIL smoke"; exit 1; }
[ -f weights/student/_smk_tS4_s20/task_emb.pt ] || { echo "S11_FAIL smoke snapshot missing"; exit 1; }
rm -rf weights/student/_smk_tS4 weights/student/_smk_tS4_s20; echo "S11_SMOKE_OK $(date -Is)"
$P scripts/train_dl.py $ARGS --steps 18000 --save-at 2000,6000,12000 --tag tS4 > $L/train_tS4.log 2>&1 || { echo "S11_FAIL train"; exit 1; }
echo "S11_TRAIN_DONE $(date -Is)"
for R in tS4_s2000 tS4_s6000 tS4_s12000 tS4; do
  $P scripts/eval_e1.py --runs $R > $L/eval_e1_$R.log 2>&1 || echo "S11_FAIL eval_e1 $R"
  $P scripts/expr_eval.py --tag $R --decoder decg2 > $L/expr_$R.log 2>&1 || echo "S11_FAIL expr $R"
done
$P scripts/s11_ckpt_report.py > $L/ckpt_report.log 2>&1 || echo "S11_FAIL ckpt_report"
echo "S11_CKPT_EVALS_DONE $(date -Is)"
$P scripts/sharp_ratio.py cartoon3d dl6g2_decg2 tS1_decg2 tS3_decg2 tS4_decg2 > results/sharp_tS4_decg2.txt 2>&1 || echo "S11_FAIL sharp"
for E in v2 real selfie; do
  EVAL_SET=$E $P scripts/student_infer.py --tag tS4 --decoder decg2 --style cartoon3d --no-perf >> $L/quality.log 2>&1 && \
  EVAL_SET=$E $P scripts/eval_quality.py --tag student_tS4_decg2 --styles cartoon3d >> $L/quality.log 2>&1 || echo "S11_FAIL quality $E"
done
EVAL_SET=real STYLE=cartoon3d TAGS="klein_personal student_dl6g2_decg2 student_tS1_decg2 student_tS3_decg2 student_tS4_decg2" OUT=_tS4_decg2_real \
  $P scripts/skin_compare.py > $L/skin.log 2>&1 || echo "S11_FAIL skin"
$P scripts/s8_split_eval.py --tags dl6g2 tS1 tS3 tS4 --out tS4 > $L/split_tS4.log 2>&1 || echo "S11_FAIL split_tS4"
$P scripts/s5_visual_tS2.py tS1 tS3 dl6g2 tS4 > $L/visual.log 2>&1 || echo "S11_FAIL visual"
chmod -R a+rwX weights/student/tS4* data/pairs/cartoon3d/${NAME}* outputs results grids $L 2>/dev/null
echo "S11_ALL_DONE $(date -Is)"
