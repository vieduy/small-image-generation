#!/bin/bash
# Step 2 of the teacher style fix (user 2026-11-25: "yes do step 2 with rule pick"), dev-duyhv 1x H200, ~4.5 h.
# Rule pick from scripts/s6_cfg_style_sweep.py: teacher tB1 CFG img 1.0 / txt 2.5 at 8 steps (klein-level style 0.774,
# tune expr 0.794, likeness 0.253) instead of img 3 / txt 4 (style 0.669).
#  a) full teacher eval at the pick (eval_e1 24 expression faces + 400 avatars), GATE: expr_all >= 0.70 (the img3/txt4 teacher scored
#     0.727 at 8 steps; 0.03 = about the CI half-width) else stop and report.
#  b) all 53,930 synthetic teacher pairs at the pick (fixed phone noise) -> data/pairs/cartoon3d/teacher_tB1_s8_img1.pt (no real-photo pairs:
#     they added nothing in tS2).
#  c) tS3 = tS1 recipe with ONLY the teacher setting changed (no R1/R3 filter, as tS1): dl6g2 init, lr 2e-5, 6000 steps
#     (copied from tS1, not curve-based), then the tS1 evals + style scores.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-s6 TRITON_CACHE_DIR=/tmp/triton-cache-s6
P=$VENV/bin/python
L=logs/s6; mkdir -p $L
export CUDA_VISIBLE_DEVICES=0
echo "S6_START $(date -Is)"
$P scripts/teacher_eval.py --tag tB1 --g-img 1.0 --g-txt 2.5 --steps 8 > $L/teacher_eval.log 2>&1 || { echo "S6_FAIL teacher_eval"; exit 1; }
R=results/teacher/tB1_final_img1.0_txt2.5_s8.json
$P -c "import json,sys; e=json.load(open('$R'))['expr_acc']['all']; print('teacher expr_all', e); sys.exit(0 if e >= 0.70 else 1)" \
  || { echo "S6_GATE_FAIL teacher expr below 0.70 $(date -Is)"; exit 1; }
echo "S6_TEACHER_OK $(date -Is)"
$P scripts/s3_teacher_pairs.py --tag tB1 --g-img 1.0 --g-txt 2.5 --steps 8 --name teacher_tB1_s8_img1 > $L/pairs.log 2>&1 || { echo "S6_FAIL pairs"; exit 1; }
echo "S6_PAIRS_DONE $(date -Is)"
ARGS="--style cartoon3d --manifest manifest_v2.jsonl --expr --init-from dl6g2 --lr 2e-5 --val-every 1000
  --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --gan-w 0.2 --gan-start 200
  --teacher-latents data/pairs/cartoon3d/teacher_tB1_s8_img1.pt"
$P scripts/train_dl.py $ARGS --steps 20 --tag _smk_tS3 > $L/smoke.log 2>&1 || { echo "S6_FAIL smoke"; exit 1; }
rm -rf weights/student/_smk_tS3; echo "S6_SMOKE_OK $(date -Is)"
$P scripts/train_dl.py $ARGS --steps 6000 --tag tS3 > $L/train_tS3.log 2>&1 || { echo "S6_FAIL train"; exit 1; }
echo "S6_TRAIN_DONE $(date -Is)"
$P scripts/eval_e1.py --runs tS3 > $L/eval_e1.log 2>&1 || echo "S6_FAIL eval_e1"
$P scripts/expr_eval.py --tag tS3 --decoder decg2 > $L/expr.log 2>&1 || echo "S6_FAIL expr"
$P scripts/sharp_ratio.py cartoon3d dl6g2_decg2 tS1_decg2 tS3_decg2 > results/sharp_tS3_decg2.txt 2>&1 || echo "S6_FAIL sharp"
for E in v2 real selfie; do
  EVAL_SET=$E $P scripts/student_infer.py --tag tS3 --decoder decg2 --style cartoon3d --no-perf >> $L/quality.log 2>&1 && \
  EVAL_SET=$E $P scripts/eval_quality.py --tag student_tS3_decg2 --styles cartoon3d >> $L/quality.log 2>&1 || echo "S6_FAIL quality $E"
done
EVAL_SET=real STYLE=cartoon3d TAGS="klein_personal student_dl6g2_decg2 student_tS1_decg2 student_tS3_decg2" OUT=_tS3_decg2_real \
  $P scripts/skin_compare.py > $L/skin.log 2>&1 || echo "S6_FAIL skin"
chmod -R a+rwX weights/student/tS3 data/pairs/cartoon3d/teacher_tB1_s8_img1* outputs results grids $L 2>/dev/null
echo "S6_ALL_DONE $(date -Is)"
