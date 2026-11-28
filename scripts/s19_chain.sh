#!/bin/bash
# tS6 (2026-11-27, user "yes go with the fixed side fix tS6"), dev-duyhv-gpu GPU 0: fixed hand side for thumbs / wave / ok teacher pairs
# (scripts/s19_fixed_side.py) -> tS6 = tS4 recipe (dl6g2 init, lr 2e-5, 18000 steps = tS4's, same losses) with the fixed-side pairs + drop
# list -> tS4's evals + split evals + ghost metric (s15 inputs rendered by scripts/s19_student_s15.py, OWLv2 s18_owl_hands.py, s18_ghost_rate.py).
# Pass rule (fixed now): extra-hand rate on thumbs/wave/ok <= half of tS4's (4.0% -> <= 2.0%) and no-hand rate <= 2%, without losing
# expressions (eval_e1 all below tS4's CI lower bound 0.65) or likeness (v2 / real / selfie > 0.01 below tS4).
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-s19 TRITON_CACHE_DIR=/tmp/triton-cache-s19
P=$VENV/bin/python
L=logs/s19; mkdir -p $L
export CUDA_VISIBLE_DEVICES=0
echo "S19_START $(date -Is)"
NAME=teacher_tB1_s8_img3.0_k1
D=data/pairs/cartoon3d
$P scripts/s19_fixed_side.py > $L/fixed_side.log 2>&1 || { echo "S19_FAIL fixed_side"; exit 1; }
echo "S19_SIDE_DONE $(tail -1 $L/fixed_side.log | cut -c1-600) $(date -Is)"
$P -c "import json,sys; d=json.load(open('$D/${NAME}_side.json')); n=sum(v['rows'] for v in d['per_option'].values()); r=d['gesture_rows_dropped']/n; print('gesture rows dropped', d['gesture_rows_dropped'], n, round(r,3)); sys.exit(0 if r <= 0.25 else 1)" \
  || { echo "S19_GATE_FAIL too many gesture rows dropped $(date -Is)"; exit 1; }
ARGS="--style cartoon3d --manifest manifest_v2.jsonl --expr --init-from dl6g2 --lr 2e-5 --val-every 1000
  --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --gan-w 0.2 --gan-start 200
  --teacher-latents $D/${NAME}_side.pt --drop-list $D/${NAME}_side_drop.json"
$P scripts/train_dl.py $ARGS --steps 30 --tag _smk_tS6 > $L/smoke.log 2>&1 || { echo "S19_FAIL smoke"; exit 1; }
rm -rf weights/student/_smk_tS6; echo "S19_SMOKE_OK $(date -Is)"
$P scripts/train_dl.py $ARGS --steps 18000 --tag tS6 > $L/train_tS6.log 2>&1 || { echo "S19_FAIL train"; exit 1; }
echo "S19_TRAIN_DONE $(date -Is)"
$P scripts/eval_e1.py --runs tS6 > $L/eval_e1.log 2>&1 || echo "S19_FAIL eval_e1"
$P scripts/expr_eval.py --tag tS6 --decoder decg2 > $L/expr.log 2>&1 || echo "S19_FAIL expr"
$P scripts/sharp_ratio.py cartoon3d dl6g2_decg2 tS4_decg2 tS6_decg2 > results/sharp_tS6_decg2.txt 2>&1 || echo "S19_FAIL sharp"
for E in v2 real selfie; do
  EVAL_SET=$E $P scripts/student_infer.py --tag tS6 --decoder decg2 --style cartoon3d --no-perf >> $L/quality.log 2>&1 && \
  EVAL_SET=$E $P scripts/eval_quality.py --tag student_tS6_decg2 --styles cartoon3d >> $L/quality.log 2>&1 || echo "S19_FAIL quality $E"
done
$P scripts/s8_split_eval.py --tags dl6g2 tS4 tS6 --out tS6 > $L/split.log 2>&1 || echo "S19_FAIL split"
$P scripts/s19_student_s15.py tS6 > $L/render_s15.log 2>&1 || echo "S19_FAIL render_s15"
$P scripts/s18_owl_hands.py > $L/owl.log 2>&1 || echo "S19_FAIL owl"
$P scripts/s18_ghost_rate.py > $L/ghost_rate.log 2>&1 || echo "S19_FAIL ghost_rate"
$P scripts/s5_visual_tS2.py dl6g2 tS4 tS6 > $L/visual.log 2>&1 || echo "S19_FAIL visual"
chmod -R a+rwX weights/student/tS6 $D/${NAME}_side* outputs results grids $L 2>/dev/null
echo "S19_ALL_DONE $(date -Is)"
