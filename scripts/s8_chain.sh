#!/bin/bash
# Stage-2 follow-up (2026-11-26, user: "yes do A first, when training pls: per-row consistency check ... and report evals split by
# input type"), dev-duyhv 1x H200.
#  0) split-by-input-type evals of the existing models (dl6g2, tS1, tS3) from saved outputs
#  1) sweep A: scripts/s8_attr_sweep.py (pick rule fixed in its docstring). STOP here unless its "proceed" is true.
#  2) teacher eval at the pick (8 steps, interval), GATE expr_all >= 0.70 (as s6)
#  3) 53,930 synthetic teacher pairs at the pick (fixed phone noise)
#  4) per-row consistency check vs klein (scripts/s8_consistency.py, drop if detail > 1.5x klein's on either view)
#  5) tS4 = tS3 recipe (dl6g2 init, lr 2e-5, 6000 steps: copied from tS1/tS3, not curve-based; tS3's val was still falling at 6000)
#     with the new pairs + the consistency drop list; then the tS3 evals + split evals for dl6g2 / tS1 / tS3 / tS4.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-s8 TRITON_CACHE_DIR=/tmp/triton-cache-s8
P=$VENV/bin/python
L=logs/s8; mkdir -p $L
export CUDA_VISIBLE_DEVICES=0
echo "S8_START $(date -Is)"
$P scripts/s8_split_eval.py --tags dl6g2 tS1 tS3 --out base > $L/split_base.log 2>&1 && echo "S8_SPLIT_BASE_DONE $(date -Is)" || echo "S8_FAIL split_base"
$P scripts/s8_attr_sweep.py > $L/attr_sweep.log 2>&1 || { echo "S8_FAIL sweep"; exit 1; }
S=results/teacher/tB1_attr_sweep.json
read PROCEED G K <<< $($P -c "import json; d=json.load(open('$S')); p=d['pick']; r=d['grid'].get(p, {}) if p else {}; print(int(d['proceed']), r.get('g_img', 0), r.get('k_img', 0))")
echo "S8_SWEEP_DONE proceed=$PROCEED g_img=$G k_img=$K $(date -Is)"
[ "$PROCEED" = "1" ] || { echo "S8_STOP rule: no setting beats base_img1 (see $S) $(date -Is)"; exit 0; }
NAME=teacher_tB1_s8_img${G}_k${K}
$P scripts/s8_teacher_eval.py --tag tB1 --g-img $G --g-txt 2.5 --steps 8 --k-img $K > $L/teacher_eval.log 2>&1 || { echo "S8_FAIL teacher_eval"; exit 1; }
R=results/teacher/tB1_final_img${G}_txt2.5_s8_k${K}.json
$P -c "import json,sys; e=json.load(open('$R'))['expr_acc']['all']; print('teacher expr_all', e); sys.exit(0 if e >= 0.70 else 1)" \
  || { echo "S8_GATE_FAIL teacher expr below 0.70 $(date -Is)"; exit 1; }
echo "S8_TEACHER_OK $(date -Is)"
$P scripts/s8_teacher_pairs.py --tag tB1 --g-img $G --g-txt 2.5 --k-img $K --steps 8 --name $NAME > $L/pairs.log 2>&1 || { echo "S8_FAIL pairs"; exit 1; }
echo "S8_PAIRS_DONE $(date -Is)"
$P scripts/s8_consistency.py --pairs $NAME > $L/consistency.log 2>&1 || { echo "S8_FAIL consistency"; exit 1; }
echo "S8_CONSISTENCY_DONE $(tail -1 $L/consistency.log | cut -c1-300) $(date -Is)"
ARGS="--style cartoon3d --manifest manifest_v2.jsonl --expr --init-from dl6g2 --lr 2e-5 --val-every 1000
  --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --gan-w 0.2 --gan-start 200
  --teacher-latents data/pairs/cartoon3d/$NAME.pt --drop-list data/pairs/cartoon3d/${NAME}_consistency_drop.json"
$P scripts/train_dl.py $ARGS --steps 20 --tag _smk_tS4 > $L/smoke.log 2>&1 || { echo "S8_FAIL smoke"; exit 1; }
rm -rf weights/student/_smk_tS4; echo "S8_SMOKE_OK $(date -Is)"
$P scripts/train_dl.py $ARGS --steps 6000 --tag tS4 > $L/train_tS4.log 2>&1 || { echo "S8_FAIL train"; exit 1; }
echo "S8_TRAIN_DONE $(date -Is)"
$P scripts/eval_e1.py --runs tS4 > $L/eval_e1.log 2>&1 || echo "S8_FAIL eval_e1"
$P scripts/expr_eval.py --tag tS4 --decoder decg2 > $L/expr.log 2>&1 || echo "S8_FAIL expr"
$P scripts/sharp_ratio.py cartoon3d dl6g2_decg2 tS1_decg2 tS3_decg2 tS4_decg2 > results/sharp_tS4_decg2.txt 2>&1 || echo "S8_FAIL sharp"
for E in v2 real selfie; do
  EVAL_SET=$E $P scripts/student_infer.py --tag tS4 --decoder decg2 --style cartoon3d --no-perf >> $L/quality.log 2>&1 && \
  EVAL_SET=$E $P scripts/eval_quality.py --tag student_tS4_decg2 --styles cartoon3d >> $L/quality.log 2>&1 || echo "S8_FAIL quality $E"
done
EVAL_SET=real STYLE=cartoon3d TAGS="klein_personal student_dl6g2_decg2 student_tS1_decg2 student_tS3_decg2 student_tS4_decg2" OUT=_tS4_decg2_real \
  $P scripts/skin_compare.py > $L/skin.log 2>&1 || echo "S8_FAIL skin"
$P scripts/s8_split_eval.py --tags dl6g2 tS1 tS3 tS4 --out tS4 > $L/split_tS4.log 2>&1 || echo "S8_FAIL split_tS4"
chmod -R a+rwX weights/student/tS4 data/pairs/cartoon3d/${NAME}* outputs results grids $L 2>/dev/null
echo "S8_ALL_DONE $(date -Is)"
