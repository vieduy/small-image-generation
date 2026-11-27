#!/bin/bash
# tS4 chain (2026-11-26, user: "yes start tS4 with g img 3 step 1, and compare with the best model we have for cartoon").
# = scripts/s8_chain.sh steps 2-5 with the teacher setting fixed to img guidance 3.0 on the first of 8 steps (txt 2.5) - the user's
# pick after s9 (sweep A's rule picked nothing; s9's rule picked img2_k2 on a lighting-confounded skin metric). Then a visual
# comparison vs dl6g2 (shipped cartoon3d model) + tS1 / tS3: scripts/s5_visual_tS2.py tS1 tS3 dl6g2 tS4.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-s10 TRITON_CACHE_DIR=/tmp/triton-cache-s10
P=$VENV/bin/python
L=logs/s10; mkdir -p $L
export CUDA_VISIBLE_DEVICES=0
echo "S10_START $(date -Is)"
G=3.0; K=1  # fixed by the user (README 2026-11-26 s9), not by the sweep rule
NAME=teacher_tB1_s8_img${G}_k${K}
$P scripts/s8_teacher_eval.py --tag tB1 --g-img $G --g-txt 2.5 --steps 8 --k-img $K > $L/teacher_eval.log 2>&1 || { echo "S10_FAIL teacher_eval"; exit 1; }
R=results/teacher/tB1_final_img${G}_txt2.5_s8_k${K}.json
$P -c "import json,sys; e=json.load(open('$R'))['expr_acc']['all']; print('teacher expr_all', e); sys.exit(0 if e >= 0.70 else 1)" \
  || { echo "S10_GATE_FAIL teacher expr below 0.70 $(date -Is)"; exit 1; }
echo "S10_TEACHER_OK $(date -Is)"
$P scripts/s8_teacher_pairs.py --tag tB1 --g-img $G --g-txt 2.5 --k-img $K --steps 8 --name $NAME > $L/pairs.log 2>&1 || { echo "S10_FAIL pairs"; exit 1; }
echo "S10_PAIRS_DONE $(date -Is)"
$P scripts/s8_consistency.py --pairs $NAME > $L/consistency.log 2>&1 || { echo "S10_FAIL consistency"; exit 1; }
echo "S10_CONSISTENCY_DONE $(tail -1 $L/consistency.log | cut -c1-300) $(date -Is)"
ARGS="--style cartoon3d --manifest manifest_v2.jsonl --expr --init-from dl6g2 --lr 2e-5 --val-every 1000
  --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --gan-w 0.2 --gan-start 200
  --teacher-latents data/pairs/cartoon3d/$NAME.pt --drop-list data/pairs/cartoon3d/${NAME}_consistency_drop.json"
$P scripts/train_dl.py $ARGS --steps 20 --tag _smk_tS4 > $L/smoke.log 2>&1 || { echo "S10_FAIL smoke"; exit 1; }
rm -rf weights/student/_smk_tS4; echo "S10_SMOKE_OK $(date -Is)"
$P scripts/train_dl.py $ARGS --steps 6000 --tag tS4 > $L/train_tS4.log 2>&1 || { echo "S10_FAIL train"; exit 1; }
echo "S10_TRAIN_DONE $(date -Is)"
$P scripts/eval_e1.py --runs tS4 > $L/eval_e1.log 2>&1 || echo "S10_FAIL eval_e1"
$P scripts/expr_eval.py --tag tS4 --decoder decg2 > $L/expr.log 2>&1 || echo "S10_FAIL expr"
$P scripts/sharp_ratio.py cartoon3d dl6g2_decg2 tS1_decg2 tS3_decg2 tS4_decg2 > results/sharp_tS4_decg2.txt 2>&1 || echo "S10_FAIL sharp"
for E in v2 real selfie; do
  EVAL_SET=$E $P scripts/student_infer.py --tag tS4 --decoder decg2 --style cartoon3d --no-perf >> $L/quality.log 2>&1 && \
  EVAL_SET=$E $P scripts/eval_quality.py --tag student_tS4_decg2 --styles cartoon3d >> $L/quality.log 2>&1 || echo "S10_FAIL quality $E"
done
EVAL_SET=real STYLE=cartoon3d TAGS="klein_personal student_dl6g2_decg2 student_tS1_decg2 student_tS3_decg2 student_tS4_decg2" OUT=_tS4_decg2_real \
  $P scripts/skin_compare.py > $L/skin.log 2>&1 || echo "S10_FAIL skin"
$P scripts/s8_split_eval.py --tags dl6g2 tS1 tS3 tS4 --out tS4 > $L/split_tS4.log 2>&1 || echo "S10_FAIL split_tS4"
chmod -R a+rwX weights/student/tS4 data/pairs/cartoon3d/${NAME}* outputs results grids $L 2>/dev/null
$P scripts/s5_visual_tS2.py tS1 tS3 dl6g2 tS4 > $L/visual.log 2>&1 || echo "S10_FAIL visual"
echo "S10_ALL_DONE $(date -Is)"
