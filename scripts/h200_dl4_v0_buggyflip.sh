#!/bin/bash
# Detail follow-up to dl3 (user-approved 2026-11-19 ~03:30 UTC): the two untested loss ideas from the 2026-11-19 loss research,
# each as a 2000-step continuation of the dl3 main stage (lr 1e-5, the dl3 losses kept), against a control with no new term.
#   GPU 0 (after the dl3 chain):  dl4c = control (dl3 losses only)  ->  dl4f = + Focal Frequency Loss (--ffl-w 2, log matrix)
#   GPU 1 (after the dl3L chain): dl4g = + frozen DINOv2-base critic (--gan-w 0.1, critic-only warm-up 200 steps)
# Weights from the smoke (logs/fflgan_smoke.log): FFL ~0.045 -> x2 ~0.09; generator hinge ~0.88 -> x0.1 ~0.09; both about
# half the 256 px LPIPS term (~0.2). Note: the critic uses DINOv2-base, the same model as the style_dino metric, so style_dino
# is not an independent check for dl4g (use detail_bands, SFace likeness, CLIP expression accuracy, skin, eyes).
# Acceptance (as before): fine band up vs dl4c, likeness drop <= 0.05, no skin regression, no crosshatch at pixel zoom.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
L="--lp512-w 1 --eye-w 1 --eye-lpips-w 0.25"
B="--style cartoon3d --manifest manifest_v2.jsonl --expr --init-from dl3 --lr 1e-5 --steps 2000 --val-every 250 $L"
waitfor() { until grep -avE '^\+' logs/h200_dl3.log | grep -qaE "$1|DL3_ALL_DONE"; do sleep 60; done; }  # $1 = this chain's end/fail markers
evals() {  # $1 gpu, $2 tag
  g=$1; t=$2
  CUDA_VISIBLE_DEVICES=$g $P scripts/expr_eval.py --tag $t || echo "EXPR_FAIL $t"
  for ES in v2 real; do
    CUDA_VISIBLE_DEVICES=$g EVAL_SET=$ES $P scripts/student_infer.py --tag $t --style cartoon3d --no-perf && \
      CUDA_VISIBLE_DEVICES=$g EVAL_SET=$ES $P scripts/eval_quality.py --tag student_$t --styles cartoon3d || echo "EVAL_FAIL $t $ES"
  done
  EVAL_SET=real STYLE=cartoon3d TAGS="klein_personal student_dl3 student_$t" OUT=_${t}_real $P scripts/skin_compare.py || echo "SKIN_FAIL $t"
}
run() {  # $1 gpu, $2 tag, rest = extra args
  g=$1; t=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g $P scripts/train_dl.py $B --tag $t "$@" > logs/dl4_$t.log 2>&1 || { echo "DL4_FAIL $t train"; return 1; }
  echo "DL4_TRAIN_DONE $t $(date -Is)"
  evals $g $t >> logs/dl4_$t.log 2>&1
  echo "DL4_EVAL_DONE $t $(date -Is)"
}
waitfor "DL3_CHAIN_DONE dl3 |DL3_FAIL dl3 |DL3_FAIL dl3_2s "
[ -f weights/student/dl3/task_emb.pt ] || { echo "DL4_SKIP: no dl3 weights"; exit 1; }
( run 0 dl4c; run 0 dl4f --ffl-w 2 ) &
( waitfor "DL3_CHAIN_DONE dl3L |DL3_FAIL dl3L |DL3_FAIL dl3L_2s "; run 1 dl4g --gan-w 0.1 --gan-start 200 ) &
wait
until grep -qa DL3_ALL_DONE logs/h200_dl3.log; do sleep 60; done   # dl3's own detail/iris/sheet step ends first
T="student_dl3 student_dl4c student_dl4f student_dl4g"
$P scripts/detail_bands.py student_dl2 $T --eval-set v2 || echo "DETAIL_FAIL"
$P scripts/iris_colour.py klein_personal $T || echo "IRIS_FAIL"
$P scripts/zoom_detail_sheet.py --tags klein_personal,student_dl4c,student_dl4f,student_dl4g --names fe2_157,fe2_232,fe2_007,rp_232 --out grids/zoom_dl4.jpg || echo "SHEET_FAIL"
$P scripts/compare_students_sheet.py --tags klein_personal,student_dl4c,student_dl4f,student_dl4g --out grids/dl4_sheet.jpg || echo "SHEET_FAIL"
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "DL4_ALL_DONE $(date -Is)"
