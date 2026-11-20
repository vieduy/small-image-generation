#!/bin/bash
# Detail follow-up (user-approved 2026-11-19 ~03:30 UTC; REWIRED ~04:05 to start from dl3F, the proper-flip model, with
# --proper-flip --tgt-cache; first version kept as h200_dl4_v0_buggyflip.sh, never ran): the two untested loss ideas from the 2026-11-19 loss research,
# each as a 2000-step continuation of the dl3F main stage (lr 1e-5, the dl3 losses kept), against a control with no new term.
#   ROLE=cf (dev-duyhv-2 GPU 1, right after the dl3F main stage): dl4c = control (dl3 losses only) -> dl4f = + FFL (--ffl-w 2, log matrix)
#   ROLE=g  (dev-duyhv GPU 0, after the dl3 chain):            dl4g = + frozen DINOv2-base critic (--gan-w 0.1, critic-only warm-up 200 steps)
#   (2 extra H200 on dev-duyhv-2 since 2026-11-19 ~04:25 UTC; logs/h200_dl4_<role>.log). ROLE=cf also builds the final sheets.
# Weights from the smoke (logs/fflgan_smoke.log): FFL ~0.045 -> x2 ~0.09; generator hinge ~0.88 -> x0.1 ~0.09; both about
# half the 256 px LPIPS term (~0.2). Note: the critic uses DINOv2-base, the same model as the style_dino metric, so style_dino
# is not an independent check for dl4g (use detail_bands, SFace likeness, CLIP expression accuracy, skin, eyes).
# Acceptance (as before): fine band up vs dl4c, likeness drop <= 0.05, no skin regression, no crosshatch at pixel zoom.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
SPEED=${SPEED:-}
L="--lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --tgt-cache $SPEED"
B="--style cartoon3d --manifest manifest_v2.jsonl --expr --init-from dl3F --lr 1e-5 --steps 2000 --val-every 250 $L"
waitfor() { until cat logs/h200_dl3.log logs/h200_dl3F.log 2>/dev/null | grep -avE '^\+' | grep -qaE "$1"; do sleep 60; done; }
evals() {  # $1 gpu, $2 tag
  g=$1; t=$2
  CUDA_VISIBLE_DEVICES=$g $P scripts/expr_eval.py --tag $t || echo "EXPR_FAIL $t"
  for ES in v2 real; do
    CUDA_VISIBLE_DEVICES=$g EVAL_SET=$ES $P scripts/student_infer.py --tag $t --style cartoon3d --no-perf && \
      CUDA_VISIBLE_DEVICES=$g EVAL_SET=$ES $P scripts/eval_quality.py --tag student_$t --styles cartoon3d || echo "EVAL_FAIL $t $ES"
  done
  EVAL_SET=real STYLE=cartoon3d TAGS="klein_personal student_dl3F student_$t" OUT=_${t}_real $P scripts/skin_compare.py || echo "SKIN_FAIL $t"
}
run() {  # $1 gpu, $2 tag, rest = extra args
  g=$1; t=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g $P scripts/train_dl.py $B --tag $t "$@" > logs/dl4_$t.log 2>&1 || { echo "DL4_FAIL $t train"; return 1; }
  echo "DL4_TRAIN_DONE $t $(date -Is)"
  evals $g $t >> logs/dl4_$t.log 2>&1
  echo "DL4_EVAL_DONE $t $(date -Is)"
}
ROLE=${ROLE:?set ROLE=cf or ROLE=g}
waitfor "DL3F_MAIN_EVAL_DONE|DL3F_FAIL|DL3F_SMOKE_FAIL"
[ -f weights/student/dl3F/task_emb.pt ] || { echo "DL4_SKIP: no dl3F weights"; exit 1; }
if [ "$ROLE" = g ]; then
  waitfor "DL3_CHAIN_DONE dl3 |DL3_FAIL dl3 |DL3_FAIL dl3_2s "
  run 0 dl4g --gan-w 0.1 --gan-start 200
  echo "DL4_ROLE_G_DONE $(date -Is)"
  exit 0
fi
run 1 dl4c
run 1 dl4f --ffl-w 2
until grep -qaE "DL4_ROLE_G_DONE|DL4_FAIL dl4g|DL4_SKIP" logs/h200_dl4_g.log 2>/dev/null; do sleep 60; done
until grep -qa "DL3F_ALL_DONE" logs/h200_dl3F.log; do sleep 60; done   # dl3F's own detail/iris/sheet step ends first
T="student_dl3F student_dl4c student_dl4f student_dl4g"
$P scripts/detail_bands.py student_dl3 $T --eval-set v2 || echo "DETAIL_FAIL"
$P scripts/iris_colour.py klein_personal $T || echo "IRIS_FAIL"
$P scripts/zoom_detail_sheet.py --tags klein_personal,student_dl4c,student_dl4f,student_dl4g --names fe2_157,fe2_232,fe2_007,rp_232 --out grids/zoom_dl4.jpg || echo "SHEET_FAIL"
$P scripts/compare_students_sheet.py --tags klein_personal,student_dl4c,student_dl4f,student_dl4g --out grids/dl4_sheet.jpg || echo "SHEET_FAIL"
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "DL4_ALL_DONE $(date -Is)"
