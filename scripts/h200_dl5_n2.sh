#!/bin/bash
# Detail-loss experiments on dev-duyhv-2 (2 extra H200 NVL, h200-gpu-fleet; user 2026-11-19 ~04:30 UTC: "dev-duyhv2 use for
# experiment those losses setup that u think it help the model generate more detail"). Replaces the dl4 plan (from dl3F,
# never ran) and the dl3FL idea. Starts NOW from dl2 (exists) instead of waiting ~1.5 h for dl3F; all runs use the FIXED
# pipeline (--proper-flip --tgt-cache), so the control already includes the flip fix and only the loss term differs:
#   GPU 0: dl5c  = control: dl3 losses (--lp512-w 1 --eye-w 1 --eye-lpips-w 0.25)        -> dl5f  = + Focal Frequency Loss (--ffl-w 2)
#   GPU 1: dl5g  = + frozen DINOv2-base critic (--gan-w 0.1, 200-step critic warm-up)      -> dl5fg = + FFL + critic (both)
# Each: 3000 steps from dl2, lr 1e-5, bs 16, seed 0. Weights from the smoke (FFL ~0.045 x2, hinge ~0.88 x0.1: each ~half the
# 256 px LPIPS term). Acceptance: fine band up vs dl5c, likeness drop <= 0.05, no skin regression, no crosshatch at pixel zoom.
# Caveat: the critic uses DINOv2-base like the style_dino metric, so style_dino is not independent for dl5g/dl5fg.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-dl5   # node-local: dev-duyhv-2 cannot write the shared NFS inductor-cache (created by root on dev-duyhv)
P=$VENV/bin/python
until grep -qa FLIP_CACHE_DONE logs/flip_cache.log && [ -f configs/speed_flags.txt ]; do sleep 30; done
SPEED=$(cat configs/speed_flags.txt); echo "SPEED flags: '$SPEED'"
L="--lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --tgt-cache $SPEED"
B="--style cartoon3d --manifest manifest_v2.jsonl --expr --init-from dl2 --lr 1e-5 --steps 3000 --val-every 500 $L"
evals() {  # $1 gpu, $2 tag
  g=$1; t=$2
  CUDA_VISIBLE_DEVICES=$g $P scripts/expr_eval.py --tag $t || echo "EXPR_FAIL $t"
  for ES in v2 real; do
    CUDA_VISIBLE_DEVICES=$g EVAL_SET=$ES $P scripts/student_infer.py --tag $t --style cartoon3d --no-perf && \
      CUDA_VISIBLE_DEVICES=$g EVAL_SET=$ES $P scripts/eval_quality.py --tag student_$t --styles cartoon3d || echo "EVAL_FAIL $t $ES"
  done
  EVAL_SET=real STYLE=cartoon3d TAGS="klein_personal student_dl2 student_$t" OUT=_${t}_real $P scripts/skin_compare.py || echo "SKIN_FAIL $t"
}
run() {  # $1 gpu, $2 tag, rest = extra args
  g=$1; t=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g $P scripts/train_dl.py $B --tag $t "$@" > logs/dl5_$t.log 2>&1 || { echo "DL5_FAIL $t train"; return 1; }
  echo "DL5_TRAIN_DONE $t $(date -Is)"
  evals $g $t >> logs/dl5_$t.log 2>&1
  echo "DL5_EVAL_DONE $t $(date -Is)"
}
# smoke: both new terms together, 3 steps
CUDA_VISIBLE_DEVICES=1 $P scripts/train_dl.py $B --tag smoke_dl5 --steps 3 --ffl-w 2 --gan-w 0.1 --gan-start 1 > logs/dl5_smoke.log 2>&1 \
  || { echo "DL5_SMOKE_FAIL"; tail -20 logs/dl5_smoke.log; exit 1; }
rm -rf weights/student/smoke_dl5
echo "DL5_SMOKE_OK $(date -Is)"
( run 0 dl5c; run 0 dl5f --ffl-w 2 ) &
( run 1 dl5g --gan-w 0.1 --gan-start 200; run 1 dl5fg --ffl-w 2 --gan-w 0.1 --gan-start 200 ) &
wait
T="student_dl2 student_dl5c student_dl5f student_dl5g student_dl5fg"
$P scripts/detail_bands.py $T --eval-set v2 || echo "DETAIL_FAIL"
$P scripts/iris_colour.py klein_personal $T || echo "IRIS_FAIL"
$P scripts/zoom_detail_sheet.py --tags klein_personal,student_dl5c,student_dl5f,student_dl5g,student_dl5fg --names fe2_157,fe2_232,fe2_007,rp_232 --out grids/zoom_dl5.jpg || echo "SHEET_FAIL"
$P scripts/compare_students_sheet.py --tags klein_personal,student_dl5c,student_dl5f,student_dl5g,student_dl5fg --out grids/dl5_sheet.jpg || echo "SHEET_FAIL"
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "DL5_ALL_DONE $(date -Is)"
