#!/bin/bash
# Candidate final recipe + critic-weight test on dev-duyhv-2 (user choice 2026-11-19 ~13:10 VN: "Final recipe + weight test").
# dl5 result: only the frozen-DINOv2 critic added real detail (fine 0.52 -> 0.68 vs control, likeness up, no artifacts,
# expression -0.025); FFL did nothing and FFL+critic cost expression accuracy. dl3 vs dl3F: the flip fix does not change detail
# but is correct, so the base is dl3F (fixed flip + full-res LPIPS + eye terms, 16k steps).
#   GPU 0: dl6g1 = dl3F + critic 0.1 (= dl5g settings), 3000 steps, lr 1e-5  -> the candidate final regression stage
#   GPU 1: dl6g2 = dl3F + critic 0.2                                          -> more detail? at what expression cost?
# Same code path as dl5g (no new smoke). Evals incl. selfie set for the candidate; sheets vs dl3F and dl5g.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-dl5   # node-local (dev-duyhv-2 cannot write the shared NFS inductor cache)
P=$VENV/bin/python
L="--lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --tgt-cache --compile"
B="--style cartoon3d --manifest manifest_v2.jsonl --expr --init-from dl3F --lr 1e-5 --steps 3000 --val-every 500 $L"
[ -f weights/student/dl3F/task_emb.pt ] || { echo "DL6_SKIP: no dl3F weights"; exit 1; }
evals() {  # $1 gpu, $2 tag
  g=$1; t=$2
  CUDA_VISIBLE_DEVICES=$g $P scripts/expr_eval.py --tag $t || echo "EXPR_FAIL $t"
  for ES in v2 real selfie; do
    CUDA_VISIBLE_DEVICES=$g EVAL_SET=$ES $P scripts/student_infer.py --tag $t --style cartoon3d --no-perf && \
      CUDA_VISIBLE_DEVICES=$g EVAL_SET=$ES $P scripts/eval_quality.py --tag student_$t --styles cartoon3d || echo "EVAL_FAIL $t $ES"
  done
  EVAL_SET=real STYLE=cartoon3d TAGS="klein_personal student_dl3F student_$t" OUT=_${t}_real $P scripts/skin_compare.py || echo "SKIN_FAIL $t"
}
run() {  # $1 gpu, $2 tag, rest = extra args
  g=$1; t=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g $P scripts/train_dl.py $B --tag $t "$@" > logs/dl6_$t.log 2>&1 || { echo "DL6_FAIL $t train"; return 1; }
  echo "DL6_TRAIN_DONE $t $(date -Is)"
  evals $g $t >> logs/dl6_$t.log 2>&1
  echo "DL6_EVAL_DONE $t $(date -Is)"
}
run 0 dl6g1 --gan-w 0.1 --gan-start 200 &
run 1 dl6g2 --gan-w 0.2 --gan-start 200 &
wait
T="student_dl3F student_dl5g student_dl6g1 student_dl6g2"
$P scripts/detail_bands.py $T --eval-set v2 || echo "DETAIL_FAIL"
$P scripts/iris_colour.py klein_personal $T || echo "IRIS_FAIL"
$P scripts/zoom_detail_sheet.py --tags klein_personal,student_dl3F,student_dl6g1,student_dl6g2 --names fe2_157,fe2_232,fe2_007,rp_232 --out grids/zoom_dl6.jpg || echo "SHEET_FAIL"
$P scripts/compare_students_sheet.py --tags klein_personal,student_dl3F,student_dl6g1,student_dl6g2 --out grids/dl6_sheet.jpg || echo "SHEET_FAIL"
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "DL6_ALL_DONE $(date -Is)"
