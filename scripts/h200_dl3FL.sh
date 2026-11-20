#!/bin/bash
# dl3FL = dl3F with 24k steps instead of 16k (the postponed under-training test, now on the fixed flip pipeline).
# Runs on dev-duyhv-2 GPU 0 (2 extra H200 NVL on h200-gpu-fleet, user request 2026-11-19 ~04:20 UTC). Same data, losses,
# proper flip, target cache and 2-step DMD as dl3F; dl3FL vs dl3F isolates training length.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
M=manifest_v2.jsonl
SPEED=${SPEED:-}
L="--lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --tgt-cache $SPEED"
R=$(cat configs/dmd_recipe.txt)
until grep -qa FLIP_CACHE_DONE logs/flip_cache.log && grep -qa SPEED_BENCH_DONE logs/speed_bench.log; do sleep 30; done
evals() {  # $1 gpu, $2 tag, $3 comparison tags
  g=$1; t=$2
  CUDA_VISIBLE_DEVICES=$g $P scripts/expr_eval.py --tag $t || echo "EXPR_FAIL $t"
  for ES in v2 real selfie; do
    CUDA_VISIBLE_DEVICES=$g EVAL_SET=$ES $P scripts/student_infer.py --tag $t --style cartoon3d --no-perf && \
      CUDA_VISIBLE_DEVICES=$g EVAL_SET=$ES $P scripts/eval_quality.py --tag student_$t --styles cartoon3d || echo "EVAL_FAIL $t $ES"
    EVAL_SET=$ES STYLE=cartoon3d TAGS="klein_personal $3 student_$t" OUT=_${t}_$ES $P scripts/skin_compare.py || echo "SKIN_FAIL $t $ES"
  done
}
CUDA_VISIBLE_DEVICES=0 $P scripts/train_dl.py --style cartoon3d --tag smoke_dl3FL --manifest $M --expr --steps 3 $L > logs/dl3FL_smoke.log 2>&1 \
  || { echo "DL3FL_SMOKE_FAIL"; tail -20 logs/dl3FL_smoke.log; exit 1; }
rm -rf weights/student/smoke_dl3FL
echo "DL3FL_SMOKE_OK $(date -Is)"
CUDA_VISIBLE_DEVICES=0 $P scripts/train_dl.py --style cartoon3d --tag dl3FL --manifest $M --expr --steps 24000 $L > logs/dl3FL.log 2>&1 \
  || { echo "DL3FL_FAIL train"; exit 1; }
echo "DL3FL_MAIN_DONE $(date -Is)"
evals 0 dl3FL "student_dl2 student_dl3" >> logs/dl3FL.log 2>&1
echo "DL3FL_MAIN_EVAL_DONE $(date -Is)"
CUDA_VISIBLE_DEVICES=0 $P scripts/train_dmd.py --style cartoon3d --manifest $M --expr --max-pairs 5000 --expr-frac 0.6 \
  --init-from dl3FL --fake-warmup 200 --lr 1e-4 --ms-sigmas 1.0,0.5 --lp512-w 1 --tag dl3FL_2s --steps 2000 --ckpt-every 250 $R \
  > logs/dl3FL_2s.log 2>&1 || { echo "DL3FL_FAIL dmd"; exit 1; }
echo "DL3FL_DMD_DONE $(date -Is)"
evals 0 dl3FL_2s "student_dl3_2s student_dl3FL" >> logs/dl3FL_2s.log 2>&1
CUDA_VISIBLE_DEVICES=0 EVAL_SET=v2 $P scripts/student_infer.py --tag dl3FL_2s --style cartoon3d --no-perf --dl-k 1 >> logs/dl3FL_2s.log 2>&1 && \
  CUDA_VISIBLE_DEVICES=0 EVAL_SET=v2 $P scripts/eval_quality.py --tag student_dl3FL_2s_k1 --styles cartoon3d >> logs/dl3FL_2s.log 2>&1 || echo "EVAL_FAIL dl3FL_2s_k1"
echo "DL3FL_CHAIN_DONE $(date -Is)"
until grep -qa DL3F_ALL_DONE logs/h200_dl3F.log; do sleep 60; done
T="student_dl3F student_dl3FL student_dl3F_2s student_dl3FL_2s_k1 student_dl3FL_2s"
$P scripts/detail_bands.py $T --eval-set v2 || echo "DETAIL_FAIL"
$P scripts/iris_colour.py klein_personal student_dl3F student_dl3FL student_dl3F_2s student_dl3FL_2s || echo "IRIS_FAIL"
$P scripts/zoom_detail_sheet.py --tags klein_personal,student_dl3F,student_dl3FL,student_dl3F_2s,student_dl3FL_2s --names fe2_157,fe2_232,fe2_007,rp_232 --out grids/zoom_dl3FL.jpg || echo "SHEET_FAIL"
$P scripts/compare_students_sheet.py --tags klein_personal,student_dl3F,student_dl3FL,student_dl3F_2s,student_dl3FL_2s --out grids/dl3FL_sheet.jpg || echo "SHEET_FAIL"
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "DL3FL_ALL_DONE $(date -Is)"
