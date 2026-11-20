#!/bin/bash
# Combined DreamLite retrain (user request 2026-11-19 ~02:10 UTC, both H200s). Data v2 (manifest_v2.jsonl, 53,930 pairs).
# Main stage = every proven fix: full-res crop LPIPS (--lp512-w 1, probe: fine 0.65->0.71, streaks gone) + eye terms
# (probe 2 "lo": --eye-w 1 --eye-lpips-w 0.25, brown irises, no global cost). Finishing stage = 2-step DMD
# (--ms-sigmas 1.0,0.5; dl2_2s: +29% fine detail) with the full-res LPIPS also in the regression anchor.
#   GPU 0: dl3  = main stage 16k steps (as dl1/dl2) -> evals -> dl3_2s (2000 DMD steps) -> evals
#   GPU 1: dl3L = same losses, 24k steps (dl2 val LPIPS still falling at 16k; 4.5x more data than dl1) -> evals -> dl3L_2s -> evals
# Each stage is evaluated separately (the v2 DMD pass softened every student so far; keep whichever stage is sharper).
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
M=manifest_v2.jsonl
L="--lp512-w 1 --eye-w 1 --eye-lpips-w 0.25"
R=$(cat configs/dmd_recipe.txt)
evals() {  # $1 gpu, $2 tag, $3 comparison tags
  g=$1; t=$2
  CUDA_VISIBLE_DEVICES=$g $P scripts/expr_eval.py --tag $t || echo "EXPR_FAIL $t"
  for ES in v2 real selfie; do
    CUDA_VISIBLE_DEVICES=$g EVAL_SET=$ES $P scripts/student_infer.py --tag $t --style cartoon3d --no-perf && \
      CUDA_VISIBLE_DEVICES=$g EVAL_SET=$ES $P scripts/eval_quality.py --tag student_$t --styles cartoon3d || echo "EVAL_FAIL $t $ES"
    EVAL_SET=$ES STYLE=cartoon3d TAGS="klein_personal $3 student_$t" OUT=_${t}_$ES $P scripts/skin_compare.py || echo "SKIN_FAIL $t $ES"
  done
}
chain() {  # $1 gpu, $2 tag, $3 steps
  g=$1; t=$2
  CUDA_VISIBLE_DEVICES=$g $P scripts/train_dl.py --style cartoon3d --tag $t --manifest $M --expr --steps $3 $L > logs/dl3_${t}.log 2>&1 \
    || { echo "DL3_FAIL $t train"; return 1; }
  echo "DL3_MAIN_DONE $t $(date -Is)"
  evals $g $t "student_dl2 student_dl2_dmd" >> logs/dl3_${t}.log 2>&1
  echo "DL3_MAIN_EVAL_DONE $t $(date -Is)"
  CUDA_VISIBLE_DEVICES=$g $P scripts/train_dmd.py --style cartoon3d --manifest $M --expr --max-pairs 5000 --expr-frac 0.6 \
    --init-from $t --fake-warmup 200 --lr 1e-4 --ms-sigmas 1.0,0.5 --lp512-w 1 --tag ${t}_2s --steps 2000 --ckpt-every 250 $R \
    > logs/dl3_${t}_2s.log 2>&1 || { echo "DL3_FAIL ${t}_2s train"; return 1; }
  echo "DL3_DMD_DONE ${t}_2s $(date -Is)"
  evals $g ${t}_2s "student_dl2_2s student_$t" >> logs/dl3_${t}_2s.log 2>&1
  CUDA_VISIBLE_DEVICES=$g EVAL_SET=v2 $P scripts/student_infer.py --tag ${t}_2s --style cartoon3d --no-perf --dl-k 1 >> logs/dl3_${t}_2s.log 2>&1 && \
    CUDA_VISIBLE_DEVICES=$g EVAL_SET=v2 $P scripts/eval_quality.py --tag student_${t}_2s_k1 --styles cartoon3d >> logs/dl3_${t}_2s.log 2>&1 || echo "EVAL_FAIL ${t}_2s_k1"
  echo "DL3_CHAIN_DONE $t $(date -Is)"
}
# smoke tests of the new loss combinations (3 steps each)
CUDA_VISIBLE_DEVICES=0 $P scripts/train_dl.py --style cartoon3d --tag smoke_dl3 --manifest $M --expr --steps 3 $L > logs/dl3_smoke.log 2>&1 \
  || { echo "DL3_SMOKE_FAIL main"; tail -20 logs/dl3_smoke.log; exit 1; }
CUDA_VISIBLE_DEVICES=1 $P scripts/train_dmd.py --style cartoon3d --manifest $M --expr --max-pairs 5000 --expr-frac 0.6 --init-from dl2 \
  --lr 1e-4 --ms-sigmas 1.0,0.5 --lp512-w 1 --tag smoke_dl3_2s --steps 3 --fake-warmup 2 --val-every 1000 $R >> logs/dl3_smoke.log 2>&1 \
  || { echo "DL3_SMOKE_FAIL dmd"; tail -20 logs/dl3_smoke.log; exit 1; }
rm -rf weights/student/smoke_dl3 weights/student/smoke_dl3_2s
echo "DL3_SMOKE_OK $(date -Is)"
chain 0 dl3 16000 &
chain 1 dl3L 24000 &
wait
$P scripts/detail_bands.py student_dl1_dmd student_dl2 student_dl2_2s student_dl3 student_dl3_2s_k1 student_dl3_2s student_dl3L student_dl3L_2s_k1 student_dl3L_2s --eval-set v2 || echo "DETAIL_FAIL"
$P scripts/iris_colour.py klein_personal student_dl2_2s student_dl3 student_dl3_2s student_dl3L student_dl3L_2s || echo "IRIS_FAIL"
$P scripts/zoom_detail_sheet.py --tags klein_personal,student_dl2_2s,student_dl3,student_dl3_2s,student_dl3L_2s --names fe2_157,fe2_232,fe2_007,rp_232 --out grids/zoom_dl3.jpg || echo "SHEET_FAIL"
$P scripts/compare_students_sheet.py --tags klein_personal,student_dl2_2s,student_dl3,student_dl3_2s,student_dl3L_2s --out grids/dl3_sheet.jpg || echo "SHEET_FAIL"
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "DL3_ALL_DONE $(date -Is)"
