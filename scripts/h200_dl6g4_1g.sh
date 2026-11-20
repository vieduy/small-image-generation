#!/bin/bash
# dl6g4 on ONE GPU (user "go all" 2026-11-19 ~19:10 VN). Same recipe as dl6g2 except critic 0.4: from dl3F, 3000 steps, bs 16, lr 1e-5,
# --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --tgt-cache --compile, --gan-w 0.4 from step 200. Same code path as dl6g2
# (which ran single-GPU), so no smoke. GPU via $G (default 0). Acceptance vs dl6g2: more detail in zoom with no grain/crosshatch,
# no over-aging, likeness drop <= 0.05, expr drop <= 0.02. Evals then also with the decg2 decoder (is the critic gain additive?).
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-dl6g4
P=$VENV/bin/python
G=${G:-0}
t=dl6g4
L="--lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --tgt-cache --compile"
CUDA_VISIBLE_DEVICES=$G $P scripts/train_dl.py --style cartoon3d --manifest manifest_v2.jsonl --expr --init-from dl3F --lr 1e-5 $L \
  --tag $t --steps 3000 --val-every 500 --gan-w 0.4 --gan-start 200 > logs/dl6g4.log 2>&1 || { echo "DL6G4_FAIL train"; exit 1; }
echo "DL6G4_TRAIN_DONE $(date -Is)"
{
  CUDA_VISIBLE_DEVICES=$G $P scripts/expr_eval.py --tag $t || echo "EXPR_FAIL $t"
  for ES in v2 real selfie; do
    CUDA_VISIBLE_DEVICES=$G EVAL_SET=$ES $P scripts/student_infer.py --tag $t --style cartoon3d --no-perf && \
      CUDA_VISIBLE_DEVICES=$G EVAL_SET=$ES $P scripts/eval_quality.py --tag student_$t --styles cartoon3d || echo "EVAL_FAIL $t $ES"
  done
  EVAL_SET=real STYLE=cartoon3d TAGS="klein_personal student_dl6g2 student_$t" OUT=_${t}_real $P scripts/skin_compare.py || echo "SKIN_FAIL $t"
  CUDA_VISIBLE_DEVICES=$G EVAL_SET=v2 $P scripts/student_infer.py --tag $t --decoder decg2 --style cartoon3d --no-perf || echo "EVAL_FAIL ${t}_decg2"
} >> logs/dl6g4.log 2>&1
echo "DL6G4_EVAL_DONE $(date -Is)"
T="student_dl6g2 student_dl6g4 student_dl6g2_decg2 student_dl6g4_decg2"
$P scripts/detail_bands.py $T --eval-set v2 || echo "DETAIL_FAIL"
$P scripts/iris_colour.py klein_personal $T || echo "IRIS_FAIL"
$P scripts/zoom_detail_sheet.py --tags klein_personal,student_dl6g2,student_dl6g4,student_dl6g2_decg2,student_dl6g4_decg2 --names fe2_157,fe2_232,fe2_007,rp_232 --out grids/zoom_dl6g4.jpg || echo "SHEET_FAIL"
$P scripts/compare_students_sheet.py --tags klein_personal,student_dl6g2,student_dl6g4 --out grids/dl6g4_sheet.jpg || echo "SHEET_FAIL"
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "DL6G4_ALL_DONE $(date -Is)"
