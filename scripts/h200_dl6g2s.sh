#!/bin/bash
# Skin-tone fix "C" (user "go all" 2026-11-19): dl6g2s = EXACT dl6g2 recipe (from dl3F, 3000 steps, critic 0.2) with the only change
# --drop-list data/pairs/cartoon3d/drop_skin_lighten.json (scripts/make_skin_drop.py 55 15: 6718 pairs whose klein target lightened
# input skin L* < 55 by > 15, 97% real photos). Root cause: real-photo dark-quartile lightening dL +32 (klein +6.7) learned from those
# pairs. Key metric: skin_compare on EVAL_SET=real (dL_dark25, dL_median) vs dl6g2; plus likeness/expr/detail must hold.
# NOTE: dl3F (init) was trained on the full data, so this only tests whether 3000 clean steps undo it.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-dl6g2s
P=$VENV/bin/python
G=${G:-0}
t=dl6g2s
L="--lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --tgt-cache --compile"
CUDA_VISIBLE_DEVICES=$G $P scripts/train_dl.py --style cartoon3d --manifest manifest_v2.jsonl --expr --init-from dl3F --lr 1e-5 $L \
  --tag $t --steps 3000 --val-every 500 --gan-w 0.2 --drop-list data/pairs/cartoon3d/drop_skin_lighten.json --gan-start 200 > logs/dl6g2s.log 2>&1 || { echo "DL6G2S_FAIL train"; exit 1; }
echo "DL6G2S_TRAIN_DONE $(date -Is)"
{
  CUDA_VISIBLE_DEVICES=$G $P scripts/expr_eval.py --tag $t || echo "EXPR_FAIL $t"
  for ES in v2 real selfie; do
    CUDA_VISIBLE_DEVICES=$G EVAL_SET=$ES $P scripts/student_infer.py --tag $t --style cartoon3d --no-perf && \
      CUDA_VISIBLE_DEVICES=$G EVAL_SET=$ES $P scripts/eval_quality.py --tag student_$t --styles cartoon3d || echo "EVAL_FAIL $t $ES"
  done
  for ES in real selfie v2; do EVAL_SET=$ES STYLE=cartoon3d TAGS="klein_personal student_dl6g2 student_$t" OUT=_${t}_$ES $P scripts/skin_compare.py || echo "SKIN_FAIL $t $ES"; done
  CUDA_VISIBLE_DEVICES=$G EVAL_SET=v2 $P scripts/student_infer.py --tag $t --decoder decg2 --style cartoon3d --no-perf || echo "EVAL_FAIL ${t}_decg2"
} >> logs/dl6g2s.log 2>&1
echo "DL6G2S_EVAL_DONE $(date -Is)"
T="student_dl6g2 student_dl6g2s student_dl6g2_decg2 student_dl6g2s_decg2"
$P scripts/detail_bands.py $T --eval-set v2 || echo "DETAIL_FAIL"
$P scripts/iris_colour.py klein_personal $T || echo "IRIS_FAIL"
$P scripts/zoom_detail_sheet.py --tags klein_personal,student_dl6g2,student_dl6g2s,student_dl6g2_decg2,student_dl6g2s_decg2 --names fe2_157,fe2_232,fe2_007,rp_232 --out grids/zoom_dl6g2s.jpg || echo "SHEET_FAIL"
$P scripts/compare_students_sheet.py --tags klein_personal,student_dl6g2,student_dl6g2s --out grids/dl6g2s_sheet.jpg || echo "SHEET_FAIL"
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "DL6G2S_ALL_DONE $(date -Is)"
