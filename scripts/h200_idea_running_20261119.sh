#!/bin/bash
# Novelty ideas A/B (user "go all" 2026-11-19). Usage: G=<gpu> T=<tag> DEC=<eval decoder tag or ""> STEPS=<n> bash h200_idea.sh <extra train args>
#   A1 dlA16: --train-decoder decg2 --wide-latent 16 --pix-l1-w 1      (wide latent, UNet + decoder jointly; DEC=dlA16)
#   A0 dlA4 : --train-decoder decg2 --pix-l1-w 1                       (control: same joint training, 4 channels; DEC=dlA4)
#   B  dlB  : --shared-trunk (no compile), lr 2e-5, 6000 steps         (down/mid shared across menu options; DEC=decg2)
# All from dl6g2 with its losses (critic 0.2, lp512, eye terms, proper flip). Compare vs dl6g2 + decg2 (current best).
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-$T TRITON_CACHE_DIR=/tmp/triton-cache-$T
P=$VENV/bin/python
L="--lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --tgt-cache"
CUDA_VISIBLE_DEVICES=$G $P scripts/train_dl.py --style cartoon3d --manifest manifest_v2.jsonl --expr --init-from dl6g2 --lr ${LR:-1e-5} $L \
  --gan-w 0.2 --gan-start 200 --tag $T --steps ${STEPS:-3000} --val-every 500 "$@" > logs/idea_$T.log 2>&1 || { echo "IDEA_FAIL $T train"; exit 1; }
echo "IDEA_TRAIN_DONE $T $(date -Is)"
DA=${DEC:+--decoder $DEC}; O=$T${DEC:+_$DEC}
{
  CUDA_VISIBLE_DEVICES=$G $P scripts/expr_eval.py --tag $T $DA || echo "EXPR_FAIL $O"
  for ES in v2 real selfie; do
    CUDA_VISIBLE_DEVICES=$G EVAL_SET=$ES $P scripts/student_infer.py --tag $T $DA --style cartoon3d --no-perf && \
      CUDA_VISIBLE_DEVICES=$G EVAL_SET=$ES $P scripts/eval_quality.py --tag student_$O --styles cartoon3d || echo "EVAL_FAIL $O $ES"
  done
  EVAL_SET=real STYLE=cartoon3d TAGS="klein_personal student_dl6g2_decg2 student_$O" OUT=_${O}_real $P scripts/skin_compare.py || echo "SKIN_FAIL $O"
} >> logs/idea_$T.log 2>&1
echo "IDEA_EVAL_DONE $T $(date -Is)"
X="student_dl6g2 student_dl6g2_decg2 student_$O"
$P scripts/detail_bands.py $X --eval-set v2 > results/detail_$O.txt 2>&1 || echo "DETAIL_FAIL"
$P scripts/iris_colour.py klein_personal $X > results/iris_$O.txt 2>&1 || echo "IRIS_FAIL"
$P scripts/zoom_detail_sheet.py --tags klein_personal,student_dl6g2_decg2,student_$O --names fe2_157,fe2_232,fe2_007,rp_232 --out grids/zoom_$O.jpg || echo "SHEET_FAIL"
$P scripts/compare_students_sheet.py --tags klein_personal,student_dl6g2_decg2,student_$O --out grids/sheet_$O.jpg || echo "SHEET_FAIL"
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "IDEA_ALL_DONE $T $(date -Is)"
