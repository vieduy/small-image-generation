#!/bin/bash
# New chat styles (user 2026-11-20 "run training both joint style and each style lora"): same data, same losses (dl6g2 recipe:
# lp512, eye L1 + eye LPIPS, critic 0.2, proper flip; no --tgt-cache, see prep_style_data.py), all from dl6g2, decoder decg2.
#   pack  : one LoRA pack per style, r32 on mid_block + up_blocks.0 (an_mu winner), lr 1e-4, 6000 steps (16 x 6000 = 96k samples)
#   joint : ONE UNet for cartoon3d + comic_line + chibi, 66 option tokens, lr 2e-5, 18000 steps (~96k samples per style)
# Usage:  G=0 S=comic_line T=pk_comic bash scripts/style_train.sh pack
#         G=0 T=dl7 bash scripts/style_train.sh joint
#         G=0 T=<tag> S=<style> bash scripts/style_train.sh eval        (expr_eval + v2/real/selfie quality for one style)
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-$T TRITON_CACHE_DIR=/tmp/triton-cache-$T
P=$VENV/bin/python
L="--expr --manifest manifest_v2.jsonl --init-from dl6g2 --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --gan-w 0.2 --gan-start 200 --val-every 1000"

ev() {  # $1 tag  $2 style
  CUDA_VISIBLE_DEVICES=$G $P scripts/expr_eval.py --tag $1 --decoder decg2 || echo "EXPR_FAIL $1"
  for ES in v2 real selfie; do
    CUDA_VISIBLE_DEVICES=$G EVAL_SET=$ES $P scripts/student_infer.py --tag $1 --decoder decg2 --style $2 --no-perf && \
      CUDA_VISIBLE_DEVICES=$G EVAL_SET=$ES $P scripts/eval_quality.py --tag student_$1_decg2 --styles $2 || echo "EVAL_FAIL $1 $ES"
  done
}

case $1 in
pack)
  CUDA_VISIBLE_DEVICES=$G $P scripts/train_dl.py --style $S $L --tag $T --lora-rank 32 --lora-scope mid_up0 --lr 1e-4 --steps 6000 \
    > logs/style_$T.log 2>&1 || { echo "STYLE_FAIL $T train"; exit 1; }
  echo "STYLE_TRAIN_DONE $T $(date -Is)"
  ev $T $S >> logs/style_$T.log 2>&1 ;;
joint)
  CUDA_VISIBLE_DEVICES=$G $P scripts/train_dl.py --style cartoon3d --joint-styles comic_line,chibi $L --tag $T --lr 2e-5 --steps 18000 \
    > logs/style_$T.log 2>&1 || { echo "STYLE_FAIL $T train"; exit 1; }
  echo "STYLE_TRAIN_DONE $T $(date -Is)"
  { ev $T cartoon3d; ev ${T}_comic_line comic_line; ev ${T}_chibi chibi
    EVAL_SET=real STYLE=cartoon3d TAGS="klein_personal student_dl6g2_decg2 student_${T}_decg2" OUT=_${T}_decg2_real $P scripts/skin_compare.py || echo "SKIN_FAIL"
    $P scripts/detail_bands.py student_dl6g2_decg2 student_${T}_decg2 --eval-set v2 > results/detail_${T}_decg2.txt 2>&1 || echo "DETAIL_FAIL"
  } >> logs/style_$T.log 2>&1 ;;
eval)
  ev $T $S >> logs/style_$T.log 2>&1 ;;
esac
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "STYLE_ALL_DONE $1 $T $(date -Is)"
exit 0
