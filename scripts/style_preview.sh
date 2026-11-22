#!/bin/bash
# New chat styles, preview (user 2026-11-20 "generate data for training new style for chat app"), dev-duyhv-2.
# 24 synthetic photos x {avatar, sticker, happy, thumbs, love} per style with the v2 prompts (klein teacher, same seeds as
# cartoon3d v2). comic_line is new (8 klein reference images via gen_style.py). Full runs only after the user OKs the look.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-sty TRITON_CACHE_DIR=/tmp/triton-cache-sty
prev() {  # $1 GPU $2 style
  O="--out data/pairs_v2/$2_preview --style $2"
  $P scripts/gen_pairs_v2.py plan-style $O --photos 24 --avatar-n 1 --tasks expr_happy,expr_thumbs,expr_love && \
  CUDA_VISIBLE_DEVICES=$1 $P scripts/gen_pairs_v2.py gen $O --compile 0 --check-compile 0 > logs/style_prev_gen_$2.log 2>&1 && \
  CUDA_VISIBLE_DEVICES=$1 SCORE_THREADS=8 $P scripts/gen_pairs_v2.py score $O > logs/style_prev_score_$2.log 2>&1 || echo "PREV_FAIL $2"
}
(
  [ -f styles/comic_line/style.yaml ] || CUDA_VISIBLE_DEVICES=0 $P scripts/gen_style.py --name comic_line --seed 32000 --short "comic illustration character" \
    --desc "digital comic illustration, bold clean ink outlines, cel shading with soft gradients, expressive friendly face, vibrant colors" \
    > logs/style_gen_comic_line.log 2>&1 || echo "PREV_FAIL comic_line refs"
  prev 0 comic_line
  prev 0 chibi
) &
( prev 1 toy3d; prev 1 anime; prev 1 flat_vector ) &
wait
$P scripts/style_preview_sheet.py comic_line chibi toy3d anime flat_vector > logs/style_prev_sheet.log 2>&1 || echo "SHEET_FAIL"
chmod -R a+rwX data/pairs_v2 styles grids results logs 2>/dev/null
echo "STYLE_PREVIEW_DONE $(date -Is)"
