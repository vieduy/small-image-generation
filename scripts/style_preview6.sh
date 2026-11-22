#!/bin/bash
# Re-preview with prompt v6 (2026-11-20, user: focus cartoon3d, chibi, comic_line): caption age, no named grey hair,
# chibi without the adult-body sentence and without "pastel colors".
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
prev() {
  O="--out data/pairs_v2/$2_preview6 --style $2"
  $P scripts/gen_pairs_v2.py plan-style $O --photos 24 --avatar-n 1 --tasks expr_happy,expr_thumbs,expr_love && \
  CUDA_VISIBLE_DEVICES=$1 $P scripts/gen_pairs_v2.py gen $O --pv 6 --compile 0 --check-compile 0 > logs/style_prev6_gen_$2.log 2>&1 && \
  CUDA_VISIBLE_DEVICES=$1 SCORE_THREADS=8 $P scripts/gen_pairs_v2.py score $O > logs/style_prev6_score_$2.log 2>&1 || echo "PREV6_FAIL $2"
}
prev 0 chibi & prev 1 comic_line & wait
chmod -R a+rwX data/pairs_v2 logs 2>/dev/null
echo "PREVIEW6_DONE $(date -Is)"
