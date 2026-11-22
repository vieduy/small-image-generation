#!/bin/bash
# User 2026-11-20: "toy3d and cartoon3d similar" -> preview a clearly different look: clay / claymation (matte plasticine).
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-sty2 TRITON_CACHE_DIR=/tmp/triton-cache-sty2
[ -f styles/clay/style.yaml ] || CUDA_VISIBLE_DEVICES=0 $P scripts/gen_style.py --name clay --seed 33000 --short "clay figure character" \
  --desc "handmade clay figure, claymation style, matte plasticine texture with subtle fingerprint marks, soft rounded sculpted shapes, simple bead eyes, warm soft lighting" \
  > logs/style_gen_clay.log 2>&1 || { echo "PREV_FAIL clay refs"; exit 1; }
O="--out data/pairs_v2/clay_preview --style clay"
$P scripts/gen_pairs_v2.py plan-style $O --photos 24 --avatar-n 1 --tasks expr_happy,expr_thumbs,expr_love && \
CUDA_VISIBLE_DEVICES=0 $P scripts/gen_pairs_v2.py gen $O --compile 0 --check-compile 0 > logs/style_prev_gen_clay.log 2>&1 && \
CUDA_VISIBLE_DEVICES=0 SCORE_THREADS=8 $P scripts/gen_pairs_v2.py score $O > logs/style_prev_score_clay.log 2>&1 || echo "PREV_FAIL clay"
chmod -R a+rwX data/pairs_v2/clay_preview styles/clay logs 2>/dev/null
echo "CLAY_PREVIEW_DONE $(date -Is)"
