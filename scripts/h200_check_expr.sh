#!/bin/bash
# Do the DMD pack / style packs keep b1's 20 expression tokens working? (expr_eval: CLIP expression acc + identity)
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
for T in c3_dmd an_b1_dmd fv_b1; do $VENV/bin/python scripts/expr_eval.py --tag $T || echo "EXPR_FAIL $T"; done
chmod -R a+rwX outputs results grids logs 2>/dev/null
echo "CHECK_EXPR_DONE $(date -Is)"
