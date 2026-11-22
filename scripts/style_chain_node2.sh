#!/bin/bash
# dev-duyhv-2 (2026-11-20): GPU 0 = comic_line pack after its data prep; GPU 1 = chibi scoring (style_full_node2.sh) -> prep -> pack.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
(
  until grep -q "PREP_DONE comic_line" logs/prep_comic_line.log 2>/dev/null; do sleep 60; done
  G=0 S=comic_line T=pk_comic bash scripts/style_train.sh pack
) > logs/style_chain_node2_gpu0.out 2>&1 &
(
  until grep -q "STYLE_SCORE_DONE chibi" logs/style_full_node2.out 2>/dev/null; do sleep 60; done
  CUDA_VISIBLE_DEVICES=1 $P scripts/prep_style_data.py chibi > logs/prep_chibi.log 2>&1 || { echo "PREP_FAIL chibi"; exit 1; }
  G=1 S=chibi T=pk_chibi bash scripts/style_train.sh pack
) > logs/style_chain_node2_gpu1.out 2>&1 &
wait
echo "STYLE_NODE2_DONE $(date -Is)"
exit 0
