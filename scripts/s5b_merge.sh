#!/bin/bash
# Replaces the rest of s5_chain.sh (stopped after the smoke test, 2026-11-25 09:23 UTC): wait for the 2 pair workers, merge
# (skin stats per quartile), then STOP: the skin filter for training is decided after seeing the full statistics.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
until [ -f data/pairs/real_teacher/part_0of2.pt ] && [ -f data/pairs/real_teacher/part_1of2.pt ]; do
  pgrep -f "s5_real_pairs_m[t].py --part" > /dev/null || { [ -f data/pairs/real_teacher/part_0of2.pt ] && [ -f data/pairs/real_teacher/part_1of2.pt ] || { echo "S5B_FAIL worker died"; exit 1; }; }
  sleep 60; done
$VENV/bin/python scripts/s5_real_pairs.py --nparts 2 --merge > logs/s5/merge.log 2>&1 && echo "S5B_MERGED $(date -Is)" || echo "S5B_FAIL merge"
chmod -R a+rwX data/pairs/real_teacher logs/s5 2>/dev/null
