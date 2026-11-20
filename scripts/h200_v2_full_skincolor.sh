#!/bin/bash
# Synthetic run colour pass, relaunched 2026-11-18 ~07:55 UTC with 1 thread per pool worker (_pool_init); the first launch
# (64 workers x ~319 threads) starved the GPU jobs. The real-photo run waits for SKINCOLOR_DONE in logs/v2full_skincolor.log.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
OMP_NUM_THREADS=1 $VENV/bin/python scripts/gen_pairs_v2.py skincolor --out data/pairs_v2/cartoon3d --workers 128 \
  && chmod -R a+rwX data/pairs_v2 && echo "SKINCOLOR_DONE $(date -Is)"
