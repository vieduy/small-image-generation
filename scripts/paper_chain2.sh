#!/bin/bash
# Paper task (c) trained: half-head students from dlB (2026-11-19, dev-duyhv). Each waits for its GPU's eval job.
#   GPU 1: hhB0 = --half-head 0 (2.2x per 22-option set)   GPU 0: hhB4 = --half-head 4 (2.1x, ablation)
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
(
  until grep -q HH_ALL_DONE logs/paper_halfhead.log; do sleep 30; done
  G=1 T=hhB0 DEC=decg2 STEPS=3000 LR=2e-5 bash scripts/h200_idea.sh --init-from dlB --half-head 0 > logs/paper_hhB0.out 2>&1
) &
(
  until grep -q INFL_ALL_DONE logs/paper_influence.log; do sleep 30; done
  G=0 T=hhB4 DEC=decg2 STEPS=3000 LR=2e-5 bash scripts/h200_idea.sh --init-from dlB --half-head 4 > logs/paper_hhB4.out 2>&1
) &
wait
