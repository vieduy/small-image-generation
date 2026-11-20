#!/bin/bash
# Plan A3 (user 2026-11-20 "queue A first"): smaller students from dl6g2, units from scripts/block_scan.py (configs/prune_sets.json)
#   GPU 0: pS70 (9 units, -29% UNet FLOPs)   GPU 1: pS60 (15 units, -40%); 6000 steps, lr 2e-5, dl6g2 losses, then full evals
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
G=0 T=pS70 DEC=decg2 STEPS=6000 LR=2e-5 bash scripts/h200_idea.sh --prune S70 > logs/planA_pS70.out 2>&1 &
G=1 T=pS60 DEC=decg2 STEPS=6000 LR=2e-5 bash scripts/h200_idea.sh --prune S60 > logs/planA_pS60.out 2>&1 &
wait
