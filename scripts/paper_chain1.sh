#!/bin/bash
# Paper tasks on dev-duyhv (user 2026-11-19: dev-duyhv = paper, dev-duyhv-2 = production). Eval-only, no training.
#   GPU 0: influence map (dl6g2, neutral avatar then mean; then dlBd up-path)   GPU 1: half-head exactness (dlB, dlBd, dl6g2)
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
mkdir -p logs
{
  for N in avatar mean; do CUDA_VISIBLE_DEVICES=0 $P scripts/influence_map.py --tag dl6g2 --decoder decg2 --neutral $N || echo "INFL_FAIL dl6g2 $N"; done
  CUDA_VISIBLE_DEVICES=0 $P scripts/influence_map.py --tag dlBd --decoder decg2 --neutral avatar || echo "INFL_FAIL dlBd"
  echo "INFL_ALL_DONE $(date -Is)"
} > logs/paper_influence.log 2>&1 &
{
  for T in dlB dlBd dl6g2; do CUDA_VISIBLE_DEVICES=1 $P scripts/halfhead_check.py --tag $T --decoder decg2 || echo "HH_FAIL $T"; done
  echo "HH_ALL_DONE $(date -Is)"
} > logs/paper_halfhead.log 2>&1 &
wait
chmod -R a+rwX results grids logs 2>/dev/null
