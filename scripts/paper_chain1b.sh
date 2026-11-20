#!/bin/bash
# Re-run of the influence-map half of paper_chain1.sh after a hook fix (2026-11-19). GPU 0 on dev-duyhv.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
{
  for N in avatar mean; do CUDA_VISIBLE_DEVICES=0 $P scripts/influence_map.py --tag dl6g2 --decoder decg2 --neutral $N || echo "INFL_FAIL dl6g2 $N"; done
  CUDA_VISIBLE_DEVICES=0 $P scripts/influence_map.py --tag dlBd --decoder decg2 --neutral avatar || echo "INFL_FAIL dlBd"
  echo "INFL_ALL_DONE $(date -Is)"
} > logs/paper_influence.log 2>&1
chmod -R a+rwX results grids logs 2>/dev/null
