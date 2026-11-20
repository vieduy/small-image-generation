#!/bin/bash
# Teacher consistency check (user-approved 2026-11-18): bash scripts/h200_consistency.sh <shard> <nshards> [score]
# One shard per GPU/container; the shard-0 host also scores once every shard has written GEN_DONE.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
P=$VENV/bin/python
$P scripts/teacher_consistency.py gen --shard $1 --nshards $2 || { echo "CONS_GEN_FAIL shard$1"; exit 1; }
if [ "$3" = "score" ]; then
  for s in $(seq 0 $(($2 - 1))); do
    until grep -q "GEN_DONE shard$s\|CONS_GEN_FAIL shard$s" logs/consistency_shard$s.log 2>/dev/null; do sleep 10; done
  done
  $P scripts/teacher_consistency.py score || echo "CONS_SCORE_FAIL"
fi
chmod -R a+rwX data/consistency results grids logs 2>/dev/null
echo "CONS_DONE $(date -Is)"
