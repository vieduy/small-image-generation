#!/bin/bash
# Plan A1 (user 2026-11-20 "queue A first if GPU available"): wait for a free GPU on this node, then run the block scan.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-a1 TRITON_CACHE_DIR=/tmp/triton-cache-a1
G=""
while [ -z "$G" ]; do
  for i in 0 1; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i)
    if [ "$m" -lt 1000 ]; then sleep 60; m2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i); [ "$m2" -lt 1000 ] && { G=$i; break; }; fi
  done
  [ -n "$G" ] || sleep 60
done
echo "A1_START gpu=$G $(date -Is)"
CUDA_VISIBLE_DEVICES=$G $P scripts/block_scan.py --tag dl6g2 --decoder decg2 > logs/planA1_scan.log 2>&1 && echo "A1_DONE $(date -Is)" || echo "A1_FAIL"
chmod a+rw results/block_scan_* logs/planA1_scan.log 2>/dev/null
