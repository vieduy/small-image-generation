#!/bin/bash
# Re-run the v2 eval set that failed on an NFS I/O error (2026-11-20 ~08:32): $1 tag, $2 style, $3 GPU, $4 done-marker file, $5 marker text
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
until grep -q "$5" $4 2>/dev/null; do sleep 60; done
CUDA_VISIBLE_DEVICES=$3 EVAL_SET=v2 $P scripts/student_infer.py --tag $1 --decoder decg2 --style $2 --no-perf && \
  CUDA_VISIBLE_DEVICES=$3 EVAL_SET=v2 $P scripts/eval_quality.py --tag student_$1_decg2 --styles $2 || echo "EVAL_FAIL $1 v2"
echo "RERUN_V2_DONE $1 $(date -Is)"
exit 0
