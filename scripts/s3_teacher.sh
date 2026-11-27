#!/bin/bash
# Stage 3 (2026-11-23): smoke (96 rows -> sheet), then all 53,930 cartoon3d rows x (plain + flipped photo), tB1 8 steps CFG 3/4.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
echo "S3_START $(date -Is)"
CUDA_VISIBLE_DEVICES=0 $P scripts/s3_teacher_pairs.py --tag tB1 --max-rows 96 --shard 96 > logs/s3_smoke.log 2>&1 || { echo "S3_FAIL smoke"; exit 1; }
mv data/pairs/cartoon3d/teacher_tB1_s8.pt data/pairs/cartoon3d/_smoke_teacher_tB1_s8.pt; rm -rf data/pairs/cartoon3d/teacher_tB1_s8
echo "S3_SMOKE_OK $(date -Is)"
CUDA_VISIBLE_DEVICES=0 $P scripts/s3_teacher_pairs.py --tag tB1 > logs/s3_pairs.log 2>&1 || { echo "S3_FAIL pairs"; exit 1; }
chmod -R a+rwX data/pairs/cartoon3d/teacher_tB1_s8* logs/s3_* 2>/dev/null
echo "S3_ALL_DONE $(date -Is)"
