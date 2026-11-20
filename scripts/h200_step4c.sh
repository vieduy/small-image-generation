#!/bin/bash
# Step 4c: fixed subject filter (3909 pairs). s3d = no GAN (control), s3g = + projected-VGG GAN (w 0.1 from step 2000). ~85 min.
set -e
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
echo "START $(date -Is)"
for T in s3d s3g; do
  EXTRA=""; [ $T = s3g ] && EXTRA="--gan-w 0.1 --gan-start 2000"
  $VENV/bin/python scripts/train_student.py --style cartoon3d --tag $T --param direct --steps 10000 $EXTRA
  $VENV/bin/python scripts/student_infer.py --tag $T --style cartoon3d
  $VENV/bin/python scripts/eval_quality.py --tag student_$T --styles cartoon3d
done
TAGS="klein_personal student_s2d student_s3d student_s3g" OUT=_v3 $VENV/bin/python scripts/skin_compare.py
chmod -R a+rwX outputs results grids weights data/pairs 2>/dev/null || true
echo "STEP4C_DONE $(date -Is)"
