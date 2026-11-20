#!/bin/bash
# Step 4b: student s2d on data v2 (direct param, 10k steps) + eval; rescore earlier runs with the subject metrics. ~50 min.
set -e
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
echo "START $(date -Is)"
$VENV/bin/python scripts/train_student.py --style cartoon3d --tag s2d --param direct --steps 10000
$VENV/bin/python scripts/student_infer.py --tag s2d --style cartoon3d
for T in student_s2d student_s2d_sdvae student_s1d student_s1 klein_text; do
  $VENV/bin/python scripts/eval_quality.py --tag $T --styles cartoon3d
done
TAGS="klein_text klein_personal student_s1d student_s2d student_s2d_sdvae" OUT=_v2 $VENV/bin/python scripts/skin_compare.py
chmod -R a+rwX outputs results grids weights data/pairs 2>/dev/null || true
echo "STEP4B_DONE $(date -Is)"
