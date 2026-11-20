#!/bin/bash
# Step 3d+4e: faces 5000 -> 8000 (avatar only; sticker-faces stays 5000), same filters (id top-5, |dL*| <= 12),
# then s5d = s3d recipe (direct, no GAN, 10k steps) on the larger filtered set. ~90 min.
set -e
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
echo "START $(date -Is)"
$VENV/bin/python scripts/gen_pairs.py --style cartoon3d --faces 8000 --subjects 1200 --sticker-faces 5000 --max-dl 12
$VENV/bin/python -c "
import json,collections
rows=[json.loads(l) for l in open('data/pairs/cartoon3d/manifest.jsonl')]
c=collections.Counter((r['task'],'face' if r['is_face'] else 'subj',r['keep']) for r in rows); print('PAIRS', sorted(c.items()))"
$VENV/bin/python scripts/train_student.py --style cartoon3d --tag s5d --param direct --steps 10000
$VENV/bin/python scripts/student_infer.py --tag s5d --style cartoon3d
$VENV/bin/python scripts/eval_quality.py --tag student_s5d --styles cartoon3d
TAGS="klein_personal student_s3d student_s4d student_s5d" OUT=_v5 $VENV/bin/python scripts/skin_compare.py
chmod -R a+rwX outputs results grids weights data/pairs 2>/dev/null || true
echo "STEP3D4E_DONE $(date -Is)"
