#!/bin/bash
# Step 3c+4d: +2500 face stickers (sticker-faces 2500 -> 5000), skin-drift filter |dL*| <= 12, then student s4d (direct, no GAN,
# 6k steps: val LPIPS is flat after ~5k). ~80 min.
set -e
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
echo "START $(date -Is)"
$VENV/bin/python scripts/gen_pairs.py --style cartoon3d --faces 5000 --subjects 1200 --sticker-faces 5000 --max-dl 12
$VENV/bin/python -c "
import json,collections
rows=[json.loads(l) for l in open('data/pairs/cartoon3d/manifest.jsonl')]
c=collections.Counter((r['task'],'face' if r['is_face'] else 'subj',r['keep']) for r in rows); print('PAIRS', sorted(c.items()))
d=[r['dL'] for r in rows if r.get('dL') is not None and r.get('id_pass')]
import statistics as st; print('dL id_pass n',len(d),'median',st.median(d),'|dL|>12',sum(abs(x)>12 for x in d))"
$VENV/bin/python scripts/train_student.py --style cartoon3d --tag s4d --param direct --steps 6000
$VENV/bin/python scripts/student_infer.py --tag s4d --style cartoon3d
$VENV/bin/python scripts/eval_quality.py --tag student_s4d --styles cartoon3d
TAGS="klein_personal student_s3d student_s3g student_s4d" OUT=_v4 $VENV/bin/python scripts/skin_compare.py
chmod -R a+rwX outputs results grids weights data/pairs 2>/dev/null || true
echo "STEP3C4D_DONE $(date -Is)"
