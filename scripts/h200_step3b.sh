#!/bin/bash
# Step 3b (data v2): subject-prompt fix + scale-up. ~80 min on 1x H200. Separate job from training (one GPU job at a time).
set -e
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
echo "START $(date -Is)"
$VENV/bin/python scripts/gen_pairs.py --style cartoon3d --faces 5000 --subjects 1200 --sticker-faces 2500
$VENV/bin/python -c "
import json,collections
c=collections.Counter()
for l in open('data/pairs/cartoon3d/manifest.jsonl'):
    r=json.loads(l); c[(r['task'],'face' if r['is_face'] else 'subj',r['keep'])]+=1
print('PAIRS', sorted(c.items()))"
# teacher reference with the fixed subject prompt (subject stickers only are regenerated)
rm -f outputs/klein_personal/cartoon3d/sticker/subj_*
PERSONAL=1 MODEL=klein KLEIN_REF=0 PROMPTS_FAMILY=klein_text $VENV/bin/python scripts/eval_quality.py --tag klein_personal --styles cartoon3d --generate
chmod -R a+rwX data/pairs outputs results grids styles 2>/dev/null || true
echo "STEP3B_DONE $(date -Is)"
