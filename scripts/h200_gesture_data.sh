#!/bin/bash
# Round 4 (hands): +3000 faces (f02500-f05499) x 3 of the 7 hand-gesture expressions, klein teacher, ADULT-AGE prompt (pilot B), same filters.
# Same args as round 3b (h200_gpu2_real.sh R2) so every existing pair stays in the manifest; existing images are skipped.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
until grep -qaE "^CHECK_REAL_DONE" logs/h200_check_real.log; do sleep 30; done
echo "GD_START $(date -Is)"
$P scripts/gen_pairs.py --style cartoon3d --faces 10000 --subjects 1200 --sticker-faces 5000 --max-dl 12 --verify-extra \
   --expr-faces 2500 --expr-k 4 --selfies 3000 --selfie-stickers 1500 --real-n 6597 --real-stickers 2000 \
   --expr2-from 2500 --expr2-to 5500 --expr2-k 3 --expr2-adult --expr2-set think,thumbs,wave,ok,facepalm,shrug,thanks
cp data/pairs/cartoon3d/manifest.jsonl data/pairs/cartoon3d/manifest_round4.jsonl
$P -c "
import json,collections
g=['think','thumbs','wave','ok','facepalm','shrug','thanks']
c=collections.Counter(); k=collections.Counter()
for l in open('data/pairs/cartoon3d/manifest_round4.jsonl'):
    r=json.loads(l); c[r['task']]+=1; k[r['task']]+=r['keep']
print('ROWS', sum(c.values()), 'KEPT', sum(k.values()))
for e in g: print('GESTURE', e, 'generated', c['expr_'+e], 'kept', k['expr_'+e])"
chmod -R a+rwX data logs 2>/dev/null
echo "GESTURE_DATA_DONE $(date -Is)"
