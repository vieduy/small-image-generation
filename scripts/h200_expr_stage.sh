#!/bin/bash
# PRIORITY #1 sticker variety (runs first after h200_queue_1to4). ~2 h:
#  expression pairs: 2500 f* faces x 4 of the 20 expressions (klein, per-person prompt + phrase, attribute-verified),
#  filters: sticker pass, identity top-10, |dL*|<=12; then student s5e = s5d + 20 expression tokens (6k steps), eval.
set -e
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
echo "EXPR_START $(date -Is)"
$P scripts/gen_pairs.py --style cartoon3d --faces 10000 --subjects 1200 --sticker-faces 5000 --max-dl 12 --verify-extra \
   --expr-faces 2500 --expr-k 4
$P -c "
import json,collections
rows=[json.loads(l) for l in open('data/pairs/cartoon3d/manifest.jsonl')]
e=[r for r in rows if r['task'].startswith('expr_')]
print('EXPR_PAIRS', len(e), 'kept', sum(r['keep'] for r in e), sorted(collections.Counter(r['task'] for r in e if r['keep']).items()))"
cp data/pairs/cartoon3d/manifest.jsonl data/pairs/cartoon3d/manifest_expr.jsonl
$P scripts/train_student.py --style cartoon3d --tag s5e --param direct --steps 6000 --init-from s5d --expr --manifest manifest_expr.jsonl
$P scripts/expr_eval.py --tag s5e
chmod -R a+rwX data outputs results grids weights logs 2>/dev/null || true
echo "EXPR_DONE $(date -Is)"
