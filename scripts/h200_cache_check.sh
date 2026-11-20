#!/bin/bash
# Verify the gen_pairs face-score cache: run 1 fills it, run 2 should only reuse. Both manifests must equal the s5d one.
set -e
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
D=data/pairs/cartoon3d
cp $D/manifest.jsonl $D/manifest_s5d.jsonl   # exact training set of s5d, kept for reproducibility
ARGS="--style cartoon3d --faces 10000 --subjects 1200 --sticker-faces 5000 --max-dl 12"
for RUN in 1 2; do
  S=$(date +%s); $VENV/bin/python scripts/gen_pairs.py $ARGS 2>&1 | grep -E "^\[pairs\] (scoring|face cache|scored, total)"
  echo "RUN$RUN wall $(( $(date +%s) - S ))s"
  $VENV/bin/python -c "
import json
a=[json.loads(l) for l in open('$D/manifest_s5d.jsonl')]; b=[json.loads(l) for l in open('$D/manifest.jsonl')]
k=('id','task','keep','id_rank100','dL','id_cos')
diff=[(x['id'],x['task']) for x,y in zip(a,b) if any(x.get(f)!=y.get(f) for f in k)]
print('RUN$RUN rows',len(a),len(b),'diffs',len(diff),diff[:5])"
done
chmod -R a+rwX $D logs 2>/dev/null || true
echo CACHE_CHECK_DONE
