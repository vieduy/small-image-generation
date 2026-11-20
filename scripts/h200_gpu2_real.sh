#!/bin/bash
# GPU 2 chain after h200_gpu2_after_expr.sh:
#  R1 (~20 min): REAL-photo baselines (300 held-out real identities): klein (server prompt = measured skin tone only), s5d, s5e
#  X1 (~45 min): s7e = expression student trained from the SDXS init (12k steps) vs s5e (fine-tuned from s5d)
#  R2 (~80 min): round 3b real pairs (after GPU 1's round 3 releases the manifest): 6597 real inputs -> avatars + 2000 stickers
set -e
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
until grep -qa "GPU2_EXP_DONE" logs/h200_gpu2_after_expr.log 2>/dev/null; do sleep 60; done
echo "R1_START $(date -Is)"
export EVAL_SET=real
PERSONAL=1 MODEL=klein KLEIN_REF=0 PROMPTS_FAMILY=klein_text $P scripts/eval_quality.py --tag klein_personal --styles cartoon3d --generate
for T in s5d s5e; do
  $P scripts/student_infer.py --tag $T --style cartoon3d --no-perf
  $P scripts/eval_quality.py --tag student_$T --styles cartoon3d
done
TAGS="klein_personal student_s5d student_s5e" OUT=_real STYLE=cartoon3d $P scripts/skin_compare.py
unset EVAL_SET
echo "X1_START $(date -Is)"
$P scripts/train_student.py --style cartoon3d --tag s7e --param direct --steps 12000 --expr --manifest manifest_expr.jsonl
$P scripts/expr_eval.py --tag s7e
echo "R2_WAIT $(date -Is)"
until grep -qa "ROUND3_DATA_DONE" logs/h200_round3_data.log 2>/dev/null; do sleep 60; done
echo "R2_START $(date -Is)"
$P scripts/gen_pairs.py --style cartoon3d --faces 10000 --subjects 1200 --sticker-faces 5000 --max-dl 12 --verify-extra \
   --expr-faces 2500 --expr-k 4 --selfies 3000 --selfie-stickers 1500 --real-n 6597 --real-stickers 2000
$P -c "
import json,collections
rows=[json.loads(l) for l in open('data/pairs/cartoon3d/manifest.jsonl')]
print('PAIRS_R3B', sorted(collections.Counter((r['task'] if not r['task'].startswith('expr_') else 'expr', r['id'][0], r['keep']) for r in rows).items()))"
cp data/pairs/cartoon3d/manifest.jsonl data/pairs/cartoon3d/manifest_round3b.jsonl
chmod -R a+rwX data outputs results grids weights logs 2>/dev/null || true
echo "GPU2_REAL_DONE $(date -Is)"
