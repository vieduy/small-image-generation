#!/bin/bash
# Round 3 (synthetic only): realistic close-up selfie inputs + CLIP attribute verification, selfie-style eval set,
# baseline scores of existing models on the selfie set. ~70 min. Starts after h200_queue_1to4.sh (QUEUE_DONE).
set -e
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
# PRIORITY #1 (sticker variety) runs on the 2nd H200 (dev-duyhv-2); both write data/pairs/cartoon3d/manifest.jsonl -> wait
until grep -qa "EXPR_DONE" logs/h200_expr_stage.log 2>/dev/null; do sleep 60; done
echo "START $(date -Is)"
[ -f data/faces_selfie/fs_199.png ] || $P scripts/gen_eval_selfie.py
cp data/pairs/cartoon3d/manifest.jsonl data/pairs/cartoon3d/manifest_pre_round3.jsonl
$P scripts/gen_pairs.py --style cartoon3d --faces 10000 --subjects 1200 --sticker-faces 5000 --max-dl 12 \
   --selfies 3000 --selfie-stickers 1500 --verify-extra --expr-faces 2500 --expr-k 4
$P -c "
import json,collections
rows=[json.loads(l) for l in open('data/pairs/cartoon3d/manifest.jsonl')]
print('PAIRS_R3', sorted(collections.Counter((r['task'],r['id'][0],r['keep']) for r in rows).items()))
f=[r for r in rows if r['is_face'] and r['task']=='avatar']
print('EXTRA', collections.Counter((r['extra_verified'], r['extra_ok']) for r in f))"
cp data/pairs/cartoon3d/manifest.jsonl data/pairs/cartoon3d/manifest_round3.jsonl
export EVAL_SET=selfie
for S in cartoon3d flat_vector; do
  VERIFY_EXTRA=1 PERSONAL=1 MODEL=klein KLEIN_REF=0 PROMPTS_FAMILY=klein_text $P scripts/eval_quality.py --tag klein_personal --styles $S --generate
done
for T in s5d s6d; do
  [ -f weights/student/$T/config.json ] && $P scripts/student_infer.py --tag $T --style cartoon3d --no-perf && \
     $P scripts/eval_quality.py --tag student_$T --styles cartoon3d
done
[ -f weights/student/fv_lora/config.json ] && $P scripts/student_infer.py --tag fv_lora --style flat_vector --no-perf && \
   $P scripts/eval_quality.py --tag student_fv_lora --styles flat_vector
TAGS="klein_personal student_s5d student_s6d" OUT=_selfie STYLE=cartoon3d $P scripts/skin_compare.py
chmod -R a+rwX data outputs results grids weights logs 2>/dev/null || true
echo "ROUND3_DATA_DONE $(date -Is)"
