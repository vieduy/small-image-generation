#!/bin/bash
# Items 1-4 queue (2026-11-16). Stages run one after another (one GPU job at a time); each stage < 2 h.
#  A (~45 min): eval set v2 (300 faces) + klein teacher refs on v2 (cartoon3d, flat_vector) + flat_vector pairs (style 2)
#  B (~75 min): best-of-3 teacher selection for cartoon3d (4000 avatars, 2000 face stickers)
#  C (~75 min): students s6d (best-of-K data), fv_lora (style pack on s5d), fv_full (full fine-tune control)
#  D (~45 min): inference + v2 eval of all students, skin compare
set -e
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
stage() { echo "=== STAGE $1 $(date -Is)"; }
stage A
[ -f data/faces_eval2/fe2_299.png ] || $P scripts/gen_eval2.py
mkdir -p data/pairs/flat_vector/inputs
for i in $(seq -f "%05g" 0 2399); do cp -n data/pairs/cartoon3d/inputs/f$i.png data/pairs/flat_vector/inputs/; done
for i in $(seq -f "%05g" 0 599); do cp -n data/pairs/cartoon3d/inputs/o$i.png data/pairs/flat_vector/inputs/; done
$P scripts/gen_pairs.py --style flat_vector --faces 2400 --subjects 600 --sticker-faces 1200 --max-dl 12
$P -c "
import json,collections
rows=[json.loads(l) for l in open('data/pairs/flat_vector/manifest.jsonl')]
print('PAIRS_FV', sorted(collections.Counter((r['task'],'face' if r['is_face'] else 'subj',r['keep']) for r in rows).items()))"
export EVAL_SET=v2
for S in cartoon3d flat_vector; do
  PERSONAL=1 MODEL=klein KLEIN_REF=0 PROMPTS_FAMILY=klein_text $P scripts/eval_quality.py --tag klein_personal --styles $S --generate
done
unset EVAL_SET
stage B
$P scripts/best_of_k.py --style cartoon3d --k 3 --avatar-n 4000 --sticker-n 2000 --manifest-in manifest_s5d.jsonl --manifest-out manifest_bestk.jsonl
stage C
$P scripts/train_student.py --style cartoon3d --tag s6d --param direct --steps 10000 --manifest manifest_bestk.jsonl
$P scripts/train_student.py --style flat_vector --tag fv_lora --param direct --steps 6000 --init-from s5d --lora-rank 32 --lr 1e-4
$P scripts/train_student.py --style flat_vector --tag fv_full --param direct --steps 6000 --init-from s5d
stage D
export EVAL_SET=v2
for T in s3d s5d s6d; do
  $P scripts/student_infer.py --tag $T --style cartoon3d --no-perf
  $P scripts/eval_quality.py --tag student_$T --styles cartoon3d
done
for T in fv_lora fv_full; do
  $P scripts/student_infer.py --tag $T --style flat_vector --no-perf
  $P scripts/eval_quality.py --tag student_$T --styles flat_vector
done
TAGS="klein_personal student_s3d student_s5d student_s6d" OUT=_v6 STYLE=cartoon3d $P scripts/skin_compare.py
TAGS="klein_personal student_fv_lora student_fv_full" OUT=_fv STYLE=flat_vector $P scripts/skin_compare.py
chmod -R a+rwX data outputs results grids weights logs 2>/dev/null || true
echo "QUEUE_DONE $(date -Is)"
