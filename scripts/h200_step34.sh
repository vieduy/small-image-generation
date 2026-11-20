#!/bin/bash
# Steps 3+4 pilot on H200: teacher pairs (per-person prompts) -> student s1 (eps_x0) + s1d (direct) -> eval vs teacher.
set -e
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
echo "== start $(date -Is)"
$VENV/bin/python scripts/gen_pairs.py --style cartoon3d --faces 1600 --subjects 400 --sticker-faces 800
python3 -c "import json;r=[json.loads(l) for l in open('data/pairs/cartoon3d/manifest.jsonl')];import collections;print('PAIRS', len(r), 'keep', collections.Counter((x['task'],x['keep']) for x in r))"
# teacher on the eval set with the same per-person prompts (the bar the student must approach) + rescore old run with retrieval
PERSONAL=1 MODEL=klein KLEIN_REF=0 PROMPTS_FAMILY=klein_text $VENV/bin/python scripts/eval_quality.py --tag klein_personal --styles cartoon3d --generate
$VENV/bin/python scripts/eval_quality.py --tag klein_text
for T in s1:eps_x0 s1d:direct; do tag=${T%%:*}; par=${T#*:}
  $VENV/bin/python scripts/train_student.py --style cartoon3d --tag $tag --param $par --steps 4000
  $VENV/bin/python scripts/student_infer.py --tag $tag --style cartoon3d
  $VENV/bin/python scripts/eval_quality.py --tag student_$tag --styles cartoon3d
  $VENV/bin/python scripts/eval_quality.py --tag student_${tag}_sdvae --styles cartoon3d
done
echo "STEP34_DONE $(date -Is)"
