#!/bin/bash
# New style "anime" (generic anime look; no studio/franchise/artist names) as a 25 MB LoRA style pack on s5d. ~75 min.
set -e
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
echo "ANIME_START $(date -Is)"
[ -f styles/anime/style.yaml ] || $P scripts/gen_style.py --name anime --seed 32000 --short "anime character" \
  --desc "anime illustration, clean cel shading, crisp line art, expressive anime eyes, glossy hair highlights, vibrant colors"
$P - <<'PY'
import yaml
from pathlib import Path
p = Path("styles/anime/style.yaml"); y = yaml.safe_load(p.read_text())
y["subject_description"] = "anime illustration style, clean cel shading, crisp line art, vibrant colors"  # no face/eye/hair words
p.write_text(yaml.safe_dump(y, sort_keys=False, width=1000)); print("style", y["name"], "|", y["description"])
PY
mkdir -p data/pairs/anime/inputs
for i in $(seq -f "%05g" 0 2399); do cp -n data/pairs/cartoon3d/inputs/f$i.png data/pairs/anime/inputs/; done
for i in $(seq -f "%05g" 0 599); do cp -n data/pairs/cartoon3d/inputs/o$i.png data/pairs/anime/inputs/; done
$P scripts/gen_pairs.py --style anime --faces 2400 --subjects 600 --sticker-faces 1200 --max-dl 12 --verify-extra
$P -c "
import json,collections
rows=[json.loads(l) for l in open('data/pairs/anime/manifest.jsonl')]
print('PAIRS_ANIME', sorted(collections.Counter((r['task'],'face' if r['is_face'] else 'subj',r['keep']) for r in rows).items()))"
$P scripts/train_student.py --style anime --tag an_lora --param direct --steps 6000 --init-from s5d --lora-rank 32 --lr 1e-4
export EVAL_SET=v2
VERIFY_EXTRA=1 PERSONAL=1 MODEL=klein KLEIN_REF=0 PROMPTS_FAMILY=klein_text $P scripts/eval_quality.py --tag klein_personal --styles anime --generate
$P scripts/student_infer.py --tag an_lora --style anime --no-perf
$P scripts/eval_quality.py --tag student_an_lora --styles anime
TAGS="klein_personal student_an_lora" OUT=_anime STYLE=anime $P scripts/skin_compare.py
chmod -R a+rwX styles data outputs results grids weights logs 2>/dev/null || true
echo "ANIME_DONE $(date -Is)"
