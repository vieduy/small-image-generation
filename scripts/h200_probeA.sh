#!/bin/bash
# Probe A (user-approved 2026-11-18 ~15:35 UTC): is the DMD softening of dl2_dmd caused by soft real-photo inputs?
# Same DMD pass as dl2_dmd (from dl2, dmd_recipe.txt, 5000-pair subset, 60% expressions, same seed and val tiles) but
# training rows restricted to SYNTHETIC photos (--src-prefixes fc). 500 generator steps (dl2_dmd's sharpness fell
# 708 -> 380 by step 500). Compare the sharpness series with dl2_dmd's, plus v2 eval + detail_bands.
# Queued on GPU 0 after h200_dl2_2s.sh.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
until grep -avE '^\+' logs/h200_dl2_2s.log | grep -qaE "DL2_2S_DONE|DL2_2S_FAIL|DL2_2S_SMOKE_FAIL|DL2_2S_SKIP"; do sleep 30; done
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
R=$(cat configs/dmd_recipe.txt); echo "RECIPE $R"
$P scripts/train_dmd.py --style cartoon3d --manifest manifest_v2.jsonl --expr --max-pairs 5000 --expr-frac 0.6 \
  --init-from dl2 --fake-warmup 200 --lr 1e-4 --src-prefixes fc --tag probeA_syn --steps 500 $R || { echo "PROBEA_FAIL train"; exit 1; }
$P scripts/dmd_sharpness_series.py dl1_dmd dl2_dmd probeA_syn
EVAL_SET=v2 $P scripts/student_infer.py --tag probeA_syn --style cartoon3d --no-perf && \
  EVAL_SET=v2 $P scripts/eval_quality.py --tag student_probeA_syn --styles cartoon3d || echo "EVAL_FAIL probeA"
$P scripts/detail_bands.py student_dl2 student_dl2_dmd student_probeA_syn || echo "DETAIL_FAIL"
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "PROBEA_DONE $(date -Is)"
