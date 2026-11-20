#!/bin/bash
# DreamLite anime DMD (user-approved 2026-11-17): dl_an_dmd = continue the dl_an pack with exactly the SDXS an_b1_dmd recipe
# (2000 steps, fake warm-up 200, lr 4e-5, configs/dmd_recipe.txt). Evals: v2 anime quality + skin vs klein / SDXS.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
R=$(cat configs/dmd_recipe.txt); echo "RECIPE $R"
$P scripts/train_dmd.py --style anime --init-from dl_an --tag dl_an_dmd --steps 2000 --fake-warmup 200 --lr 4e-5 --ckpt-every 500 $R \
   || { echo "DLANDMD_FAIL"; exit 1; }
export EVAL_SET=v2
$P scripts/student_infer.py --tag dl_an_dmd --style anime --no-perf && $P scripts/eval_quality.py --tag student_dl_an_dmd --styles anime || echo "EVAL_FAIL"
STYLE=anime TAGS="klein_personal student_an_b1_dmd student_dl_an student_dl_an_dmd" OUT=_dl_an_dmd_v2 $P scripts/skin_compare.py || echo "SKIN_FAIL"
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "DLANDMD_DONE $(date -Is)"
