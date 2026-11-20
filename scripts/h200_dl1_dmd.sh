#!/bin/bash
# DreamLite stage 2 (user request 2026-11-17, after dl1 beat SDXS b1 on expressions/age/props): dl1_dmd = DMD pack on dl1
# with exactly the c3_dmd3 recipe, subset and step count, so dl1_dmd vs c3_dmd3 is apples to apples. Queued after the
# SD-Turbo probe on GPU1. A 3-step smoke run first (the DreamLite DMD path has not run end to end yet).
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
until grep -qa PROBE_GPU1_DONE logs/h200_probe_gpu1.log; do sleep 30; done
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
R=$(cat configs/dmd_recipe.txt); echo "RECIPE $R"
D="--style cartoon3d --manifest manifest_round4.jsonl --expr --max-pairs 5000 --expr-frac 0.6 --init-from dl1 --fake-warmup 200 --lr 1e-4"
$P scripts/train_dmd.py $D --tag smoke_dl1_dmd --steps 3 --fake-warmup 2 --val-every 1000 $R || { echo "DL1DMD_SMOKE_FAIL"; exit 1; }
rm -rf weights/student/smoke_dl1_dmd
$P scripts/train_dmd.py $D --tag dl1_dmd --steps 2000 --ckpt-every 250 $R
$P scripts/expr_eval.py --tag dl1_dmd || echo "EXPR_FAIL dl1_dmd"
export EVAL_SET=v2
$P scripts/student_infer.py --tag dl1_dmd --style cartoon3d --no-perf
$P scripts/eval_quality.py --tag student_dl1_dmd --styles cartoon3d
TAGS="klein_personal student_b1 student_c3_dmd3 student_dl1 student_dl1_dmd" OUT=_dl1dmd_v2 STYLE=cartoon3d $P scripts/skin_compare.py
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "DL1DMD_DONE $(date -Is)"
