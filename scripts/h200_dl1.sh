#!/bin/bash
# DreamLite bake-off, stage 1 (regression; mirrors b1): dl1 = full fine-tune of DreamLite-mobile UNet as a one-step, text-free
# student on manifest_round4 (same data as c3_dmd3), then the same evals as b1 (v2, expressions).
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
$P scripts/train_dl.py --style cartoon3d --tag dl1 --manifest manifest_round4.jsonl --expr --steps 16000
$P scripts/expr_eval.py --tag dl1 || echo "EXPR_FAIL dl1"
export EVAL_SET=v2
$P scripts/student_infer.py --tag dl1 --style cartoon3d --no-perf
$P scripts/eval_quality.py --tag student_dl1 --styles cartoon3d
TAGS="klein_personal student_b1 student_dl1" OUT=_dl1_v2 STYLE=cartoon3d $P scripts/skin_compare.py
chmod -R a+rwX outputs results grids weights logs data/pairs 2>/dev/null
echo "DL1_DONE $(date -Is)"
