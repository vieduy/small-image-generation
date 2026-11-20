#!/bin/bash
# DreamLite anime style pack (user request 2026-11-17): dl_an = LoRA r32 on the frozen dl1 base, regression on the anime
# teacher pairs, the same recipe as the SDXS anime pack an_b1 (6000 steps, lr 1e-4, manifest.jsonl, 1546 kept pairs).
# Queued on GPU1 after the eye-fix run. Evals: v2 anime quality + skin vs klein and SDXS an_b1_dmd.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
until grep -qaE "DL1EYE_DONE|EYE_FAIL" logs/h200_dl1_dmd_eye.log; do sleep 30; done
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
$P scripts/train_dl.py --style anime --tag dl_an --manifest manifest.jsonl --init-from dl1 --lora-rank 32 --lr 1e-4 \
   --steps 6000 --val-every 1000 || { echo "DLAN_FAIL"; exit 1; }
export EVAL_SET=v2
$P scripts/student_infer.py --tag dl_an --style anime --no-perf && $P scripts/eval_quality.py --tag student_dl_an --styles anime || echo "EVAL_FAIL dl_an"
STYLE=anime TAGS="klein_personal student_an_b1 student_an_b1_dmd student_dl_an" OUT=_dl_an_v2 $P scripts/skin_compare.py || echo "SKIN_FAIL"
chmod -R a+rwX outputs results grids weights logs data/pairs 2>/dev/null
echo "DLAN_DONE $(date -Is)"
