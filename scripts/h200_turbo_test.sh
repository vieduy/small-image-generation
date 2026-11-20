#!/bin/bash
# Base-capacity ceiling test (user-approved 2026-11-16): SD-Turbo UNet (0.87B, Stability Community License - research
# test only, license/legal check needed before any product use) vs SDXS (0.33B), same lineage as s5d -> an_lora.
#  t5d       : SD-Turbo, cartoon3d manifest_s5d (5096 pairs), 6000 steps (s5d: SDXS, 10000 steps)
#  t_an_lora : LoRA r32 on t5d, anime pairs, 4000 steps (an_lora: on s5d, 6000 steps)
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
until grep -qaE "ANIME_4STEP_DONE|Traceback" logs/h200_anime_4step.log; do sleep 60; done
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
export EVAL_SET=v2
$P scripts/train_student.py --style cartoon3d --tag t5d --base stabilityai/sd-turbo --param direct --steps 6000 --manifest manifest_s5d.jsonl
$P scripts/student_infer.py --tag t5d --style cartoon3d --no-perf
$P scripts/eval_quality.py --tag student_t5d --styles cartoon3d
$P scripts/train_student.py --style anime --tag t_an_lora --param direct --steps 4000 --init-from t5d --lora-rank 32 --lr 1e-4
$P scripts/student_infer.py --tag t_an_lora --style anime --no-perf
$P scripts/eval_quality.py --tag student_t_an_lora --styles anime
echo TURBO_DONE $(date -Is)
