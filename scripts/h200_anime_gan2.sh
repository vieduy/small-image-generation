#!/bin/bash
# Detail test: the anime LoRA with a milder adversarial loss (w 0.2; w 0.5 over-sharpened and cost ~0.15 identity).
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
$P scripts/train_student.py --style anime --tag an_lora_g2 --param direct --steps 6000 --init-from s5d --lora-rank 32 --lr 1e-4 --gan-w 0.2 --gan-start 1000 --d-lr 2e-4
export EVAL_SET=v2
$P scripts/student_infer.py --tag an_lora_g2 --style anime --no-perf
$P scripts/eval_quality.py --tag student_an_lora_g2 --styles anime
$P scripts/eval_quality.py --tag student_an_lora_g_sdvae --styles anime
echo ANIME_GAN2_DONE $(date -Is)
