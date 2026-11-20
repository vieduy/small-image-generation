#!/bin/bash
# 4-step anime student (same data/LoRA/init as an_lora, + 3 refinement steps at t=749/499/249), evaluated at 1, 2 and 4 steps.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
$P scripts/train_student.py --style anime --tag an_lora_m4 --param direct --steps 6000 --init-from s5d --lora-rank 32 --lr 1e-4 --msteps 4
export EVAL_SET=v2
$P scripts/student_infer.py --tag an_lora_m4 --style anime --no-perf
$P scripts/student_infer.py --tag an_lora_m4 --style anime --no-perf --ts 999,499
$P scripts/student_infer.py --tag an_lora_m4 --style anime --no-perf --ts 999
for t in an_lora_m4 an_lora_m4_sdvae an_lora_m4_k2 an_lora_m4_k2_sdvae an_lora_m4_k1; do
  $P scripts/eval_quality.py --tag student_$t --styles anime
done
echo ANIME_4STEP_DONE $(date -Is)
