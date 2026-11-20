#!/bin/bash
# Smoke test for h200_probe_base.sh on SD 1.5 (tiny step counts; outputs smoke_* are deleted afterwards).
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
B=stable-diffusion-v1-5/stable-diffusion-v1-5
C="--style cartoon3d --manifest manifest_round4.jsonl --expr --max-pairs 4000 --expr-frac 0.5 --param direct --T 999 --bs 16 --emb-lr 1e-3 --val-every 10 --base $B --te-from $B"
$P scripts/train_student.py --tag smoke_sd15_A $C --lr 2e-5 --steps 20 || echo SMOKE_A_FAIL
$P scripts/train_student.py --tag smoke_sd15_B $C --lr 1e-4 --steps 20 --lora-rank 32 || echo SMOKE_B_FAIL
$P scripts/expr_eval.py --tag smoke_sd15_B || echo SMOKE_EXPR_FAIL
R=$(cat configs/dmd_recipe.txt)
$P scripts/train_dmd.py --style cartoon3d --manifest manifest_round4.jsonl --expr --max-pairs 5000 --expr-frac 0.6 \
   --init-from smoke_sd15_A --tag smoke_sd15_C --steps 3 --fake-warmup 2 --lr 1e-4 --val-every 1000 $R || echo SMOKE_C_FAIL
echo "SMOKE_DONE $(date -Is)"
