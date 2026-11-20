#!/bin/bash
# Full SD 1.5 student (user-approved 2026-11-17 after the base probe): mirrors the SDXS production path b1 -> c3_dmd3.
#   Stage 1 sd15_b1 : full fine-tune from stable-diffusion-v1-5, 16k steps, bs 16, ALL round-4 pairs (+20 expressions),
#                     tokens from SD 1.5's own text encoder (768-d).  (b1 used round 3b; dl1 used round 4.)
#   Stage 2 sd15_dmd: DMD pack on sd15_b1 with the c3_dmd3 recipe, subset (5000, 60% expressions) and steps (2000).
# Queued on GPU2 after the SDXS probe.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
until grep -qa PROBE_GPU2_DONE logs/h200_probe_gpu2.log; do sleep 30; done
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
B=stable-diffusion-v1-5/stable-diffusion-v1-5
echo "SD15_START $(date -Is)"
$P scripts/train_student.py --style cartoon3d --tag sd15_b1 --manifest manifest_round4.jsonl --expr --param direct --T 999 \
   --steps 16000 --bs 16 --lr 2e-5 --emb-lr 1e-3 --val-every 1000 --base $B --te-from $B || { echo "SD15_B1_FAIL"; exit 1; }
$P scripts/expr_eval.py --tag sd15_b1 || echo "EXPR_FAIL sd15_b1"
EVAL_SET=v2 $P scripts/student_infer.py --tag sd15_b1 --style cartoon3d --no-perf && \
  EVAL_SET=v2 $P scripts/eval_quality.py --tag student_sd15_b1 --styles cartoon3d || echo "EVAL_FAIL sd15_b1"
R=$(cat configs/dmd_recipe.txt); echo "RECIPE $R"
$P scripts/train_dmd.py --style cartoon3d --manifest manifest_round4.jsonl --expr --max-pairs 5000 --expr-frac 0.6 \
   --init-from sd15_b1 --tag sd15_dmd --steps 2000 --fake-warmup 200 --lr 1e-4 --ckpt-every 250 $R || { echo "SD15_DMD_FAIL"; exit 1; }
$P scripts/expr_eval.py --tag sd15_dmd || echo "EXPR_FAIL sd15_dmd"
EVAL_SET=v2 $P scripts/student_infer.py --tag sd15_dmd --style cartoon3d --no-perf && \
  EVAL_SET=v2 $P scripts/eval_quality.py --tag student_sd15_dmd --styles cartoon3d || echo "EVAL_FAIL sd15_dmd"
TAGS="klein_personal student_b1 student_c3_dmd3 student_dl1 student_dl1_dmd student_sd15_b1 student_sd15_dmd" OUT=_sd15_v2 STYLE=cartoon3d \
  EVAL_SET=v2 $P scripts/skin_compare.py || echo "SKIN_FAIL"
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "SD15_DONE $(date -Is)"
