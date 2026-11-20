#!/bin/bash
# Round 4 (user-approved 2026-11-16): base v1 "b1" = same SDXS 1-step student, trained from the SDXS init on the full
# round-3b manifest (synthetic + selfie + legal-approved real + expressions), then style packs rebuilt on b1.
set -ex
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
echo "B1_START $(date -Is)"
$P scripts/train_student.py --style cartoon3d --tag b1 --param direct --steps 16000 --expr --manifest manifest_round3b.jsonl
for ES in v2 selfie real; do
  export EVAL_SET=$ES
  $P scripts/student_infer.py --tag b1 --style cartoon3d --no-perf
  $P scripts/eval_quality.py --tag student_b1 --styles cartoon3d
  TAGS="klein_personal student_s5d student_b1" OUT=_b1_$ES STYLE=cartoon3d $P scripts/skin_compare.py
done
unset EVAL_SET
$P scripts/expr_eval.py --tag b1 || echo "EXPR_EVAL_FAILED"
echo "B1_TRAIN_DONE $(date -Is)"
# style packs on b1 (same recipe as an_lora / fv_lora)
export EVAL_SET=v2
for S in anime flat_vector; do
  T=$([ $S = anime ] && echo an_b1 || echo fv_b1)
  $P scripts/train_student.py --style $S --tag $T --param direct --steps 6000 --init-from b1 --lora-rank 32 --lr 1e-4
  $P scripts/student_infer.py --tag $T --style $S --no-perf
  $P scripts/eval_quality.py --tag student_$T --styles $S
done
STYLE=anime TAGS="klein_personal student_an_lora student_an_b1" OUT=_an_b1 $P scripts/skin_compare.py
STYLE=flat_vector TAGS="klein_personal student_fv_lora student_fv_b1" OUT=_fv_b1 $P scripts/skin_compare.py
chmod -R a+rwX outputs results grids weights logs 2>/dev/null || true
echo "B1_ALL_DONE $(date -Is)"
