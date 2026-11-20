#!/bin/bash
# Base probe (user request 2026-11-17): fixed-budget screen of a public base, instead of a full b1-style run per base.
# Every base gets the SAME data subset, steps, LRs and evals; SDXS is re-run under the same probe as the reference, and
# SD-Turbo as the calibration check (its full-run result is known: equal to SDXS).
#   A: regression sprint  - full fine-tune from the public base, 3000 steps, 4000-pair round-4 subset (50% expression rows)
#   B: frozen-base LoRA   - base frozen, LoRA r32 (+ tokens), 2000 steps, same subset  -> quality of the pretrained prior
#   C: DMD response       - fresh LoRA pack on A, 500 generator steps, c3_dmd3 recipe and subset
# Evals per test: expr_eval (24 faces x 20 expressions, gestures) + v2 quality (likeness, style) + skin_compare.
# usage: scripts/h200_probe_base.sh <name> <base repo> [text-encoder repo for the token init]
set -x
N=$1; BASE=$2; TE=${3:+--te-from $3}
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
COMMON="--style cartoon3d --manifest manifest_round4.jsonl --expr --max-pairs 4000 --expr-frac 0.5 --param direct --T 999 \
 --bs 16 --emb-lr 1e-3 --val-every 500 --base $BASE $TE"
ev() {
  $P scripts/expr_eval.py --tag $1 || echo "EXPR_FAIL $1"
  EVAL_SET=v2 $P scripts/student_infer.py --tag $1 --style cartoon3d --no-perf && \
    EVAL_SET=v2 $P scripts/eval_quality.py --tag student_$1 --styles cartoon3d || echo "EVAL_FAIL $1"
}
echo "PROBE_START $N $BASE $(date -Is)"
$P scripts/train_student.py --tag pr_${N}_A $COMMON --lr 2e-5 --steps 3000 && ev pr_${N}_A
$P scripts/train_student.py --tag pr_${N}_B $COMMON --lr 1e-4 --steps 2000 --lora-rank 32 && ev pr_${N}_B
R=$(cat configs/dmd_recipe.txt); echo "RECIPE $R"
if [ -f weights/student/pr_${N}_A/unet/config.json ]; then
  $P scripts/train_dmd.py --style cartoon3d --manifest manifest_round4.jsonl --expr --max-pairs 5000 --expr-frac 0.6 \
     --init-from pr_${N}_A --tag pr_${N}_C --steps 500 --fake-warmup 200 --lr 1e-4 $R && ev pr_${N}_C
fi
TAGS="klein_personal student_pr_${N}_A student_pr_${N}_B student_pr_${N}_C" OUT=_probe_${N} STYLE=cartoon3d EVAL_SET=v2 \
  $P scripts/skin_compare.py || echo "SKIN_FAIL $N"
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "PROBE_DONE $N $(date -Is)"
