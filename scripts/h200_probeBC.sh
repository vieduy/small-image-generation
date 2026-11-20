#!/bin/bash
# Probes B and C (user request 2026-11-19): what makes the data-v2 DMD pass soften the students?
#   B (GPU 0) probeB_oldprompt: DMD from dl2, data v2 images/targets/subset, but the klein-base judge gets OLD round-4-style
#     prompts (manifest_v2_oldprompt.jsonl: TP.avatar_prompt / sticker_prompt / expr_sticker_prompt, median 85 vs 159 words,
#     no "keep the face ... expression ... exactly"). Stays sharp -> the v2 prompts cause it.
#   C (GPU 1) probeC_dl1start: DMD from dl1 (old-data student) with data v2 and v2 prompts. Softens -> prompts/data of the
#     v2 pass; stays sharp -> the start model (dl2) causes it.
# Both = the dl2_dmd pass (dmd_recipe.txt, 5000-pair subset, 60% expressions, seed 0, same val tiles), 500 steps.
# Read-out: val-tile sharpness series (vs dl2_dmd 708->448->380, dl1_dmd 904->1011->909), v2 likeness, detail_bands, expr_eval.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
R=$(cat configs/dmd_recipe.txt); echo "RECIPE $R"
D="--style cartoon3d --manifest manifest_v2.jsonl --expr --max-pairs 5000 --expr-frac 0.6 --fake-warmup 200 --lr 1e-4 --steps 500"
run() {  # $1 gpu, $2 tag, rest = extra args
  g=$1; t=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g $P scripts/train_dmd.py $D --tag $t "$@" $R > logs/probe_$t.log 2>&1 || { echo "PROBE_FAIL $t train"; return 1; }
  CUDA_VISIBLE_DEVICES=$g EVAL_SET=v2 $P scripts/student_infer.py --tag $t --style cartoon3d --no-perf >> logs/probe_$t.log 2>&1 && \
    CUDA_VISIBLE_DEVICES=$g EVAL_SET=v2 $P scripts/eval_quality.py --tag student_$t --styles cartoon3d >> logs/probe_$t.log 2>&1 || echo "EVAL_FAIL $t"
  CUDA_VISIBLE_DEVICES=$g $P scripts/expr_eval.py --tag $t >> logs/probe_$t.log 2>&1 || echo "EXPR_FAIL $t"
  echo "PROBE_DONE $t $(date -Is)"
}
run 0 probeB_oldprompt --init-from dl2 --prompt-manifest manifest_v2_oldprompt.jsonl &
run 1 probeC_dl1start --init-from dl1 &
wait
$P scripts/dmd_sharpness_series.py dl1_dmd dl2_dmd probeA_syn probeB_oldprompt probeC_dl1start
$P scripts/detail_bands.py student_dl1_dmd student_dl2_dmd student_probeA_syn student_probeB_oldprompt student_probeC_dl1start --eval-set v2 || echo "DETAIL_FAIL"
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "PROBEBC_DONE $(date -Is)"
