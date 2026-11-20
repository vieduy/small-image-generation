#!/bin/bash
# DreamLite eye fix (user request 2026-11-17): dl1_dmd draws solid black eyes (the chroma-only eye term cannot see a luminance
# error; the TinyVAE round-trip keeps eyes intact, grids/vae_eye_roundtrip.jpg). Continue the dl1_dmd pack for 750 DMD steps
# with an extra 4x4-pooled luminance term in the eye masks (--eye-luma-w), everything else = the c3_dmd3 recipe.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
R=$(cat configs/dmd_recipe.txt); echo "RECIPE $R --eye-luma-w 4"
$P scripts/train_dmd.py --style cartoon3d --manifest manifest_round4.jsonl --expr --max-pairs 5000 --expr-frac 0.6 \
   --init-from dl1_dmd --tag dl1_dmd_eye --steps 750 --fake-warmup 200 --lr 1e-4 --ckpt-every 250 $R --eye-luma-w 4 || { echo "EYE_FAIL"; exit 1; }
$P scripts/expr_eval.py --tag dl1_dmd_eye || echo "EXPR_FAIL dl1_dmd_eye"
for ES in v2 real; do
  EVAL_SET=$ES $P scripts/student_infer.py --tag dl1_dmd_eye --style cartoon3d --no-perf && \
    EVAL_SET=$ES $P scripts/eval_quality.py --tag student_dl1_dmd_eye --styles cartoon3d || echo "EVAL_FAIL $ES"
  EVAL_SET=$ES STYLE=cartoon3d TAGS="klein_personal student_c3_dmd3 student_dl1_dmd student_dl1_dmd_eye" OUT=_dl1eye_$ES $P scripts/skin_compare.py || echo "SKIN_FAIL $ES"
done
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "DL1EYE_DONE $(date -Is)"
