#!/bin/bash
# 2-step DreamLite student for output DETAIL (user request 2026-11-18 ~14:30 UTC). Queued on GPU 0 after h200_dl2.sh.
# dl_arch.forward(sigmas=[1.0, 0.5]): step 1 as the 1-step student (t=1000); its x0 is re-noised to sigma 0.5 with a
# fixed per-step noise and denoised again with the photo latent beside it (photo conditions BOTH steps; the failed
# an_lora_m4 SDXS test re-noised without the photo). Trained with DMD (DMD2 backward simulation: random step k per
# iteration, earlier steps without gradient), same recipe/subset as dl2_dmd, continued from dl2_dmd, 1500 steps.
# Evals: 2-step vs dl2_dmd on expressions + v2/real/selfie; the same weights at 1 step (k1) separates "2nd step" from
# "more DMD"; detail_bands (fine-detail energy vs klein) + iris colour + side-by-side sheet.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
until grep -qaE "DL2DMD_DONE|DL2DMD_FAIL|DL2_FAIL|DL2DMD_SMOKE_FAIL" logs/h200_dl2.log; do sleep 30; done
grep -qa DL2DMD_DONE logs/h200_dl2.log || { echo "DL2_2S_SKIP: dl2_dmd did not finish"; exit 1; }
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
R=$(cat configs/dmd_recipe.txt); echo "RECIPE $R"
D="--style cartoon3d --manifest manifest_v2.jsonl --expr --max-pairs 5000 --expr-frac 0.6 --init-from dl2_dmd --lr 1e-4 --ms-sigmas 1.0,0.5"
$P scripts/train_dmd.py $D --tag smoke_dl2_2s --steps 3 --fake-warmup 2 --val-every 1000 $R || { echo "DL2_2S_SMOKE_FAIL"; exit 1; }
rm -rf weights/student/smoke_dl2_2s
$P scripts/train_dmd.py $D --tag dl2_2s --steps 1500 --fake-warmup 200 --ckpt-every 250 $R || { echo "DL2_2S_FAIL train"; exit 1; }
echo "DL2_2S_TRAIN_DONE $(date -Is)"
$P scripts/expr_eval.py --tag dl2_2s || echo "EXPR_FAIL dl2_2s"
for ES in v2 real selfie; do
  EVAL_SET=$ES $P scripts/student_infer.py --tag dl2_2s --style cartoon3d --no-perf && \
    EVAL_SET=$ES $P scripts/eval_quality.py --tag student_dl2_2s --styles cartoon3d || echo "EVAL_FAIL dl2_2s $ES"
  EVAL_SET=$ES STYLE=cartoon3d TAGS="klein_personal student_dl2_dmd student_dl2_2s" OUT=_dl2_2s_$ES $P scripts/skin_compare.py || echo "SKIN_FAIL $ES"
done
EVAL_SET=v2 $P scripts/student_infer.py --tag dl2_2s --style cartoon3d --no-perf --dl-k 1 && \
  EVAL_SET=v2 $P scripts/eval_quality.py --tag student_dl2_2s_k1 --styles cartoon3d || echo "EVAL_FAIL dl2_2s_k1"
$P scripts/detail_bands.py student_dl1_dmd student_dl2 student_dl2_dmd student_dl2_2s_k1 student_dl2_2s || echo "DETAIL_FAIL"
$P scripts/iris_colour.py student_dl1_dmd student_dl2_dmd student_dl2_2s || echo "IRIS_FAIL"
$P scripts/compare_students_sheet.py --tags klein_personal,student_dl2_dmd,student_dl2_2s --out grids/dl2_2s_sheet.jpg || echo "SHEET_FAIL"
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "DL2_2S_DONE $(date -Is)"
