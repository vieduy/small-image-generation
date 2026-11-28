#!/bin/bash
# DMD2 for ghost hands (2026-11-27, user "go with C, start DMD2"). Usage: TAG=dmA REGW=0.25 [WAIT=logs/s16_chain.out:S16_ALL_DONE] bash scripts/s21_dmd.sh
# = tS4 init + scripts/train_dl_dmd.py: real score tB1 (tS4 CFG setting), fake score = trainable tB1 copy (lr 5e-6, 2 updates/step,
# 200 warm-up), dmd_w 1, regression terms x REGW (dmA 0.25 = anchored; dmB 0 = DMD2-faithful: DMD + DINOv2 critic only), student lr 5e-6
# (tS4 used 2e-5 from dl6g2; lower for a fine-tune of a converged model), 4000 steps (earlier project DMD runs: 2000; unverified),
# snapshots 1000 / 2000 / 3000. Smoke 2026-11-27: 1.85 s/step on H200.
# Evals: eval_e1 + expr_eval + ghost metric (s15 inputs, OWLv2) per checkpoint; final: sharpness, quality v2/real/selfie, split, visual.
# Pass rule (fixed before running): extra-hand <= 2.0% (tS4 4.0%) and no-hand <= 2% on thumbs/wave/ok; expr_all >= 0.65 (tS4 CI low);
# likeness v2 / real / selfie within 0.01 of tS4 (0.214 / 0.157 / 0.183); sharpness >= 0.69 (tS4 0.72 - 0.03).
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-$TAG TRITON_CACHE_DIR=/tmp/triton-cache-$TAG
P=$VENV/bin/python
L=logs/s21_$TAG; mkdir -p $L
export CUDA_VISIBLE_DEVICES=${GPU:-0}
echo "S21_START $TAG reg_w=$REGW $(date -Is)"
if [ -n "$WAIT" ]; then f=${WAIT%%:*}; m=${WAIT##*:}; until grep -q "$m" $f 2>/dev/null; do sleep 60; done; echo "S21_WAIT_OVER $(date -Is)"; fi
NAME=teacher_tB1_s8_img3.0_k1
D=data/pairs/cartoon3d
$P scripts/train_dl_dmd.py --style cartoon3d --manifest manifest_v2.jsonl --expr --init-from tS4 --lr 5e-6 --val-every 1000 \
  --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --gan-w 0.2 --gan-start 200 \
  --teacher-latents $D/$NAME.pt --drop-list $D/${NAME}_consistency_drop.json --reg-w $REGW --dmd-w 1 --fake-warmup 200 \
  --steps 4000 --save-at 1000,2000,3000 --tag $TAG > $L/train.log 2>&1 || { echo "S21_FAIL train"; exit 1; }
echo "S21_TRAIN_DONE $(date -Is)"
RUNS="${TAG}_s1000 ${TAG}_s2000 ${TAG}_s3000 $TAG"
for R in $RUNS; do
  $P scripts/eval_e1.py --runs $R > $L/eval_e1_$R.log 2>&1 || echo "S21_FAIL eval_e1 $R"
  $P scripts/expr_eval.py --tag $R --decoder decg2 > $L/expr_$R.log 2>&1 || echo "S21_FAIL expr $R"
  $P scripts/s19_student_s15.py $R > $L/render_$R.log 2>&1 || echo "S21_FAIL render $R"
done
$P scripts/s18_owl_hands.py $RUNS > $L/owl.log 2>&1 || echo "S21_FAIL owl"
$P scripts/s18_ghost_rate.py > $L/ghost_rate.log 2>&1 || echo "S21_FAIL ghost_rate"
echo "S21_CKPT_EVALS_DONE $(date -Is)"
$P scripts/sharp_ratio.py cartoon3d dl6g2_decg2 tS4_decg2 ${TAG}_decg2 > results/sharp_${TAG}_decg2.txt 2>&1 || echo "S21_FAIL sharp"
for E in v2 real selfie; do
  EVAL_SET=$E $P scripts/student_infer.py --tag $TAG --decoder decg2 --style cartoon3d --no-perf >> $L/quality.log 2>&1 && \
  EVAL_SET=$E $P scripts/eval_quality.py --tag student_${TAG}_decg2 --styles cartoon3d >> $L/quality.log 2>&1 || echo "S21_FAIL quality $E"
done
$P scripts/s8_split_eval.py --tags dl6g2 tS4 $TAG --out $TAG > $L/split.log 2>&1 || echo "S21_FAIL split"
$P scripts/s5_visual_tS2.py dl6g2 tS4 $TAG > $L/visual.log 2>&1 || echo "S21_FAIL visual"
chmod -R a+rwX weights/student/${TAG}* outputs results grids $L 2>/dev/null
echo "S21_ALL_DONE $(date -Is)"
