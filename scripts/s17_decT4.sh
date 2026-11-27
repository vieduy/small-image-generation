#!/bin/bash
# decT4 (2026-11-27, user "yes start decT4 on dev-duyhv-gpu"): phone decoder for tS4 (frozen) = decg2's recipe (L1 + LPIPS full 512 +
# DINOv2 critic 0.1 from step 500, lr 1e-4, bs 16, 10000 steps - copied from decg2, unverified; 1 GPU instead of 2) but the TARGET is
# the teacher's image for the same photo / flip / fixed noise decoded by decg2 (scripts/train_decoder_t.py --target-latents), init decg2.
# Pass rule (= decg2's acceptance): keep decT4 only if tS4 + decT4 beats tS4 + decg2 on sharpness (sharp_ratio mean) without losing
# expression accuracy (eval_e1 all, beyond its CI) or likeness (v2 / real / selfie, > 0.01 drop).
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
L=logs/s17; mkdir -p $L
export CUDA_VISIBLE_DEVICES=0
echo "S17_START $(date -Is)"
TGT=data/pairs/cartoon3d/teacher_tB1_s8_img3.0_k1.pt
$P scripts/train_decoder_t.py --student tS4 --build-latents-only > $L/latents.log 2>&1 || { echo "S17_FAIL latents"; exit 1; }
echo "S17_LATENTS_DONE $(date -Is)"
$P scripts/train_decoder_t.py --student tS4 --tag _smk_decT4 --init decg2 --target-latents $TGT --steps 30 --val-every 20 --gan-w 0.1 --gan-start 10 \
  > $L/smoke.log 2>&1 || { echo "S17_FAIL smoke"; exit 1; }
rm -rf weights/decoder/_smk_decT4; echo "S17_SMOKE_OK $(date -Is)"
$P scripts/train_decoder_t.py --student tS4 --tag decT4 --init decg2 --target-latents $TGT --steps 10000 --bs 16 --lr 1e-4 \
  --gan-w 0.1 --gan-start 500 > $L/train_decT4.log 2>&1 || { echo "S17_FAIL train"; exit 1; }
echo "S17_TRAIN_DONE $(date -Is)"
# eval alias so results/e1/tS4.json (decg2) is not overwritten
A=weights/student/tS4d; mkdir -p $A; [ -e $A/unet ] || ln -s ../tS4/unet $A/unet; cp weights/student/tS4/task_emb.pt weights/student/tS4/config.json $A/
$P scripts/eval_e1.py --runs tS4d --decoder decT4 > $L/eval_e1.log 2>&1 || echo "S17_FAIL eval_e1"
$P scripts/expr_eval.py --tag tS4 --decoder decT4 > $L/expr.log 2>&1 || echo "S17_FAIL expr"
$P scripts/sharp_ratio.py cartoon3d dl6g2_decg2 tS4_decg2 tS4_decT4 > results/sharp_decT4.txt 2>&1 || echo "S17_FAIL sharp"
for E in v2 real selfie; do
  EVAL_SET=$E $P scripts/student_infer.py --tag tS4 --decoder decT4 --style cartoon3d --no-perf >> $L/quality.log 2>&1 && \
  EVAL_SET=$E $P scripts/eval_quality.py --tag student_tS4_decT4 --styles cartoon3d >> $L/quality.log 2>&1 || echo "S17_FAIL quality $E"
done
chmod -R a+rwX weights/decoder/decT4 weights/decoder/_zs_tS4* $A outputs results grids $L 2>/dev/null
echo "S17_ALL_DONE $(date -Is)"
