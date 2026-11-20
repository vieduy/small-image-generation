#!/bin/bash
# dl6g4 = dl6g2 recipe with critic 0.4, trained on BOTH dev-duyhv GPUs (user 2026-11-19 ~14:55 VN: "do stronger critic 0.4",
# "do A": one run split over 2 GPUs). torchrun DDP in train_dl.py (global bs 16 = 8 per GPU, same lr/steps as dl6g2).
# From dl3F, 3000 steps, lr 1e-5, --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --tgt-cache --compile, --gan-w 0.4.
# Acceptance vs dl6g2: more detail in zoom with no grain/crosshatch, no over-aging, likeness drop <= 0.05, expr drop <= 0.02.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
TR="$VENV/bin/torchrun --nproc_per_node 2 --master_port 29541"
L="--lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --tgt-cache --compile"
B="--style cartoon3d --manifest manifest_v2.jsonl --expr --init-from dl3F --lr 1e-5 $L"
CUDA_VISIBLE_DEVICES=0,1 $TR scripts/train_dl.py $B --tag smoke_dl6g4 --steps 60 --val-every 50 --gan-w 0.4 --gan-start 20 \
  > logs/dl6g4_smoke.log 2>&1 || { echo "DL6G4_SMOKE_FAIL"; tail -30 logs/dl6g4_smoke.log; exit 1; }
rm -rf weights/student/smoke_dl6g4
echo "DL6G4_SMOKE_OK $(date -Is)"
CUDA_VISIBLE_DEVICES=0,1 $TR scripts/train_dl.py $B --tag dl6g4 --steps 3000 --val-every 500 --gan-w 0.4 --gan-start 200 \
  > logs/dl6g4.log 2>&1 || { echo "DL6G4_FAIL train"; exit 1; }
echo "DL6G4_TRAIN_DONE $(date -Is)"
t=dl6g4
( for ES in v2 real; do
    CUDA_VISIBLE_DEVICES=0 EVAL_SET=$ES $P scripts/student_infer.py --tag $t --style cartoon3d --no-perf && \
      CUDA_VISIBLE_DEVICES=0 EVAL_SET=$ES $P scripts/eval_quality.py --tag student_$t --styles cartoon3d || echo "EVAL_FAIL $t $ES"
  done ) >> logs/dl6g4.log 2>&1 &
( CUDA_VISIBLE_DEVICES=1 $P scripts/expr_eval.py --tag $t || echo "EXPR_FAIL $t"
  CUDA_VISIBLE_DEVICES=1 EVAL_SET=selfie $P scripts/student_infer.py --tag $t --style cartoon3d --no-perf && \
    CUDA_VISIBLE_DEVICES=1 EVAL_SET=selfie $P scripts/eval_quality.py --tag student_$t --styles cartoon3d || echo "EVAL_FAIL $t selfie" ) >> logs/dl6g4.log 2>&1 &
wait
EVAL_SET=real STYLE=cartoon3d TAGS="klein_personal student_dl6g2 student_$t" OUT=_${t}_real $P scripts/skin_compare.py >> logs/dl6g4.log 2>&1 || echo "SKIN_FAIL $t"
echo "DL6G4_EVAL_DONE $(date -Is)"
T="student_dl6g1 student_dl6g2 student_dl6g4"
$P scripts/detail_bands.py $T --eval-set v2 || echo "DETAIL_FAIL"
$P scripts/iris_colour.py klein_personal $T || echo "IRIS_FAIL"
$P scripts/zoom_detail_sheet.py --tags klein_personal,student_dl6g1,student_dl6g2,student_dl6g4 --names fe2_157,fe2_232,fe2_007,rp_232 --out grids/zoom_dl6g4.jpg || echo "SHEET_FAIL"
$P scripts/compare_students_sheet.py --tags klein_personal,student_dl6g2,student_dl6g4 --out grids/dl6g4_sheet.jpg || echo "SHEET_FAIL"
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "DL6G4_ALL_DONE $(date -Is)"
