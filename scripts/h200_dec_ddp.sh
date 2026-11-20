#!/bin/bash
# ONE phone-decoder fine-tune on BOTH dev-duyhv-2 GPUs (user 2026-11-19 ~14:05 VN: "choose one and training on 2 GPU to prove the
# decoder fine-tune help"). Student = dl6g2 (best: fine 0.77, likeness 0.243). Recipe = L1 + LPIPS(full 512) + DINOv2 critic 0.1
# (from step 500), 10k steps, global bs 16 (8 per GPU), lr 1e-4. The dl6g1 dec1/dec2 runs on dev-duyhv were stopped for this.
# Proof = dl6g2 with the original decoder vs dl6g2 + this decoder on the same eval sets (detail_bands + zoom, likeness v2/real/selfie,
# expressions, skin, iris). Acceptance: mid/fine clearly up with no artifacts, likeness drop <= 0.05, no skin regression, expr drop <= 0.02.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-dl5
P=$VENV/bin/python
TR="$VENV/bin/torchrun --nproc_per_node 2 --master_port 29531"
S=dl6g2
D=decg2
CUDA_VISIBLE_DEVICES=0 $P scripts/train_decoder.py --student $S --build-latents-only > logs/decddp_latents.log 2>&1 || { echo "DECDDP_FAIL latents"; exit 1; }
echo "DECDDP_LATENTS_DONE $(date -Is)"
CUDA_VISIBLE_DEVICES=0,1 $TR scripts/train_decoder.py --student $S --tag smoke_decddp --steps 60 --val-every 50 --gan-w 0.1 --gan-start 20 \
  > logs/decddp_smoke.log 2>&1 || { echo "DECDDP_SMOKE_FAIL"; tail -20 logs/decddp_smoke.log; exit 1; }
rm -rf weights/decoder/smoke_decddp
echo "DECDDP_SMOKE_OK $(date -Is)"
CUDA_VISIBLE_DEVICES=0,1 $TR scripts/train_decoder.py --student $S --tag $D --gan-w 0.1 --gan-start 500 > logs/decddp_$D.log 2>&1 \
  || { echo "DECDDP_FAIL train"; exit 1; }
echo "DECDDP_TRAIN_DONE $(date -Is)"
o=${S}_$D
CUDA_VISIBLE_DEVICES=0 $P scripts/expr_eval.py --tag $S --decoder $D >> logs/decddp_$D.log 2>&1 || echo "EXPR_FAIL $o"
for ES in v2 real selfie; do
  CUDA_VISIBLE_DEVICES=0 EVAL_SET=$ES $P scripts/student_infer.py --tag $S --decoder $D --style cartoon3d --no-perf >> logs/decddp_$D.log 2>&1 && \
    CUDA_VISIBLE_DEVICES=0 EVAL_SET=$ES $P scripts/eval_quality.py --tag student_$o --styles cartoon3d >> logs/decddp_$D.log 2>&1 || echo "EVAL_FAIL $o $ES"
done
EVAL_SET=real STYLE=cartoon3d TAGS="klein_personal student_$S student_$o" OUT=_${o}_real $P scripts/skin_compare.py >> logs/decddp_$D.log 2>&1 || echo "SKIN_FAIL $o"
echo "DECDDP_EVAL_DONE $(date -Is)"
T="student_$S student_$o"
$P scripts/detail_bands.py $T --eval-set v2 || echo "DETAIL_FAIL"
$P scripts/iris_colour.py klein_personal $T || echo "IRIS_FAIL"
$P scripts/zoom_detail_sheet.py --tags klein_personal,student_$S,student_$o --names fe2_157,fe2_232,fe2_007,rp_232 --out grids/zoom_decg2.jpg || echo "SHEET_FAIL"
$P scripts/compare_students_sheet.py --tags klein_personal,student_$S,student_$o --out grids/decg2_sheet.jpg || echo "SHEET_FAIL"
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "DECDDP_ALL_DONE $(date -Is)"
