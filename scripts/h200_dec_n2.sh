#!/bin/bash
# Phone-decoder fine-tune (user "go ahead" 2026-11-19 ~13:40 VN) on dev-duyhv-2 after dl6. STUDENT = frozen student whose latents
# the decoder learns to decode (rule from the plan: dl6g1 if at least as good as dl5g, else dl5g; set by env at launch).
#   GPU 0: dec1 = L1 + LPIPS (full 512 px) vs klein PNG              -> how much comes from retargeting the decoder alone
#   GPU 1: dec2 = dec1 + frozen-DINOv2 critic 0.1 (from step 500)     -> does the critic add the crispness?
# 10k steps, bs 16, lr 1e-4. Smoke first (60 steps, exercises val + save). Evals: student_<S>_decN on v2/real/selfie, expressions,
# skin, detail_bands + zoom + sheet vs the student with the original decoder.
# Acceptance: mid/fine detail clearly up with no artifacts (zoom), likeness drop <= 0.05, no skin regression, expr drop <= 0.02.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-dl5
P=$VENV/bin/python
S=${STUDENT:?set STUDENT}
CUDA_VISIBLE_DEVICES=0 $P scripts/train_decoder.py --student $S --build-latents-only > logs/dec_latents.log 2>&1 || { echo "DEC_FAIL latents"; exit 1; }
echo "DEC_LATENTS_DONE $(date -Is)"
CUDA_VISIBLE_DEVICES=0 $P scripts/train_decoder.py --student $S --tag smoke_dec --steps 60 --val-every 50 --gan-w 0.1 --gan-start 20 \
  > logs/dec_smoke.log 2>&1 || { echo "DEC_SMOKE_FAIL"; tail -20 logs/dec_smoke.log; exit 1; }
rm -rf weights/decoder/smoke_dec
echo "DEC_SMOKE_OK $(date -Is)"
# baseline student (original decoder) on any eval set it is missing (dl5g had no selfie run)
for ES in v2 real selfie; do
  [ -n "$(ls outputs/student_$S/cartoon3d/avatar 2>/dev/null)" ] && [ "$ES" != selfie ] && continue
  CUDA_VISIBLE_DEVICES=1 EVAL_SET=$ES $P scripts/student_infer.py --tag $S --style cartoon3d --no-perf > /dev/null 2>&1
done
evals() {  # $1 gpu, $2 decoder tag
  g=$1; dtag=$2; o=${S}_$dtag
  CUDA_VISIBLE_DEVICES=$g $P scripts/expr_eval.py --tag $S --decoder $dtag || echo "EXPR_FAIL $o"
  for ES in v2 real selfie; do
    CUDA_VISIBLE_DEVICES=$g EVAL_SET=$ES $P scripts/student_infer.py --tag $S --decoder $dtag --style cartoon3d --no-perf && \
      CUDA_VISIBLE_DEVICES=$g EVAL_SET=$ES $P scripts/eval_quality.py --tag student_$o --styles cartoon3d || echo "EVAL_FAIL $o $ES"
  done
  EVAL_SET=real STYLE=cartoon3d TAGS="klein_personal student_$S student_$o" OUT=_${o}_real $P scripts/skin_compare.py || echo "SKIN_FAIL $o"
}
run() {  # $1 gpu, $2 decoder tag, rest = extra args
  g=$1; dtag=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g $P scripts/train_decoder.py --student $S --tag $dtag "$@" > logs/dec_$dtag.log 2>&1 || { echo "DEC_FAIL $dtag train"; return 1; }
  echo "DEC_TRAIN_DONE $dtag $(date -Is)"
  evals $g $dtag >> logs/dec_$dtag.log 2>&1
  echo "DEC_EVAL_DONE $dtag $(date -Is)"
}
run 0 dec1 &
run 1 dec2 --gan-w 0.1 --gan-start 500 &
wait
T="student_$S student_${S}_dec1 student_${S}_dec2"
$P scripts/detail_bands.py $T --eval-set v2 || echo "DETAIL_FAIL"
$P scripts/iris_colour.py klein_personal $T || echo "IRIS_FAIL"
$P scripts/zoom_detail_sheet.py --tags klein_personal,student_$S,student_${S}_dec1,student_${S}_dec2 --names fe2_157,fe2_232,fe2_007,rp_232 --out grids/zoom_dec.jpg || echo "SHEET_FAIL"
$P scripts/compare_students_sheet.py --tags klein_personal,student_$S,student_${S}_dec1,student_${S}_dec2 --out grids/dec_sheet.jpg || echo "SHEET_FAIL"
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "DEC_ALL_DONE $(date -Is)"
