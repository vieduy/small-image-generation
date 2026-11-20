#!/bin/bash
# Production quick wins on dev-duyhv-2 (user 2026-11-20: "2a + 2b RGBA decoder and wider decoder; 3a 2-step avatar").
# GPU 0, decoders from decg2 on dl6g2 latents, 5000 steps each, run together:
#   decC = decg2 + 5000 more steps (control: more training, same size)   decW = width 80 (2b)   decD = +1 block per stage (2b)
#   decA = RGBA, 4th channel = cut-out alpha (2a)
# GPU 1, refiners (3a): step-2-only LoRA r64 on frozen dl6g2, avatar pairs only, 3000 steps, dl6g2 losses:
#   rf50 = sigma 0.5   rf30 = sigma 0.3
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
mkdir -p logs
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-prod1 TRITON_CACHE_DIR=/tmp/triton-cache-prod1
DEC="--student dl6g2 --init decg2 --steps 5000 --gan-w 0.1 --val-every 1000 --workers 6"
L="--style cartoon3d --manifest manifest_v2.jsonl --expr --init-from dl6g2 --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --tgt-cache --gan-w 0.2 --gan-start 200 --val-every 500"

evalset() {  # $1 = GPU, $2 = student tag, $3 = decoder tag (""), $4 = output tag
  for ES in v2 real selfie; do
    CUDA_VISIBLE_DEVICES=$1 EVAL_SET=$ES $P scripts/student_infer.py --tag $2 ${3:+--decoder $3} --style cartoon3d --no-perf && \
      CUDA_VISIBLE_DEVICES=$1 EVAL_SET=$ES $P scripts/eval_quality.py --tag student_$4 --styles cartoon3d || echo "EVAL_FAIL $4 $ES"
  done
  EVAL_SET=real STYLE=cartoon3d TAGS="klein_personal student_dl6g2_decg2 student_$4" OUT=_$4_real $P scripts/skin_compare.py || echo "SKIN_FAIL $4"
}

# ---------------- GPU 0: decoders ----------------
(
  # alpha labels first (single process, BiRefNet_lite on decg2 renders), then the 4 trainings together
  CUDA_VISIBLE_DEVICES=0 $P scripts/train_decoder.py $DEC --tag decA --alpha > logs/dec_decA.log 2>&1 &
  CUDA_VISIBLE_DEVICES=0 $P scripts/train_decoder.py $DEC --tag decC > logs/dec_decC.log 2>&1 &
  CUDA_VISIBLE_DEVICES=0 $P scripts/train_decoder.py $DEC --tag decW --width 80 > logs/dec_decW.log 2>&1 &
  CUDA_VISIBLE_DEVICES=0 $P scripts/train_decoder.py $DEC --tag decD --extra-blocks 1,1,1,1 > logs/dec_decD.log 2>&1 &
  wait
  echo "DEC_TRAIN_DONE $(date -Is)"
  for D in decC decW decD; do
    [ -f weights/decoder/$D/diffusion_pytorch_model.safetensors ] || { echo "DEC_FAIL $D"; continue; }
    ( CUDA_VISIBLE_DEVICES=0 $P scripts/expr_eval.py --tag dl6g2 --decoder $D || echo "EXPR_FAIL $D"
      evalset 0 dl6g2 $D dl6g2_$D ) > logs/dec_eval_$D.log 2>&1 &
  done
  if [ -f weights/decoder/decA/diffusion_pytorch_model.safetensors ]; then
    ( for ES in v2 real; do CUDA_VISIBLE_DEVICES=0 EVAL_SET=$ES $P scripts/student_infer.py --tag dl6g2 --decoder decA --style cartoon3d --no-perf; done
      CUDA_VISIBLE_DEVICES=0 $P scripts/alpha_eval.py --name dl6g2_decA \
        --dirs outputs/student_dl6g2_decA/cartoon3d/sticker,outputs/student_dl6g2_decA/cartoon3d/avatar || echo "ALPHA_FAIL"
    ) > logs/dec_eval_decA.log 2>&1 &
  fi
  wait
  X="student_dl6g2_decg2 student_dl6g2_decC student_dl6g2_decW student_dl6g2_decD"
  $P scripts/detail_bands.py $X --eval-set v2 > results/detail_decoders_CWD.txt 2>&1 || echo "DETAIL_FAIL"
  $P scripts/iris_colour.py klein_personal $X > results/iris_decoders_CWD.txt 2>&1 || echo "IRIS_FAIL"
  $P scripts/zoom_detail_sheet.py --tags klein_personal,student_dl6g2_decg2,student_dl6g2_decC,student_dl6g2_decW,student_dl6g2_decD \
    --names fe2_157,fe2_232,fe2_007,rp_232 --out grids/zoom_decoders_CWD.jpg || echo "SHEET_FAIL"
  echo "DEC_ALL_DONE $(date -Is)"
) > logs/prod_chain1_gpu0.log 2>&1 &

# ---------------- GPU 1: 2-step avatar refiners ----------------
(
  for S in 50 30; do
    CUDA_VISIBLE_DEVICES=1 $P scripts/train_dl.py $L --tag rf$S --lora-rank 64 --lr 1e-4 --refine-sigma 0.$S --only-tasks avatar \
      --steps 3000 > logs/refine_rf$S.log 2>&1 &
  done
  wait
  echo "RF_TRAIN_DONE $(date -Is)"
  for S in 50 30; do
    [ -f weights/student/rf$S/config.json ] || { echo "RF_FAIL rf$S"; continue; }
    evalset 1 rf$S decg2 rf${S}_decg2 > logs/refine_eval_rf$S.log 2>&1 &
  done
  wait
  X="student_dl6g2_decg2 student_rf50_decg2 student_rf30_decg2"
  $P scripts/detail_bands.py $X --eval-set v2 > results/detail_refiners.txt 2>&1 || echo "DETAIL_FAIL"
  $P scripts/iris_colour.py klein_personal $X > results/iris_refiners.txt 2>&1 || echo "IRIS_FAIL"
  $P scripts/zoom_detail_sheet.py --tags klein_personal,student_dl6g2_decg2,student_rf50_decg2,student_rf30_decg2 \
    --names fe2_157,fe2_232,fe2_007,rp_232 --out grids/zoom_refiners.jpg || echo "SHEET_FAIL"
  echo "RF_ALL_DONE $(date -Is)"
) > logs/prod_chain1_gpu1.log 2>&1 &
wait
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
