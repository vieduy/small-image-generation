#!/bin/bash
# Plan B2 + B3 (user 2026-11-20: "queue b3, not to ask me again"), run as dstack task b23-sana (2x H200NVL).
#   B2: DC-AE latents of cartoon3d manifest_v2 (2 parts in parallel, then merge)
#   B3: sD1 = Sana-Sprint student, diptych [noise | photo] (GPU 0); sC1 = channel concat (GPU 1). 15000 steps, lr 5e-5, tokens 1e-3,
#       bs 16, losses mse + LPIPS256 + lp512 + DINOv2 critic 0.2, tiny decoder tdec1r (all NOT curve-based; checkpoints every 2500)
#   Eval: scripts/sana_eval.py (tiny encoder = phone path, and full DC-AE encoder) + scripts/eval_e1.py --runs dl6g2 (same metrics)
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-b23 TRITON_CACHE_DIR=/tmp/triton-cache-b23
P=$VENV/bin/python
L=logs/b23
mkdir -p $L
echo "B23_START $(date -Is)"
if [ ! -f data/pairs/cartoon3d/sana_lat_manifest_v2.pt ]; then
  CUDA_VISIBLE_DEVICES=0 $P scripts/sana_encode.py --part 0 --nparts 2 > $L/enc0.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 $P scripts/sana_encode.py --part 1 --nparts 2 > $L/enc1.log 2>&1 &
  wait
  $P scripts/sana_encode.py --nparts 2 --merge > $L/enc_merge.log 2>&1 || { echo "B2_FAIL"; exit 1; }
fi
echo "B2_DONE $(date -Is)"
CUDA_VISIBLE_DEVICES=0 $P scripts/train_sana.py --tag sD1 --cond diptych --tiny-dec tdec1r > $L/train_sD1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $P scripts/train_sana.py --tag sC1 --cond concat --tiny-dec tdec1r > $L/train_sC1.log 2>&1 &
wait
echo "B3_TRAIN_DONE $(date -Is)"
for T in sD1 sC1; do [ -d weights/sana_student/$T/final ] || echo "B3_FAIL $T"; done
( [ -d weights/sana_student/sD1/final ] && for E in tiny dcae; do CUDA_VISIBLE_DEVICES=0 $P scripts/sana_eval.py --tag sD1 --enc $E; done
  CUDA_VISIBLE_DEVICES=0 $P scripts/eval_e1.py --runs dl6g2 ) > $L/eval_gpu0.log 2>&1 &
( [ -d weights/sana_student/sC1/final ] && for E in tiny dcae; do CUDA_VISIBLE_DEVICES=1 $P scripts/sana_eval.py --tag sC1 --enc $E; done ) > $L/eval_gpu1.log 2>&1 &
wait
chmod -R a+rwX weights/sana_student outputs/sana results/sana grids/sana outputs/e1 results/e1 grids/e1 data/pairs/cartoon3d/sana_lat_* $L 2>/dev/null
echo "B23_ALL_DONE $(date -Is)"
