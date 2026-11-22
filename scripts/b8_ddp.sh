#!/bin/bash
# dl8 one-model multi-style on BOTH GPUs (user 2026-11-21 "use both gpu to accelerate training"): replaces b8_onemodel.sh
# (stopped during setup, before any step). Same dl8 recipe; torchrun DDP, --bs 16 = global batch (8 per GPU) -> same optimisation,
# ~2x faster. dl8n (no-FiLM ablation) dropped for now. Then dec8 (GPU 0) || dl8 evals with the specialists' decoders (GPU 1),
# then dl8 + dec8 evals.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-b8 TRITON_CACHE_DIR=/tmp/triton-cache-b8
P=$VENV/bin/python
L=logs/b8; mkdir -p $L
COMMON="--style cartoon3d --manifest manifest_v2.jsonl --expr --init-from dl6g2 --joint-styles comic_line,chibi --joint-manifest manifest_v2.jsonl
  --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --gan-w 0.2 --gan-start 200 --lr 2e-5 --steps 14000 --val-every 1000 --bs 16
  --style-tokens 32 --distill-w 1.0
  --distill-latents cartoon3d=weights/decoder/_zs_dl6g2_manifest_v2.pt,comic_line=weights/decoder/_zs_pk_comic_manifest_v2.pt,chibi=weights/decoder/_zs_pk_chibi_manifest_v2.pt"
evalv() {  # $1 GPU  $2 view tag  $3 style  $4 decoder
  CUDA_VISIBLE_DEVICES=$1 $P scripts/expr_eval.py --tag $2 --decoder $4 || echo "EXPR_FAIL $2 $4"
  CUDA_VISIBLE_DEVICES=$1 EVAL_SET=v2 $P scripts/student_infer.py --tag $2 --decoder $4 --style $3 --no-perf && \
    CUDA_VISIBLE_DEVICES=$1 EVAL_SET=v2 $P scripts/eval_quality.py --tag student_$2_$4 --styles $3 || echo "Q_FAIL $2 $4"
  $P scripts/sharp_ratio.py $3 $2_$4 > results/sharp_$2_$4.txt 2>&1 || echo "SHARP_FAIL $2 $4"
}
echo "B8_START $(date -Is)"
CUDA_VISIBLE_DEVICES=0,1 $VENV/bin/torchrun --nproc_per_node 2 --master_port 29541 scripts/train_dl.py $COMMON --style-film --tag dl8 > $L/train_dl8.log 2>&1 \
  || { echo "B8_FAIL dl8 train"; exit 1; }
echo "B8_P1_DONE $(date -Is)"
( CUDA_VISIBLE_DEVICES=0 $P scripts/train_decoder_ms.py --student dl8 --tag dec8 > $L/train_dec8.log 2>&1 || echo "DEC8_FAIL" ) &
( evalv 1 dl8 cartoon3d decg2; evalv 1 dl8_comic_line comic_line decCm; evalv 1 dl8_chibi chibi decg2 ) > $L/eval_spec.log 2>&1 &
wait
echo "B8_P2_DONE $(date -Is)"
if [ -f weights/decoder/dec8/style_film_dec.pt ]; then
  ( evalv 0 dl8 cartoon3d dec8; evalv 0 dl8_chibi chibi dec8_chibi ) > $L/eval_dec8_a.log 2>&1 &
  ( evalv 1 dl8_comic_line comic_line dec8_comic_line ) > $L/eval_dec8_b.log 2>&1 &
  wait
fi
chmod -R a+rwX weights/student/dl8* weights/decoder/dec8* weights/decoder/_zs_dl8* outputs results grids $L 2>/dev/null
echo "B8_ALL_DONE $(date -Is)"
