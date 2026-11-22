#!/bin/bash
# LOCAL H100 copy (GPUs 1+2; H200 fleet host hung, user 2026-11-21 "check again or submit new job"). dl9 = dl8 + comic emphasis: continue dl8 on both GPUs (DDP), comic sampled 2x (style weights 1,2,1),
# comic distillation to pk_comic x3 (1,3,1), lr 1e-5 (dl6g2's fine-tune lr), 4000 steps (not curve-based; ~0.6 comic epochs).
# Evals: all 3 styles with the one decoder dec8 (regression check) + comic with decCm.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-b9l TRITON_CACHE_DIR=/tmp/triton-cache-b9l
P=$VENV/bin/python
L=logs/b9; mkdir -p $L
evalv() {  # $1 GPU  $2 view tag  $3 style  $4 decoder
  CUDA_VISIBLE_DEVICES=$1 $P scripts/expr_eval.py --tag $2 --decoder $4 || echo "EXPR_FAIL $2 $4"
  CUDA_VISIBLE_DEVICES=$1 EVAL_SET=v2 $P scripts/student_infer.py --tag $2 --decoder $4 --style $3 --no-perf && \
    CUDA_VISIBLE_DEVICES=$1 EVAL_SET=v2 $P scripts/eval_quality.py --tag student_$2_$4 --styles $3 || echo "Q_FAIL $2 $4"
  $P scripts/sharp_ratio.py $3 $2_$4 > results/sharp_$2_$4.txt 2>&1 || echo "SHARP_FAIL $2 $4"
}
echo "B9_START $(date -Is)"
ARGS="--style cartoon3d --manifest manifest_v2.jsonl --expr --init-from dl8 --joint-styles comic_line,chibi --joint-manifest manifest_v2.jsonl
  --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --gan-w 0.2 --gan-start 200 --lr 1e-5 --steps 4000 --val-every 500 --bs 16
  --style-tokens 32 --style-film --distill-w 1.0 --style-weights 1,2,1 --distill-style-w 1,3,1
  --distill-latents cartoon3d=weights/decoder/_zs_dl6g2_manifest_v2.pt,comic_line=weights/decoder/_zs_pk_comic_manifest_v2.pt,chibi=weights/decoder/_zs_pk_chibi_manifest_v2.pt
  --tag dl9"
CUDA_VISIBLE_DEVICES=1,2 timeout 2h $VENV/bin/torchrun --nproc_per_node 2 --master_port 29542 scripts/train_dl.py $ARGS > $L/train_dl9.log 2>&1 \
  || { echo "B9_DDP_FAIL -> single GPU $(date -Is)"; CUDA_VISIBLE_DEVICES=1 $P scripts/train_dl.py $ARGS > $L/train_dl9_1gpu.log 2>&1 || { echo "B9_FAIL dl9 train"; exit 1; }; }
echo "B9_TRAIN_DONE $(date -Is)"
( evalv 1 dl9_comic_line comic_line dec8_comic_line; evalv 1 dl9 cartoon3d dec8 ) > $L/eval_a.log 2>&1 &
( evalv 2 dl9_comic_line comic_line decCm; evalv 2 dl9_chibi chibi dec8_chibi ) > $L/eval_b.log 2>&1 &
wait
chmod -R a+rwX weights/student/dl9* outputs results grids $L 2>/dev/null
echo "B9_ALL_DONE $(date -Is)"
