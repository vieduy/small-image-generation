#!/bin/bash
# dev-duyhv, production ideas (user 2026-11-20 "do 4" + "if still have gpu, do the tasks not run"):
#  #4 style packs on the production base dl6g2, anime data (avatar+sticker), 6000 steps, same recipe as dl_an:
#     an_all = LoRA r32 everywhere (today's pack)  an_mu = LoRA r32 only in mid_block + up_blocks.0  an_tok = tokens only
#  #5 dropX = hhB0 with the cross-attention of down blocks + up_blocks.1 removed (option has no effect there), fine-tuned
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-prod2 TRITON_CACHE_DIR=/tmp/triton-cache-prod2
A="--style anime --manifest manifest.jsonl --init-from dl6g2 --steps 6000 --val-every 1000"
anim_eval() {  # $1 GPU $2 tag
  for ES in v2 real; do
    CUDA_VISIBLE_DEVICES=$1 EVAL_SET=$ES $P scripts/student_infer.py --tag $2 --style anime --no-perf && \
      CUDA_VISIBLE_DEVICES=$1 EVAL_SET=$ES $P scripts/eval_quality.py --tag student_$2 --styles anime || echo "EVAL_FAIL $2 $ES"
  done
}
(
  CUDA_VISIBLE_DEVICES=0 $P scripts/train_dl.py $A --tag an_all --lora-rank 32 --lr 1e-4 > logs/pack_an_all.log 2>&1 &
  CUDA_VISIBLE_DEVICES=0 $P scripts/train_dl.py $A --tag an_mu --lora-rank 32 --lr 1e-4 --lora-scope mid_up0 > logs/pack_an_mu.log 2>&1 &
  wait
  anim_eval 0 an_all > logs/pack_eval_an_all.log 2>&1 &
  anim_eval 0 an_mu > logs/pack_eval_an_mu.log 2>&1 &
  wait
  echo "PACK_GPU0_DONE $(date -Is)"
) > logs/prod_chain2_gpu0.log 2>&1 &
(
  CUDA_VISIBLE_DEVICES=1 $P scripts/flops_drop.py down,up1 > logs/flops_drop.log 2>&1 || echo "FLOPS_FAIL"
  CUDA_VISIBLE_DEVICES=1 $P scripts/train_dl.py $A --tag an_tok --tokens-only > logs/pack_an_tok.log 2>&1 &
  G=1 T=dropX DEC=decg2 STEPS=3000 LR=2e-5 bash scripts/h200_idea.sh --init-from hhB0 --drop-attn2 down,up1 > logs/prod_dropX.out 2>&1 &
  wait
  anim_eval 1 an_tok > logs/pack_eval_an_tok.log 2>&1
  echo "GPU1_DONE $(date -Is)"
) > logs/prod_chain2_gpu1.log 2>&1 &
wait
TAGS="klein_personal student_dl_an student_an_all student_an_mu student_an_tok"
for ES in v2 real; do EVAL_SET=$ES STYLE=anime TAGS="$TAGS" OUT=_an_packs_$ES $P scripts/skin_compare.py > logs/pack_skin_$ES.log 2>&1 || echo "SKIN_FAIL $ES"; done
du -sh weights/student/an_*/unet_lora weights/student/an_tok/task_emb.pt > results/pack_sizes.txt 2>&1
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "PROD2_ALL_DONE $(date -Is)"
