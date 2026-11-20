#!/bin/bash
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-smk2 TRITON_CACHE_DIR=/tmp/triton-cache-smk2
L="--style cartoon3d --manifest manifest_v2.jsonl --expr --init-from dl6g2 --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --tgt-cache --gan-w 0.2 --gan-start 5"
CUDA_VISIBLE_DEVICES=1 $P scripts/train_dl.py $L --tag _smkR --lora-rank 64 --lr 1e-4 --refine-sigma 0.5 --only-tasks avatar --steps 20 --val-every 10 > logs/smoke_refine.log 2>&1 && echo SMK_R_OK || echo SMK_R_FAIL
CUDA_VISIBLE_DEVICES=1 EVAL_SET=v2 $P scripts/student_infer.py --tag _smkR --decoder decg2 --style cartoon3d --no-perf >> logs/smoke_refine.log 2>&1 && echo SMK_RI_OK || echo SMK_RI_FAIL
echo SMK_R_DONE
