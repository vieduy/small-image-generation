#!/bin/bash
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-smk TRITON_CACHE_DIR=/tmp/triton-cache-smk
DEC="--student dl6g2 --init decg2 --steps 60 --gan-w 0.1 --gan-start 20 --val-every 50 --workers 4"
CUDA_VISIBLE_DEVICES=0 $P scripts/train_decoder.py $DEC --tag _smkA --alpha --alpha-n 64 > logs/smoke_decA.log 2>&1 && echo SMK_A_OK || echo SMK_A_FAIL
CUDA_VISIBLE_DEVICES=0 $P scripts/train_decoder.py $DEC --tag _smkW --width 80 --extra-blocks 1,1,1,1 > logs/smoke_decW.log 2>&1 && echo SMK_W_OK || echo SMK_W_FAIL
echo SMK_DEC_DONE
