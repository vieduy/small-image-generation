#!/bin/bash
# D (user 2026-11-20 "let do D"): comic_line phone decoder decCm = decg2 fine-tuned on pk_comic's own latents -> klein comic
# targets (L1 + LPIPS 512 + DINOv2 critic 0.1 from 500, lr 1e-4, 5000 steps copied from decC = decg2 + 5000, unverified).
# Then expr_eval + v2 quality with pk_comic + decCm, and sharpness vs teacher next to pk_comic + decg2.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
G=${G:-1}
CUDA_VISIBLE_DEVICES=$G $P scripts/train_decoder.py --student pk_comic --style comic_line --manifest manifest_v2.jsonl --init decg2 \
  --tag decCm --steps 5000 --gan-w 0.1 --gan-start 500 > logs/dec_decCm.log 2>&1 || { echo "DEC_FAIL train"; exit 1; }
echo "DEC_TRAIN_DONE $(date -Is)"
CUDA_VISIBLE_DEVICES=$G $P scripts/expr_eval.py --tag pk_comic --decoder decCm >> logs/dec_decCm.log 2>&1 || echo "EXPR_FAIL"
CUDA_VISIBLE_DEVICES=$G EVAL_SET=v2 $P scripts/student_infer.py --tag pk_comic --decoder decCm --style comic_line --no-perf >> logs/dec_decCm.log 2>&1 && \
  CUDA_VISIBLE_DEVICES=$G EVAL_SET=v2 $P scripts/eval_quality.py --tag student_pk_comic_decCm --styles comic_line >> logs/dec_decCm.log 2>&1 || echo "EVAL_FAIL"
python3 scripts/sharp_ratio.py comic_line pk_comic_decg2 pk_comic_decCm > results/sharp_comic_decCm.txt 2>&1
chmod -R a+rwX weights/decoder/decCm outputs results grids logs 2>/dev/null
echo "DEC_ALL_DONE $(date -Is)"
exit 0
