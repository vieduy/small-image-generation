#!/bin/bash
# Stage 2 (user 2026-11-23 "do it"): tB1 final -> CFG sweep (teacher_eval.py --sweep picks g_img/g_txt) -> full eval at the pick
# (28 steps) -> cheaper settings at the pick: 12 and 8 steps; 28 steps without guidance (1/1). dev-duyhv, 1x H200.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
L=logs/s2_teacher; mkdir -p $L
echo "S2_START $(date -Is)"
CUDA_VISIBLE_DEVICES=0 $P scripts/teacher_eval.py --tag tB1 --sweep > $L/sweep.log 2>&1 || { echo "S2_FAIL sweep"; exit 1; }
# the --sweep run also does the full 28-step eval at its pick ("[sweep] pick img<g>_txt<g> {...}")
PICK=$(grep -oE "\[sweep\] pick img[0-9.]+_txt[0-9.]+" $L/sweep.log | tail -1)
GI=$(echo "$PICK" | grep -oE "img[0-9.]+" | cut -c4-); GT=$(echo "$PICK" | grep -oE "txt[0-9.]+" | cut -c4-)
echo "S2_PICK g_img=$GI g_txt=$GT ($PICK)"
[ -n "$GI" ] && [ -n "$GT" ] || { echo "S2_FAIL no pick parsed"; exit 1; }
for S in 12 8; do  # 28 steps at the pick = the sweep run's full eval
  CUDA_VISIBLE_DEVICES=0 $P scripts/teacher_eval.py --tag tB1 --g-img $GI --g-txt $GT --steps $S > $L/eval_s$S.log 2>&1 || echo "S2_FAIL eval s$S"
done
CUDA_VISIBLE_DEVICES=0 $P scripts/teacher_eval.py --tag tB1 --g-img 1 --g-txt 1 --steps 28 > $L/eval_s28_nocfg.log 2>&1 || echo "S2_FAIL eval nocfg"
chmod -R a+rwX results/teacher grids/teacher outputs logs/s2_teacher 2>/dev/null
echo "S2_ALL_DONE $(date -Is)"
