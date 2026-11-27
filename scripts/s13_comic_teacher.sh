#!/bin/bash
# Comic teacher tC1 (2026-11-26, user picked "Comic teacher" for an idle H200): = tB1 recipe (scripts/s1_teacher.sh: DreamLite-base,
# 20000 steps, bs 32, lr 1e-5; steps copied from tB1, unverified for comic - tB1's val flattened from ~10k) on the 52,750 comic_line klein
# pairs. Then teacher eval at the tS4 setting (img guidance 3 on the first of 8 steps, txt 2.5), GATE expr_all >= 0.70, then the
# 52,750 comic teacher pairs (fixed phone noise) split over GPU 0 (forward) + GPU 1 (reverse helper) of dev-duyhv-gpu. Student design later.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
L=logs/s13; mkdir -p $L
echo "S13_START $(date -Is)"
export CUDA_VISIBLE_DEVICES=0
[ -f weights/student/_dl_tokens_comic_line_22.pt ] || $P scripts/s13_tokens.py comic_line > $L/tokens.log 2>&1 || { echo "S13_FAIL tokens"; exit 1; }
$P scripts/train_teacher_dl.py --tag _smk_tC1 --styles comic_line --steps 20 --bs 32 --max-rows 256 --ckpt-every 1000 > $L/smoke.log 2>&1 || { echo "S13_FAIL smoke"; exit 1; }
rm -rf weights/teacher/_smk_tC1; echo "S13_SMOKE_OK $(date -Is)"
$P scripts/train_teacher_dl.py --tag tC1 --styles comic_line --steps 20000 --bs 32 > $L/teacher_tC1.log 2>&1 || { echo "S13_FAIL train"; exit 1; }
echo "S13_TRAIN_DONE $(date -Is)"
$P scripts/s8_teacher_eval.py --tag tC1 --g-img 3.0 --g-txt 2.5 --steps 8 --k-img 1 > $L/teacher_eval.log 2>&1 || { echo "S13_FAIL teacher_eval"; exit 1; }
R=results/teacher/tC1_final_img3.0_txt2.5_s8_k1.json
$P -c "import json,sys; e=json.load(open('$R'))['expr_acc']['all']; print('teacher expr_all', e); sys.exit(0 if e >= 0.70 else 1)" \
  || { echo "S13_GATE_FAIL comic teacher expr below 0.70 $(date -Is)"; exit 1; }
echo "S13_TEACHER_OK $(date -Is)"
NAME=teacher_tC1_s8_img3.0_k1
mkdir -p data/pairs/comic_line/$NAME && chmod a+rwX data/pairs/comic_line/$NAME
A="--tag tC1 --style comic_line --g-img 3.0 --g-txt 2.5 --k-img 1 --steps 8 --name $NAME"
CUDA_VISIBLE_DEVICES=1 $P scripts/s8_teacher_pairs_rev.py $A > $L/pairs_rev.log 2>&1 &
$P scripts/s8_teacher_pairs.py $A > $L/pairs.log 2>&1 || { echo "S13_FAIL pairs"; exit 1; }
wait
echo "S13_PAIRS_DONE $(date -Is)"
chmod -R a+rwX weights/teacher/tC1 data/pairs/comic_line/${NAME}* $L 2>/dev/null
echo "S13_ALL_DONE $(date -Is)"
