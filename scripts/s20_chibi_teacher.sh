#!/bin/bash
# Chibi teacher tH1 (2026-11-27, user "start chibi teacher on dev-duyhv-gpu GPU 1") = scripts/s13_comic_teacher.sh for chibi:
# tB1 recipe on the 22,451 chibi klein pairs, but 12000 steps (not 20000): chibi has 42% of cartoon3d's rows, 12k steps x bs 32 ~ 17 epochs
# (tB1: 20k ~ 12 epochs; its val flattened from ~10k) - unverified. Gate expr_all >= 0.70 at img3-k1, then chibi teacher pairs on GPU 1 only
# (GPU 0 is kept for tS6; the reverse helper is not used).
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
L=logs/s20; mkdir -p $L
echo "S20_START $(date -Is)"
export CUDA_VISIBLE_DEVICES=1
[ -f weights/student/_dl_tokens_chibi_22.pt ] || $P scripts/s13_tokens.py chibi > $L/tokens.log 2>&1 || { echo "S20_FAIL tokens"; exit 1; }
$P scripts/train_teacher_dl.py --tag _smk_tH1 --styles chibi --steps 20 --bs 32 --max-rows 256 --ckpt-every 1000 > $L/smoke.log 2>&1 || { echo "S20_FAIL smoke"; exit 1; }
rm -rf weights/teacher/_smk_tH1; echo "S20_SMOKE_OK $(date -Is)"
$P scripts/train_teacher_dl.py --tag tH1 --styles chibi --steps 12000 --bs 32 > $L/teacher_tH1.log 2>&1 || { echo "S20_FAIL train"; exit 1; }
echo "S20_TRAIN_DONE $(date -Is)"
$P scripts/s8_teacher_eval.py --tag tH1 --g-img 3.0 --g-txt 2.5 --steps 8 --k-img 1 > $L/teacher_eval.log 2>&1 || { echo "S20_FAIL teacher_eval"; exit 1; }
R=results/teacher/tH1_final_img3.0_txt2.5_s8_k1.json
$P -c "import json,sys; e=json.load(open('$R'))['expr_acc']['all']; print('teacher expr_all', e); sys.exit(0 if e >= 0.70 else 1)" \
  || { echo "S20_GATE_FAIL comic teacher expr below 0.70 $(date -Is)"; exit 1; }
echo "S20_TEACHER_OK $(date -Is)"
NAME=teacher_tH1_s8_img3.0_k1
mkdir -p data/pairs/chibi/$NAME && chmod a+rwX data/pairs/chibi/$NAME
A="--tag tH1 --style chibi --g-img 3.0 --g-txt 2.5 --k-img 1 --steps 8 --name $NAME"
$P scripts/s8_teacher_pairs.py $A > $L/pairs.log 2>&1 || { echo "S20_FAIL pairs"; exit 1; }
echo "S20_PAIRS_DONE $(date -Is)"
chmod -R a+rwX weights/teacher/tH1 data/pairs/chibi/${NAME}* $L 2>/dev/null
echo "S20_ALL_DONE $(date -Is)"
