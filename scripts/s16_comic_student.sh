#!/bin/bash
# Comic student pkC1 (2026-11-27, user: "start comic student training first"), dev-duyhv 1x H200.
# Agreed new-style recipe (memory style-pack-recipe): LoRA pack r32 on mid + up_blocks.0, on the shared base = tS4 (new cartoon3d model),
# lr 1e-4 and losses = pk_comic (eye, lp512, critic, proper flip). Data = 52,750 comic TEACHER pairs (tC1, img 3 on step 1 of 8, fixed phone
# noise) after the per-row consistency filter. 12000 steps (pk_comic used 6000 on klein data; tS4's curve showed 6k under-trained -
# unverified for packs), snapshot 6000. Evals: packs merged into their base (scripts/s16_merge_pack.py), compared with pk_comic (old klein
# comic pack on dl6g2) and dl9_comic_line (one-model shipped comic view), all with decoder decg2 (decCm / dec8 were trained on the old
# models' latents, so decg2 keeps the comparison fair).
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-s16 TRITON_CACHE_DIR=/tmp/triton-cache-s16
P=$VENV/bin/python
L=logs/s16; mkdir -p $L
export CUDA_VISIBLE_DEVICES=0
echo "S16_START $(date -Is)"
NAME=teacher_tC1_s8_img3.0_k1
D=data/pairs/comic_line
[ -f $D/${NAME}_consistency_drop.json ] || $P scripts/s8_consistency.py --pairs $NAME --style comic_line > $L/consistency.log 2>&1 || { echo "S16_FAIL consistency"; exit 1; }
echo "S16_CONSISTENCY_DONE $(tail -1 $L/consistency.log | cut -c1-500) $(date -Is)"
ARGS="--style comic_line --manifest manifest_v2.jsonl --expr --init-from tS4 --lora-rank 32 --lora-scope mid_up0 --lr 1e-4 --val-every 1000
  --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --gan-w 0.2 --gan-start 200
  --teacher-latents $D/$NAME.pt --drop-list $D/${NAME}_consistency_drop.json"
$P scripts/train_dl.py $ARGS --steps 30 --save-at 20 --tag _smk_pkC1 > $L/smoke.log 2>&1 || { echo "S16_FAIL smoke"; exit 1; }
[ -d weights/student/_smk_pkC1_s20/unet_lora ] || { echo "S16_FAIL smoke snapshot"; exit 1; }
$P scripts/s16_merge_pack.py _smk_pkC1_s20 > $L/smoke_merge.log 2>&1 || { echo "S16_FAIL smoke merge"; exit 1; }
rm -rf weights/student/_smk_pkC1 weights/student/_smk_pkC1_s20 weights/student/_smk_pkC1_s20_m; echo "S16_SMOKE_OK $(date -Is)"
$P scripts/train_dl.py $ARGS --steps 12000 --save-at 6000 --tag pkC1 > $L/train_pkC1.log 2>&1 || { echo "S16_FAIL train"; exit 1; }
echo "S16_TRAIN_DONE $(date -Is)"
$P scripts/s16_merge_pack.py pkC1_s6000 pkC1 pk_comic > $L/merge.log 2>&1 || { echo "S16_FAIL merge"; exit 1; }
for R in pk_comic_m dl9_comic_line pkC1_s6000_m pkC1_m; do
  $P scripts/eval_e1.py --runs $R > $L/eval_e1_$R.log 2>&1 || echo "S16_FAIL eval_e1 $R"
  $P scripts/expr_eval.py --tag $R --decoder decg2 > $L/expr_$R.log 2>&1 || echo "S16_FAIL expr $R"
done
$P scripts/sharp_ratio.py comic_line pk_comic_m_decg2 dl9_comic_line_decg2 pkC1_m_decg2 > results/sharp_pkC1.txt 2>&1 || echo "S16_FAIL sharp"
for T in pk_comic_m dl9_comic_line pkC1_m; do
  for E in v2 real selfie; do
    EVAL_SET=$E $P scripts/student_infer.py --tag $T --decoder decg2 --style comic_line --no-perf >> $L/quality.log 2>&1 && \
    EVAL_SET=$E $P scripts/eval_quality.py --tag student_${T}_decg2 --styles comic_line >> $L/quality.log 2>&1 || echo "S16_FAIL quality $T $E"
  done
done
$P scripts/s16_sheet.py pk_comic_m dl9_comic_line pkC1_m > $L/sheet.log 2>&1 || echo "S16_FAIL sheet"
chmod -R a+rwX weights/student/pkC1* weights/student/pk_comic_m outputs results grids $L 2>/dev/null
echo "S16_ALL_DONE $(date -Is)"
