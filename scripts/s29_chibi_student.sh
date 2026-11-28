#!/bin/bash
# Chibi pack pkH1 (2026-11-27, user "start chibi pack"), dev-duyhv-gpu GPU 0. = comic pkC1 recipe (scripts/s16_comic_student.sh): LoRA r32
# mid + up_blocks.0 on tS4, lr 1e-4, pk_* losses, chibi TEACHER pairs (tH1, img 3 on step 1, fixed noise) after the consistency filter;
# 12000 steps + snapshot 6000 (copied from pkC1; chibi has 42% of comic's rows - unverified). Refs: pk_chibi (old klein chibi pack on dl6g2,
# merged), dl9_chibi (one-model view); all with decg2. Fast eval form: evals in parallel; sticker id / quality for the final models.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-s29 TRITON_CACHE_DIR=/tmp/triton-cache-s29
P=$VENV/bin/python
L=logs/s29; mkdir -p $L
export CUDA_VISIBLE_DEVICES=0
echo "S29_START $(date -Is)"
NAME=teacher_tH1_s8_img3.0_k1
D=data/pairs/chibi
[ -f $D/${NAME}_consistency_drop.json ] || $P scripts/s8_consistency.py --pairs $NAME --style chibi > $L/consistency.log 2>&1 || { echo "S29_FAIL consistency"; exit 1; }
echo "S29_CONSISTENCY_DONE $(tail -1 $L/consistency.log | cut -c1-400) $(date -Is)"
ARGS="--style chibi --manifest manifest_v2.jsonl --expr --init-from tS4 --lora-rank 32 --lora-scope mid_up0 --lr 1e-4 --val-every 1000
  --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --gan-w 0.2 --gan-start 200
  --teacher-latents $D/$NAME.pt --drop-list $D/${NAME}_consistency_drop.json"
$P scripts/train_dl.py $ARGS --steps 30 --save-at 20 --tag _smk_pkH1 > $L/smoke.log 2>&1 || { echo "S29_FAIL smoke"; exit 1; }
rm -rf weights/student/_smk_pkH1 weights/student/_smk_pkH1_s20; echo "S29_SMOKE_OK $(date -Is)"
$P scripts/train_dl.py $ARGS --steps 12000 --save-at 6000 --tag pkH1 > $L/train_pkH1.log 2>&1 || { echo "S29_FAIL train"; exit 1; }
echo "S29_TRAIN_DONE $(date -Is)"
$P scripts/s16_merge_pack.py pkH1_s6000 pkH1 pk_chibi > $L/merge.log 2>&1 || { echo "S29_FAIL merge"; exit 1; }
for R in pk_chibi_m dl9_chibi pkH1_s6000_m pkH1_m; do
  ( $P scripts/eval_e1.py --runs $R > $L/eval_e1_$R.log 2>&1 || echo "S29_FAIL eval_e1 $R" ) &
done
for R in pk_chibi_m dl9_chibi pkH1_m; do
  ( $P scripts/expr_eval.py --tag $R --decoder decg2 > $L/expr_$R.log 2>&1 || echo "S29_FAIL expr $R" ) &
  ( for E in v2 real selfie; do EVAL_SET=$E $P scripts/student_infer.py --tag $R --decoder decg2 --style chibi --no-perf >> $L/infer_$R.log 2>&1 || echo "S29_FAIL infer $R $E"; done
    for E in v2 real selfie; do ( EVAL_SET=$E $P scripts/eval_quality.py --tag student_${R}_decg2 --styles chibi > $L/quality_${R}_$E.log 2>&1 || echo "S29_FAIL quality $R $E" ) & done; wait ) &
done
wait
$P scripts/sharp_ratio.py chibi pk_chibi_m_decg2 dl9_chibi_decg2 pkH1_m_decg2 > results/sharp_pkH1.txt 2>&1 || echo "S29_FAIL sharp"
$P scripts/s29_sheet.py pk_chibi_m dl9_chibi pkH1_m > $L/sheet.log 2>&1 || echo "S29_FAIL sheet"
chmod -R a+rwX weights/student/pkH1* weights/student/pk_chibi_m outputs results grids $L 2>/dev/null
echo "S29_ALL_DONE $(date -Is)"
