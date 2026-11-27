#!/bin/bash
# tS4o (2026-11-26, user: "parallel experiment start with base checkpoint dreamlite original on H100 in gpu 0"), LOCAL H100 GPU 0.
# = tS4 (scripts/s11_chain.sh) with ONE change: no --init-from (start from carlofkl/DreamLite-mobile@diffusers, like dl3F; option tokens from
# the Qwen3-VL init cache) instead of dl6g2. Tests whether the ghost hands are leftovers of dl6g2's old hand positions.
# Same pairs (teacher_tB1_s8_img3.0_k1), same consistency drop list, same losses / lr 2e-5 / bs 16 / 18000 steps (dl3F's from-original
# budget was 16000 at lr 2e-5), snapshots 2000 / 6000 / 12000 (mid-schedule). Then eval_e1 + expr_eval per checkpoint + report.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-s12 TRITON_CACHE_DIR=/tmp/triton-cache-s12
P=$VENV/bin/python
L=logs/s12; mkdir -p $L
export CUDA_VISIBLE_DEVICES=0
echo "S12_START $(date -Is)"
NAME=teacher_tB1_s8_img3.0_k1
DROP=data/pairs/cartoon3d/${NAME}_consistency_drop.json
until [ -f $DROP ] && [ -f data/pairs/cartoon3d/${NAME}_consistency.json ]; do sleep 60; done
sleep 30
echo "S12_DROP_LIST_READY $(date -Is)"
ARGS="--style cartoon3d --manifest manifest_v2.jsonl --expr --lr 2e-5 --val-every 1000
  --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --gan-w 0.2 --gan-start 200
  --teacher-latents data/pairs/cartoon3d/$NAME.pt --drop-list $DROP"
$P scripts/train_dl.py $ARGS --steps 30 --save-at 20 --tag _smk_tS4o > $L/smoke.log 2>&1 || { echo "S12_FAIL smoke"; exit 1; }
rm -rf weights/student/_smk_tS4o weights/student/_smk_tS4o_s20; echo "S12_SMOKE_OK $(date -Is)"
$P scripts/train_dl.py $ARGS --steps 18000 --save-at 2000,6000,12000 --tag tS4o > $L/train_tS4o.log 2>&1 || { echo "S12_FAIL train"; exit 1; }
echo "S12_TRAIN_DONE $(date -Is)"
for R in tS4o_s2000 tS4o_s6000 tS4o_s12000 tS4o; do
  $P scripts/eval_e1.py --runs $R > $L/eval_e1_$R.log 2>&1 || echo "S12_FAIL eval_e1 $R"
  $P scripts/expr_eval.py --tag $R --decoder decg2 > $L/expr_$R.log 2>&1 || echo "S12_FAIL expr $R"
done
$P scripts/s12_ckpt_report.py > $L/ckpt_report.log 2>&1 || echo "S12_FAIL ckpt_report"
echo "S12_ALL_DONE $(date -Is)"
