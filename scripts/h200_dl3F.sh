#!/bin/bash
# dl3F = dl3 with PROPER flip augmentation (user choice 2026-11-19 ~04:00 UTC: "A/B on GPU 1"; dl3L stopped at ~15% for it).
# Root cause: train_dl.py flipped the cached latents; the TinyVAE is not flip-equivariant (decode(flip z) vs flip(PNG)
# 22.6 dB vs 33.4 dB round trip; scripts/flip_aug_check.py, grids/flip_aug_check.jpg), so half of all regression targets
# were damaged (blocky edges, eye colour fringes). dl3F uses latents of the flipped images (scripts/build_flip_cache.py).
# Everything else = dl3 (16k steps, public base, --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25, same 2-step DMD), so dl3F vs dl3
# isolates the flip fix. Also --tgt-cache (decoded 512 px targets from disk, uint8 61 dB; speed only) and $SPEED (from the
# speed bench). Afterwards the user-approved detail follow-up (dl4: control / +FFL / +DINOv2 critic) runs from dl3F on the
# fixed pipeline: GPU 1 after this chain, GPU 0 after the dl3 chain.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
M=manifest_v2.jsonl
SPEED=${SPEED:-}
L="--lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --tgt-cache $SPEED"
R=$(cat configs/dmd_recipe.txt)
until grep -qa FLIP_CACHE_DONE logs/flip_cache.log && grep -qa SPEED_BENCH_DONE logs/speed_bench.log; do sleep 30; done
evals() {  # $1 gpu, $2 tag, $3 comparison tags
  g=$1; t=$2
  CUDA_VISIBLE_DEVICES=$g $P scripts/expr_eval.py --tag $t || echo "EXPR_FAIL $t"
  for ES in v2 real selfie; do
    CUDA_VISIBLE_DEVICES=$g EVAL_SET=$ES $P scripts/student_infer.py --tag $t --style cartoon3d --no-perf && \
      CUDA_VISIBLE_DEVICES=$g EVAL_SET=$ES $P scripts/eval_quality.py --tag student_$t --styles cartoon3d || echo "EVAL_FAIL $t $ES"
    EVAL_SET=$ES STYLE=cartoon3d TAGS="klein_personal $3 student_$t" OUT=_${t}_$ES $P scripts/skin_compare.py || echo "SKIN_FAIL $t $ES"
  done
}
CUDA_VISIBLE_DEVICES=1 $P scripts/train_dl.py --style cartoon3d --tag smoke_dl3F --manifest $M --expr --steps 3 $L > logs/dl3F_smoke.log 2>&1 \
  || { echo "DL3F_SMOKE_FAIL"; tail -20 logs/dl3F_smoke.log; exit 1; }
rm -rf weights/student/smoke_dl3F
echo "DL3F_SMOKE_OK $(date -Is)"
CUDA_VISIBLE_DEVICES=1 $P scripts/train_dl.py --style cartoon3d --tag dl3F --manifest $M --expr --steps 16000 $L > logs/dl3F.log 2>&1 \
  || { echo "DL3F_FAIL train"; exit 1; }
echo "DL3F_MAIN_DONE $(date -Is)"
evals 1 dl3F "student_dl2 student_dl3" >> logs/dl3F.log 2>&1
echo "DL3F_MAIN_EVAL_DONE $(date -Is)"
CUDA_VISIBLE_DEVICES=1 $P scripts/train_dmd.py --style cartoon3d --manifest $M --expr --max-pairs 5000 --expr-frac 0.6 \
  --init-from dl3F --fake-warmup 200 --lr 1e-4 --ms-sigmas 1.0,0.5 --lp512-w 1 --tag dl3F_2s --steps 2000 --ckpt-every 250 $R \
  > logs/dl3F_2s.log 2>&1 || { echo "DL3F_FAIL dmd"; exit 1; }
echo "DL3F_DMD_DONE $(date -Is)"
evals 1 dl3F_2s "student_dl3_2s student_dl3F" >> logs/dl3F_2s.log 2>&1
CUDA_VISIBLE_DEVICES=1 EVAL_SET=v2 $P scripts/student_infer.py --tag dl3F_2s --style cartoon3d --no-perf --dl-k 1 >> logs/dl3F_2s.log 2>&1 && \
  CUDA_VISIBLE_DEVICES=1 EVAL_SET=v2 $P scripts/eval_quality.py --tag student_dl3F_2s_k1 --styles cartoon3d >> logs/dl3F_2s.log 2>&1 || echo "EVAL_FAIL dl3F_2s_k1"
echo "DL3F_CHAIN_DONE $(date -Is)"
until grep -qa DL3_ALL_DONE logs/h200_dl3.log; do sleep 60; done
T="student_dl2 student_dl3 student_dl3F student_dl3_2s_k1 student_dl3_2s student_dl3F_2s_k1 student_dl3F_2s"
$P scripts/detail_bands.py $T --eval-set v2 || echo "DETAIL_FAIL"
$P scripts/iris_colour.py klein_personal student_dl2 student_dl3 student_dl3F student_dl3_2s student_dl3F_2s || echo "IRIS_FAIL"
$P scripts/zoom_detail_sheet.py --tags klein_personal,student_dl3,student_dl3F,student_dl3_2s,student_dl3F_2s --names fe2_157,fe2_232,fe2_007,rp_232 --out grids/zoom_dl3F.jpg || echo "SHEET_FAIL"
$P scripts/compare_students_sheet.py --tags klein_personal,student_dl3,student_dl3F,student_dl3_2s,student_dl3F_2s --out grids/dl3F_sheet.jpg || echo "SHEET_FAIL"
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "DL3F_ALL_DONE $(date -Is)"
