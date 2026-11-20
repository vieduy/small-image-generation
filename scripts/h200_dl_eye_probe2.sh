#!/bin/bash
# DreamLite eye probe 2 (follow-up of h200_dl_eye_probe.sh). Probe 1 (from the public base, eye-w 4 + eye-lpips-w 1) gave
# brown irises (iris a* 4.4 vs ctrl 1.0, klein 5.0) but ghost/double eyes on ~1 in 3 faces: the heavy eye term pulls an eye
# to the target's landmark before the face layout has formed. Probe 2 starts from dl1 (layout already learned), 2000 steps
# at lr 1e-5, two weights in parallel: dlp2_hi (4 / 1, as probe 1) and dlp2_lo (1 / 0.25). Baseline = dl1.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
B="--style cartoon3d --manifest manifest_round4.jsonl --expr --val-every 1000 --init-from dl1 --lr 1e-5 --steps 2000"
$P scripts/train_dl.py $B --tag dlp2_lo --eye-w 1 --eye-lpips-w 0.25 > logs/dlp2_lo.log 2>&1 &
C=$!
$P scripts/train_dl.py $B --tag dlp2_hi --eye-w 4 --eye-lpips-w 1 > logs/dlp2_hi.log 2>&1 || echo "DLP2_HI_FAIL"
wait $C || echo "DLP2_LO_FAIL"
for T in dlp2_lo dlp2_hi; do
  EVAL_SET=v2 $P scripts/student_infer.py --tag $T --style cartoon3d --no-perf || echo "INFER_FAIL $T"
done
$P scripts/iris_colour.py student_c3_dmd3 student_dl1 student_dlp_eye student_dlp2_lo student_dlp2_hi || echo "IRIS_FAIL"
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "DLEYE_PROBE2_DONE $(date -Is)"
