#!/bin/bash
# DreamLite eye probe (user-approved 2026-11-17): does an explicit eye term in the REGRESSION stage give brown irises?
# dl1's irises were already dark at step 1k (grids/eye_dl1_trajectory.jpg), so 3k steps is enough to see it.
# Two runs in parallel on one H200, same seed/data/schedule as dl1 but 3000 steps: dlp_ctrl (dl1 recipe) vs
# dlp_eye (+ --eye-w 4 pixel L1 at 512 in eye masks, + --eye-lpips-w 1 LPIPS on 192 px ROI-aligned eye crops).
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
B="--style cartoon3d --manifest manifest_round4.jsonl --expr --val-every 1000"
E="--eye-w 4 --eye-lpips-w 1"
$P scripts/train_dl.py $B --tag smoke_dleye --steps 3 $E || { echo "DLEYE_SMOKE_FAIL"; exit 1; }
rm -rf weights/student/smoke_dleye
$P scripts/train_dl.py $B --tag dlp_ctrl --steps 3000 > logs/dlp_ctrl.log 2>&1 &
C=$!
$P scripts/train_dl.py $B --tag dlp_eye --steps 3000 $E > logs/dlp_eye.log 2>&1 || echo "DLP_EYE_FAIL"
wait $C || echo "DLP_CTRL_FAIL"
for T in dlp_ctrl dlp_eye; do
  EVAL_SET=v2 $P scripts/student_infer.py --tag $T --style cartoon3d --no-perf || echo "INFER_FAIL $T"
done
$P scripts/iris_colour.py student_c3_dmd3 student_dl1 student_dl1_dmd student_dlp_ctrl student_dlp_eye || echo "IRIS_FAIL"
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "DLEYE_PROBE_DONE $(date -Is)"
