#!/bin/bash
# Full-res perceptual loss probe (user request 2026-11-19: "the loss at 256 ... plan and queue to solve it"). Queued after
# h200_probeBC.sh. The global LPIPS (and the cached klein targets) are 256 px, so detail between 256 and 512 px was only
# supervised by latent MSE (averages -> blur). New train_dl.py --lp512-w: LPIPS on a random 256 px crop of the native 512 px
# output vs the TinyVAE decode of the target latent. Same setup as eye probe 2: continue dl2 at lr 1e-5 for 2000 steps,
#   GPU 0 dlp512_ctrl (no new loss)  vs  GPU 1 dlp512_on (--lp512-w 1).
# Read-out: detail_bands (v2), likeness/style (v2), zoom sheet. Positive -> include in the combined retrain.
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
until grep -avE '^\+' logs/h200_probeBC.log 2>/dev/null | grep -qaE "PROBEBC_DONE"; do sleep 30; done
source env.sh
P=$VENV/bin/python
B="--style cartoon3d --manifest manifest_v2.jsonl --expr --init-from dl2 --lr 1e-5 --val-every 500"
CUDA_VISIBLE_DEVICES=1 $P scripts/train_dl.py $B --tag smoke_lp512 --steps 3 --lp512-w 1 > logs/lp512_smoke.log 2>&1 || { echo "LP512_SMOKE_FAIL"; tail -20 logs/lp512_smoke.log; exit 1; }
rm -rf weights/student/smoke_lp512
run() {  # $1 gpu, $2 tag, rest = extra args
  g=$1; t=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g $P scripts/train_dl.py $B --tag $t --steps 2000 "$@" > logs/probe_$t.log 2>&1 || { echo "LP512_FAIL $t"; return 1; }
  CUDA_VISIBLE_DEVICES=$g EVAL_SET=v2 $P scripts/student_infer.py --tag $t --style cartoon3d --no-perf >> logs/probe_$t.log 2>&1 && \
    CUDA_VISIBLE_DEVICES=$g EVAL_SET=v2 $P scripts/eval_quality.py --tag student_$t --styles cartoon3d >> logs/probe_$t.log 2>&1 || echo "EVAL_FAIL $t"
  echo "LP512_DONE $t $(date -Is)"
}
run 0 dlp512_ctrl &
run 1 dlp512_on --lp512-w 1 &
wait
$P scripts/detail_bands.py student_dl2 student_dlp512_ctrl student_dlp512_on student_dl1_dmd --eval-set v2 || echo "DETAIL_FAIL"
$P scripts/zoom_detail_sheet.py --tags klein_personal,student_dl2,student_dlp512_ctrl,student_dlp512_on --names fe2_157,fe2_232,fe2_007 --out grids/zoom_lp512.jpg || echo "SHEET_FAIL"
chmod -R a+rwX outputs results grids weights logs 2>/dev/null
echo "LP512_PROBE_DONE $(date -Is)"
