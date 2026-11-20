#!/bin/bash
# Data v2 synthetic run, scoring step relaunched (2026-11-18 07:2x UTC). h200_v2_full.sh scored with 8 processes x ~30
# threads: CPU-bound (OpenCV YuNet/SFace have no CUDA here; sticker WebP size check), GPUs idle, ETA 4-7 h. Same score code
# and keep rule, 64 parts x 8 threads; then merge and V2FULL_DONE in logs/v2full.log (the queued skincolor, natural-skin
# A/B and real-photo run all wait for that line).
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
O="--out data/pairs_v2/cartoon3d"
N=64
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 SCORE_THREADS=8
rm -f data/pairs_v2/cartoon3d/manifest_part*.jsonl
for k in $(seq 0 $((N - 1))); do CUDA_VISIBLE_DEVICES=$((k % 2)) $P scripts/gen_pairs_v2.py score $O --part $k --nparts $N > logs/v2full_score$k.log 2>&1 & done
wait
n=$(cat logs/v2full_score*.log | grep -c SCORE_PART_DONE)
[ "$n" -eq "$N" ] || { echo "V2FULL_FAIL score $n/$N"; exit 1; }
$P scripts/gen_pairs_v2.py merge $O || { echo V2FULL_FAIL merge; exit 1; }
chmod -R a+rwX data/pairs_v2 logs 2>/dev/null
echo "V2FULL_DONE $(date -Is)"
