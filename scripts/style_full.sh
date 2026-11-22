#!/bin/bash
# New-style mass generation (user 2026-11-20: cartoon3d + chibi + comic_line; "increase size gen", same size per style).
# Same plan as the cartoon3d v2 synthetic run: 13,000 synthetic photos, avatar best-of-2, sticker x1, 2,700 per expression
# = ~93k klein images per style, identical (photo, option, seed) jobs for every style. Prompt v6.
# Usage: bash style_full.sh worker <GPU>        -> claims gen chunks (8 per style, mkdir locks), comic_line first, then chibi
#        bash style_full.sh plan | score <style>
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-full TRITON_CACHE_DIR=/tmp/triton-cache-full
NS=8
STYLES="comic_line chibi"
case "$1" in
plan)
  for S in $STYLES; do
    [ -f data/pairs_v2/$S/jobs.jsonl ] || $P scripts/gen_pairs_v2.py plan-style --out data/pairs_v2/$S --style $S --photos 13000 --avatar-n 2 --expr-per 2700
    mkdir -p data/pairs_v2/$S/locks
  done ;;
worker)
  G=$2
  for S in $STYLES; do
    for k in $(seq 0 $((NS - 1))); do
      mkdir data/pairs_v2/$S/locks/shard$k 2>/dev/null || continue   # atomic claim (NFS mkdir)
      echo "$(hostname) gpu$G $(date -Is)" > data/pairs_v2/$S/locks/shard$k/owner
      CUDA_VISIBLE_DEVICES=$G $P scripts/gen_pairs_v2.py gen --out data/pairs_v2/$S --style $S --pv 6 --shard $k --nshards $NS \
        > logs/stylefull_gen_${S}_$k.log 2>&1 || echo "GEN_FAIL $S $k"
    done
  done ;;
score)
  S=$2; IR=$([ "$S" = chibi ] && echo 10 || echo 5)
  n=$(grep -l GEN_DONE logs/stylefull_gen_${S}_*.log 2>/dev/null | wc -l)
  [ "$n" -eq "$NS" ] || { echo "SCORE_WAIT $S only $n/$NS shards done"; exit 1; }
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 SCORE_THREADS=8
  rm -f data/pairs_v2/$S/manifest_part*.jsonl
  for k in $(seq 0 63); do CUDA_VISIBLE_DEVICES=$((k % 2)) $P scripts/gen_pairs_v2.py score --out data/pairs_v2/$S --style $S --id-rank $IR \
      --part $k --nparts 64 > logs/stylefull_score_${S}_$k.log 2>&1 & done
  wait
  $P scripts/gen_pairs_v2.py merge --out data/pairs_v2/$S && \
  OMP_NUM_THREADS=1 $P scripts/gen_pairs_v2.py skincolor --out data/pairs_v2/$S --workers 128 && echo "STYLE_SCORE_DONE $S $(date -Is)"
  chmod -R a+rwX data/pairs_v2/$S 2>/dev/null || true ;;
esac
exit 0
