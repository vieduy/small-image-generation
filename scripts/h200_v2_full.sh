#!/bin/bash
# Data v2 synthetic run (user-approved 2026-11-18, ~3 h): 13k synthetic photos (f*, c*) -> 93k teacher images
# (avatar best-of-2, sticker x1, 2,700 per expression), prompts avatar v4 / stickers v5, fast klein on both GPUs,
# round-4 quality filter, one kept target per (photo, option). Output data/pairs_v2/cartoon3d (old data untouched).
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
O="--out data/pairs_v2/cartoon3d"
$P scripts/gen_pairs_v2.py plan-full $O || { echo V2FULL_FAIL plan; exit 1; }
for g in 0 1; do CUDA_VISIBLE_DEVICES=$g $P scripts/gen_pairs_v2.py caption-ids $O --part $g --nparts 2 > logs/v2full_caption$g.log 2>&1 & done
wait
grep -q CAPTION_DONE logs/v2full_caption0.log && grep -q CAPTION_DONE logs/v2full_caption1.log || { echo V2FULL_FAIL caption; exit 1; }
for g in 0 1; do CUDA_VISIBLE_DEVICES=$g $P scripts/gen_pairs_v2.py gen $O --shard $g --nshards 2 > logs/v2full_gen$g.log 2>&1 & done
wait
grep -q GEN_DONE logs/v2full_gen0.log && grep -q GEN_DONE logs/v2full_gen1.log || { echo V2FULL_FAIL gen; exit 1; }
for k in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$((k % 2)) $P scripts/gen_pairs_v2.py score $O --part $k --nparts 8 > logs/v2full_score$k.log 2>&1 & done
wait
$P scripts/gen_pairs_v2.py merge $O || { echo V2FULL_FAIL merge; exit 1; }
chmod -R a+rwX data/pairs_v2 logs 2>/dev/null
echo "V2FULL_DONE $(date -Is)"
