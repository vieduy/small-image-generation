#!/bin/bash
# Natural-skin prompt A/B (user-approved 2026-11-18): same 500 jobs/seeds/captions as data/pairs_v2/test500, style text
# "soft vibrant colors" -> "vibrant clothing and hair colors, natural skin color exactly matching the photo, not tanned and
# not more saturated". Waits for the synthetic run to finish (GPUs), then gen on 2 GPUs, score, merge, skincolor.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
O="--out data/pairs_v2/test500_nat"
until grep -qE "V2FULL_DONE|V2FULL_FAIL" logs/v2full.log; do sleep 30; done
for g in 0 1; do CUDA_VISIBLE_DEVICES=$g $P scripts/gen_pairs_v2.py gen $O --shard $g --nshards 2 --natural-skin 1 > logs/v2nat_gen$g.log 2>&1 & done
wait
for g in 0 1; do CUDA_VISIBLE_DEVICES=$g $P scripts/gen_pairs_v2.py score $O --part $g --nparts 2 > logs/v2nat_score$g.log 2>&1 & done
wait
$P scripts/gen_pairs_v2.py merge $O && $P scripts/gen_pairs_v2.py skincolor $O --workers 32 || { echo NAT_FAIL; exit 1; }
chmod -R a+rwX data/pairs_v2 logs 2>/dev/null
echo "NAT_DONE $(date -Is)"
