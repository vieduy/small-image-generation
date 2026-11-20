#!/bin/bash
# Data v2 keep-rate test (user-approved 2026-11-18): ~500 pairs, new prompts (avatar v4, stickers v5) vs round-4 keep on the
# same (photo, option) pairs + FFHQ adults. Caption on GPU 0, generation split over both GPUs, then scoring.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
O="--out data/pairs_v2/test500"
$P scripts/gen_pairs_v2.py plan-test $O || { echo V2_FAIL plan; exit 1; }
CUDA_VISIBLE_DEVICES=0 $P scripts/gen_pairs_v2.py caption $O || { echo V2_FAIL caption; exit 1; }
for g in 0 1; do CUDA_VISIBLE_DEVICES=$g $P scripts/gen_pairs_v2.py gen $O --shard $g --nshards 2 > logs/v2_test500_gen$g.log 2>&1 & done
wait
grep -h "compiled vs eager\|GEN_DONE\|Traceback" logs/v2_test500_gen*.log
CUDA_VISIBLE_DEVICES=0 $P scripts/gen_pairs_v2.py score $O || { echo V2_FAIL score; exit 1; }
chmod -R a+rwX data/pairs_v2 data/ffhq_norm logs 2>/dev/null
echo V2_TEST_DONE
