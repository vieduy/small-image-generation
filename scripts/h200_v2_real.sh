#!/bin/bash
# Data v2 REAL-photo run (user request 2026-11-18): all CelebA-derived r* (legal-approved) + 20k FFHQ adults (user-approved
# exception), 3 jobs per photo (avatar, sticker, 1 expression). Plan + captions + finalize start right away (they do not
# depend on the style decision); gen waits for the synthetic run's skincolor pass and the natural-skin A/B; style text
# chosen by a fixed rule (below), logged. Output data/pairs_v2/real. ~2.5 h.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
O="--out data/pairs_v2/real"
OMP_NUM_THREADS=1 $P scripts/gen_pairs_v2.py plan-real $O --workers 128 || { echo REAL_FAIL plan; exit 1; }
for g in 0 1; do CUDA_VISIBLE_DEVICES=$g $P scripts/gen_pairs_v2.py caption-ids $O --part $g --nparts 2 > logs/v2real_caption$g.log 2>&1 & done
wait
grep -q CAPTION_DONE logs/v2real_caption0.log && grep -q CAPTION_DONE logs/v2real_caption1.log || { echo REAL_FAIL caption; exit 1; }
$P scripts/gen_pairs_v2.py finalize-real $O || { echo REAL_FAIL finalize; exit 1; }
echo "CAPTIONS_READY $(date -Is); waiting for the natural-skin A/B"
until grep -q SKINCOLOR_DONE logs/v2full_skincolor.log 2>/dev/null && grep -qE "NAT_DONE|NAT_FAIL" logs/v2nat.log 2>/dev/null; do sleep 30; done
NAT=$($P - <<'PY'
import json, statistics as S
def load(p): return [json.loads(l) for l in open(p)]
a = load("data/pairs_v2/test500/manifest.jsonl"); b = load("data/pairs_v2/test500_nat/manifest.jsonl")
ca = S.median([r["d_chroma"] for r in a if r.get("d_chroma") is not None]); cb = S.median([r["d_chroma"] for r in b if r.get("d_chroma") is not None])
base = lambda r: r["id_pass"] and (r["task"] == "avatar" or r.get("pass_sticker")) and r["extra_ok"]
ka = sum(map(base, a)) / len(a); kb = sum(map(base, b)) / len(b)
use = int(cb <= ca - 3 and kb >= ka - 0.03)
import sys; print(f"[decision] median skin saturation change current {ca:+.1f} vs natural {cb:+.1f}; keep (likeness+sticker) "
                  f"current {ka:.1%} vs natural {kb:.1%} -> natural_skin={use}", file=sys.stderr)
print(use)
PY
)
echo "STYLE natural_skin=$NAT"
for g in 0 1; do CUDA_VISIBLE_DEVICES=$g $P scripts/gen_pairs_v2.py gen $O --shard $g --nshards 2 --natural-skin $NAT > logs/v2real_gen$g.log 2>&1 & done
wait
grep -q GEN_DONE logs/v2real_gen0.log && grep -q GEN_DONE logs/v2real_gen1.log || { echo REAL_FAIL gen; exit 1; }
# 64 parts x 8 threads (as scripts/h200_v2_full_score.sh): 8 parts were CPU-bound (OpenCV face models, WebP), 4-7 h ETA
for k in $(seq 0 63); do OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 SCORE_THREADS=8 CUDA_VISIBLE_DEVICES=$((k % 2)) \
  $P scripts/gen_pairs_v2.py score $O --part $k --nparts 64 > logs/v2real_score$k.log 2>&1 & done
wait
[ "$(cat logs/v2real_score*.log | grep -c SCORE_PART_DONE)" -eq 64 ] || { echo REAL_FAIL score; exit 1; }
$P scripts/gen_pairs_v2.py merge $O && $P scripts/gen_pairs_v2.py skincolor $O --workers 128 || { echo REAL_FAIL merge; exit 1; }
chmod -R a+rwX data/pairs_v2 data/ffhq_norm logs 2>/dev/null
echo "REAL_DONE $(date -Is)"
