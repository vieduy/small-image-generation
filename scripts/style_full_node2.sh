#!/bin/bash
# dev-duyhv-2: plan, 2 gen workers; score comic_line as soon as all its chunks are done (CPU), then chibi.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
bash scripts/style_full.sh plan || exit 1
bash scripts/style_full.sh worker 0 & bash scripts/style_full.sh worker 1 &
for S in comic_line chibi; do
  until [ $(grep -l GEN_DONE logs/stylefull_gen_${S}_*.log 2>/dev/null | wc -l) -ge 8 ]; do sleep 60; done
  bash scripts/style_full.sh score $S
done
wait
echo "STYLE_FULL_ALL_DONE $(date -Is)"
