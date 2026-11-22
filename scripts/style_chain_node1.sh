#!/bin/bash
# dev-duyhv (2026-11-20): joint 3-style model dl7 on GPU 0 once both new styles are prepared (chibi prep runs on dev-duyhv-2).
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
until grep -q "PREP_DONE chibi" logs/prep_chibi.log 2>/dev/null && grep -q "PREP_DONE comic_line" logs/prep_comic_line.log 2>/dev/null; do sleep 60; done
G=0 T=dl7 bash scripts/style_train.sh joint
echo "STYLE_NODE1_DONE $(date -Is)"
exit 0
