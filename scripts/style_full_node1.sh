#!/bin/bash
# dev-duyhv: join the gen workers once the Plan A trainings + evals are done (both IDEA_ALL_DONE).
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
until grep -q IDEA_ALL_DONE logs/planA_pS70.out 2>/dev/null && grep -q IDEA_ALL_DONE logs/planA_pS60.out 2>/dev/null; do sleep 60; done
until [ -d data/pairs_v2/chibi/locks ]; do sleep 30; done
bash scripts/style_full.sh worker 0 & bash scripts/style_full.sh worker 1 & wait
