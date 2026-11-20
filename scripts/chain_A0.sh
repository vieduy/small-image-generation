#!/bin/bash
# Start A0 (dlA4: joint UNet+decoder at 4 channels, control for dlA16) on dev-duyhv GPU 0 once dl6g4 is fully done.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
until grep -qE "DL6G4_ALL_DONE|DL6G4_FAIL" logs/run_dl6g4_1g.out; do sleep 30; done
G=0 T=dlA4 DEC=dlA4 bash scripts/h200_idea.sh --train-decoder decg2 --pix-l1-w 1 --compile
