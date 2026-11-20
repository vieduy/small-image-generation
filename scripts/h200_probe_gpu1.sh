#!/bin/bash
# Base probe queue, GPU1 (dev-duyhv): SD-Turbo = calibration (full-run result known: equal to SDXS). Waits for the dl1 evals.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
until grep -qa DL1_DONE logs/h200_dl1.log; do sleep 30; done
bash scripts/h200_probe_base.sh turbo stabilityai/sd-turbo
echo "PROBE_GPU1_DONE $(date -Is)"
