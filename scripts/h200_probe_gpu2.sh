#!/bin/bash
# Base probe queue, GPU2 (dev-duyhv-2): SD 1.5 (the candidate), then SDXS (the reference, same probe).
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
until grep -qa SMOKE_DONE logs/h200_probe_smoke.log; do sleep 30; done
bash scripts/h200_probe_base.sh sd15 stable-diffusion-v1-5/stable-diffusion-v1-5 stable-diffusion-v1-5/stable-diffusion-v1-5
bash scripts/h200_probe_base.sh sdxs IDKiro/sdxs-512-0.9
echo "PROBE_GPU2_DONE $(date -Is)"
