#!/bin/bash
# Core ML export (CPU only) of the given student tags; usage: export_night.sh TAG...
set -x
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=
for T in "$@"; do
  S=$(date +%s); $VENV/bin/python scripts/export_coreml.py --tag $T --int8 2>&1 | grep -vE "Converting|Running|it/s\]"; echo "EXPORT_SECONDS $T $(( $(date +%s) - S ))"
  echo "EXPORT_DONE $T $(date -Is)"
done
chmod -R a+rwX release logs 2>/dev/null
