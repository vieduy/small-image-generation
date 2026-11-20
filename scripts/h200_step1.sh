#!/bin/bash
# Step 1 on H200 (dev-duyhv): new placeholder style + klein quality eval on all styles.
set -e
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
[ -f styles/cartoon3d/style.yaml ] || $VENV/bin/python scripts/gen_style.py --name cartoon3d --seed 31000 \
  --short "3D cartoon avatar" \
  --desc "3D cartoon avatar, stylized rounded face, large expressive eyes, smooth soft skin shading, simple clean hair shapes, soft studio lighting, friendly look, soft vibrant colors"
MODEL=klein KLEIN_REF=0 PROMPTS_FAMILY=klein_text $VENV/bin/python scripts/eval_quality.py --tag klein_text --generate
echo STEP1_DONE
