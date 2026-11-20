#!/bin/bash
# GPU 2 follow-up (after h200_expr_stage.sh): experiments that do NOT write the pairs manifest (safe next to GPU 1's round 3).
#  E1: 8-bit weight check for the phone build (s5d fp16 vs simulated Core ML w8) on eval v2.
set -e
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=0
P=$VENV/bin/python
until grep -qa "EXPR_DONE" logs/h200_expr_stage.log 2>/dev/null; do sleep 60; done
echo "GPU2_EXP_START $(date -Is)"
export EVAL_SET=v2
$P scripts/student_infer.py --tag s5d --style cartoon3d --no-perf --fake-w8
$P scripts/eval_quality.py --tag student_s5d_w8 --styles cartoon3d
$P -c "
import numpy as np
from pathlib import Path
from PIL import Image
d=[]
for f in sorted(Path('outputs/student_s5d_w8/cartoon3d/avatar').glob('*.png')):
    b=Path('outputs/student_s5d/cartoon3d/avatar')/f.name
    if b.exists(): d.append(np.abs(np.asarray(Image.open(f),float)-np.asarray(Image.open(b),float)).mean())
print('W8_PIXEL_DIFF mean abs (0-255) over', len(d), 'avatars:', round(float(np.mean(d)),3), 'p95', round(float(np.percentile(d,95)),3))"
chmod -R a+rwX outputs results grids 2>/dev/null || true
echo "GPU2_EXP_DONE $(date -Is)"
