#!/bin/bash
# Core ML export of dmC + decg2 (2026-11-28, user "export dmC"): same steps as the 2026-11-19 dl6g2 + decg2 pack (README):
# export_coreml_dl.py --int8 (fp16 + w8 + PyTorch equivalence check + reference I/O), requant_w8.py (w8e16: task embeddings fp16), then
# release/2026-11-28_dmC_decg2/ (+ README_iOS.md adapted from the dl6g2 pack) and a zip. CPU only.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
L=logs/s32; mkdir -p $L
export CUDA_VISIBLE_DEVICES=
echo "S32_START $(date -Is)"
$P scripts/export_coreml_dl.py --tag dmC --decoder decg2 --int8 > $L/export.log 2>&1 || { echo "S32_FAIL export"; exit 1; }
echo "S32_EXPORT_DONE $(date -Is)"
$P scripts/requant_w8.py --tag dmC --decoder decg2 > $L/requant.log 2>&1 || { echo "S32_FAIL requant"; exit 1; }
echo "S32_REQUANT_DONE $(date -Is)"
R=release/2026-11-28_dmC_decg2; mkdir -p $R
cp -r weights/student/dmC/coreml_decg2/* $R/
sed -e 's/2026-11-19: `dl6g2 + decg2` (current best cartoon3d model)/2026-11-28: `dmC + decg2` (new best cartoon3d candidate)/' \
    -e 's/Student_dl6g2_decg2/Student_dmC_decg2/g' release/2026-11-19_dl6g2_decg2/README_iOS.md > $R/README_iOS.md
python3 - "$R/README_iOS.md" <<'PY'
import sys, re
p = sys.argv[1]; s = open(p).read()
i = s.index("\n| File |")
intro = """
DreamLite-mobile one-step student `dmC` with the phone decoder `decg2` (same graph, size and I/O contract as the 2026-11-19 `dl6g2 + decg2`
pack - a drop-in replacement). dmC = tS4 (distilled from the DreamLite-base teacher tB1) + best-of-N clean gesture pairs + a DMD2 stage
(reports/FLOW_cartoon3d_dmC.md). vs dl6g2 on our evals: expressions 0.710 vs 0.590, extra / ghost hands 0.7% (teacher 1.7%), missing hands
1.7% vs 12%, OK-sign 0.67 vs 0.08, sharpness 0.73 vs 0.64; likeness 0.215 vs 0.248 and sticker identity 0.329 vs 0.346 (dl6g2 still ahead).
w8 / w8e16 sizes and the PyTorch equivalence check: export_info.json. Phone latency / on-device quality not measured yet.
"""
j = s.index("\n", s.index("# Core ML pack"))
s = s[:j] + "\n" + intro + s[i:]
open(p, "w").write(s)
PY
(cd release && rm -f 2026-11-28_dmC_decg2_coreml.zip && zip -qr 2026-11-28_dmC_decg2_coreml.zip 2026-11-28_dmC_decg2)
chmod -R a+rwX weights/student/dmC $R release/2026-11-28_dmC_decg2_coreml.zip $L 2>/dev/null
echo "S32_ALL_DONE $(du -sh $R release/2026-11-28_dmC_decg2_coreml.zip | tr '\n' ' ') $(date -Is)"
