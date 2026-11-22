#!/bin/bash
# dl7 evals on 3 GPUs in parallel instead of one after another (user 2026-11-20 "utilize idle GPUs"). The sequential eval
# chain (style_chain_node1.sh -> style_train.sh joint) was stopped; train_dl.py itself keeps running.
# Usage: bash scripts/dl7_eval_parallel.sh <part> <gpu>    part = cartoon3d | comic_line | chibi
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
P=$VENV/bin/python
until grep -aq "\[dl\] view dl7_chibi" logs/style_dl7.log 2>/dev/null; do sleep 30; done
sleep 20
case $1 in
cartoon3d)
  G=$2 T=dl7 S=cartoon3d bash scripts/style_train_scope.sh eval
  { EVAL_SET=real STYLE=cartoon3d TAGS="klein_personal student_dl6g2_decg2 student_dl7_decg2" OUT=_dl7_decg2_real $P scripts/skin_compare.py || echo "SKIN_FAIL"
    $P scripts/detail_bands.py student_dl6g2_decg2 student_dl7_decg2 --eval-set v2 > results/detail_dl7_decg2.txt 2>&1 || echo "DETAIL_FAIL"
  } >> logs/style_dl7.log 2>&1 ;;
comic_line) G=$2 T=dl7_comic_line S=comic_line bash scripts/style_train_scope.sh eval ;;
chibi)      G=$2 T=dl7_chibi S=chibi bash scripts/style_train_scope.sh eval ;;
esac
echo "DL7_EVAL_DONE $1 $(date -Is)"
exit 0
