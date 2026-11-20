#!/bin/bash
# Training-speed benchmark on an idle GPU 1 (user request 2026-11-19 ~04:00 UTC). dl3 config (bs 16, lp512 + eye terms), from the public base.
# A = code as used by dl3 (target decoded twice per step), via its log timer; B..E = new train_dl.py --bench 60 (steps 11-60 timed).
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export CUDA_VISIBLE_DEVICES=1
P=$VENV/bin/python
B="--style cartoon3d --manifest manifest_v2.jsonl --expr --tag bench_speed"
L="--lp512-w 1 --eye-w 1 --eye-lpips-w 0.25"
$P scripts/_train_dl_pre_speed.py $B $L --steps 100 --val-every 100000 2>&1 | grep -a '"step": 100' | sed 's/^/A_old_code /'
$P scripts/train_dl.py $B $L --bench 60 2>&1 | grep -a BENCH | sed 's/^/B_one_decode /'
$P scripts/train_dl.py $B $L --bench 60 --tgt-cache 2>&1 | grep -a BENCH | sed 's/^/C_tgt_cache /'
$P scripts/train_dl.py $B $L --bench 80 --tgt-cache --compile 2>&1 | grep -aE "BENCH|Error" | tail -2 | sed 's/^/D_cache_compile /'
$P scripts/train_dl.py $B --bench 60 2>&1 | grep -a BENCH | sed 's/^/E_no_extra_terms /'
rm -rf weights/student/bench_speed
echo SPEED_BENCH_DONE
