#!/bin/bash
# B follow-ups (user "queue B" 2026-11-19 ~17:15 UTC) on dev-duyhv GPU 1 once dl6g2s (skin fix) is fully done. All three share the GPU:
#  * dl6g2_tf_dm / dl6g2_tf_d: TRAINING-FREE baselines = dl6g2 with the avatar token as shared context in down+mid / down only
#    (no retraining) -> shows whether training is what makes sharing work (vs Faster-Diffusion-style training-free reuse)
#  * dlBd: B retrained sharing the DOWN blocks only (33.5% of UNet FLOPs shared vs 42% for dlB) -> speed/quality trade-off
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
until grep -qE "DL6G2S_ALL_DONE|DL6G2S_FAIL" logs/run_dl6g2s.out; do sleep 30; done
G=1 T=dl6g2_tf_dm DEC=decg2 EVAL_ONLY=1 bash scripts/h200_idea.sh > logs/run_tf_dm.out 2>&1 &
G=1 T=dl6g2_tf_d DEC=decg2 EVAL_ONLY=1 bash scripts/h200_idea.sh > logs/run_tf_d.out 2>&1 &
G=1 T=dlBd DEC=decg2 STEPS=6000 LR=2e-5 bash scripts/h200_idea.sh --shared-trunk --shared-scope down > logs/run_dlBd.out 2>&1 &
wait
echo CHAIN_B2_DONE $(date -Is)
