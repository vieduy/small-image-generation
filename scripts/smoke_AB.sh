#!/bin/bash
# Smoke tests for idea A (joint wide-latent decoder) and idea B (shared trunk) in train_dl.py; then reload checks.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
source env.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-AB TRITON_CACHE_DIR=/tmp/triton-cache-AB
P=$VENV/bin/python
L="--lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 --proper-flip --tgt-cache"
B="--style cartoon3d --manifest manifest_v2.jsonl --expr --init-from dl6g2 --lr 1e-5 $L --gan-w 0.2 --gan-start 10 --steps 40 --val-every 20"
( CUDA_VISIBLE_DEVICES=0 $P scripts/train_dl.py $B --tag smoke_A --train-decoder decg2 --wide-latent 16 --pix-l1-w 1 --compile > logs/smoke_A.log 2>&1 && echo SMOKE_A_TRAIN_OK || { echo SMOKE_A_FAIL; tail -25 logs/smoke_A.log; } ) &
# B passed 2026-11-19 16:5x: ( CUDA_VISIBLE_DEVICES=1 $P scripts/train_dl.py $B --tag smoke_B --shared-trunk > logs/smoke_B.log 2>&1 && echo SMOKE_B_TRAIN_OK || { echo SMOKE_B_FAIL; tail -25 logs/smoke_B.log; } ) &
wait
CUDA_VISIBLE_DEVICES=0 $P - <<'PY' 2>&1 | grep -vE "Warning|warn|float32"
import sys, torch
sys.path.insert(0, "scripts")
import dl_arch as DLA
for tag, dec in (("smoke_A", "smoke_A"), ("smoke_B", None)):
    u = DLA.load_unet(f"weights/student/{tag}/unet").cuda().half().eval()
    te = torch.load(f"weights/student/{tag}/task_emb.pt")
    v = DLA.load_vae(torch.float16).cuda().eval()
    vd = DLA.load_vae(torch.float16, f"weights/decoder/{dec}").cuda().eval() if dec else v
    x = torch.rand(2, 3, 512, 512, device="cuda", dtype=torch.float16) * 2 - 1
    with torch.no_grad():
        zp = DLA.forward(u, DLA.encode(v, x), te["emb"][[0, 10]].cuda().half(), te["mask"][[0, 10]].cuda())
        y = DLA.decode(vd, zp)
    print(tag, "reload OK: latent", tuple(zp.shape), "out", tuple(y.shape), "finite", bool(torch.isfinite(y).all()),
          "shared_trunk", getattr(u, "_shared_trunk_modules", 0))
PY
grep -E "wide latent|joint decoder|shared trunk|pairs=" logs/smoke_A.log logs/smoke_B.log
grep -h '"step": 40' logs/smoke_A.log logs/smoke_B.log | cut -c1-300
echo SMOKE_AB_DONE
