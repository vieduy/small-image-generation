# Source before any work. Big files stay on NFS (home quota is full).
export PROJ=/mnt/project/kchat/duyhv/i2i-sticker-avatar
export HF_HOME=$PROJ/hf_cache
export UV_CACHE_DIR=/mnt/project/kchat/duyhv/flux2-klein/uv-cache
export UV_LINK_MODE=copy
export UV_PYTHON_INSTALL_DIR=/mnt/project/kchat/duyhv/flux2-klein/python
export TORCHINDUCTOR_CACHE_DIR=$PROJ/inductor-cache
export TRITON_CACHE_DIR=$PROJ/triton-cache
export XDG_CACHE_HOME=$PROJ/cache
export CUDA_VISIBLE_DEVICES=0      # user: wait for GPU 0 (2026-11-14); check it is idle first
export VENV=$PROJ/venv
# Already-cached weights (read-only reuse, no re-download):
export KLEIN_CACHE=/mnt/project/kchat/duyhv/flux2-klein/hf_cache
export SANA_CACHE=/mnt/project/kchat/duyhv/sana/hf_cache
