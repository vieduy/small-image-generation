"""Selective (fp16-only) downloads of candidate weights into $HF_HOME. Each repo < 5 GB."""
from huggingface_hub import snapshot_download

BASE = ["model_index.json", "*/config.json", "scheduler/*", "tokenizer/*", "feature_extractor/*"]
FP16 = ["text_encoder/model.fp16.safetensors", "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/diffusion_pytorch_model.fp16.safetensors"]
JOBS = [
    ("timbrooks/instruct-pix2pix", BASE + FP16),
    ("stable-diffusion-v1-5/stable-diffusion-v1-5", BASE + FP16),
    ("h94/IP-Adapter", ["models/image_encoder/config.json", "models/image_encoder/model.safetensors",
                        "models/ip-adapter-plus_sd15.safetensors", "models/ip-adapter-plus-face_sd15.safetensors"]),
    ("latent-consistency/lcm-lora-sdv1-5", ["*.safetensors", "*.json", "*.md"]),
    ("IDKiro/sdxs-512-0.9", BASE + ["text_encoder/model.safetensors", "unet/diffusion_pytorch_model.safetensors",
                                    "vae/diffusion_pytorch_model.safetensors"]),
    ("stabilityai/sd-turbo", BASE + FP16 + ["LICENSE*"]),
]
for repo, pats in JOBS:
    p = snapshot_download(repo, allow_patterns=pats)
    print("OK", repo, p, flush=True)
print("DONE", flush=True)
