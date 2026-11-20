# License register

Checked on 2026-11-14 against HF API tags (`gated`, `license:*`), model cards, and repo LICENSE files.
**Verdict key.** OK means commercial use is allowed. OK-R means commercial use is allowed with use-based restrictions (OpenRAIL). COND means conditional. NC means non-commercial or research-only. ? means unclear. N/A means nothing has been released.
Nothing here is legal advice. Every OK verdict should get a legal check before it goes into a product.

## Models: candidates and teachers

| Item | License | Verdict | Quote / evidence | Link | Used here for |
|---|---|---|---|---|---|
| FLUX.2-klein-4B (+ Qwen3-4B TE, FLUX.2 VAE) | Apache-2.0 | OK | "This model is licensed under the Apache 2.0 License" | https://huggingface.co/black-forest-labs/FLUX.2-klein-4B | teacher, baseline, synthetic data |
| FLUX.2-klein-base-4B (transformer only; TE/VAE byte-identical to klein-4B) | Apache-2.0 | OK | model card: Apache 2.0 (non-distilled base of klein-4B); downloaded 2026-11-16 with user approval (7.75 GB) | https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B | DMD real score during training only (not shipped) |
| FLUX.2-klein-9B | FLUX non-commercial | NC | gated=auto, license:other | https://huggingface.co/black-forest-labs/FLUX.2-klein-9B | not used |
| DreamLite base/mobile | weights CC BY-NC 4.0, code Apache-2.0 | **NC, research reference only** | "Commercial use and public redistribution of the model weights are strictly prohibited." Not gated (gated=False). | https://github.com/ByteVisionLab/DreamLite | research reference (pending approval) |
| BlazeEdit | none (no release) | N/A | No code or weights statement in arXiv 2605.28067 | https://arxiv.org/abs/2605.28067 | architecture reference |
| Sana / Sana-Sprint 0.6B | Apache-2.0, plus Gemma Terms for the Gemma-2-2B-IT TE | OK (Gemma terms apply) | "License: Apache License 2.0. Additional Information: Gemma Terms of Use … for Gemma-2-2B-IT" | https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers | candidate (already cached) |
| Nitro-E | MIT (denoiser). TE Llama-3.2-1B is under the Llama 3.2 Community License and is **gated (manual)** | OK / TE gated | "This project is licensed under the MIT License." | https://huggingface.co/amd/Nitro-E | needs gated access, ask |
| SDXS-512-0.9 | CreativeML OpenRAIL++-M | OK-R | license:openrail++ | https://huggingface.co/IDKiro/sdxs-512-0.9 | candidate |
| BK-SDM-Tiny | CreativeML OpenRAIL-M | OK-R | license:creativeml-openrail-m | https://huggingface.co/nota-ai/bk-sdm-tiny | student-backbone option |
| SD-Turbo | Stability AI Community License | **COND** | Free only for orgs with "annual revenue of less than US $1,000,000". Also no using outputs "to create or improve any foundational generative AI model" | https://huggingface.co/stabilityai/sd-turbo | base of pix2pix-turbo |
| img2img-turbo (pix2pix-turbo) code | MIT (code), inherits SD-Turbo for weights | COND | repo LICENSE: MIT | https://github.com/GaParmar/img2img-turbo | candidate / pilot |
| InstructPix2Pix | MIT tag, SD1.5 base under OpenRAIL-M | OK-R | license:mit | https://huggingface.co/timbrooks/instruct-pix2pix | candidate |
| SD1.5 (community mirror) | CreativeML OpenRAIL-M | OK-R | license:creativeml-openrail-m | https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 | base for IP-Adapter |
| IP-Adapter (non-FaceID) | Apache-2.0 | OK | license:apache-2.0 | https://huggingface.co/h94/IP-Adapter | candidate (style refs) |
| IP-Adapter-FaceID | research-only | NC | "released exclusively for research purposes and is not intended for commercial use" | https://huggingface.co/h94/IP-Adapter-FaceID | not used |
| InstantID / PuLID / PhotoMaker-V2 | Apache-2.0 tags, but they require InsightFace models | NC (dependency) | InsightFace models are "available for non-commercial research purposes only" | see candidates.csv | not used |
| InfiniteYou | CC BY-NC 4.0 | NC | license:cc-by-nc-4.0 | https://huggingface.co/ByteDance/InfiniteYou | not used |
| USO / InstantCharacter / Kontext-dev | built on FLUX.1-dev (non-commercial) | NC / ? | Kontext-dev gated=auto, license:other | — | not used |
| Qwen-Image-Edit-2511 | Apache-2.0 | OK | license:apache-2.0 | https://huggingface.co/Qwen/Qwen-Image-Edit-2511 | possible teacher (57.7 GB, ask) |
| FireRed-Image-Edit-1.0 | Apache-2.0 | OK | license:apache-2.0 | https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.0 | possible teacher |
| OmniGen2 | Apache-2.0 | OK | license:apache-2.0 | https://huggingface.co/OmniGen2/OmniGen2 | possible teacher |
| Z-Image-Turbo | Apache-2.0 | OK | license:apache-2.0 | https://huggingface.co/Tongyi-MAI/Z-Image-Turbo | optional data generator |
| AnimeGANv3 | custom non-commercial | **NC** | "freely available … for non-commercial purposes", commercial use needs an authorization letter | https://github.com/TachibanaYoshino/AnimeGANv3 | research reference only |
| VToonify / DualStyleGAN | S-Lab License 1.0 | NC | non-commercial | https://github.com/williamyang1991/VToonify | not used |
| White-box Cartoonization | CC BY-NC-SA 4.0 | NC | — | https://github.com/SystemErrorWang/White-box-Cartoonization | not used |

## Pipeline components

| Item | License | Verdict | Link | Use |
|---|---|---|---|---|
| BiRefNet / BiRefNet_lite | MIT | OK | https://huggingface.co/ZhengPeng7/BiRefNet | sticker cutout |
| RMBG-2.0 | BRIA non-commercial | NC | https://huggingface.co/briaai/RMBG-2.0 | not used |
| TAESD / TAEF2 | MIT | OK | https://huggingface.co/madebyollin/taef2 | mobile-decoder study |

## Evaluation-only models (never shipped)

| Item | License | Verdict | Note |
|---|---|---|---|
| OpenCV Zoo SFace (MobileFaceNet) + YuNet detector | Apache-2.0 (files in dir) | OK | Primary face-ID metric. The training-set provenance isn't stated in the README; the SFace paper has it. |
| InsightFace buffalo_l (ArcFace) | models: non-commercial research only | NC, **evaluation-only, pending user OK** | Cross-check metric only, never in the product |
| DINOv2-base | Apache-2.0 | OK | style similarity |
| OpenAI CLIP ViT-L/14 | MIT (repo) | OK | style similarity |

## Datasets

| Item | License | Verdict | Note |
|---|---|---|---|
| Self-generated synthetic faces/subjects (FLUX.2-klein-4B, Apache-2.0) | outputs owned by us | OK (planned primary) | No real-person names in prompts; prompts and seeds are logged |
| Self-generated style references | outputs owned by us | OK (planned) | `styles/<name>/` |
| SFHQ / SFHQ-T2I | informal ("feel free to use"), no formal license | ? | Not used unless approved |
| FFHQ, CelebA, any real-photo set | real people | **forbidden** by the project rules unless the user grants an exception | exceptions so far: CelebA-derived portrait_2_avatar photos (legal approval, 2026-11-16), FFHQ (user approval, 2026-11-18) |

## Added 2026-11-15 (Phase 2b/3)
| Item | License | Verdict | Use |
|---|---|---|---|
| opencv/face_detection_yunet (YuNet 2023mar) | MIT (OpenCV Zoo) | OK | eval only: face detection + skin-tone (L*) region |
| opencv/face_recognition_sface (SFace 2021dec) | Apache-2.0 | OK | eval only: identity retrieval |
| facebook/dinov2-base | Apache-2.0 | OK | eval only: style similarity |
| openai/clip-vit-large-patch14 | MIT | OK | eval only: style similarity |
| ZhengPeng7/BiRefNet_lite | MIT (remote code reviewed: no network calls in loaded modules) | OK | sticker cutout (post-process, phone-portable) |
| madebyollin/taesd | MIT | OK | student encoder/decoder (phone path) + LPIPS decode in training |
| LPIPS (richzhang/PerceptualSimilarity, pip `lpips` 0.1.4) + torchvision VGG16 weights | BSD-2 / BSD-3 | OK | training loss only |
| IDKiro/sdxs-512-0.9 UNet (student init) | CreativeML OpenRAIL++-M | OK-R | student initialization; derivative must carry the OpenRAIL use restrictions |
| Training pairs data/pairs/cartoon3d | self-generated with FLUX.2-klein-4B (Apache-2.0); inputs = synthetic faces/selfies/subjects **+ (round 3b) 6,597 real CelebA-derived photos from portrait_2_avatar, used under the user's legal approval (upstream CC BY-NC 4.0, see below)** | OK with legal sign-off | student training data; base b1 and every model built on it (c3_dmd, an_b1(_dmd), fv_b1(_dmd)) were trained on these real photos |
| onnx 1.23.2 / onnxruntime 1.32.0 / onnxscript 0.7.2 | Apache-2.0 / MIT / MIT | OK | export check of the student (tooling only) |

## Evaluated and rejected (2026-11-16): datasets in /mnt/project/kchat/thaitt/generative-cv/DreamLite/data/ (colleague's DreamLite experiments)
| Item | License | Verdict | Evidence | Use here |
|---|---|---|---|---|
| murali1729S/portrait_2_avatar (7,265 photo→anime pairs, 25 GB) | CC BY-NC 4.0 | **NC + real people: not usable** | HF tag `license:cc-by-nc-4.0`. The card says "human images are sourced from the CelebA dataset" (real celebrities; CelebA terms are non-commercial research only). Anime targets come from a GAN/diffusion method (Xiao et al., TVCG 2024) with an unstated model license. A visual check of `_peek_*.png` confirmed real celebrity photos. | none (violates the "NEVER use real people's photos" rule and is NC) |
| sticker_dataset = 5 × misshimichka/* (flower/people_flower/cat/clown faces; 332 pairs, 959 MB) | **none stated** (no license tag, no license in the cards) | **? → not usable** | All 5 HF cards have no license and no statement of where the input photos come from. No license means all rights reserved by default. The styles (flower frames, clown/cat face paint) do not match our sticker spec. | none |

### Update 2026-11-16: legal approval (owner decision)
- The project owner states that the data under `/mnt/project/kchat/thaitt/generative-cv/DreamLite/data/` is owned by our product and **approved by legal for product use**. This overrides the "evaluated and rejected" verdict above and the spec's real-photo rule for this source only. The approval document is held by the owner/legal and has not been seen here.
- Used: `portrait_2_avatar` **input photos only** (real CelebA-derived portraits), via `scripts/import_real_p2a.py`. They are deduplicated, face-gated, framing-normalised, and split by identity into `data/real_p2a/eval` (held-out real-photo eval set) and `data/real_p2a/train` (inputs for klein teacher pairs, ids `r*`). The anime targets are not used (they are not one of our styles).
- `data/` is now group-only (mode 2770), because it holds real-person images.
- `sticker_dataset` (332 pairs) is not used yet: small, and its styles do not match our sticker spec.

## FFHQ (real photos, user-approved exception 2026-11-18)
- Source: Kaggle mirror `arnaud58/flickrfaceshq-dataset-ffhq`, 52,000 PNG at 512x512 (FFHQ resized), 20.9 GB, downloaded with the user's Kaggle token. The mirror lists **CC-BY-SA-3.0**. The original NVIDIA FFHQ (NVlabs/ffhq-dataset) is **CC BY-NC-SA 4.0** for the dataset, and each image keeps its Flickr licence (CC BY 2.0, CC BY-NC 2.0, Public Domain Mark 1.0, CC0 1.0, or U.S. Government Works). These are real people who did not consent to this specific use: legal must review before anything trained on it ships.
- Use: teacher (klein) **input photos only**, in `data/ffhq/`. Kept internal on NFS, never sent to external services, never shown in demo material. Any student trained on FFHQ-derived pairs must be marked as such in this file.
