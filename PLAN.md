# Plan: small i2i models for AVATAR and STICKER (as of 2026-11-14)

Status: Phase 1 discovery is done (`candidates.csv`, `LICENSES.md`). The user asked to wait for **GPU 0** before any GPU work.

## Phase 1 findings that change the brief

1. **No commercial-OK model under 0.5B with image conditioning has public weights.** The two closest designs fail on access or license:
   * BlazeEdit (195M, 2 steps, 512, text-free) has no code or weights.
   * DreamLite (0.39B) is CC BY-NC, so it is a research reference only. It is *not* gated (HF gated=False), which differs from the brief. Its 4.3 GB Qwen3-VL text encoder is ~10× its UNet.
   * So the realistic route is **teacher, then synthetic pairs, then a small student we train ourselves**, not picking an off-the-shelf phone model.
2. **The best commercial-OK teacher is FLUX.2-klein-4B (Apache-2.0).** It does multi-reference editing (photo + style reference) in 4 steps, and it is already cached locally. Qwen-Image-Edit-2511, FireRed-Edit and OmniGen2 are Apache-2.0 alternatives, but much larger.
3. **Identity adapters are a license trap.** InstantID, PuLID, PhotoMaker and IP-Adapter-FaceID all depend on InsightFace models, which are non-commercial. The plain IP-Adapter (Apache-2.0, CLIP image encoder) is the clean option.
4. **The text encoder can be removed for the phone.** With a fixed prompt per task or style, the text embedding is a constant that we cache, the same idea as BlazeEdit's per-task constant. The student then needs no text encoder.
5. **SD-Turbo / pix2pix-turbo has two license problems.** It is free commercially only below US$1M annual revenue. Its license also forbids using outputs "to create or improve any foundational generative AI model". This needs a legal read before SD-Turbo is used as a distillation base.

## Phase 2 shortlist (benchmark set)

| # | Candidate | Why | Weights | Commercial |
|---|---|---|---|---|
| 1 | FLUX.2-klein-4B (multi-ref edit) | teacher / quality ceiling | cached | yes |
| 2 | Sana-Sprint 0.6B img2img (SDEdit) | small DiT, 1–4 steps, student-backbone probe | cached | yes (Gemma terms) |
| 3 | InstructPix2Pix (SD1.5, image-concat) | open analog of BlazeEdit-style conditioning | ~2 GB fp16 | yes (OpenRAIL-M) |
| 4 | SD1.5 + IP-Adapter Plus (+ LCM-LoRA) | swappable style **reference image** with no training | ~3 GB | yes (OpenRAIL-M) |
| 5 | SDXS-512-0.9 img2img | smallest 1-step UNet (0.32B) at 512 | ~2 GB | yes (OpenRAIL++) |
| 6 | pix2pix-turbo (SD-Turbo, 1-step) | efficiency now, quality only after Phase-3 training | ~2.6 GB | conditional |
| R | DreamLite-mobile | **research reference only** | 5.1 GB (needs approval) | **no** |

Not benchmarked:
* Nitro-E 304M: its text encoder Llama-3.2-1B is gated. Access needs your OK.
* Qwen-Image-Edit-2511: a 57.7 GB download. Needs your OK.
* AnimeGANv3: non-commercial. It could be timed as a "mobile floor" reference if you want.

## Data (synthetic only)

* `data/faces/`: 120 synthetic portraits from FLUX.2-klein-4B T2I, 512². A prompt grid covers age, gender presentation, skin tone, hair, glasses, head pose and lighting. No names of real people. Seeds and prompts go to `data/faces/manifest.jsonl`.
* `data/subjects/`: 30 pets and objects for the sticker track.
* `styles/{flat_vector,chibi,toy3d}/`: 8 references each, self-generated. Each style also has a `style.yaml` with a prompt template, reference list and optional LoRA path, so styles are config, not code.
* Faces are split 100 for eval and 20 held out for Phase-3 identity checks.

## Metrics

* **Identity:**
  * SFace cosine (OpenCV Zoo, Apache-2.0) between input and output, using YuNet detection.
  * Detection failure on stylized output is reported as its own rate, not dropped.
  * Optional InsightFace cross-check, evaluation-only (ask).
* **Style:** DINOv2 and CLIP-L image similarity to the style refs (mean of top-k). Also a content-leak check: output-to-input CLIP similarity.
* **Sticker compliance, measured after the shared post-process.** The post-process is BiRefNet_lite cutout → alpha cleanup → 8px white outline (dilate alpha) → 512² → WebP quality search to ≤100 KB. Pass rate is reported per step, and so is which steps each model *needs*.
* **Efficiency (GPU 0 only, nothing else of ours running):**
  * Timed with CUDA events: 10 warm-up runs, then median and p95 over 100 runs.
  * Sweep: batch {1,4,16} × steps {1,2,4,8} × {bf16, fp16}.
  * Also recorded: peak VRAM and GPU-s/img.
  * Models that aren't distilled are marked "timing-only" at 1–2 steps.
  * Text-encoder time is reported separately and with cached embeddings.
* **Mobile notes:**
  * params, FLOPs (fvcore / torch.utils.flop_counter), and fp16/int8 sizes;
  * ops that are hard on Core ML/NPU (e.g., RoPE with complex ops, large-head attention, GroupNorm with odd groups, dynamic shapes);
  * text-encoder dependence.
* **H100 numbers are server numbers.** They are never presented as phone numbers.

## Phase 3 pilot (top 1–2 commercial-OK, ≤ ~1 H100-day)

1. A FLUX.2-klein-4B style LoRA for one placeholder style. This tests that the teacher can lock a style while keeping identity.
2. A 1-step paired student, using the pix2pix-turbo recipe on a commercially clean base: SD1.5 / BK-SDM-Tiny / SDXS, or Sana-Sprint, depending on Phase 2. It is trained on ≤2k klein-generated (photo → avatar) pairs with a cached prompt embedding (no text encoder at inference).

Each run must stay under 2 h, or it gets approved first. No benchmarks run during training.

## Run-time budget (rough)

* Data generation: ~15 min.
* Quality runs: under 1 h total. That is about 600 images per candidate per track, and ~0.1–1 s/img.
* Efficiency sweeps: ~0.5–2 h per candidate, with klein being the long one. Sweeps run as separate jobs, each under 2 h.

---
# Conclusion of survey + Phase 2a (2026-11-15) and next plan

## What we know now
1. **No commercial-OK small (≤0.5B) image-to-image model works zero-shot.** On a 512² smoke grid (8 inputs × 3 styles):
   * SDXS: pure noise in img2img.
   * SD-Turbo: weak style, identity drift.
   * Sana-Sprint: unchanged at strength 0.5, identity lost at 0.6.
   * InstructPix2Pix: avatars OK-ish, stickers wash out white.
   * SD1.5 + IP-Adapter: weak style.
   * BlazeEdit has no weights. DreamLite is non-commercial.
2. **FLUX.2-klein-4B (Apache-2.0), photo-only with style as text, is the only usable model.** It keeps identity (head covering, braids, cap, skin tone; pets stay pets) and makes white-outlined stickers. Remaining flaws: it hallucinates glasses (prompt fixed, not yet verified), a bald person gets hair, and the style-reference-image mode copies the reference character.
3. **klein speed in our diffusers i2i server** (batch 1, 512², 4 steps, clean GPU 0):
   * CCU 1: 1.87 img/s (p50 0.53 s). Ceiling ~2.1–2.4 img/s from CCU 2 up.
   * Random arrivals: ~1.0 RPS at p95 ≤ 1 s, ~1.7 RPS at p95 ≤ 1.5 s, ~2.0 RPS at p95 ≤ 2 s.
   * For comparison, the same model in SGLang text-to-image with batching reached 7.6 img/s (bf16) and 9.1 img/s (FP8) on 512²/4 steps ([[fp8-results]]). So the serving stack, not the model, limits our i2i number.
4. The small models are 1.5–2.5× faster per image (smoke estimates), but useless without training.
   → **Strategy: klein-4B = teacher (and the server model for now); train our own small text-free student on klein-made pairs (the BlazeEdit recipe).**

## Next plan
| Step | What | GPU time | Needs |
|---|---|---|---|
| 1 | **Quality harness.** SFace identity, DINOv2/CLIP style, sticker compliance (BiRefNet_lite cutout → 8px white outline → 512² WebP ≤100 KB). Run klein on 100 faces × 3 styles × {avatar, sticker} and verify the glasses fix. | ~30 min | – |
| 2 | **Faster klein serving** (server track, RPS/CCU 1–8). SGLang edit endpoint with dynamic batching (or diffusers per-sample batching), FP8, cached text conditioning per style, torch.compile. Target ≥2–3× today's ~2 img/s. | ~1.5 h | clean GPU 0 |
| 3 | **Pair data engine.** klein makes photo→avatar and photo→sticker pairs for ONE placeholder style. Inputs: 2k synthetic photos (new seeds). Pairs are filtered by Step-1 metrics, so only good teacher outputs are kept. | ~30–40 min | – |
| 4 | **Student pilot** (Phase 3, ≤1 H100-day). One-step, text-free, image-conditioned UNet (pix2pix-turbo recipe: paired L2 + LPIPS + GAN, fixed style embedding) at 512² with a tiny AE (TAESD, MIT). Init from SDXS-512 UNet (0.32B, OpenRAIL++) or BK-SDM-Tiny (0.32B, OpenRAIL-M). SD-Turbo is avoided because of its "no improving foundation models" clause + revenue cap. | ≤1 day | **approval (>2 h job)** |
| 5 | **Evaluate the student vs klein.** Identity/style/compliance, H100 latency at batch 1/4/16, FLOPs, fp16/int8 size, and a Core ML / ONNX export test (report failed ops). Write `report.md` with ranked shortlist per track. | ~1 h | – |
| (opt) | Finish speed rows for the 5 failed small models (b8 configs only, ~45 min) as student-size reference. | ~45 min | clean GPU 0 |

Open questions for the user: which style to pilot (flat_vector suggested). Approve the 1-day training job. Is a hosted server path with klein acceptable for v1 while the student is trained (E2EE: the photo leaves the device)?
