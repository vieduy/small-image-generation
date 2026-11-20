# i2i-sticker-avatar: small image-to-image model study

`~/exp/i2i-sticker-avatar` is a symlink to `/mnt/project/kchat/duyhv/i2i-sticker-avatar`, because the home quota is full.

* Status: Phase 1 is done. Phase 2 is waiting for GPU 0. See `PLAN.md`.
* Files: `candidates.csv`, `LICENSES.md`, `PLAN.md`, `logs/env_snapshot.txt`.
* Environment: `source env.sh`. The venv is not created yet; the plan is uv, Python 3.12, and torch pinned to 2.11.0+cu128 to match the driver (CUDA 12.9).
* Hardware: 4× H100 PCIe 80GB, driver 575.57.08, CUDA 12.9 (driver), nvcc 12.8. Benchmarks run on GPU 0 only.

Commands, seeds and versions will be appended below as runs happen.

## Setup log (2026-11-14)
- venv: `uv venv --python 3.12.16 venv` + `uv pip install -r requirements.lock` (cloned from the flux2-klein venv freeze minus torchao, + opencv-python-headless, aiohttp, pyyaml). torch 2.11.0+cu128, diffusers 0.42.0.dev0 (git 0905b66), transformers 5.22.2.
- Weights: `scripts/download.py` (fp16 subsets, 12 GB total in hf_cache/). klein-4B and Sana-Sprint reused read-only from the earlier caches (see env.sh).
- Synthetic data: `scripts/gen_data.py` (klein-4B T2I, seeds 10000+/20000+/30000+, manifests in each folder). NOT RUN YET (waiting for GPU 0).
- Serving benchmark: `scripts/i2i_server.py` (one model per process, dynamic batching per (task, style)), `scripts/loadgen_i2i.py`,
  `scripts/bench_all.py` (CCU 1..8 x 30 s, then Poisson RPS at 0.3/0.5/0.7/0.85/1.0 x closed-loop peak x 60 s; pmon log of GPU per config).
- GPU: user decided to wait until GPU 0 has ~40 GB free (it dropped to ~8.9 GB at 14:13 when a second vLLM engine started). Watcher log: logs/gpu0_watch.log.

## Phase 2a run log: serving benchmark (2026-11-14, GPU 0)
GPU 0 had ~41 GB free by 14:22. The only other process was the root vLLM engine (pid 2197593, 0% SM during runs, per the pmon logs).
- Data: `scripts/gen_data.py` ran in 65 s. Output: 120 faces (seeds 10000-10119), 30 subjects (20000-20029), 3x8 style refs (30000+). Contact sheets are in `grids/inputs_*.jpg`.
- Smoke tests: `MODEL=<m> [TAG=..] scripts/smoke.py`. Grids are `grids/smoke_<tag>.jpg` (first pass in `grids/smoke_v1/`). Records are in `results/smoke*.jsonl`.
- Benchmark: `scripts/bench_all.py --configs ... --ccu-dur 30 --rate-dur 60` writes `results/serving.jsonl`. `scripts/summarize.py` writes `results/summary.md` and `results.csv`.

### Deviations / changes (logged, not silent)
1. **klein-4B style mode.** Two-image mode ([photo, style ref]) copied the reference character into the output (glasses, chibi girl), so the benchmark uses **photo-only + style text** (`KLEIN_REF=0`, `configs/prompts.yaml: klein_text`). That makes the reference image unused for klein; a style LoRA is the planned way to fix a style (Phase 3).
2. **SDXS-512** is distilled for t=999 only, and image-to-image from partial noise gives pure noise. It still produces noise with the full VAE (`vae_large`) swapped in for the tiny VAE. It is **timing-only (quality N/A)**. This is a negative result.
3. **SD-Turbo:** `steps=2` with strength 0.5 or 0.75 both run the same single low-noise step (diffusers `int(steps*strength)`). The benchmark uses steps=4 / strength=0.75, which is **3 UNet passes**.
4. **Sana-Sprint img2img:** strength 0.5 leaves the photo unchanged, and 0.6 loses identity entirely (a cliff). The benchmark uses 0.6 = **1 effective step**. It is not usable zero-shot. Text encoder (Gemma-2B) embeddings are cached per (task, style) and the encoder moved to CPU (it isn't in the measured GPU path).
5. **InstructPix2Pix:** the long prompts were replaced with short instructions (`configs/prompts.yaml: ip2p`). Stickers still wash out to white.
6. **Server fixes found in dry runs:** (a) warm-up now covers every batch size 1..MAX_BATCH x every (task, style) and runs **on the GPU executor thread**. Before this, the first call per batch size on the worker thread stalled ~3 s. (b) `gc.freeze()` after warm-up. Dry-run results in `results/dryrun*.jsonl` (not used in the report).
7. klein cannot batch requests that have different input photos (pipeline limitation), so it runs batch 1 only.
- NOTE (2026-11-15): eval seeds in the first `klein_text` eval run used Python `hash()` (randomized per process), so those images are not bit-reproducible. Fixed to `zlib.crc32("<stem>|<style>|<task>")` for all later runs.

## Phase 2b/3 log (2026-11-15, H200 `dev-duyhv` via dstack; H200 numbers are NOT comparable to H100 serving numbers)
- Access: `dstack apply -f dstack/dev_env.yaml --fleet h200-120-fleet -y -d`, then `dstack attach dev-duyhv` once (it refreshes a stale jump-host IP), then `ssh dev-duyhv`.
- New style `cartoon3d` (generic chat-app 3D cartoon avatar, no brand names): `scripts/gen_style.py`, seeds 31000-31007.
- Step 1: `scripts/eval_quality.py` + `scripts/metrics.py`. Findings:
  * **Absolute SFace cosine is invalid photo→cartoon.** klein avatars scored 0.08–0.30, below impostor photo pairs (0.28). Switched to **retrieval identity**: the true input's rank among the 100 eval inputs (top-1/top-5, chance 1%/5%).
  * Sticker post-process (BiRefNet_lite → 8 px outline → WebP) passes 99–100% of klein stickers at 19–27 KB.
- **Teacher bias found and fixed:** klein with generic style prompts lightens dark skin (median +35 L* on the 32 darkest eval faces) and erases age.
  * Rule prompts with negations ("glasses only if present") inject glasses, grey hair and beards into everyone.
  * The fix is the **per-person teacher prompt** (`scripts/teacher_prompt.py`): synthetic-generation attributes plus a measured skin-tone word. It brings ΔL* to +3 and identity top-5 from 16% to 34% (`results/prompt_ab.json`, `grids/prompt_ab_skin.jpg`). Residual issue: some shadowed mid-tone faces are rendered too dark (bin calibration is TODO).
- Steps 3+4 pipeline: `scripts/h200_step34.sh` (gen_pairs → train_student s1/s1d → student_infer → eval). Log: `logs/h200_step34.log`.
- **Step 3 v1 result (cartoon3d):** 2800 teacher pairs (1600 face avatars, 800 face stickers, 400 subject stickers) in 34 min on 1x H200. Kept: avatar 637/1600 (identity top-5 of 100), face sticker 262/800, subject sticker 400/400.
- **Step 4 v1 result.** Students are SDXS UNet 0.33B, one step, no text encoder, TAESD enc/dec. Training: 4000 steps, bs 16, 11.6 min each. Eval is on the 100 held-out eval faces, cartoon3d avatar:

  | run | id top-1 | id top-5 | mean rank | style_dino | ΔL* median (all / 25 darkest) |
  |---|---|---|---|---|---|
  | klein_text (generic prompt) | 0.07 | 0.17 | 34.2 | 0.759 | +19.7 / +37.3 |
  | klein_personal (teacher) | 0.11 | 0.26 | 27.0 | 0.709 | +6.3 / +3.1 |
  | student s1 (eps_x0, TAESD) | 0.05 | 0.18 | 34.3 | 0.642 | +8.4 / +4.7 |
  | student s1d (direct, TAESD) | 0.09 | 0.26 | 29.4 | 0.631 | +9.1 / +6.7 |
  | student s1d (direct, SD-VAE) | 0.09 | 0.26 | 27.2 | 0.629 | +8.6 / +5.1 |

  * Training overfits: s1d train MSE fell to 0.10, while val MSE plateaued at 0.23 by step ~1000. More data is the first lever.
  * Server efficiency: 9.2 ms/img at b1 and 4.3 ms/img at b16 on H200 fp16. UNet 418 GFLOPs, TAESD enc 122 and dec 141 GFLOPs. UNet is 626 MB fp16 / 313 MB int8 (`results/student_s1d_perf.json`). These are server numbers, not phone numbers.
  * Grids: `grids/compare_teacher_student_{avatar,sticker}.jpg`. Skin numbers come from `scripts/skin_compare.py` and are in `results/skin_compare.json`.
- **Deviation 8: subject stickers were humanized.** The cartoon3d `description` contains person-only words ("rounded face, large expressive eyes, hair shapes"). klein put human faces on objects and animals: 285/400 v1 subject targets have a YuNet-detected face, with balls turned into heads and dogs given human faces. The students learned this, and the sticker pass metric could not catch it.
  * Fix: `subject_description` in every `style.yaml` (no face/eye/hair/skin words), plus `TP.subject_sticker_prompt` naming the subject and its kind (animal anatomy vs inanimate object).
  * A/B on 40 training subjects (`results/subject_ab.json`, `grids/subject_ab.jpg`): human-face rate fell from 0.55 to 0.075, and CLIP subject match rose from 0.925 to 1.00.
  * New eval metrics: `subj_human_face_rate`, `subj_clip_ok`. The v1 subject pairs (`s*`) are retired, and the v2 subjects are new ids `o*` (seeds 300000+, own rng 4343).
- **Deviation 9:** `styles/cartoon3d_b/style.yaml` has a garbled description: a bad edit re-added "soft skin shading ... soft vibrant colors" and repeated phrases. So prompt A/B variant C was not the clean ablation intended. Its numbers stay in `results/prompt_ab.json`, but no conclusion is drawn from C.
- **Mobile conversion check, s1d (`scripts/export_student.py`):** photo → TAESD enc → UNet (task tokens baked in) → TAESD dec exports as **one ONNX graph per task**. It uses torch 2.11 dynamo exporter, opset 18.
  * ONNX Runtime CPU output matches PyTorch fp32: max abs 1e-5 at [-1,1] scale.
  * Size: 1264 MB fp32 per graph. A shipped build would take the task tokens as an input to share one UNet, at fp16 (~630 MB) or int8 (~315 MB).
  * x86 server CPU, 8 threads: 1.05 s. This is NOT a phone number. Core ML / TFLite / NNAPI conversion and on-device timing have not been run.
- Data v2 + student s2d: `scripts/h200_step3b.sh` (5000 faces, 2500 face stickers, 1200 new subjects; log `logs/h200_step3b.log`), then `scripts/h200_step4b.sh` (s2d, direct, 10k steps).
- **Step 3b result (data v2, 81 min):** 8700 teacher jobs. Kept 1929/5000 face avatars, 792/2500 face stickers and 1003/1200 subject stickers.
- **Deviation 10:** YuNet `human_face` gives false positives on cartoon **animal** faces. All 9 eval subjects it flagged for klein_personal are correct-looking animals.
  * In gen_pairs it wrongly vetoed 185/511 animal pairs (36%), which biased training against frontal animals.
  * Fix: the face veto applies to inanimate objects only, which re-flags 185 pairs (subjects kept: 1003 → 1188). Eval reports `subj_obj_human_face_rate` (objects) and `subj_animal_face_det_rate` (informational) separately.
  * s2d had already cached the pre-fix set (3724 pairs), so it trains on that. The next run uses the 3909-pair set.
- **Step 4b result: s2d** (direct, data v2 3724 pairs, 10k steps, 28.6 min). Eval is on the 100 eval faces, cartoon3d; n=100, so ±0.045 is ~1 SE for top-5.

  | run | avatar top-1 / top-5 / mean rank | avatar style_dino | ΔL* all / dark25 | sticker top-5 | object stickers with a face (n=14) | subj CLIP ok |
  |---|---|---|---|---|---|---|
  | klein_text | 0.07 / 0.17 / 34.2 | 0.759 | +19.7 / +37.3 | 0.20 | 0.714 | 0.967 |
  | klein_personal (teacher, v2 subject prompt) | 0.11 / 0.26 / 27.0 | 0.709 | +6.3 / +3.1 | 0.40 | — (see dev. 10) | 1.00 |
  | student s1d (data v1) | 0.09 / 0.26 / 29.4 | 0.631 | +9.1 / +6.7 | 0.20 | 0.571 | 0.933 |
  | **student s2d (data v2, TAESD)** | **0.16 / 0.30 / 25.4** | 0.629 | +7.6 / +5.1 | 0.26 | **0.000** | 1.00 |
  | student s2d (SD-VAE) | 0.15 / 0.29 / 24.3 | 0.629 | +8.2 / +3.5 | 0.26 | 0.071 | 1.00 |

  * Identity on avatars is now at or slightly above the teacher, within noise. Style (style_dino 0.63 vs 0.71) and sticker identity (0.26 vs 0.40) are still below it.
  * Objects are no longer humanized.
  * val MSE was lowest at step ~2000 (0.215) and rose to 0.234 by 10k. val LPIPS was flat at 0.152 from 4.5k. The model still overfits, so more steps don't help: data and losses do.
  * Sticker style_dino is not comparable across the subject-prompt change: subject stickers no longer look like avatar refs.
  * Teacher errors the students copy: a few lightened faces under caps, and a beard on one long-haired "person". Grids: `grids/compare_teacher_student_{avatar,sticker}_v2.jpg`.
- Step 4c: `scripts/h200_step4c.sh` trains s3d (data v2 fixed filter, 3909 pairs, no GAN) and s3g (the same + projected-VGG GAN, w 0.1 from step 2000). Log `logs/h200_step4c.log`.
- **Step 4c result: the GAN gives no clear gain.** Both runs use 3909 pairs, 10k steps.

  | run | avatar top-1 / top-5 / rank | style_dino / clip | ΔL* all / dark25 | sticker top-5 | objects with a face |
  |---|---|---|---|---|---|
  | s3d (no GAN) | 0.15 / 0.29 / 25.8 | 0.630 / 0.801 | +8.2 / +4.7 | 0.26 | 0.0 |
  | s3g (VGG projected GAN w 0.1 from 2k) | 0.12 / 0.28 / 24.0 | 0.637 / 0.809 | +9.0 / +4.7 | 0.28 | 0.0 |

  * The GAN's eyes and edges are slightly crisper in the grid, but every metric delta is within noise. d_loss ~1.6 (2.0 = chance), so the discriminator is weak at this weight.
  * Not adopted for now. A larger weight or a different discriminator is left as an open question.
  * Visually, all students are now close to the teacher (`grids/compare_teacher_student_avatar_v3.jpg`). The remaining style_dino gap (0.63 vs 0.71) may partly reflect that students keep the photo's pose and framing more than klein does. This is unverified.
- Step 3c+4d: `scripts/h200_step3c4d.sh`. It adds 2500 face stickers and a teacher skin-drift filter (`--max-dl 12` on face pairs), then trains s4d (6k steps, no GAN). Log `logs/h200_step3c4d.log`.
- **Step 3c+4d result: s4d** (skin-drift filter |ΔL*| ≤ 12 plus 2500 more face stickers, 6k steps). The filter removed 975/3510 identity-passing face pairs (teacher median ΔL* +8.2). Avatar pairs fell from 1929 to 1339, and face-sticker pairs rose from 792 to 1196.

  | run | avatar top-1 / top-5 / rank | style_dino | ΔL* all / dark25 | sticker top-5 |
  |---|---|---|---|---|
  | klein_personal (teacher) | 0.11 / 0.26 / 27.0 | 0.709 | +6.3 / +3.1 | 0.40 |
  | s3d (no skin filter, 10k) | 0.15 / 0.29 / 25.8 | 0.630 | +8.2 / +4.7 | 0.26 |
  | s4d (skin filter, 6k) | 0.09 / 0.25 / 26.7 | 0.615 | **+6.3** / +4.3 | 0.24 |

  * The skin filter works: student skin drift now equals the teacher's.
  * Identity and style dropped slightly. This is within noise, and it is confounded by 31% fewer avatar pairs and fewer steps.
  * Sticker identity did not improve with 1.5x face-sticker pairs. The sticker gap is not simply a data-count problem.
  * Next: regenerate more face inputs so that the filtered set is at least as large as the unfiltered s3d set, then re-test.
- **Step 3c+4d result.** Klein lightens training faces by median ΔL* +8.2 (n=3510 identity-passing face pairs), and 975 of them have |ΔL*| > 12. Kept: 1339 avatars (was 1929), 1196 face stickers (was 792), 1188 subjects.
  * **s4d** (6k steps): avatar top-1/top-5/rank 0.09 / 0.25 / 26.7, style_dino 0.615, **ΔL* +6.3 / dark25 +4.3** (teacher +6.3 / +3.1), sticker top-5 0.24.
  * The skin filter fixed tone drift, but the 31% fewer avatar pairs cost identity (vs s3d 0.15 / 0.29).
  * The extra face stickers did **not** raise sticker identity (0.24–0.28 vs teacher 0.40), so the sticker gap isn't about data volume.
- Step 3d+4e: `scripts/h200_step3d4e.sh`. Faces 5000 → 10000 (avatars only) with the filter, then s5d. This recovers avatar volume while keeping the skin fix.
- **Deviation 11: run collision (2026-11-15 16:08–16:15).**
  * A parallel Claude Code session started `gen_pairs.py --faces 10000 --sticker-faces 5000 --max-dl 12` (pid 188008) under the same script and log name `h200_step3d4e`.
  * This session then launched an 8000-face run with that name. That run was stopped after 7 s, before writing data.
  * The stop command (`pkill -f h200_step3d4e.sh`) also killed the other session's driver script, which orphaned the follow-up training.
  * `scripts/h200_step3d4e.sh` now holds the 8000-face version, which never ran. The data that counts comes from the 10000-face run.
  * Training continues as `scripts/h200_train_s5d.sh`, chained on the H200 after pid 188008 exits (log `logs/h200_train_s5d.log`).
  * `logs/h200_step3d4e.log` was truncated by the second launch while the first process kept writing, so its beginning is missing or garbled.
- **s5d result** (10000 faces; filters id top-5 + |ΔL*| ≤ 12; 5096 pairs: 2711 avatar, 1197 face sticker, 1188 subject; 10k steps, 28.7 min). Exact training manifest: `data/pairs/cartoon3d/manifest_s5d.jsonl`.

  | run | avatar pairs | avatar top-1 / top-5 / rank | style_dino | ΔL* all / dark25 | sticker top-5 |
  |---|---|---|---|---|---|
  | klein_personal (teacher) | — | 0.11 / 0.26 / 27.0 | 0.709 | +6.3 / +3.1 | 0.40 |
  | s3d (no skin filter) | 1929 | 0.15 / 0.29 / 25.8 | 0.630 | +8.2 / +4.7 | 0.26 |
  | s4d (skin filter, 6k steps) | 1339 | 0.09 / 0.25 / 26.7 | 0.615 | +6.3 / +4.3 | 0.24 |
  | **s5d (skin filter, 10k steps)** | 2711 | 0.09 / 0.25 / 26.7 | 0.623 | **+5.7** / +4.3 | 0.28 |

  * Doubling avatar pairs did not raise identity. Across all runs, students sit at 0.25–0.30 top-5, which is the teacher's level (0.26).
  * The differences between runs are within noise: 100 eval faces gives an SE of ~0.045 per run. Student identity looks **capped by the teacher**, not by data volume.
  * s5d has the best skin preservation and is the current default for cartoon3d.
  * Next levers: a larger eval set (to tighten the ~±9-point 95% CI), and a better teacher identity signal.
- **Scoring cache:** `metrics.FaceCache` stores the SFace feature and skin L* per image in `data/pairs/<style>/face_cache.pkl`, keyed by path + mtime + size. gen_pairs now scores only new pairs, and loads the GPU scorers only when something is new. Verification: `scripts/h200_cache_check.sh` (log `logs/h200_cache_check.log`).
  * Cache check (16,200 pairs, nothing new). The cold run fills the cache in **401 s wall**: 25,000 images, CPU-bound. A warm re-run takes **20 s wall** (3 s scoring, the rest is model load); before the cache, this step took ~15–20 min every time.
  * Equality with `manifest_s5d.jsonl`: keep / id_rank100 / dL are identical on all rows. One `id_cos` differs by 1e-4 (f01928: -0.0042 vs -0.0041), which is float noise at a rounding boundary and has no effect on keep.

## 2026-11-16: items 1–4 (best-of-K teacher, eval set v2, sticker-identity analysis, second style as LoRA pack)
- Queue: `scripts/h200_queue_1to4.sh` (log `logs/h200_queue_1to4.log`). Stages A–D run in sequence, each < 2 h:
  * A: eval set v2 (`scripts/gen_eval2.py`, 300 faces, seeds 400000+) and klein teacher refs on v2.
  * A: flat_vector pairs (inputs copied from cartoon3d f00000–f02399 and o00000–o00599).
  * B: `scripts/best_of_k.py` (K=3, first 4000 avatars + 2000 face stickers, from `manifest_s5d.jsonl` → `manifest_bestk.jsonl`).
  * C: students s6d (best-of-K), fv_lora (LoRA r=32 on s5d, 12.9M params ≈ 25 MB fp16), fv_full (full FT from s5d, control).
  * D: inference + v2 eval.
- **Eval v2:** 400 avatar faces, 200 face stickers, 30 subjects. Identity is still ranked within galleries of 100 (4 groups), so the scale matches v1, with a 95% CI of ~±4.5 points on top-5 (v1: ±9). Summaries carry `eval_set` and `id_top5_ci95`.
- Code: `train_student.py --manifest/--init-from/--lora-rank`; rows may carry `target` (latent cache keyed on it). `student_infer.py` merges LoRA packs.
- **Item 3, sticker identity (`scripts/sticker_id_analysis.py`, `results/sticker_id_analysis.json`; v1, 50 faces):**
  * Face size is the same in avatars and stickers (~0.53 of the image side), and face-crop + upscale does not change top-5. Small faces are **not** the cause.
  * The small model's stickers score like its own avatars (s5d: 0.28 vs 0.30). The gap comes from klein stickers scoring above klein avatars on these 50 faces (0.40 vs 0.28), and n=50 has a ±13-point CI. Re-check on v2 (200 face stickers).
  * Weak trend: students lose more identity on below-median face sizes (top-5 0.16–0.20 vs 0.36).
- **Teacher-prompt flaw found (sample sheet):** the privileged attributes come from the *generation prompt*, not the image.
  * When the generator did not draw an extra ("a woman … with a beard" rendered without one), the teacher prompt still says "with a beard". klein draws it, and students learn it (face_041).
  * Planned fix: verify each `extra` attribute on the input photo with CLIP zero-shot before using it.
- Product-flow sample sheets: `scripts/make_samples.py` → `grids/samples_<style>.jpg`. Selfie → avatar, and selfie/photo → final sticker WebP on a checkerboard; teacher vs small model.
- **Real-photo demo (2026-11-16, the project owner's own selfie, used with their consent).** The photo is in `private_demo/` (mode 700). It is NOT in any dataset, and nothing is sent to external services. Script: `scripts/demo_photo.py`.
  * The pipeline is product-realistic: face-centred auto-crop, then klein with the *generic* style prompt (real users have no privileged attributes), then the small model on the phone path, then the real sticker post-process.
  * Crop v1 framed on the face centre and cut off the hair. The small model then drew a nearly bald sticker. Fixed: the crop now leaves 0.9 face-heights above the YuNet box.
  * **Findings on a real close-up selfie:**
    - klein gives clean results in both styles. But with the generic prompt it lightened the skin in cartoon3d and darkened it in the flat_vector sticker.
    - The small model s5d keeps face shape and skin tone, but it is visibly weaker than on synthetic eval photos: softer hair, a plainer face, and a less defined hairstyle.
    - Cause (hypothesis): a **domain gap**. Training inputs are synthetic head-and-shoulders shots with smaller faces. A real arm's-length selfie is a tighter close-up with phone-camera texture.
  * Planned fixes:
    1. One shared framing normaliser (`crop_face`) for training inputs AND user photos.
    2. More close-up selfie framings in the synthetic inputs.
    3. A measured skin-tone word in klein's server prompt (available for real users too).
  * **Diagnosis (`scripts/demo_reframe_test.py`): the weak small-model result was mostly a FRAMING domain gap.**
    - The selfie crop had a face box of 0.84 of the side (YuNet, measured on the 512 crop). Training inputs have 0.56 (p10–p90 0.47–0.67).
    - Re-framed to 0.56 (padding the too-tight photo with a blurred copy), s5d gives a clean avatar/sticker with the real hairstyle and shirt. Both the TAESD and SD-VAE paths behave the same.
    - Residual issues: a smudge where the blurred padding is (bottom band), and a slightly shifted shirt colour.
    - Note: YuNet box sizes measured on the full-resolution photo did not match those on the 512 crop. The first reframe attempt used them and was wrong; `framing.py` measures on the 512 crop iteratively.
  * **Fix adopted:** `scripts/framing.py: normalize_face_crop()` is the shared input normaliser for product, eval and demo. `demo_photo.py` uses it.
  * Next data round:
    1. Framing augmentation (tight crops + blurred padding on the inputs, same targets), so the student is robust to padding and close-ups.
    2. A "selfie-like" robustness eval: tight-cropped / padded variants of the v2 eval faces.
  * `framing.py` hardening:
    - **Person gate:** YuNet score ≥ 0.87. Eval people score ≥ 0.889 and animal false positives ≤ 0.852, so 0/30 subjects are face-cropped (pets keep their whole body for stickers). The margin is thin; a production gate needs a proper person classifier.
    - **Detection on a ≤640 px copy:** on the full-res 4k selfie, YuNet's score fell below the gate and its box sizes were wrong.
    - **No-touch rule:** photos whose natural square crop already has face height 0.45–0.62 and top 0.08–0.28 are only square-cropped. Others are reframed to 0.56/0.18, with blurred padding only when needed.
    - Result: the owner's selfie is reframed to 0.564 (padded). On the raw eval faces, 83/400 would be reframed and 73 padded. Eval still scores the raw inputs; framing applies to product and demo photos.
- **Decision (user, 2026-11-16): synthetic data only for now.** Real photos come later, and only as a consented eval set. No scraped internet faces.
- **Round 3, synthetic realism (`scripts/h200_round3_data.sh`, chained after the queue; log `logs/h200_round3_data.log`):**
  * `gen_pairs.py --selfies 3000 --selfie-stickers 1500`: ids `c*`, seeds 500000+, rng 4545. These are realistic arm's-length front-camera selfies: tight or cut-off framing, wide-angle distortion, phone noise/flash/HDR/JPEG. They are generated at 512x640 and passed through `framing.normalize_face_crop` (the same normaliser as user photos) before klein makes the targets.
  * `--verify-extra` adds `metrics.AttrCheck`, a CLIP zero-shot check (positive vs specific negative caption) that each prompt extra (beard, glasses, cap, ...) is visible in the input.
    - New pairs never describe unverified extras.
    - Existing pairs whose teacher prompt named an unverified extra get `extra_ok=False` and are dropped.
    - Eval teacher refs use `VERIFY_EXTRA=1`.
  * Galleries for `id_rank100` are now per id prefix (f*, c*), so the existing f* ranks are unchanged.
  * Eval set "selfie" (`scripts/gen_eval_selfie.py`): 200 close-up selfies (rng 7777, seeds 600000+, framing-normalised), with galleries of 100. It is a proxy for the real-photo gap until a consented real eval set exists.
- **Core ML export, s5d (`scripts/export_coreml.py` → `release/s5d_coreml/`, iOS 16+ ML Program):**
  * The fp16 build is 631 MB; the 8-bit weight build is 316 MB (quality unverified). Inputs are `image` (RGB 512) and `task` (Int32: 0 avatar, 1 sticker); the output is `stylized` (RGB 512).
  * torch 2.11 `jit.trace` failed in coremltools 9 (1-D size tensors in attention reshapes → `aten::Int` cast error). `torch.export` (static shapes) worked.
  * The task tokens are selected with a one-hot sum; the timestep is a python int.
  * The export wrapper matches the evaluated outputs (mean abs diff 0.04/255, fp32 vs fp16).
  * Core ML cannot run on Linux, so device latency and accuracy are unmeasured; `reference/` I/O PNGs are included for on-device comparison.
  * **Device measurement (reported by the user on 2026-11-16): about 0.3 s/image on an iPhone 14 Pro**, the first real phone number; my estimate before this was 0.6–1.5 s. Still to record: which build (fp16 or w8), compute units, warm vs first run, and whether framing and sticker matting are included. On-device output quality vs `reference/` has not been checked yet.
  * App-side requirements are in `release/s5d_coreml/README_iOS.md`: Vision face framing (the same rule as `framing.py`), and the Vision foreground mask + 8 px outline for stickers.
- **Priority #1, sticker variety:** a 20-expression vocabulary is in `configs/sticker_expressions.yaml`. The klein pilot (`scripts/expr_pilot.py`: 8 faces × 20 expressions; identity top-5, CLIP expression accuracy, sticker pass) is running.
- **Priority #1 pilot result (`results/expr_pilot.json`, `grids/expr_pilot_cartoon3d.jpg`).** klein drew all 20 sticker-set expressions on 8 eval faces.
  * Means: identity top-5 0.32 (plain stickers: 0.34 on v2), CLIP expression accuracy 0.73 (8 faces per expression; CLIP is weak on hand gestures), sticker pass 1.00.
  * Problems found:
    - "OK" was drawn as a wave or peace sign.
    - "Thank you", "Good night" and "Party" zoomed out to small child-like full bodies (identity 0.12–0.25).
  * Fixed: clearer phrases, plus `teacher_prompt.expr_sticker_prompt` asks for "head and upper body, adult proportions".
- **Sticker-set training path:**
  * `gen_pairs --expr-faces N --expr-k K` (K of the 20 expressions per face, deterministic per id; identity filter top-10 for expressions).
  * `train_student --expr` adds 20 learned expression tokens (tokens 2–21, ~3 MB; tokens 0/1 stay avatar/sticker). It is opt-in, so default runs ignore expression rows.
  * `scripts/expr_eval.py` evaluates on 24 faces × 20 expressions.
  * `scripts/h200_expr_stage.sh` now runs **first** after the queue: 2500 faces × 4 = 10k klein stickers, then student s5e (s5d + expression tokens, 6k steps), then eval.
  * Round 3 data follows it (its gen_pairs flags are cumulative so the expression rows stay in the manifest).
- **Second H200 (2026-11-16 04:35, user-approved):** `dev-duyhv-2` on fleet `h200-gpu-fleet` (H200 NVL 141 GB, `dstack/dev_env_2.yaml`, same image and NFS mount).
  * GPU 2 runs priority #1 (`h200_expr_stage.sh`, log `logs/h200_expr_stage.log`).
  * GPU 1 finishes the queue, then Round 3. Round 3 waits for `EXPR_DONE` because both runs write `data/pairs/cartoon3d/manifest.jsonl`.
  * `face_cache.pkl` may be saved by two processes at once: each saves atomically, so the last writer wins and some cache entries may be lost (they are recomputed later). No correctness impact.
- **Stage B result: best-of-3 teacher** (first 4000 avatars and 2000 face stickers of the s5d manifest; 12k extra klein candidates; 110 min).
  * Kept pairs: avatars 1063 → 1334 (+25%), face stickers 493 → 728 (+48%). Best-of-3 mostly rescues pairs that failed the filters.
  * Share of kept pairs with identity rank 1: avatars 0.457 → 0.456 (unchanged), stickers 0.367 → 0.433.
  * Chosen candidate counts (c0/c1/c2): avatars 849/235/250, stickers 266/248/214.
  * Whether students gain: s6d (stage C) vs s5d on eval v2.
- **Real-photo import result (`scripts/import_real_p2a.py`):**
  * 7194/7260 unique photos kept (face gate), grouped into 3595 SFace identity clusters.
  * **Eval set: 300 identities × 1 photo** (all from 1024 px sources), in `data/real_p2a/eval`. **Train: 6597 photos** of the other identities, in `data/real_p2a/train`.
  * 5817 photos come from 1024 px sources and 1377 from 256 px. 87% were tight crops, reframed with blurred padding (like close-up selfies).
  * Skin L* p5/50/95 = 40/65/82. ~46% of captions say "girl".
- GPU 2 chain `scripts/h200_gpu2_real.sh` (log `logs/h200_gpu2_real.log`):
  1. Real-photo baselines (klein with a skin-tone-only prompt, s5d, s5e).
  2. Experiment s7e: expression student from the SDXS init, 12k steps, vs s5e.
  3. Round 3b real pairs, after GPU 1's round 3.
- **Fast scorer (`scripts/score_fast.py`).** It replaces per-image scoring (batch 1, single CPU process, ~2 rows/s with the GPU idle): GPU work is batched (DINOv2/CLIP 64, BiRefNet 16) and CPU work runs in a process pool (FaceCache misses, sticker post-process; WebP size search with method 4).
  * The first version used a `fork` pool after CUDA init and **deadlocked** (32 idle workers for 24 min, test only, killed). Fixed with a `spawn` context.
  * Equivalence test (`scripts/test_score_fast.py`, 120 rows): identical style scores, sticker pass, subject checks, face features and skin L*; WebP KB differs ≤ 1.7. Speed 66 s → 15.7 s (4.2×) at this size, which includes worker start-up.
  * Enabled by the flag file `configs/fast_score.on` (env `FAST_SCORE=0/1` overrides), so the already-running chain scripts did not need editing. The expression stage started before the change and still uses the slow path.
- **Queue items 1–4 results (eval v2: 400 avatar faces, 200 face stickers + 30 subjects; galleries of 100; 95% CI ±4–5 pts on avatar top-5).**

  | model | style | avatar top-1 / top-5 / mean rank | style_dino | ΔL* all / dark quartile | sticker top-5 |
  |---|---|---|---|---|---|
  | klein_personal | cartoon3d | 0.077 / 0.247 / 27.1 | 0.714 | +6.3 / +3.1 | 0.340 |
  | s3d | cartoon3d | 0.107 / 0.270 / 24.6 | 0.629 | +8.2 / +4.7 | 0.205 |
  | s5d | cartoon3d | 0.090 / 0.255 / 24.6 | 0.622 | +5.9 / +4.3 | 0.220 |
  | s6d (best-of-3 targets) | cartoon3d | 0.098 / 0.250 / 25.2 | 0.630 | +6.7 / +4.3 | 0.265 |
  | klein_personal | flat_vector | 0.265 / 0.500 / 9.4 | 0.611 | +9.0 / 0.0 | 0.515 |
  | **fv_lora (s5d + 25 MB LoRA)** | flat_vector | **0.370 / 0.623 / 5.9** | 0.582 | **+4.3** / −0.8 | 0.530 |
  | fv_full (full FT from s5d) | flat_vector | 0.335 / 0.588 / 6.8 | 0.579 | +5.1 / +0.8 | 0.515 |

  * **The style pack (LoRA, 25 MB) ≥ full fine-tuning, and beats the teacher on identity in flat_vector.** New styles can ship as add-ons.
  * **Style choice dominates identity:** flat_vector keeps identity about 2× better than cartoon3d (0.50–0.62 vs 0.25–0.27 top-5). The cartoon3d look (big eyes, rounded face) erases identity cues.
  * Best-of-3 (s6d) is unchanged on avatars and gives +4.5 pts on stickers (within CI). Avatar identity in cartoon3d stays at the teacher's level.
  * Remaining flaws (sample sheets `grids/samples_{cartoon3d,flat_vector}.jpg`): the black cat comes out brown in the student, colours are slightly desaturated, and the flat_vector outline lines are lighter than klein's.
- **Priority #1, s5e (`--init-from s5d --expr`, 6k steps, 20.4 min; `results/expr_eval_s5e.json`, `grids/expr_eval_s5e.jpg`).**
  * Expression pairs: 10000 generated, 2612 kept (26%), 68–191 per expression.
  * Means on 8 eval faces × 20 expressions (klein pilot in brackets): identity top-5 0.19 [0.32], CLIP expression accuracy 0.39 [0.73], sticker pass 1.00.
  * Face-only expressions are clearly readable on visual check: happy, LOL, crying, angry, wow, cool, shy, sad, sleepy.
  * Hand gestures (thumbs up, hi, OK, facepalm) come out as blurry ghost hands. This is the expected one-step regression average over many hand poses.
  * Shrug, thank-you and party shrink the body into child-like proportions. Love, wink and good-night lose their cue (no hearts, a one-eye wink drawn as a smile, no night prop).
  * Verdict: not shippable as a 20-expression set. Fixes to test: more pairs per expression, the GAN loss for sharpness, and splitting face-only expressions (on the phone) from gestures (on the server).
- **New style "anime" (`scripts/h200_style_anime.sh`; LoRA r32 on s5d, `weights/student/an_lora`; `grids/samples_anime.jpg`).**
  * Teacher problem: klein lightens skin in this style (manifest avatar dL median +15.7 L*). The |dL|≤12 filter dropped 1724 of 2400 avatar pairs, so only 1546 training pairs were used (615 avatar, 931 sticker). klein also invents light eyes sometimes.
  * Results on the 400-face v2 test (klein → an_lora):

    | Metric | klein | an_lora |
    |---|---|---|
    | Avatar identity top-5 | 0.723 | 0.905 |
    | Sticker identity top-5 | 0.77 | 0.77 |
    | Avatar style_dino | 0.618 | 0.613 |
    | Skin dL median | +14.5 | +8.2 |
    | Skin dL, dark quartile | +6.7 | +6.3 |
    | Sticker pass | 1.0 | 1.0 |
    | Sticker WebP size | 32.9 KB | 22.6 KB |

  * Visual check: the output is semi-realistic anime/manhwa line art, not classic anime: no big eyes, realistic nose and face proportions. The high identity scores come partly from this weaker stylization. The student is softer and closer to the photo than klein.
- **Decoder detail test (anime, 400 v2 avatars; `grids/anime_decoder_zoom.jpg`).** Sharpness is the median Laplacian variance of the 512 output:
  * klein 1929; student + TAESD (the current phone path) 862; student + SD-VAE (vae_large encode and decode) 2168.
  * The full decoder gives crisp line art and edges on visual check.
  * It does not bring back detail the UNet dropped: eye catchlights, freckles, specular highlights. That is the one-step regression to the mean.
  * Phone cost (published SnapFusion figure, not measured by us): the SD VAE decoder takes about 369 ms on an iPhone 14 Pro, plus about 100 MB fp16. TAESD encode + SD-VAE decode (the cheaper mix) is untested.
  * Follow-up: `scripts/h200_anime_gan.sh` trains `an_lora_g` with a stronger GAN (w 0.5 from step 1000, d_lr 2e-4). s3g used w 0.1 and its discriminator was too weak. The job also evaluates the `_sdvae` variants.
- **Deviation 12: multi-step student (user request, 2026-11-16).** `train_student.py --msteps K` and `student_infer.py --ts` add it. The architecture is unchanged; only the sampling procedure changes.
  * Step 0 is the existing photo → target map at T=999.
  * Steps 1..K-1 re-noise the current estimate to t ∈ {749, 499, 249} (forward process) and predict the target again with the same UNet and tokens.
  * Training picks a random step k each iteration. For k>0, the input is the re-noised teacher latent (prob 0.5) or the student's own detached step-0 estimate. Validation runs the full K-step chain.
  * K=1 is bit-identical to the old path (`sample_chain` with ts=[T]).
  * The run is `scripts/h200_anime_4step.sh`: `an_lora_m4` uses the same data, init (s5d) and LoRA r32 as an_lora, 6000 steps, and is evaluated at 1, 2 and 4 steps.
  * Phone cost is about K UNet passes. That is not measured; Core ML export of the loop is still to do.
- **Detail test 1: stronger GAN (`an_lora_g`: w 0.5 from step 1000, d_lr 2e-4; d_loss fell to 1.46, so the discriminator was active; `grids/anime_gan_zoom.jpg`).**
  * Identity top-5 (avatar / sticker): an_lora_sdvae 0.912 / 0.81, an_lora 0.905 / 0.77, an_lora_g 0.743 / 0.65, an_lora_g_sdvae 0.772 / 0.645. The GAN costs about 0.15 identity.
  * style_dino is unchanged (0.61).
  * Sharpness (median Laplacian variance): klein 1929, an_lora_sdvae 2168, an_lora_g with TAESD 3607, an_lora_g_sdvae 6338.
  * Visual check: the GAN brings back eye catchlights, freckles and specular shine, even with the tiny decoder. It also over-sharpens: crosshatch texture in grey hair, harsher and older-looking faces, and drifting eye colour.
  * Verdict: w 0.5 is too strong. A milder weight (0.15–0.25) is worth testing; with TAESD it adds no phone cost.
- **Base-capacity test, SD-Turbo (user-approved 2026-11-16; `scripts/h200_turbo_test.sh`, GPU2, queued after the 4-step run).**
  * `train_student.py --base` lets training start from another UNet. SD-Turbo is the same SD2.x latent space with 1024-d cross-attention, so the latent cache and task tokens carry over.
  * Runs:
    - t5d: SD-Turbo on cartoon3d manifest_s5d, 6000 steps; s5d used 10000 steps on SDXS. Steps were cut to keep the test under about 2 h; SD-Turbo is about 2.6× slower per step.
    - t_an_lora: LoRA r32 on t5d, anime, 4000 steps; an_lora used 6000 steps.
  * SD-Turbo is under the Stability AI Community License: free only below US$1M/yr revenue, and distilling a product model from it needs a legal check. This is a ceiling test only, not a product candidate until that check is done. Its weights were already in hf_cache, so there was no new download.
- **Detail test 2: mild GAN (`an_lora_g2`, w 0.2; `grids/anime_gan2_zoom.jpg`).**
  * Identity top-5 avatar / sticker: 0.828 / 0.665, vs an_lora 0.905 / 0.77.
  * Sharpness: 2768.
  * It still shows hair crosshatch and drifting eye colour.
  * It fails the user's rule (drop GAN if it costs more than about 0.05 likeness or shows artifacts), so **the GAN is dropped**. Detail is pursued through multi-step and DMD instead.
- **Detail test 3: 4-step student (`an_lora_m4`, deviation 12; `grids/anime_m4_zoom.jpg`). Negative result.**
  * Avatar identity top-5 (cos mean):
    - Run at 1 step (k1): 0.90 (0.366), the same as an_lora.
    - Run at 2 steps (999, 499): 0.347 (0.175).
    - Run at 4 steps: 0.133 (0.127).
  * Sticker identity top-5: 0.695 at 1 step, 0.255 at 2 steps, 0.09 at 4 steps.
  * style_dino rises to 0.69 at 4 steps, but the faces become generic: soft, washed-out, beards lost. Sharpness falls (Laplacian 862 → 457 → 180).
  * Cause: the refinement steps see only the re-noised estimate, never the photo. At t ≥ 499 (alpha_bar ≤ 0.28) most identity information is gone, and with MSE+LPIPS regression each step predicts a posterior mean, which is blurrier and more average.
  * Conclusion: under a regression objective, more steps do not recover detail. Together with the GAN results, this points to the training objective, so the next step is DMD. Revisit multi-step only with photo conditioning at every step and a distribution-matching loss.
- **Deviation 13: DMD training (`scripts/train_dmd.py`, user-approved 2026-11-16; real score FLUX.2-klein-base-4B, Apache-2.0, 7.75 GB transformer downloaded with approval; its text encoder and VAE are byte-identical to klein-4B's).**
  * The student keeps its phone architecture (1-step SDXS UNet + LoRA style pack + tokens). Only the training signal changes.
  * DMD2-style gradient: (fake score − real score) on the student's TAESD-decoded output, re-encoded into the klein latent space.
    - The real score is klein-base, conditioned on the photo reference and the per-pair teacher prompt, with CFG 3.
    - The fake score is the same transformer + a LoRA r64 (to_q/k/v/out, add_*, single-block qkv+mlp and to_out), updated 2× per generator step after 200 fake-only warm-up steps.
  * The MSE+LPIPS regression to the filtered klein targets stays at weight 1 as the identity/skin anchor.
  * Smoke tests: about 2.3 s/iteration at bs 4 on the H200, losses stable.
  * Run: `scripts/h200_dmd_anime.sh`, an_lora → an_dmd, 2000 generator steps, evaluated on v2 with skin_compare.
- **s7e (expressions from scratch, 12k steps):** identity top-5 0.196, CLIP expression accuracy 0.531 (s5e 0.39, klein 0.73), sticker pass 1.0.
- **Selfie eval set (200 synthetic close-up selfies), avatar identity top-5:**
  * cartoon3d: klein 0.24, s5d 0.325, s6d 0.29.
  * flat_vector: klein 0.555, fv_lora 0.675.
  * Skin dL, dark quartile: klein +5.7, s5d +3.9.
- **Real-photo eval (300 legal-approved CelebA-derived photos), cartoon3d avatar identity top-5:** klein 0.15, s5d 0.35, s5e 0.34.
  * Skin dL, dark quartile: klein +6.7, s5d +12.2. The student lightens darker real skin; the round-3b real-photo pairs target this.
- **DMD run 1 (`an_dmd`: dmd-w 1, reg-w 1, CFG 3, 2000 steps, 80.9 min, peak 24.6 GB; `grids/anime_dmd_zoom.jpg`).**
  * Identity top-5 avatar / sticker: 0.902 / 0.765 (an_lora 0.905 / 0.77).
  * style_dino: 0.616 (an_lora 0.613).
  * Skin dL median / dark quartile: +8.6 / +6.5 (an_lora +8.2 / +6.3).
  * Sharpness: 1035 (an_lora 862, klein 1929).
  * Visual check: crisper eye outlines and line art. Klein's catchlights, freckles and shine do not come back. The run plateaued by about step 750, with the regression anchor dominating.
  * Verdict: safe (no identity or skin cost) but a small gain. Run 2 (`an_dmd2`: dmd-w 4, reg-w 0.25, CFG 4.5, lr 4e-5) follows on GPU1.
- **Base-capacity result: SD-Turbo 0.87B ≈ SDXS 0.33B (`grids/anime_turbo_zoom.jpg`).**
  * cartoon3d t5d vs s5d: avatar identity top-5 0.255 vs 0.255, style_dino 0.618 vs 0.622, sticker top-5 0.26 vs 0.22.
  * anime t_an_lora vs an_lora: avatar 0.93 vs 0.905, sticker 0.795 vs 0.77, both within the about ±0.045 CI. style_dino 0.613 vs 0.613.
  * Sharpness: 929 vs 862 (klein 1929).
  * Visual check: about the same softness. Freckles are faintly visible in one row; catchlights and shine are absent in both.
  * Conclusion: 2.6× UNet capacity does not recover detail, so the bottleneck is the training objective, not the base size. **Keep SDXS** (license, 0.3 s on the phone, 316 MB w8). SD-Turbo is dropped (its license also needs a legal check).
- **DMD run 2 (`an_dmd2`: dmd-w 4, reg-w 0.25, CFG 4.5, lr 4e-5, 2000 steps, 81.1 min, peak 24.6 GB; `grids/anime_dmd2_zoom.jpg`).**
  * Identity:
    - top-5 avatar / sticker: 0.882 / 0.83 (an_lora 0.905 / 0.77). Avatar is −0.023, inside the ±0.032 CI.
    - top-1: 0.682 / 0.585 (an_lora 0.733 / 0.47).
  * style_dino avatar 0.615 (an_lora 0.613).
  * Skin dL median / dark quartile: +11.8 / +7.5. That is lighter than an_lora (+8.2 / +6.3); klein is +14.5 / +6.7.
  * Sharpness: 1587 (an_lora 862, run 1 1035, klein 1929).
  * Visual check:
    - Gains: much crisper eyes with clean whites and highlights, sharper line art, a cleaner cel look, no artifacts.
    - Losses: iris colour is lost (red/hazel eyes come out black), some fine wrinkles are smoothed, freckles still missing, and skin is slightly lighter.
  * Verdict: passes the likeness rule and is the best detail so far, but it costs colour and skin tone. Run 3 (`an_dmd3`: dmd-w 4, reg-w 0.5, CFG 3) tests whether a stronger anchor and lower CFG keep the gain without that cost.
- **DMD run 3 (`an_dmd3`: dmd-w 4, reg-w 0.5, CFG 3, lr 4e-5, 2000 steps, 80.9 min; `grids/anime_dmd3_zoom.jpg`). Negative result.**
  * Identity top-5 avatar / sticker: 0.863 / 0.77. Avatar is −0.042 vs an_lora, at the edge of the ±0.034 CI; the sticker gain of run 2 is gone.
  * Skin dL median / dark quartile: +10.6 / +7.5.
  * Sharpness: 1214 (run 2: 1587).
  * Visual check: iris colour is still lost (black anime eyes) and the result is softer than run 2.
  * Conclusion: neither CFG nor the regression anchor weight explains the lost iris colour. The likely cause is the klein-base anime prior pulling toward dark eyes. **Run 2 stays the best DMD setting.** A fix would need an eye-region-weighted colour anchor.
- **Overnight chain (user-approved 2026-11-16, results due 09:00 GMT+7 on 2026-11-17).**
  * **Round 4, base v1 `b1`** (`scripts/h200_base_v1.sh`, GPU2):
    - Same SDXS 1-step student, trained from the SDXS init, 16k steps, `--expr`, on `manifest_round3b.jsonl`.
    - Kept pairs: 3317 avatar + 2792 sticker + about 3.5k expression rows, covering synthetic, selfie, legal-approved real and expression sources.
    - Evaluated on v2, selfie and real, plus expr_eval.
    - Then the style packs are rebuilt on b1 with the an_lora / fv_lora recipe: `an_b1`, `fv_b1`.
  * **Deviation 14: DMD colour terms** (`train_dmd.py --color-w --eye-w`). Both anchor only low-frequency colour, so DMD keeps its detail gain:
    - `--color-w`: L1 between the 8×8 avg-pooled student and klein-target pixels (global colour and tone).
    - `--eye-w`: L1 on 2×2-pooled pixels inside soft Gaussian eye masks. Masks come from YuNet 5-point eye landmarks on the klein target (σ = 0.07 × face width); targets without a detected face get no mask.
    - Smoke test: landmarks found on 113/128 targets.
  * **Other `train_dmd.py` additions:**
    - `--max-pairs`: subset for the Qwen3 embed cache.
    - `--init-from <full student>`: starts a fresh zero-init LoRA r32 pack. This is how the cartoon3d base gets its DMD pass as a style pack; the phone base UNet is unchanged.
  * **GPU1** (`scripts/h200_night_gpu1.sh`):
    - `an_dmd4`: the run-2 settings + `--color-w 4 --eye-w 8`.
    - The recipe is chosen from an_dmd4 and written to `configs/dmd_recipe.txt`.
    - `c3_dmd`: DMD pack on b1, 3000 pairs, lr 1e-4 because the LoRA starts at zero.
    - `fv_b1_dmd`.
  * **GPU2** (`scripts/h200_night_gpu2.sh`): `an_b1_dmd` after b1.
- **DMD run 4 (`an_dmd4`: run-2 settings + `--color-w 4 --eye-w 8`, full-RGB 2×2 eye term; 80.9 min; `grids/anime_dmd4_zoom.jpg`).**
  * Identity top-5 avatar / sticker: **0.925** / 0.765 (an_lora 0.905 / 0.77, run 2 0.882 / 0.83).
  * Skin dL median / dark quartile: **+6.3 / +4.3**, the best of any model (klein +14.5 / +6.7, run 2 +11.8 / +7.5).
  * Eye landmarks were found on 1053/1546 targets.
  * Visual check: iris colour and skin tone come back. **Sharpness, however, falls back to 931** (run 2: 1587). The full-RGB 2×2 eye term is close to a pixel regression, so it brings the posterior-mean blur back into the eyes.
  * The overnight chain was stopped before its recipe step, and run 5 was started.
- **Run 5 (`an_dmd5`, `scripts/h200_dmd_anime5.sh`):** `--color-w 2 --eye-w 4 --eye-chroma`. The eye term compares chroma only (pixel minus its grey level, 4×4-pooled), which leaves luminance edges and highlights to DMD.
- **Round 4 result, base v1 `b1`** (8721 kept pairs incl. 3.4k expression rows, 16k steps, 48.9 min):

  | set / metric | klein | s5d (old base) | **b1** |
  |---|---|---|---|
  | v2 avatar / sticker top-5 | – | 0.255 / 0.22 | 0.255 / 0.22 |
  | selfie avatar / sticker top-5 | 0.24 / 0.34 | 0.325 / 0.21 | 0.165 / 0.19 |
  | real avatar / sticker top-5 | 0.15 / 0.187 | 0.35 / 0.20 | 0.14 / 0.153 |
  | real style_dino avatar | 0.862 | 0.736 | 0.852 |
  | skin dL dark quartile, v2 / selfie / real | 3.1 / 5.7 / 6.7 | 4.3 / 3.9 / **12.2** | 3.9 / 5.5 / **8.6** |
  | expression CLIP acc (expr_eval) | 0.73 | – (s7e 0.531) | **0.544** |

  * Visual check on real photos (`grids/b1_real_avatar.jpg`): b1 gives clean, complete cartoon3d avatars, as klein does. s5d's real-photo outputs are often half-formed or washed out, with the photo leaking through (e.g. a ghost face or a melted head).
  * s5d's higher identity score on selfie and real photos therefore comes from under-stylisation. b1 matches klein's style strength (dino 0.85 vs 0.86) and so also matches klein's identity on these sets.
  * **Verdict: b1 replaces s5d as the base.** It is more robust on real photos, has less dark-skin lightening on real photos (+8.6 vs +12.2) and better expressions. Identity in cartoon3d is limited by the teacher's generic face style; the same holds for klein.
- **DMD run 5 (`an_dmd5`: run-2 settings + `--color-w 2 --eye-w 4 --eye-chroma`; `grids/anime_dmd5_zoom.jpg`). Chosen as the production recipe.**
  * Identity top-5 avatar / sticker: 0.892 / 0.795 (an_lora 0.905 / 0.77).
  * Skin dL median / dark quartile: **+5.9 / +3.5**, the best so far.
  * Sharpness: 1084 (run 2 1587, run 4 931, an_lora 862).
  * Visual check: crisp eyes come back (whites, pupils, highlights), iris colour partly returns, and skin tone is the most faithful of any run.
  * The trade-off against run 2: somewhat softer, but correct skin tone and some eye colour. No further tuning tonight because of the 09:00 deadline.
  * Recipe file `configs/dmd_recipe.txt` = `--dmd-w 4 --reg-w 0.25 --cfg 4.5 --color-w 2 --eye-w 4 --eye-chroma`.
  * Production runs:
    - `c3_dmd` (GPU1, `scripts/h200_night_gpu1b.sh`).
    - `an_b1_dmd`, then `fv_b1_dmd` (GPU2, `h200_night_gpu2.sh` + `h200_night_gpu2b.sh`).
- **Core ML export of b1** (`scripts/export_night.sh b1`, CPU, 127 s): fp16 634 MB, w8 318 MB. It carries 22 task tokens (0 avatar, 1 sticker, 2–21 expressions).
- **Overnight production results (finished 2026-11-16 22:56 UTC; full write-up in `report.md`, table in `results_student.csv`).**
  * `c3_dmd`: cartoon3d DMD pack on b1, a fresh LoRA r32 at lr 1e-4, 3000 pairs, 81 min.
    - v2: avatar / sticker top-5 0.247 / 0.255; skin +5.5 / +2.9.
    - selfie: 0.175 / 0.22; skin dark quartile +4.1.
    - real: 0.12 / 0.167; skin dark quartile **+6.3** (b1 +8.6, s5d +12.2, klein +6.7).
    - Sharpness 104 (b1 85, klein 205). Crisper eyes and brows.
  * `an_b1_dmd`: avatar / sticker 0.902 / 0.80 (an_b1 0.895 / 0.74); skin +9.0 / +5.5 (an_b1 +9.8 / +7.1); sharpness 1097 (an_b1 888).
  * `fv_b1_dmd`: **fails the likeness rule**, 0.412 / 0.37 (fv_b1 0.598 / 0.545, klein 0.50 / 0.515). Visually it is flatter and cleaner, closer to klein, but loses face texture. **flat_vector ships `fv_b1` without DMD.**
  * Final lineup: `c3_dmd`, `an_b1_dmd`, `fv_b1`. Core ML exports are in `weights/student/<tag>/coreml/` (fp16 634 MB, w8 318 MB, 71–129 s on CPU).
  * Sheets: `grids/final_<style>_{avatar,sticker,zoom}.jpg`, `grids/final_real_cartoon3d.jpg`.
  * LICENSES.md: added FLUX.2-klein-base-4B. Noted that the b1-family training data includes the legal-approved real CelebA-derived photos.
  * int8-weight simulation (`scripts/w8_check.sh`, v2 set), mean |fp16 − w8| per pixel: c3_dmd 1.08 / 0.98, an_b1_dmd 1.19 / 1.24, fv_b1 1.18 / 1.18 (avatar / sticker, /255; p95 ≤ 1.9). Same level as s5d (0.96); the w8 builds are safe on the server side.
- **Checks after the overnight run (2026-11-17).**
  * **Expressions** (`expr_eval.py`, which now supports packs): CLIP expression accuracy c3_dmd **0.575** (b1 0.544), so the DMD pack did not hurt expressions. an_b1_dmd 0.371 and fv_b1 0.338: those styles have no expression training yet.
  * Per-expression accuracy, klein pilot → b1:
    - Weak in the teacher too: love 0.5 → 0, wink 0.38 → 0.12, ok 0.25 → 0.12.
    - Student-only weakness: party 0.88 → 0.29, thanks 0.62 → 0.17, facepalm 1.0 → 0.58.
    - Sleepy and night are confused by CLIP (both closed-eye).
  * **Anime / flat_vector on the selfie and real sets** (`scripts/h200_check_real.sh`; klein baselines generated), avatar top-5:
    - an_b1_dmd: selfie 0.73 (klein 0.67), real 0.45 (klein 0.477).
    - fv_b1: selfie 0.53 (klein selfie 0.555, from the earlier run), real 0.403 (klein 0.503).
    - Skin dL dark quartile on real photos: anime 9.0 (klein 10.2), flat_vector 0.0 (klein −2.4).
    - Both styles already hold up on real photos at roughly klein level without real-photo training pairs. The user asked to focus on cartoon3d first.
- **Hand fix (user request: both GPUs on the sticker hands).**
  * GPU1 `c3_dmd2` (`scripts/h200_c3_dmd_expr.sh`): `train_dmd.py --expr` (new option: expression tasks included, `--expr-frac 0.5` of a 4000-pair subset), init b1, run-5 recipe.
  * GPU2 `scripts/h200_gesture_data.sh`: new `gen_pairs.py --expr2-*` option (a second expression batch on faces f02500–f05499, 3 of the 7 hand gestures each, about 9000 klein stickers). Same filters; existing pairs are untouched. Output `manifest_round4.jsonl`.
    - Before this batch, kept pairs per gesture were think 127, thumbs 113, wave 91, ok 130, facepalm 110, shrug 82, thanks 68.
- **Deviation 15: adult-age expression prompt (user request 2026-11-17).**
  * Problem: klein drew older adults as children or babies in gesture stickers, even though the prompt already said "adult proportions, same age".
  * Pilot (`scripts/age_prompt_pilot.py`, 8 older faces × thanks/shrug, same seeds; `grids/age_prompt_pilot.jpg`):
    - A, current prompt: SFace cos 0.099.
    - **B, + "adult in their <decade>, adult head-to-body ratio/shoulders/hands, keep visible signs of age, never a child or a baby": 0.195.** Grey hair and wrinkles return.
    - C, B + chest-up crop: 0.191.
    - Gender drift (an attribute "person" face drawn as a man) is not fixed by any variant.
  * B → `TP.expr_sticker_prompt_adult`, used by `gen_pairs.py --expr2-adult`.
  * The gesture batch was restarted with it. The 4420 stickers already drawn with the old prompt were moved to `data/pairs/cartoon3d/_backup_expr2_old_prompt/` (kept, not deleted), so every manifest prompt matches its image.
  * Check on fresh outputs: `grids/gesture_adult_prompt_check.jpg`.
  * The first expression batch (f00000–f02499, all 20 expressions) still uses the old prompt.
- **Hand-fix result `c3_dmd2`** (DMD pack on b1 with `--expr`, round 3b data, `grids/hand_fix_c3_dmd2.jpg`, `grids/hand_fix_zoom.jpg`):
  * Expression accuracy 0.552 (c3_dmd 0.575; 24 faces, so within noise). Hand gestures: think 0.62 → 0.75, thumbs 0.62 → 0.75, wave 0.75 → 0.79, ok 0.25 → 0.17.
  * Visual check: clearly better finger separation on "Hi", slightly cleaner thumbs. OK is still drawn as an open palm. A partial fix.
- **Round 4 gesture data** (`manifest_round4.jsonl`, adult-age prompt): 48297 rows, 11901 kept. Kept per gesture: think 602, thumbs 658, wave 521, ok 674, facepalm 607, shrug 445, thanks 394, about 5× round 3b.
  * `c3_dmd3` (GPU2, `scripts/h200_c3_dmd3.sh`): DMD on b1, round 4, 60% expression rows, checkpoints and gesture progress sheets every 250 steps. Also the SDXS baseline of the bake-off.
- **Base-model research (user request 2026-11-17: technical merit only, licences decided later).** FLOPs measured per step at 512 px, batch 1:

  | model | params | GFLOPs |
  |---|---|---|
  | SDXS (current) | 328M | 406 |
  | SD 2.1 / SD-Turbo | 866M | 678 |
  | Sana-600M / Sana-Sprint-0.6B | 592M | 271 |
  | PixArt-α-512 | 611M | 1078 |
  | Segmind-Vega | 745M | 644 |
  | **DreamLite-mobile** | **390M** | **304 single / 606 diptych** |
  | TAESD enc+dec (added to every SD/SDXL-family model) | – | 264 |

  * Official SD 2.1 repos are gone from HF (community mirrors only). SD-Turbo is under the Stability Community License (<US$1M revenue). Sana weights are now Apache-2.0. Nitro-E: MIT, DC-AE 32× + Llama-3.2-1B.
  * Apple's ml-stable-diffusion benchmarks: Neural Engine speed is about the same from 14 Pro to 15 Pro Max. The iPhone 17 Pro gains come mainly from the GPU (Argmax).
  * **Pick: DreamLite-mobile** (native diptych editing, DMD2-distilled, 390M, TinyVAE decoder, official Core ML deploy code). License CC BY-NC 4.0.
- **Deviation 16: DreamLite one-step student** (`scripts/dl_arch.py`, `train_dl.py`, `dl_sanity.py`; `--arch` paths in `train_dmd.py`, `student_infer.py`, `expr_eval.py`). Weights `carlofkl/DreamLite-mobile@diffusers` (UNet 780 MB + TinyVAE 5 MB + Qwen3-VL text encoder 4.26 GB, the encoder used only to initialise tokens).
  * Student: [fixed noise (seed 1234) | photo latent] → one step at t=1000 → x0 = noise − v (left panel) → TinyVAE decode.
  * Text-free task tokens [22, 418, 2048] plus a fixed mask, initialised from Qwen3-VL edit-mode embeddings of each task's instruction, averaged over 16 photos.
  * **Deviation from the paper:** the VL encoder also sees the source photo; on the phone only the latent diptych is kept.
  * Sanity check (`grids/dl_sanity.jpg`): one-step conventions match the official pipeline (aligned left panel). The official 4-step output at 512 px is zoomed in, with weak identity (DreamLite is 1024-native).
  * Run `dl1` (GPU1, `scripts/h200_dl1.sh`): full fine-tune, round 4, 16k steps, bs 16, 0.4 s/step, 63 GB. Then the DMD pack, as for c3_dmd3.
  * Data loading is now parallel (`build_cache_dl`): the 11901-pair cache builds in 2 min instead of about 15 with the old serial loader.
- **Deviation 17: base probe instead of full runs per base (user request 2026-11-17).** User priorities for the base: (1) under 1B params, (2) a popular, community-backed pretrained base, (3) easy to fine-tune and to run on mobile. HF stats (30-day downloads / likes, fetched 2026-11-17): SD 1.5 1,127,453 / 1,370; SD-Turbo 389,992 / 471; SDXS 749 / 110; DreamLite-mobile 498 / 12. SD 1.5 also has official mobile builds from Apple (Core ML) and Qualcomm AI Hub (w8a16 for Snapdragon 8 Gen 3 / 7 Gen 4 / 8 Elite).
  * `scripts/h200_probe_base.sh <name> <base> [te]`: the same fixed budget for every base on manifest_round4 (4000-pair subset, 50% expression rows; val set unchanged):
    - A: full fine-tune from the public base, 3000 steps, lr 2e-5.
    - B: frozen public base + LoRA r32 (+ tokens), 2000 steps, lr 1e-4 (saved merged). LoRA size scales with the base: 12.9M params on SDXS, 29.2M on SD 1.5.
    - C: DMD pack on A, 500 generator steps, `configs/dmd_recipe.txt`, the c3_dmd3 subset.
    - Evals: expr_eval, v2 quality, skin_compare; `scripts/probe_summary.py` writes `results/probe_base.{csv,md}`.
  * Calibration: SD-Turbo is probed too. Its full-run result is known (equal to SDXS); the probe is trusted only if it agrees.
  * Queues: GPU1 `h200_probe_gpu1.sh` (turbo), GPU2 `h200_probe_gpu2.sh` (sd15, then sdxs). Smoke test `h200_probe_smoke.sh` (SD 1.5, 20 steps; `smoke_*` outputs deleted afterwards).
  * `train_student.py` additions: `--te-from` (token init from the base's own text encoder; SD 1.5 = CLIP ViT-L, 768-d; asserts the width matches the UNet), `--max-pairs` / `--expr-frac` (same subset rule as train_dmd), and frozen-public-base LoRA runs save a merged `unet/`.
  * The DreamLite DMD pass is dropped: DreamLite fails criterion 2 (498 downloads/30d) and is non-commercial. `dl1` (regression stage) is kept and evaluated.
- **dl1 result (DreamLite regression stage, round 4, 16k steps) vs SDXS** (`grids/dl1_vs_sdxs_avatar_zoom.jpg`, `grids/dl1_vs_sdxs_gestures.jpg`):
  * expr_eval (24 faces × 20): expression accuracy dl1 0.644, c3_dmd3 0.569, b1 0.544; love 0.38 (SDXS 0), party 0.96 (0.33), facepalm 0.79 (0.58); gestures (mean of 7) 0.566 vs c3_dmd3 0.637. OK 0.04.
  * v2: avatar / sticker likeness 0.263 / 0.275 (c3_dmd3 0.25 / 0.25, within CI); style_dino 0.584 (c3_dmd3 0.665, klein 0.714); skin +5.1 / +2.0 (c3_dmd3 +4.7 / +2.4).
  * Visual: keeps age (grey hair, wrinkles; SDXS turns an older woman into a child in shrug/thanks) and draws props (hearts, confetti). Flaws: grey/teal irises instead of brown, slightly washed-out colour, ghosted hands on some waves, faint horizontal streaks on cheeks.
  * Confound: b1 was trained on round 3b; dl1 on round 4 (5× gesture pairs, adult-age prompt).
  * dl1_dmd queued (`scripts/h200_dl1_dmd.sh`, GPU1 after the SD-Turbo probe): the c3_dmd3 recipe (incl. the eye-chroma term that targets iris colour), subset and steps.
- **Full SD 1.5 student (user-approved 2026-11-17)**, `scripts/h200_sd15_full.sh` on GPU2 after the SDXS probe: `sd15_b1` (full fine-tune from stable-diffusion-v1-5, 16k steps, all round-4 pairs, 768-d tokens from its own text encoder), then `sd15_dmd` (DMD pack, the c3_dmd3 recipe/subset/steps), evals on v2 + expr_eval + skin_compare against b1, c3_dmd3, dl1, dl1_dmd. Data note: b1 was trained on round 3b, sd15_b1 and dl1 on round 4.
- **Real-photo / selfie evals (2026-11-17)**, `scripts/h200_real_eval.sh`, cartoon3d avatar / sticker likeness top-5 and skin dL all / darkest quarter:
  * real (300): klein 0.15 / 0.187, +9.0 / +6.7; c3_dmd3 0.10 / 0.16, +9.4 / +7.8; **dl1_dmd 0.137 / 0.173, +5.5 / +3.5**; style_dino 0.866 (klein 0.862, c3_dmd3 0.864).
  * selfie (200): klein 0.24 / 0.34, +7.1 / +5.7; c3_dmd3 0.19 / 0.25, +6.3 / +4.3; **dl1_dmd 0.185 / 0.26, +5.1 / +2.7**.
  * DreamLite's lead holds on real photos: same likeness and style, and much less skin lightening.
- **Deviation 18: eye-luminance term (`train_dmd.py --eye-luma-w`).** dl1_dmd draws solid black eyes from DMD step 250 on (`grids/dl1_dmd_eye_progress.jpg`). The DreamLite TinyVAE round-trip keeps eyes intact (mean error 2.6/255 vs TAESD 4.2; `grids/vae_eye_roundtrip.jpg`), so the cause is training: the chroma-only eye term cannot see a luminance error. New term: L1 on 4x4-pooled luminance inside the eye masks. Run `dl1_dmd_eye` (`scripts/h200_dl1_dmd_eye.sh`): continue the dl1_dmd pack 750 steps with `--eye-luma-w 4`.
- **DreamLite anime pack `dl_an` (user request 2026-11-17)**, `scripts/h200_dl_anime.sh` (GPU1, after the eye fix): `train_dl.py --init-from dl1 --lora-rank 32` (new options: LoRA style pack on a frozen trained DreamLite student, tokens continued from it), anime manifest.jsonl, 6000 steps, lr 1e-4: the same recipe as the SDXS anime pack an_b1. Regression only; a DMD pass (as an_b1_dmd) would follow if it looks promising.
- **Eye fix results (2026-11-17).** v1 (`--eye-luma-w 4`, 4x4): grey, blurred irises. v2 (`dl1_dmd_eye2`, `--eye-luma-w 2 --eye-luma-pool 8`): eyes mostly black again; both cost skin tone on real photos (dl1_dmd +5.5/+3.5 → v1 +6.7/+5.5, v2 +7.3/+6.7) and some expression accuracy (0.662 → 0.656 / 0.633). `grids/eye_fix_v2_zoom.jpg`. dl1_dmd stays the best DreamLite checkpoint.
  * v3 (`dl1_dmd_eye3`, `scripts/h200_dl1_dmd_eye3.sh`): new `--eye-no-dmd` blocks the DMD gradient inside the (widened) eye masks, so eyes follow the regression + eye terms; continued from dl1_dmd, 750 steps.
    Result (done 16:05 UTC): **not fixed**. Irises are still near-black. Real photos show a faint dark-brown tint at most, with soft halo edges on some eyes, and no brown ring or catch-light like klein/SDXS (`grids/eye_fix_v3_zoom.jpg`, `grids/eye_fix_v3_zoom2x.jpg`). Likeness v2 avatar/sticker 0.315/0.285 (dl1_dmd 0.31/0.295), real 0.153/0.147 (0.137/0.173). Expression accuracy 0.637 (dl1_dmd 0.662). Skin lightening dL median/dark-25% v2 6.3/3.7 (dl1_dmd 4.7/1.2), real 7.1/5.9 (dl1_dmd 5.5/3.5, SDXS 9.4/7.8): the same skin cost as v2. Not exported. Conclusion: black irises are not driven by the DMD gradient alone. The brown is already lost in the dl1 regression stage: dl1 irises are dark grey-green with scratchy texture and no brown (checked on v2 face_007/fe2_042/face_000). A loss tweak on top of dl1_dmd is the wrong lever. Next candidate: eye-weighted regression when training dl1 itself (see vae_eye_roundtrip.jpg for the decoder check).
  * Eye root-cause study (2026-11-17, no training): `grids/eye_by_stage.jpg` shows SDXS b1 and SD1.5 regression stages keep brown irises (blurry), DreamLite dl1 does not (grey-green, streaky); `grids/eye_dl1_trajectory.jpg` shows dl1 irises dark from step 1k to 16k on val, so not under-training. Same data/lr/steps/losses as sd15_b1; TinyVAE round-trip keeps brown (decoder ruled out). `scripts/iris_sensitivity.py` (CPU, 64 klein targets, brown->dark iris, mask catches only the inner iris so these are lower bounds): LPIPS@256 changes 0.001 (train LPIPS ~0.12-0.14, <1%), latent MSE 0.003 TAESD / 0.004 TinyVAE (train MSE 0.13 SDXS / 0.25 DL, 1-2%). The iris is ~0.2-0.5% of the image and ~1-2% of the regression loss for every student; DreamLite's latent is not less sensitive. SDXS gets brown from its prior/full-res attention; DreamLite (no attention at 64x64, conv-only last up block) does not, and nothing in the loss pushes it. Fix must be in the regression stage: explicit eye terms (pixel L1 + chroma at 512, LPIPS on upsampled eye crops, GFPGAN-style).
- **DreamLite anime + DMD (`dl_an_dmd`, the an_b1_dmd recipe):** v2 anime avatar / sticker likeness 0.877 / 0.78 (SDXS an_b1_dmd 0.902 / 0.80, within CI), style 0.612 (0.617), skin +8.2 / +5.1 (+9.0 / +5.5). DMD removed the cross-hatch noise and blotches of dl_an; look now on par with SDXS anime (`grids/dl_anime_dmd_compare.jpg`).
- **Detail study (user request 2026-11-17): students keep only 55-75% of klein's fine detail.** `scripts/detail_bands.py` (radial FFT energy of the face region vs klein, v2 avatars; also logs to `results/detail_bands.jsonl`), fine band (0.3-0.6 Nyquist): c3_dmd3 0.58, sd15_dmd 0.65, dl1_dmd 0.67. Decoder ceilings (klein image round-trip): TAESD 0.87, DreamLite TinyVAE 0.97, so most of the loss is the training objective, not the decoder. Visual (`grids/detail_zoom_compare.jpg`): flat hair without strand groups/highlights, soft beard/hair edges. dl1's finest-band 1.45 is noise (scratchy streaks), not detail.
  * Evidence for the cause: DMD runs 1-3 sharpness falls as the regression anchor rises (reg 1: 1035, 0.5: 1214, 0.25: 1587); DMD2 (arXiv 2405.14867) reports oversmoothed images without its GAN loss and drops the regression loss.
  * Step 1 `dl1_det1` (`scripts/h200_detail_s1.sh`, GPU2): fresh DMD pack on dl1, 1000 steps, `--reg-w 0.1 --fake-updates 5`; baseline dl1_dmd_s1000 (same start, data, steps).
  * Step 2 `dl1_det2` (`scripts/h200_detail_s2.sh`, queued after step 1): + `--gan-w 5e-3`. New `train_dmd.py` options `--gan-w/--gan-block/--d-lr`: a 13.6M-param conv head on the klein-base fake-score transformer's single-block-9 features of noised latents (the transformer is frozen for this, forward stops at the hooked block; LADD/DMD2-style), real = klein targets re-encoded at 512 px, non-saturating loss, one head update per iteration. Smoke test passed (peak 54.8 GB).
  * Acceptance: fine-band energy up with no likeness drop > ~0.05, no skin regression, no crosshatch.
- **DreamLite eye probe (2026-11-17, user-approved, GPU 1 / dev-duyhv, `scripts/h200_dl_eye_probe.sh`).** New in `train_dl.py`:
  `--eye-w` (L1 on 2x2-pooled RGB at 512 px in soft YuNet eye masks; target = TinyVAE decode of the target latent) and
  `--eye-lpips-w` / `--eye-crop` (LPIPS on ROI-aligned eye crops upsampled to 192 px; a GFPGAN-style component loss
  without the discriminator). Both default off, so dl1-style runs are unchanged (backup: scratchpad `train_dl.py.bak_eye`).
  Eye landmarks were found on 10,955 of 11,901 targets (the rest are non-face subjects). Two 3k-step runs from the public
  DreamLite base, run in parallel, same seed/data/lr as dl1: `dlp_ctrl` (dl1 recipe) vs `dlp_eye` (`--eye-w 4 --eye-lpips-w 1`).
  The eye run takes ~1.0 s/step vs 0.7 for the control. At step 100 the weighted eye terms (~2.0 + 0.5) outweigh mse + lpips
  (~0.8), so watch for distortion. New metric `scripts/iris_colour.py`: Lab chroma of the dark pixels in each eye disc
  (klein 7.5, c3_dmd3 6.1, dl1_dmd 3.6). A first HSV "brown fraction" version did not separate the models (it counted
  skin) and was discarded.
  * Probe 1 result (17:39): iris colour fixed on the metric. Lab a*/b*/chroma of dark eye pixels on 200 v2 faces:
    klein 5.0 / 5.1 / 7.4, c3_dmd3 4.2 / 4.5 / 6.6, dl1 0.7 / 4.9 / 6.0 (greenish), dlp_ctrl 1.0 / 4.5 / 5.6,
    **dlp_eye 4.4 / 5.3 / 7.2**. Visually (`grids/eye_probe_dl.jpg`) the grey-green streaky texture is gone and irises are
    warm brown, but blurry, with no catch-light. About 1 face in 3 has ghost/double eyes. Val MSE/LPIPS 0.356/0.209 vs
    ctrl 0.334/0.193. Not usable as is. The eye term works, but at this weight from a fresh base it pulls eyes to the
    target landmarks before the layout forms.
  * Probe 2 (`scripts/h200_dl_eye_probe2.sh`, started ~17:50, GPU 1): from dl1, lr 1e-5, 2000 steps, dlp2_hi (4 / 1)
    vs dlp2_lo (1 / 0.25).
- **Detail step 1 result (`dl1_det1`, done 17:35): negative.** Fine-band energy vs klein (detail_bands, 700 avatars,
  coarse/mid/fine/finest): dl1_dmd_s1000 0.97/0.80/0.58/0.61, dl1_det1 0.94/0.70/0.48/0.48 (Laplacian variance 218 -> 149).
  Visually (`grids/detail_s1_zoom.jpg`) det1 is slightly smoother (hair strands, skin), not sharper. Likeness v2 avatar
  0.318 -> 0.29, expression 0.633 -> 0.65, skin unchanged (real 6.3/4.3 both). A lower regression weight plus more critic
  updates does not add detail at 1000 steps. Step 2 (`dl1_det2`, + DMD2-style GAN) keeps the step-1 settings, so
  step 2 vs step 1 still isolates the GAN.
  * Probe 2 result (18:17): **eyes fixed at the regression stage without ghosting.** From dl1, 2000 steps:
    dlp2_lo (eye-w 1, eye-lpips 0.25) a*/b*/chroma 4.0/4.6/6.8; dlp2_hi (4/1) 4.3/4.2/6.6 (klein 5.0/5.1/7.4,
    c3_dmd3 4.2/4.5/6.6, dl1 0.7/4.9/6.0). No ghost eyes in 24 v2 faces each. Streaky grey-green texture gone; irises
    clean dark brown, still softer than klein (no catch-light). Global val MSE/LPIPS 0.367/0.186 (lo) vs dl1 0.360/0.185:
    no global cost. lo and hi look the same; lo preferred (gentler). `grids/eye_probe2_dl.jpg`. DMD on top not yet run.
- **Detail step 2 result (`dl1_det2`, done 19:39): the GAN recovers step 1's loss, but no gain over the baseline.**
  Fine/finest band vs klein: s1000 0.58/0.61, det1 0.48/0.48, det2 0.59/0.66 (lapvar 218/149/242). No crosshatch at 3x
  pixel zoom (`grids/detail_s2_pixels.jpg`); edges slightly crisper than det1, about the same as the baseline; klein's hair
  structure is still missing (`grids/detail_s2_zoom.jpg`). Likeness v2 avatar 0.345 (s1000 0.318, within CI),
  expression 0.631 (0.633), skin real 5.9/3.9 (s1000 6.3/4.3). The discriminator won from step ~600 (d_real 0.99 /
  d_fake 0.02 at the end) and gan-w 5e-3 x g_gan ~4 = 0.02, small next to dmd 4 x ~0.4, so the GAN term was weak.
- **Teacher consistency check (2026-11-18, user-approved, no training).** Why the students cannot look like klein:
  klein gives a different shirt colour, framing/face size, hair layout and beard per seed for the same photo
  (`grids/teacher_variance_3seeds.jpg`, data/pairs avatar vs avatar_c1/avatar_c2). With one random target per photo, a
  regression student learns the average, so unpredictable details turn into blur. `scripts/teacher_consistency.py`
  measures how predictable the target is for three prompt settings (current / keep framing+clothes / keep + simpler
  style). 200 avatar photos from manifest_round4 (150 f*, 50 c*), 3 seeds each, 1,800 klein images, 4 steps at 512 px.
  Metrics: seed LPIPS, framing jitter, torso-colour spread, SFace id. Measured klein speed: ~5 img/s per H200
  (round 3: 10,600 targets in 2,144 s). Ran on the two existing 1-GPU containers (`h200_consistency.sh`, shard 0 on
  dev-duyhv, shard 1 on dev-duyhv-2): the 2-GPU dev-duyhv re-create was blocked because onprem-ai-dstack.qualgo.dev
  does not resolve from the workstation.
  * **Consistency result (03:30 UTC, 2026-11-18).** The first run was invalid: my script left KLEIN_REF at its default,
    so klein also got a style reference image, unlike gen_pairs (KLEIN_REF=0, PROMPTS_FAMILY=klein_text). Fixed and
    re-run on the new 2-GPU dev-duyhv; "current" seed 0 now reproduces the training targets exactly (pixel diff 0.0
    on 20 checked). 200 photos x 3 seeds:
    current seed-LPIPS 0.209, framing jitter 0.0108 / log-size 0.026, shirt seed std dE 8.1, SFace id 0.243;
    keep    0.148, 0.0073 / 0.020, 7.4, id 0.291;
    simple  0.159, 0.0078 / 0.021, 7.3, id 0.263.
    Two seeds of klein on the same photo differ by LPIPS 0.209, more than dl1's val LPIPS to its target (0.185): the
    regression student is already at the noise floor of the current targets, so loss tweaks cannot close the gap.
    The "keep" prompt (keep framing/pose/head size and clothing colours, plain light-grey background) cuts seed
    variance by 29% and raises likeness by 20%. Visually (`grids/teacher_consistency_sheet.jpg`) pose and framing
    follow the photo and clothes mostly follow it. Hair colour and some shirt colours still vary by seed. "simple" adds
    nothing over keep (lower id) and is dropped.
- **Infra (2026-11-18).** dstack server moved to https://onprem-ai-dstack.octoprexis.dev (~/.dstack/config.yml updated).
  dev-duyhv was re-created with 2x H200 NVL (dstack/dev_env.yaml gpu H200NVL:2, fleet h200-120-fleet); dev-duyhv-2
  was stopped. Run scripts take GPU=<0|1> where supported (h200_consistency.sh); older scripts still pin GPU 0.
- **FFHQ download (2026-11-18, user-approved exception to the "no real photos" rule; >5 GB approved).** Kaggle
  `arnaud58/flickrfaceshq-dataset-ffhq` (52k x 512 px, 20.9 GB) -> `data/ffhq/`, via Kaggle CLI 2.2.5 in
  `/mnt/project/kchat/duyhv/tools/kaggle_venv` (separate from the pinned project venv). The token lives in
  /mnt/project/kchat/duyhv/.kaggle (mode 600) and is never logged. Licences in LICENSES.md. Intended use: klein input photos
  for the regenerated, consistent dataset ("keep" prompt). Log: logs/ffhq_download.log.
- **Teacher prompt v2 -> v4 (2026-11-18, visual tests before any regeneration; `scripts/teacher_consistency.py`, `scripts/caption_photos.py`).**
  Problems in the original prompt: clothing never mentioned, "centered" re-frames, background colour unspecified,
  hair colour and head pose not in the keep list, real photos described only as "a person with <tone> skin", and person
  stickers say "Full subject". Local attributes come from the Qwen3-VL-2B inside DreamLite-mobile (no external API).
  * v2 (8 photos x 2 seeds, `grids/teacher_consistency_stageA_sheet.jpg`): naming "glasses / facial hair" in the keep list
    made klein ADD glasses to every face. The 2B VLM is wrong on baldness, buzz cuts and moustaches. Rejected.
  * v3: generator attributes for gender/age/hair where we have them (gender from the VLM: one synthetic photo's
    generator said "woman" for a man), VLM for clothing, and a keep list that never names absent items. 8 train photos:
    id 0.295 (keep 0.293), shirt-vs-photo dE 16.8 (keep 24.5), no invented glasses, bald/moustache/sunglasses/headphones correct.
  * v3 on FFHQ (6 photos): babies in FFHQ (VLM called one "woman, early twenties"); hair lost without a description; the
    L*-based skin word mislabels real-photo lighting both ways (medium-brown -> "light", fair -> "medium brown").
  * v4: VLM keys child / bald / skin_tone. Children are dropped (2 of 12 FFHQ). For real photos the VLM's hair (or "a bald
    or shaved head"), headwear ("exactly where it is in the photo") and skin-tone word are used. 10 FFHQ photos
    (`grids/teacher_consistency_ffhqA_sheet.jpg`): v4 id 0.149 vs v3 0.126; skin mostly right, curly hair, graduation cap and
    knit hat kept. Misses: one medium-brown woman drawn too dark, one light-brown hair drawn darker.
  * Full check running: current / keep / v4 on the same 200 train photos + 100 FFHQ, 3 seeds (`logs/consistency_v4.log`).
  * **Full v4 check result (04:10 UTC):** 200 train photos, 3 seeds: current / keep / **v4** seed-LPIPS 0.209 / 0.148 /
    **0.150**, framing jitter 0.0108 / 0.0073 / **0.0073**, shirt seed std dE 8.1 / 7.4 / **4.2**, shirt-vs-photo dE
    22.5 / 20.6 / **15.4**, SFace id 0.243 / 0.291 / **0.303**. 74 adult FFHQ photos (26 of 100 dropped as children; a
    contact-sheet check found 25 true children and 1 adult with a child at the frame edge): v4 seed-LPIPS 0.197, id 0.177
    (small faces inside the normaliser's blur padding; the torso dE is not valid for FFHQ because that region is blur).
- **Describer speed (2026-11-18, `scripts/caption_photos.py` caption_batch, `scripts/bench_caption*.py`).** Sequential
  1.12 s/photo. Batched (left padding, bs 32, greedy) agrees with sequential on gender/child/bald 100% and on
  age/skin/hair/clothing 95-98% (64 FFHQ). The first generate() call has a ~37 s one-off warm-up that hid the gain in the
  first benchmark. Warm, a batch of 32 takes 1.34 s = 0.042 s/photo (~26x), so ~38k FFHQ adults take ~27 min on one H200.
- **v4 sticker / expression prompts (2026-11-18, test running: `scripts/h200_sticker_prompt_test.sh`, tag stickA).** Same v4
  person description (+ clothing), one safe keep list, "head and upper body" for every person sticker (the old plain-sticker
  prompt said "Full subject"), the round-4 adult sentence whenever the age is known, expression phrase as "Expression and
  pose: ...", and expression words from synthetic attributes ("smiling broadly") removed for expression stickers (the old
  prompts asked for "smiling broadly" and "crying" at once). Baseline = the exact old pipeline prompt per photo. 8 train +
  6 FFHQ photos x 2 seeds x {sticker, cry, love, thumbs, shrug}.
- **Fast teacher (2026-11-18, `scripts/klein_batch.py`, `scripts/bench_klein_batch.py`, `scripts/bench_klein_profile.py`).**
  Stock Flux2KleinPipeline treats a list of images as references for every sample, so i2i_server.Klein ran batch 1. The
  KleinBatch override gives sample i only photo i. Batch 1 equals the stock adapter pixel for pixel (diff 0.0); batch 8 vs
  batch 1 differs by 0.26/255. Per H200: 3.94 img/s at bs 1, 4.54 at bs 8, **5.93 with torch.compile** (one-off compile
  ~20 s; compiled vs eager 0.59/255). The text encoder is 13 ms/img. With prefetch and background PNG writes the generator
  sustains ~5.7 img/s per GPU (11.4 on 2), vs ~2.7 per GPU for the old gen_pairs loop.
- **Data v2 generator (`scripts/gen_pairs_v2.py`) + 500-pair keep-rate test (`scripts/h200_v2_test500.sh`,
  data/pairs_v2/test500).** The round-4 keep rule is reproduced exactly from manifest_round4 fields: id rank <= 5
  (expressions <= 10) among 100 same-source inputs, sticker cut-out pass, extra_ok, and **|dL| <= 12**. Round-4 drops:
  identity 31,862 / skin 3,436 / extras 1,073 / sticker 13. Same 400 (photo, option) pairs, old prompt -> new: avatar
  12.5 -> 27.5%, sticker 22.5 -> 32.5%, expressions 23.8 -> 36.7%, **all 21.2 -> 34.0%**. By source: f 22.6 -> 37.0%,
  c 12.9 -> 29.0%, CelebA r 16.7 -> 14.3%; FFHQ adults (new) 10% (avatars 0/20). On real photos most fail both checks:
  median dL +15 (CelebA) / +19 (FFHQ) vs +7.5 synthetic. That is partly real klein lightening, partly the L* measurement
  (shadows/exposure, `q51310` looks fine at +22.7). Likeness is also lower on real photos: klein-made synthetic faces are
  in klein's own vocabulary, and FFHQ inputs are small faces in blur padding (CelebA is 178x218 upscaled). Without the skin
  rule FFHQ would keep 33%. Real-photo pairs are deferred until there is a lighting-robust skin check and better teacher
  inputs. All FFHQ adults can go to DMD (no targets needed).
- **Data v2 synthetic run (2026-11-18 04:53 UTC, user-approved ~3 h, `scripts/h200_v2_full.sh` -> data/pairs_v2/cartoon3d).**
  13,000 synthetic photos (f 10k, c 3k), 93,000 jobs: avatar best-of-2 seeds (one kept target per photo), plain sticker x1,
  2,700 per expression (<= 5 per photo). Expected ~31k kept pairs (~6.6k avatars, ~4.3k stickers, ~1k per expression)
  vs 11,901 in round 4. Captions in 2 parts, gen on 2 GPUs, scoring in 8 parts (regression: parallel scoring of test500
  gives identical keep flags, 500/500).
  Deviation (07:20 UTC): gen finished 07:15 (93,000 images, 11.6 img/s on 2 GPUs). Scoring in 8 parts was CPU-bound
  (OpenCV YuNet/SFace have no CUDA backend in this venv; sticker WebP size search; ~30 threads per process, machine 30% busy,
  GPUs 0%) with an ETA of 4-7 h. Stopped after ~5 min and relaunched as `scripts/h200_v2_full_score.sh`: same score code
  and keep rule, 64 parts x 8 threads (`SCORE_THREADS`), then merge and V2FULL_DONE in logs/v2full.log.
  Second deviation (07:53 UTC): the CPU pools (skincolor, plan-real FFHQ framing; 64 workers x ~319 OpenCV/torch threads,
  ~43k threads, load 1,700) starved the klein A/B gen on the same box (5.9 -> 0.26 img/s, GPUs 0%). Both pools stopped and
  relaunched with 1 thread per worker (`_pool_init`), 128 workers (`scripts/h200_v2_full_skincolor.sh`); A/B gen then
  finished at once. No result changes (same per-image code).
- **Per-source skin limit (user decision 2026-11-18, `gen_pairs_v2.py` DL_LIMIT / keep_rule / `rekeep` stage).**
  Synthetic f/c keep |dL| <= 12. Real photos (CelebA r, FFHQ q) get |dL| <= 23 = 12 + the measured median lighting offset
  (+18.6 FFHQ / +15.3 CelebA vs +7.5 synthetic). Pairs kept only thanks to the looser limit get `provisional: true`.
  test500 re-keyed (no re-scoring; old flags in manifest_dl12.jsonl): synthetic flags unchanged; CelebA 14 -> 29%
  (6 provisional of 42); FFHQ 10 -> 19% (9 provisional of 100). Real-photo ceiling with no skin limit is 31-33% (likeness).
  The running synthetic run's scoring uses the new code with the same limit (12) for its photos.
- **Skin filter = colour only (user decision 2026-11-18).** After `grids/skin_problems_examples.jpg` (row 1 lightening,
  row 2 over-saturation/yellowing, row 3 good): the user bypasses the brightness (dL) check entirely and filters skin on colour
  only. Row 2 is the reported "East Asian face turns orange-tan / South Asian" problem. `gen_pairs_v2.py` USE_DL = False (dL is
  still computed and stored). The colour limit (CHROMA_MAX) is set after the natural-skin prompt A/B and a visual check.
  Noted risk: the visual check also showed real lightening of darker skin (q20422 brown -> light pink, r00891), not only
  lighting artefacts on fair skin. The describer's skin label could not separate the two (r05483 and q20422 both "medium").
  Only 5% of test500 pairs keep skin within +-6 on both brightness and saturation (2% within +-4, 0 avatars).
- **Data v2 REAL-photo run (queued 2026-11-18 ~07:00 UTC, user request; `scripts/h200_v2_real.sh` -> data/pairs_v2/real).**
  All 6,597 CelebA-derived photos (r*, legal-approved) + 27.5k random FFHQ photos reframed like user photos, children dropped
  by the VLM, FFHQ capped at 20k adults. 3 jobs per photo (avatar, plain sticker, 1 expression spread evenly), ~80k images,
  ~2.5 h incl. captions/scoring. Starts after the synthetic run's skincolor pass and the natural-skin A/B. Style text is
  chosen by a fixed rule: natural-skin wording if its median skin saturation change is >= 3 lower and its keep rate
  (likeness + sticker) is not more than 3 points lower; the decision is printed in logs/v2real.log. Same keep rule as
  synthetic (likeness + sticker; brightness bypassed; colour limit set later). FFHQ inputs still use the user-photo
  normaliser (small face in blur padding), so expect lower likeness than synthetic.
  Changed 07:43 UTC (user asked why it waits): plan, captions and finalize start at once (they do not depend on the style
  decision); only gen waits for the A/B. Scoring 64 parts x 8 threads (see the synthetic run's deviation).
  Synthetic run result (V2FULL_DONE 07:38): 40,059 kept pairs of 93,000 before any colour limit (avatar 8,619 = 66% of
  photos, sticker 5,917, expressions 25,523, 895 thanks to 1,737 sleepy). Rejections are almost all likeness (48,400).
- **Natural-skin prompt A/B (2026-11-18, negative result).** Same 500 jobs/seeds, style text "natural skin color exactly
  matching the photo, not tanned and not more saturated": median skin saturation change +9.2 -> +11.1 (worse), yellow +9.0 ->
  +10.0, brightness +9.0 -> +7.5, keep (likeness+sticker) 41.6 -> 44.6%. Naming skin makes klein push it more (like
  mention-induced glasses). Real-photo run keeps the current style text (fixed rule, natural_skin=0).
- **Colour limit CHROMA_MAX = +17 (user decision 2026-11-18 ~08:10 UTC, after grids/chroma_limit_sheet.jpg).** Bands
  <= +12 natural, +13..+17 warmer but same ethnicity, +18..+22 the reported orange/tan shift. Synthetic re-keyed (`rekeep`;
  old flags in manifest_nocolor.jsonl): 40,059 -> **30,471 pairs** (avatar 6,827, sticker 4,363, expressions 19,281;
  639 thanks to 1,325 sleepy). The real-photo run applies the same limit in its final skincolor pass.
- **Data v2 REAL-photo run result (REAL_DONE 2026-11-18 10:39 UTC).** 6,532 CelebA + 20,000 FFHQ adults (4,809 children
  dropped: CelebA 65, FFHQ 4,744), style natural_skin=0, 79,596 teacher images (gen 6,831 s at 11.7 img/s on 2 GPUs).
  Pass likeness+sticker 34.4%; with colour <= +17: **23,459 pairs** (29.5%; CelebA 5,803, FFHQ 17,656; avatar 7,069,
  sticker 7,876, expressions 8,514). Colour removed 3,912; median skin saturation change +6.9 (synthetic +10.5).
  **Data v2 total: 53,930 pairs** (synthetic 30,471 + real 23,459): avatar 13,896, sticker 12,239, expressions 918
  (thanks) to 1,880 (sleepy). Round 4 had 11,901.
- **Student retrain on data v2 (user request 2026-11-18 10:50 UTC, `scripts/h200_dl2.sh`, dev-duyhv GPU 0).**
  `data/pairs/cartoon3d/manifest_v2.jsonl` (53,930 rows, built from the selected rows of data/pairs_v2/{cartoon3d,real};
  targets point into data/pairs_v2, FFHQ inputs into data/ffhq_norm via a new optional per-row `input` field read by
  train_dl.py / train_dmd.py; backups in scratchpad `*.bak_v2data`). Exact dl1 / dl1_dmd recipes and steps (16k regression,
  2000 DMD on a 5000-pair subset), so the data is the only change. No eye terms, no FFHQ-unpaired DMD (separate decisions).
  Eval sets (data/faces, faces_eval2, real_p2a/eval, faces_selfie) are disjoint from all v2 training photos. Val split =
  first 32/4 rows per task of the (task, id)-sorted manifest, so val numbers are not comparable with dl1's; compare on
  the fixed eval sets. Expected ~4 h (regression ~2 h + evals, DMD ~1 h + evals).
- **dl2 results, regression stage (DL2_DONE 2026-11-18 ~13:05 UTC).** Data v2 only change vs dl1.
  * Likeness (SFace cos, avatar/sticker): v2 0.215/0.178 (dl1 0.130/0.116, dl1_dmd 0.139/0.131, klein 0.341/0.356);
    real 0.152/0.155 (dl1_dmd 0.104/0.128, klein 0.208/0.238); selfie 0.152/0.144 (dl1_dmd 0.116/0.133, klein 0.299/0.330).
    Expression-sticker id top-5 0.365 (dl1 0.271, dl1_dmd 0.294). Style-ref similarity (style_dino) fell to klein's level
    (real 0.71 vs klein 0.70, dl1_dmd 0.87): less "generic template face".
  * Expression accuracy 0.596 (dl1 0.644, dl1_dmd 0.662); gestures 0.637 (dl1 0.566). Losses on sleepy/night/sad/facepalm
    (closed eyes / covered faces); hypothesis: the likeness filter keeps milder expressions (unchecked).
  * Skin: chroma/yellow drift mostly fixed (real Q1-Q3 +0.6..+3.9 vs dl1_dmd +13..+16). skin_compare "dark-25% dL +32" on
    the real set is a METRIC ARTEFACT: `grids/dark_skin_check_real.jpg` shows the darkest-25% real inputs are mostly
    fair-skinned people in under-exposed photos (rp_243/276/063/204/072/216/160), which klein_personal and dl1_dmd turned
    into dark-skinned avatars (old teacher prompt took skin tone from photo L*); dl2 draws them fair. Genuinely
    dark-skinned inputs stay dark in dl2 (rp_053 +15, rp_297 +15, rp_095 +5; klein +32/+12/+14). Input-L*-based dL is not a
    skin-fidelity measure on badly exposed photos.
- **SDXS retrain on data v2 (user request 2026-11-18 ~13:50 UTC, `scripts/h200_sx2.sh`, dev-duyhv GPU 1, parallel to dl2).**
  Same manifest_v2.jsonl; exact SDXS recipes: `sx2` = train_student.py --param direct 16k steps (as b1), `sx2_dmd` = DMD
  pack from sx2 (dmd_recipe.txt, 5000-pair subset, 60% expressions, 2000 steps; as c3_dmd3). train_student.py build_cache
  reads the optional per-row `input` (backup scratchpad train_student.py.bak_v2data). Evals vs c3_dmd3 / dl2_dmd / klein.
  ~2.5-3 h (b1 trained 16k steps in 49 min; cache of 53.9k pairs is serial).
- **2-step DreamLite student for detail (user request 2026-11-18 ~14:20 UTC, `scripts/h200_dl2_2s.sh`, queued on GPU 0
  after h200_dl2.sh).** New `dl_arch.forward(..., sigmas=[1.0, 0.5], k)`: step 1 = the 1-step student; its x0 is re-noised
  to sigma 0.5 with a fixed per-step noise (seed 1234+j, a phone constant) and denoised again with the photo latent beside
  it, so the photo conditions every step (the failed SDXS an_lora_m4 test re-noised without the photo and lost identity).
  `train_dmd.py --ms-sigmas 1.0,0.5`: random step k per iteration, earlier steps without gradient (DMD2 backward
  simulation); `student_infer.py --dl-k` runs the first k steps; expr_eval/student_infer read `ms_sigmas` from the
  student config. CPU check (`scripts/test_dl_multistep.py`): 1-step path bit-identical to the old code (max diff 0.0), k=1
  of the 2-step student == 1-step, gradient reaches the last step. Run `dl2_2s`: from dl2_dmd, dl2_dmd recipe/subset, 1500
  DMD steps; evals vs dl2_dmd, plus the same weights at 1 step (`dl2_2s_k1`, separates "2nd step" from "more DMD"),
  detail_bands, iris colour, `grids/dl2_2s_sheet.jpg`. Phone cost: 2 UNet passes per image. Backups: scratchpad *.bak_2step.
- **dl2_dmd results (DL2DMD_DONE 2026-11-18 ~14:56 UTC; `grids/dl2_dmd_sheet.jpg`).** DMD pack on dl2, dl1_dmd recipe.
  * Likeness (SFace cos avatar/sticker): v2 0.229/0.189 (dl2 0.215/0.178, dl1_dmd 0.139/0.131, klein 0.341/0.356); real
    0.153/0.164 (dl1_dmd 0.104/0.128, klein 0.208/0.238); selfie 0.159/0.156 (dl1_dmd 0.116/0.133, klein 0.299/0.330).
    Expression-sticker id top-5 0.390 (dl2 0.365, dl1_dmd 0.294).
  * Expression accuracy 0.598 (dl2 0.596, dl1_dmd 0.662); gestures 0.530 (dl2 0.637, dl1_dmd 0.631): think 0.67->0.42,
    thumbs 0.62->0.42, ok 0.38->0.17, thanks 0.54->0.21; happy/facepalm/party/sad/wink recover vs dl2. Night 0.04.
  * Skin chroma Q1-Q4 real +6.5/+3.1/+3.4/+4.2 (dl2 +3.9/+0.6/+0.6/+2.0, dl1_dmd +15.6/+16.1/+13.1/+2.8); DMD (klein score)
    adds a little saturation back. Style-ref similarity up again (real 0.79; dl2 0.71, dl1_dmd 0.87): bigger, darker
    cartoon eyes with catch-lights, crisper edges; pose/clothing/skin-tone fixes of dl2 kept.
- **Detail check after dl2_dmd (user: "resolution not high like klein", 2026-11-18 ~15:15 UTC).** `detail_bands` (900
  avatars, energy vs klein coarse/mid/fine/finest, lapvar): dl1_dmd 1.00/0.83/0.63/0.69, 269; dl2 0.76/0.56/0.56/1.29, 375
  (finest >1 = streak noise); **dl2_dmd 0.96/0.55/0.34/0.40, 69** (smoothest; my earlier "crisper edges" from thumbnails was
  wrong). `grids/zoom_dl2_dmd.jpg` (2x face crops): klein crisp eye outlines, brown irises, catch-lights; dl2 scratchy
  streaked eyes/stubble; dl2_dmd soft, black-dot eyes. Teacher targets are NOT softer: face-crop lapvar / fine-to-coarse
  energy, 300 synthetic each: avatar round4 490 / 0.123 vs v2 562 / 0.130; expressions 775 / 0.138 vs 743 / 0.127. So the
  detail is lost in the student (1-step + this DMD pass), not in the data. dl2 val LPIPS still creeping down at 16k
  (0.1937 @10k -> 0.1908 @16k) under the decaying LR: mild under-training possible, not the main cause.
- **sx2 results (SDXS regression on data v2, SX2_DONE ~15:30 UTC; `grids/sx2_sheet.jpg`, `grids/zoom_sx2.jpg`).**
  Likeness avatar/sticker v2 0.233/0.174 (c3_dmd3 0.133/0.130, dl2_dmd 0.229/0.189), real 0.149/0.144, selfie 0.157/0.142;
  expression-sticker id top-5 0.275 (c3_dmd3 0.204). Expression acc 0.523, gestures 0.566 (c3_dmd3 0.569/0.637).
  Detail vs klein coarse/mid/fine/finest 0.78/0.56/0.42/0.45, lapvar 105 (c3_dmd3 0.91/0.86/0.61/0.55, 228; dl2_dmd
  0.34 fine, 69). Eyes: brown irises with catch-lights (SDXS keeps them; DreamLite loses them). Skin chroma real Q1-Q4
  +10.7/+6.7/+6.8/+6.9 (c3_dmd3 +17.8/+15.6/+9.0/+1.4).
- **Why dl2_dmd is soft (diagnosis 2026-11-18 ~15:45 UTC, no GPU).** Sharpness (Laplacian variance, face crop of the 8 val
  sample tiles train_dmd.py saves every 250 steps): dl2_dmd 708 (step 0 = dl2) -> 448 -> 380 -> 322 (step 750) -> ~340-410
  to the end; same recipe dl1_dmd 904 -> 1160 and c3_dmd3 953 -> 1148 (klein targets ~1700). So THIS DMD pass smooths the
  student (the regression stage is not the main loss), in its first ~750 steps. Teacher targets are equally sharp for every
  source (avatar lapvar f/c 544, CelebA 521, FFHQ 579). What changed vs the dl1_dmd pass: data mix (round 4: f 76%, c 6%,
  o 10%, r 8% -> v2: f 46%, c 11%, q 33%, r 11%, i.e. 8% -> 44% real-photo inputs; CelebA inputs are very soft, face
  lapvar 58 vs synthetic 593; FFHQ inputs sit in blur padding), longer v4/v5 teacher prompts (median 74 -> 117 words, same
  style text), and the start model. Hypothesis (untested): the real score (klein-base) is conditioned on the input photo;
  soft/blurred reference photos pull the DMD target softer. dl2_2s continues from dl2_dmd with the same DMD settings.
- **Probe A queued (user-approved 2026-11-18 ~15:35 UTC, `scripts/h200_probeA.sh`, GPU 0 after dl2_2s).** The dl2_dmd DMD
  pass from dl2 (same recipe, seed, subset size, val tiles) with training rows restricted to synthetic photos (new
  `train_dmd.py --src-prefixes fc`), 500 generator steps. Read-out: `scripts/dmd_sharpness_series.py` (val-tile sharpness
  per 250 steps vs dl2_dmd 708 -> 448 -> 380), v2 eval, detail_bands. Stays sharp -> soft real-photo inputs cause the
  softening (fix: DMD on sharp-input pairs only); also softens -> look at prompts / start model.
- **dl2_2s results (2-step DreamLite, DL2_2S_DONE ~16:35 UTC; `grids/zoom_dl2_2s.jpg`, `grids/dl2_2s_sheet.jpg`).**
  detail_bands, v2 400 avatars, vs klein coarse/mid/fine/finest | lapvar: dl1_dmd 1.02/0.86/0.67/0.74 | 260; dl2_dmd
  0.95/0.64/0.43/0.48 | 104; **dl2_2s at 1 step (k1) 0.97/0.64/0.44/0.50 | 98; dl2_2s at 2 steps 0.97/0.73/0.55/0.62 | 168.**
  The second step adds the detail (+29% fine band, +71% lapvar vs its own 1-step output); 1,500 more DMD steps alone add
  none (k1 = dl2_dmd). Val-tile sharpness rose 279 -> 598 over training, still rising. Visually crisper beard/brow/lip/eye
  outlines; irises still dark (iris chroma 4.4; dl2_dmd 4.7, klein 7.8), no catch-lights; still well short of klein.
  Likeness avatar v2 0.199 (k1 0.202, dl2_dmd 0.229), real 0.143 (0.153), selfie 0.149 (0.159); expression-sticker id
  top-5 0.338 (0.390): the drop comes with the extra DMD training (k1 shows it too), not the second step. Expression acc
  0.598 (=), gestures 0.583 (dl2_dmd 0.530). Skin chroma slightly lower than dl2_dmd on all sets.
- **Probe A result (PROBEA_DONE ~17:03 UTC): NEGATIVE for the real-photo hypothesis.** DMD from dl2 on synthetic photos only
  (30,327 rows, 500 steps): val-tile sharpness 708 -> 502 -> 453 (dl2_dmd 708 -> 448 -> 380); detail_bands v2 0.94/0.60/
  0.42/0.48, lapvar 92 (dl2_dmd after 2000 steps 0.95/0.64/0.43/0.48, 104); likeness v2 avatar/sticker 0.218/0.188. Soft
  real-photo inputs are at most a small part. sx2_dmd (SDXS, data v2) also softened 612 -> 397 while c3_dmd3 (old data)
  sharpened 953 -> 1148: the softening is architecture-independent and tied to the v2 DMD setup. Remaining suspects:
  the v4/v5 teacher prompts given to the klein-base real score (117 vs 74 words, "keep ... expression ... exactly"), or
  the start model (dl2/sx2 trained on v2 targets).
- **sx2_dmd results (SX2DMD_DONE ~17:13 UTC).** Expression acc 0.565, gestures 0.601 (sx2 0.523/0.566, c3_dmd3 0.569/0.637);
  likeness avatar v2/real/selfie 0.222/0.152/0.162 (c3_dmd3 0.133/0.097/0.110); skin chroma real Q1-Q4 +6.8/+4.4/+3.7/+5.2.
  Val-tile sharpness fell 612 -> 397 during DMD (c3_dmd3 953 -> 1148): softened like dl2_dmd.
- **Stopped here (user, 2026-11-18 ~17:15 UTC): no probes B/C.** State: data v2 (53,930 pairs) fixes likeness and skin
  colour on both architectures; the v2 DMD pass softens both students and weakens gestures; 2 steps (dl2_2s) is the only
  measured detail gain. Open: cause of the DMD softening (suspects: v4/v5 prompts in the real score, or the start model).
- **Probes B and C (user request 2026-11-19, `scripts/h200_probeBC.sh`, both GPUs in parallel).** What makes the data-v2
  DMD pass soften the students? Both = the dl2_dmd pass (recipe, 5000-pair subset, seed, val tiles), 500 steps.
  B `probeB_oldprompt`: from dl2, data v2, but the klein-base judge gets old round-4-style prompts (new
  `manifest_v2_oldprompt.jsonl`, built with TP.avatar_prompt/sticker_prompt/expr_sticker_prompt from the generator
  attributes and input L*; median 85 vs 159 words; no "keep the face ... expression ... exactly"; new
  `train_dmd.py --prompt-manifest`). C `probeC_dl1start`: from dl1 (old-data student), data v2 + v2 prompts.
  Read-out: val-tile sharpness series, v2 likeness, detail_bands, expr_eval. Backup scratchpad train_dmd.py.bak_probeBC.
- **Full-res perceptual loss (user request 2026-11-19; root cause of the base blur of every student).** The global LPIPS and
  the colour term compare student and klein at 256 px (targets cached at 256 only), so detail between 256 and 512 px was
  supervised only by latent MSE, which averages. New `--lp512-w` / `--lp512-crop` (default 256) in `train_dl.py` (main
  stage) and `train_dmd.py` (DMD regression anchor, DreamLite and SDXS): LPIPS on one random 256 px crop per sample of the
  native 512 px output vs the decoded target latent (TinyVAE keeps 0.97 of klein's fine detail; TAESD 0.87), same box on
  both; crop origin x 48-208, y 32-192 (face area). No new cache. Backups scratchpad *.bak_lp512. train_student.py (SDXS
  main stage) not patched yet.
  Probe queued after probes B/C (`scripts/h200_lp512_probe.sh`): continue dl2 at lr 1e-5 for 2000 steps, dlp512_ctrl (GPU 0)
  vs dlp512_on --lp512-w 1 (GPU 1); 3-step smoke first; read-out detail_bands v2, likeness, `grids/zoom_lp512.jpg`.
- **Probes B/C results (PROBEBC_DONE 2026-11-19 ~01:30 UTC): prompts explain only a small part; cause still open.**
  Val-tile sharpness (8 synthetic tiles) 0/250/500: B old prompts 708/702/605, C dl1 start 768/595/454, A 708/502/453, dl2_dmd
  708/448/380. But on the 400-avatar v2 eval (detail_bands coarse/mid/fine/finest | lapvar): dl2_dmd (2000 steps)
  0.95/0.64/0.43/0.48 | 104; A 0.94/0.60/0.42/0.48 | 92; **B 0.91/0.65/0.45/0.51 | 111; C 0.82/0.53/0.37/0.45 | 81**
  (dl1_dmd 1.02/0.86/0.67/0.74 | 260; dl2 before DMD lapvar 427, finest 1.41 = streak noise). So the early 8-tile signal
  overstated the prompt effect: old prompts help a little, a dl1 start does not help, and every v2 DMD pass ends near
  lapvar 80-110. Expressions: B 0.598 (=dl2_dmd), gestures 0.536; C 0.575/0.512; no expression gain from old prompts.
  Likeness v2 avatar B 0.207, C 0.204 (dl2_dmd 0.229). Remaining suspect: the v2 data inside the DMD pass (targets for
  the regression anchor/colour term, subset), not prompts, photos (probe A) or start model (probe C). The full-res loss
  (lp512) addresses the base blur independently; its probe runs next.
- **Full-res loss probe result (LP512_PROBE_DONE 2026-11-19 01:49 UTC): POSITIVE.** From dl2, lr 1e-5, 2000 steps, v2 400
  avatars, detail_bands coarse/mid/fine/finest | lapvar: dl2 0.81/0.66/0.65/1.41 | 427; dlp512_ctrl 0.81/0.66/0.65/1.39 | 421;
  **dlp512_on (--lp512-w 1) 0.83/0.73/0.71/0.99 | 309** (dl1_dmd 1.02/0.86/0.67/0.74 | 260). Fine band 0.71 = highest of any
  student; finest 1.39 -> 0.99 = the streak noise is gone (lapvar falls because the streaks were most of it).
  `grids/zoom_lp512.jpg`: no horizontal streaks in eyes/beard, clean eye outlines and beard shape; irises still dark, no
  catch-lights. Likeness v2 avatar/sticker 0.212/0.176 (ctrl 0.217/0.178, within noise); style_dino 0.527/0.547 (ctrl
  0.493/0.505; klein 0.618/0.601). Val LPIPS(256) 0.1877 @1500 vs ctrl 0.1913 @2000.
- **Loss research (2026-11-19, web, no GPU).** DMD2 (arXiv 2405.14867) drops DMD's regression loss ("upper-bounds the
  student by the teacher") for TTUR + a GAN on real data; CDM (arXiv 2605.06376) reports DMD2 over-smoothing / detail loss,
  attributed to discrete timestep anchoring and mode-seeking reverse KL (remedy: continuous dynamic schedule,
  off-trajectory DM); pix2pix-turbo (arXiv 2403.12036, closest one-step i2i): L1 + LPIPS + CLIP-backbone GAN + zero-conv
  encoder->decoder skips for high-frequency detail; ADD/LADD (2403.12015): DINOv2 / latent discriminators, noise level
  biases toward detail; vision-aided GAN (2112.09130): CLIP/DINO discriminator ensembles; OSEDiff (2406.08177) / PiSA-SR
  (2412.03017): MSE + LPIPS + score distillation for one-step SR; Focal Frequency Loss (ICCV 2021). Follow-up candidates:
  frozen DINOv2/CLIP-feature GAN on klein targets at 512 px; FFL; lower / no DMD regression anchor.
- **Combined DreamLite retrain (user request 2026-11-19 ~02:10 UTC, `scripts/h200_dl3.sh`, both H200s).** Main stage
  --lp512-w 1 --eye-w 1 --eye-lpips-w 0.25 on manifest_v2; GPU 0 `dl3` 16k steps, GPU 1 `dl3L` 24k steps (under-training
  check). Then each: 2-step DMD (`--ms-sigmas 1.0,0.5 --lp512-w 1`, dmd_recipe, 2000 steps) -> `dl3_2s` / `dl3L_2s`; every
  stage evaluated (expr, v2/real/selfie, 1-step k1 of the 2-step model, detail_bands, iris, `grids/zoom_dl3.jpg`,
  `grids/dl3_sheet.jpg`). 3-step smoke tests of both new loss combinations first.

- 2026-11-19 ~03:05 UTC — dl3 relaunch: the first launch stopped at the DMD smoke test (`DL3_SMOKE_FAIL dmd`, device mismatch in `noisy()`).
  Cause: in the new `--lp512-w` block of `train_dmd.py` the crop corner was named `x0`, overwriting the student latent `x0` that the
  fake-score update reads next (first use of lp512 inside DMD; the lp512 probe ran only `train_dl.py`, so its result is unaffected).
  Fix: renamed to `cx`/`cy` (backup `train_dmd.py.bak_dl3fix` in scratchpad). Old logs kept as `logs/h200_dl3_fail1.log`, `logs/dl3_smoke_fail1.log`.
  No training steps were lost (failure was in the 3-step smoke); ETA shifts by ~55 min.
- **Detail follow-up dl4 (user-approved 2026-11-19 ~03:30 UTC, `scripts/h200_dl4.sh`, queued after dl3).** New off-by-default
  `train_dl.py` options (backup `train_dl.py.bak_fflgan` in scratchpad): `--ffl-w/--ffl-alpha/--ffl-log` = Focal Frequency Loss
  (Jiang et al. ICCV 2021; fft2 ortho per RGB channel on the full 512 px output vs TinyVAE decode of the target; weight matrix
  = detached error magnitude^alpha, **log(1+w) option on** = FFL's log_matrix, a deviation from the paper default, chosen so
  high-frequency errors are not drowned by low-frequency ones); `--gan-w/--gan-layers/--gan-start/--d-lr` = vision-aided
  critic: frozen DINOv2-base (local cache, no download), 4 spectral-norm conv heads (7.1M) on patch tokens of layers 2,5,8,11,
  random 448 px crops, hinge loss, real = decoded klein target, fake = student 512 px output, one critic update per step,
  generator term after a 200-step critic warm-up. `lp512_loss` now takes the decoded target (decoded once per step; same
  random draws). Training sheets now add **native 512 px face crops** (klein 1:1 | student 1:1) as columns 4-5.
  Smoke (bs 8, 100 steps beside dl3, `logs/fflgan_smoke.log`): FFL 0.04-0.05, generator hinge 0.88, critic loss 0.87 after
  50 steps (2.0 = chance). Weights: --ffl-w 2, --gan-w 0.1 (each ~0.09, about half the 256 px LPIPS term).
  Runs (each 2000 steps from dl3, lr 1e-5, dl3 losses kept): GPU 0 dl4c (control) -> dl4f (+FFL); GPU 1 dl4g (+critic).
  Caveat: the critic uses DINOv2-base, as does the style_dino metric, so style_dino is not independent for dl4g.
- **ROOT-CAUSE BUG (found 2026-11-19 ~03:50 UTC while profiling): flip augmentation in latent space damages half the
  regression targets of every student.** `train_dl.py` and `train_student.py` (SDXS) flipped the cached latents (zi, zt)
  with p = 0.5. Neither VAE commutes with a horizontal flip (`scripts/flip_aug_check.py`, 64 v2 targets):
  | VAE | round trip decode(z) vs PNG | decode(flip z) vs flip(PNG) (what training used) | decode(enc(flip PNG)) |
  |---|---|---|---|
  | DreamLite TinyVAE | 33.4 dB | **22.6 dB** | 33.4 dB |
  | SD VAE -> TAESD (SDXS) | 33.1 dB | **22.5 dB** | 33.1 dB |
  Latent MSE between flip(z) and enc(flip PNG) = 0.45 (DL; latent var 1.24), larger than dl3's training MSE (~0.39).
  `grids/flip_aug_check.jpg`: blocky/jagged edges, colour fringes in the eyes, smeared hair and beards; the error map sits
  on edges, eyes and hair texture (the detail we have been chasing). On flipped samples the latent MSE, full-res LPIPS and eye
  targets were damaged while the 256 px LPIPS target (pixel flip) was clean: conflicting targets. Effect on the students is
  not yet measured (plausible contributor to blur / streaks; A/B below). DMD (`train_dmd.py`) does not flip.
  Fix: `scripts/build_flip_cache.py` encodes the flipped images (`latents_dl_manifest_v2_flip.pt`, zi_f/zt_f) and writes the
  decoded 512 px target cache (`tgt512_dl_manifest_v2.u8`, uint8 npy memmap, 42 GB, 61 dB quantisation). New `train_dl.py`
  options (off by default): `--proper-flip`, `--tgt-cache`, `--compile`, `--bench N`. The eye terms now reuse the one target
  decode per step (was decoded twice; same values). Backup `train_dl.py.bak_flipfix` (scratchpad).
  User choice ("A/B on GPU 1"): dl3 (old flip) keeps running on GPU 0 as the control; dl3L stopped at ~15% (under-training
  test postponed); GPU 1 runs `scripts/h200_dl3F.sh`: dl3F = dl3 + --proper-flip (+ --tgt-cache, speed only), same 2-step DMD.
  The dl4 detail follow-up now starts from dl3F with the fixed pipeline (`scripts/h200_dl4.sh`; old version kept as
  `h200_dl4_v0_buggyflip.sh`, never ran). SDXS (`train_student.py`) still has the bug (not patched; only if SDXS is revisited).
- **Training speed (2026-11-19, idle H200 NVL, dl3 config bs 16, `scripts/h200_speed_bench.sh`, `logs/speed_bench.log`).**
  s/step: dl3 code ~0.55 (log timer, incl. warm-up) | one target decode per step 0.483 | + `--tgt-cache` 0.459 |
  **+ `--compile` 0.328 (peak 45 GB vs 64.5)** | no lp512/eye terms 0.400. => ~1.7x vs dl3. dl3F and dl5 use
  `--tgt-cache --compile` (speed only; same losses/data). Note: the 3-step smokes do not reach the val/sample code under compile.
- **2 extra H200 NVL (user request 2026-11-19 ~04:25 UTC):** `dstack/dev_env_2.yaml` -> run `dev-duyhv-2` on h200-gpu-fleet
  (dev-duyhv is on h200-120-fleet), same image + /mnt/project/kchat volume. User: use it for the detail-loss experiments ->
  `scripts/h200_dl5_n2.sh`: from dl2, 3000 steps, lr 1e-5, fixed pipeline (--proper-flip --tgt-cache --compile), dl3 losses;
  GPU 0 dl5c (control) -> dl5f (+FFL 2); GPU 1 dl5g (+DINOv2 critic 0.1) -> dl5fg (both). Replaces the dl4 plan (never ran)
  and the dl3FL idea. The node was stopped by accident once (~05:00) during the smoke and resubmitted; no training lost.
- **dl5 results, part 1 (2026-11-19 ~05:20 UTC; detail_bands on the same 400 v2 avatars; `grids/zoom_dl5c_vs_lp512.jpg`, `grids/zoom_dl5cg.jpg`).**
  | model | coarse/mid/fine/finest | lapvar | v2 / real avatar id | expr acc | dL dark25 (real) |
  |---|---|---|---|---|---|
  | dl2 | 0.81/0.66/0.65/1.41 | 427 | — | 0.596 | +32.2 |
  | dlp512_on (probe, old flip) | 0.83/0.73/0.71/0.99 | 309 | 0.212 / — | — | — |
  | dl5c (control: fixed flip + lp512 + eye) | 0.82/0.61/0.57/0.73 | 207 | 0.203 / 0.150 | 0.594 | +33.7 |
  | **dl5g (+ DINOv2 critic 0.1)** | **0.93/0.76/0.72/0.84** | 285 | **0.221 / 0.157** | 0.569 | +32.5 |
  **Metric correction:** the zoom shows dl2 (and less so dlp512_on) with horizontal scan-line streaks in the irises and jagged
  edges; dl5c has none. Fine/finest band energy counts those artifacts, so the earlier "dlp512_on fine 0.71 = best" was partly
  artifact energy; detail_bands must be read together with the zoom sheet. Whether the streak removal comes from the flip fix
  or the eye terms is separated by dl3 vs dl3F (both have eye terms).
  **dl5g passes the acceptance rule:** fine +0.15 vs control with no streaks/crosshatch (stubble texture, nose/cheek shading,
  slight brown in irises), likeness up, skin unchanged; expression accuracy -0.025 (watch). style_dino not counted (same
  DINOv2 as the critic). Still missing vs klein: big round eyes with catch-lights, crisp eyeliner/lashes (shape/style, not blur).
- **Flip-fix A/B result (dl3 vs dl3F main stages, 2026-11-19 ~06:00 UTC; `grids/zoom_dl3_vs_dl3F.jpg`): NEGATIVE for detail.**
  detail_bands (400 v2 avatars): dl3 0.84/0.64/0.59/0.72 | 211, dl3F 0.83/0.63/0.58/0.71 | 215 (same). Likeness v2 avatar
  0.214 vs 0.218, expression acc 0.569 vs 0.600 (+0.031 for dl3F; check against CI). Zoom: the two look the same; both are free
  of dl2's iris streaks, so the streak removal comes from the eye terms / full-res LPIPS, not from the flip fix. The latent
  flip damaged half the targets yet the student output barely changed: the student evidently learned the flipped latents as a
  separate domain that clean inputs never trigger. The fix stays (correct, free) but it is not a detail lever.
  dl5f (+FFL 2): 0.79/0.54/0.51/0.65 vs control 0.81/0.56/0.52/0.67 (700 avatars), likeness 0.205 / 0.152, expr 0.590: NEGATIVE,
  FFL dropped. Only the DINOv2 critic (dl5g) adds real detail so far.
- **dl5 final (2026-11-19 ~05:50 UTC, 700 v2 avatars):** dl5fg (critic 0.1 + FFL 2) 0.90/0.68/0.67/0.77 | 265, likeness 0.224 /
  0.157, expr **0.533** (vs dl5g 0.569, control 0.594): no extra detail over dl5g and costs expressions -> FFL dropped entirely.
  Iris chroma (klein 7.4): dl2 4.95, dl5c 6.08, dl5f 6.39, dl5g 5.62, dl5fg 5.96.
- **dl6 (user choice "Final recipe + weight test", ~06:10 UTC, `scripts/h200_dl6_n2.sh`, dev-duyhv-2):** from dl3F, 3000 steps,
  lr 1e-5, dl3 losses + fixed flip; GPU 0 dl6g1 critic 0.1 (candidate final regression stage), GPU 1 dl6g2 critic 0.2.
- **2026-11-19 ~06:50 UTC (user: "drop it"):** dl3_2s / dl3F_2s DMD finishing stages stopped at it ~1400 / ~1150 of 2200 (no result;
  every v2 DMD run so far softened detail; user chose not to spend the GPU time). DMD is out of the current pipeline.
- **dl6 (critic on dl3F) result:** dl6g1 (critic 0.1) detail 0.93/0.72/0.69/0.79 | 277 (700 v2 avatars; dl5g 0.92/0.70/0.68/0.78),
  likeness v2/real/selfie 0.237/0.160/0.170 (dl5g 0.221/0.157/-), expr 0.577 (dl5g 0.569), skin dark25 +32.2. **Best student so
  far** -> student for the decoder fine-tune (plan rule). dl6g2 (critic 0.2) pending.
- **Phone-decoder fine-tune (user "go ahead", `scripts/train_decoder.py`, `scripts/h200_dec_n2.sh`, launched on dev-duyhv
  ~06:55 UTC with STUDENT=dl6g1):** frozen dl6g1 latents (+ properly flipped photos) -> klein ORIGINAL 512 px PNG; only the
  TinyVAE decoder trains (encoder/student frozen; same decoder size/speed on the phone). dec1 = L1 + LPIPS(vgg, full 512);
  dec2 = + DINOv2 critic 0.1 from step 500; 10k steps, bs 16, lr 1e-4. Precedent: YONOS-SR (2401.17258), CCSR (2401.00877),
  sd-vae-ft-mse. New options (default off): `student_infer.py --decoder`, `expr_eval.py --decoder`, `dl_arch.load_vae(path=)`.
- **dl6g2 (critic 0.2) = best student (2026-11-19 ~07:00 UTC):** 0.96/0.79/0.77/0.86 | 320 (700 v2 avatars; dl6g1 0.93/0.72/0.69/0.79),
  likeness v2/real/selfie 0.243/0.161/0.170, expr 0.579, skin dark25 +32.2; zoom (`grids/zoom_dl6.jpg`): crisper stubble/hair
  strands/lids, no crosshatch. Fine detail now 77% of klein (54% without critic).
- **Decoder fine-tune, revised (user ~14:05 VN: "choose one and training on 2 GPU to prove the decoder fine-tune help"):**
  dl6g1 dec1/dec2 on dev-duyhv stopped before training. One run on BOTH dev-duyhv-2 GPUs (`scripts/h200_dec_ddp.sh`, torchrun,
  DDP added to `train_decoder.py`: --bs = global batch split over GPUs, eval/save on rank 0, critic-head grads all-reduced;
  backup `train_decoder.py.bak_1gpu`): student dl6g2, decoder `decg2` = L1 + LPIPS(full 512) + DINOv2 critic 0.1 from step 500,
  10k steps, global bs 16, lr 1e-4. Proof = student_dl6g2 (original decoder) vs student_dl6g2_decg2 on the same eval sets.
- **dl6g4 = critic 0.4 on 2 GPUs (user 2026-11-19 ~14:55 VN: "do stronger critic 0.4", one run split over both dev-duyhv GPUs).**
  `train_dl.py` now supports torchrun DDP (--bs = global batch split over GPUs; task tokens and critic-head grads all-reduced by
  hand; val/samples/save on rank 0 through the raw, non-DDP UNet; collective NaN abort; single-process behaviour unchanged;
  backup `train_dl.py.bak_ddp`). `scripts/h200_dl6g4.sh`: 2-GPU 60-step smoke first, then dl3F + critic 0.4, 3000 steps,
  global bs 16, lr 1e-5, dl6g2 recipe otherwise; evals split over both GPUs; sheets vs dl6g1/dl6g2 (`grids/zoom_dl6g4.jpg`).
- **Decoder fine-tune RESULT (decg2 on dl6g2, 2026-11-19 ~08:00 UTC): PASS, the largest detail gain of the project.**
  | (v2/real/selfie) | coarse/mid/fine/finest, lapvar (900 v2 avatars) | likeness v2 / real / selfie | expr acc | skin dark25 |
  |---|---|---|---|---|
  | dl6g2 + original TinyVAE decoder | 0.96/0.78/0.76/0.86, 328 | 0.243 / 0.161 / 0.170 | 0.579 | +32.2 |
  | **dl6g2 + decg2 (fine-tuned decoder)** | **1.01/0.94/1.01/0.99, 428** | **0.248 / 0.165 / 0.179** | 0.583 | +32.5 |
  Zoom (`grids/zoom_decg2.jpg`): crisp eyelids/eyeliner, brown irises, white catch-lights now drawn, clean stubble texture, no
  streaks/crosshatch. Fine-band energy reaches klein's level with clean edges (not artifact energy as with dl2). Same 1.22M-param
  decoder (phone cost unchanged). Acceptance rule met on every item. Training: `logs/decddp_decg2.log`, val LPIPS vs klein
  0.2236 -> 0.2135. Not changed: face shape / eye size / pose (the decoder sharpens what the student draws).
- Infra 2026-11-19: dev-duyhv's 2 GPUs (h200-120-fleet) faulted ~07:19 UTC during the first 2-GPU NCCL init of dl6g4 ("GPU
  requires reset", Channel/TPC repair pending); resubmits fail with "executor error" (same GPUs) -> host reset needed. dl6g4 not run.
  dev-duyhv-3 (1 GPU, h200-gpu-fleet) stayed in "pulling" then stopped. SSH to dev-duyhv-2 lost ~07:55 UTC ("Can't connect to the
  remote host") while its jobs kept running (results read via NFS).
- **Core ML export of the best model, dl6g2 + decg2 (user request 2026-11-19).** `scripts/export_coreml_dl.py` gained `--decoder <tag>`
  (loads `weights/decoder/<tag>`; output `weights/student/<tag>/coreml_<decoder>/`, backup scratchpad `export_coreml_dl.py.bak_dec`).
  Run on the local CPU (shared box, 2974 s). Equivalence vs `dl_arch.forward` with the decg2 decoder: mean 0.0000/255, max 0.015/255.
  fp16 785.7 MB, w8 394.5 MB (w8 quality unverified), same I/O contract as earlier packs (image 512 RGB + task Int32 0–21).
  Pack: `release/2026-11-19_dl6g2_decg2/` (+ `README_iOS.md`, `reference/`) and `release/2026-11-19_dl6g2_decg2_coreml.zip` (1.1 GB).
  Phone latency/quality of this pack not measured yet (decoder same size as stock TinyVAE, so speed should equal the dl1 pack).
- **w8 vs fp16 check of the dl6g2+decg2 Core ML pack (user request 2026-11-19), `scripts/w8_check.py` (GPU 3).** Core ML cannot run
  on Linux, so the script reads the real int8 weights + per-channel scales from the w8 .mlpackage (514 constexpr_affine_dequantize
  tensors, all mapped to torch weights) and runs the export graph in fp16 compute; 40 synthetic faces × 7 tasks = 280 images.
  * fp16 vs trained fp32: PSNR 50.8 dB, LPIPS 0.0003, lapvar 370.3 vs 370.2 → **fp16 = identical in practice**.
  * w8 vs fp16: PSNR 34.2 dB (median 34.0, 5th pct 28.9, worst 23.3 on face_027 thumbs), LPIPS 0.008, lapvar 364.9 (−1.5%).
    By task (PSNR / LPIPS): avatar 38.3/0.005, sticker 36.4/0.005, happy 35.1/0.006, love 33.6/0.007, cool 34.5/0.007,
    thumbs 30.9/0.012, wave 30.3/0.016. Visually the same picture; differences are 1–3 px shifts of edges/pose, largest on gestures
    (`grids/w8_check_dl6g2_decg2/`). The 22 task embeddings were also quantized (per-task scale: mean relative error 18%, small
    values rounded to 0), a likely cause of the gesture drift. Results `results/w8_check_dl6g2_decg2.json`.
- **Infra (2026-11-19 ~12:00 UTC):** dev-duyhv resubmitted on h200-120-fleet → error again (GPUs there still need a reset). Resubmitted
  on **h200-gpu-fleet** (`dstack/dev_env_gpufleet.yaml`, same config + `fleets: [h200-gpu-fleet]`) → running, SSH OK, 2× H200 NVL
  healthy, venv + NFS read/write OK. SSH to dev-duyhv-2 (same fleet) works again too (both GPUs idle). 4 H200 usable.
- **"go all" batch (user 2026-11-19 ~16:30 UTC): dl6g4, w8 re-export, ideas A/B/C. Status at launch:**
  * **w8e16** (`scripts/requant_w8.py`, from the fp16 package, task embeddings kept fp16, 66 s CPU): 412 MB. w8_check vs fp16:
    PSNR 35.9 dB (first w8 34.2), LPIPS 0.006 (0.008), thumbs/wave 34.4/33.4 dB (30.9/30.3). Added to the release pack + zip.
  * **C root cause (skin):** skin_compare dL = L*(out) − L*(in), so positive = LIGHTER (my earlier "darkens" wording was wrong).
    dl6g2 matches klein on synthetic sets (v2 dark-quartile +3.1 vs +3.1; selfie +4.7 vs +5.7) but on real photos lightens the
    darkest quartile +32 (klein_personal +6.7). Cause: real-photo training pairs (CelebA/FFHQ) have klein targets +40/+31 L*
    lighter for input skin L* <40 / 40–50 (brightness filter bypassed 2026-11-18). Fix run **dl6g2s** = dl6g2 recipe +
    `--drop-list` (new train_dl flag; `scripts/make_skin_drop.py 55 15`: 6718 pairs with input L* < 55 and dL > 15, 97% real).
  * **B profile** (FlopCounter, 512² diptych, 683 GFLOPs): down 33.5%, mid 8.7%, up 57.2% → down+mid (42%) shareable across a
    sticker set. `dl_arch.install_shared_trunk` (18 down/mid cross-attn modules read one learned context; shared_ctx.pt next to
    unet/, auto-installed by load_unet). Exactness test `scripts/test_dl_AB.py`: shared=own token → max diff 0.0; mid-block
    features identical across tasks. Run **dlB**: from dl6g2, `--shared-trunk`, lr 2e-5, 6000 steps (no compile).
  * **A** (`train_dl --train-decoder decg2 --wide-latent 16 --pix-l1-w 1`): UNet conv_out 4→16 (extra = 0.1×std random),
    TinyVAE decoder input 4→16 (extra weights 0) → starts equal to dl6g2+decg2; UNet + decoder trained jointly (dec lr 1e-4).
    Runs **dlA16** (wide) and control **dlA4** (joint, 4 ch; queued after dl6g4 via `scripts/chain_A0.sh`).
  * dl6g4 single-GPU (`scripts/h200_dl6g4_1g.sh`). dev-duyhv now on h200-gpu-fleet (`dstack/dev_env_gpufleet.yaml`).
    Compile on dev-duyhv-2 needs node-local TRITON_CACHE_DIR (NFS triton-cache not writable): set in `scripts/h200_idea.sh`.
- **Literature check for ideas A/B (2026-11-19, web search, not exhaustive).**
  * A (wide latent + joint decoder) is **largely covered**: LCUDiff (arXiv 2602.04406) upgrades a one-step model's latent 4→16
    channels, keeping the first 4 aligned and using the extra ones for high-frequency detail; REPA-E (ICCV 2025) trains VAE +
    diffusion end-to-end; Mage-Flow (arXiv 2607.19064) jointly tunes a one-step encoder–decoder; MobileDiffusion uses an 8-ch
    latent + slim decoder; YONOS-SR/CCSR fine-tune the decoder. Our variant (student emits extra channels, encoder input stays
    4 ch, identity start, phone decoder) is incremental → keep A as a product lever, not a paper headline.
  * B (task-independent down/mid shared across a closed menu) has **no direct match found**. Closest: Faster Diffusion (NeurIPS
    2024) reuses UNet-encoder features across *timesteps* and runs decoder passes in parallel; DeepCache / Clockwork (CVPR 2024)
    reuse low-res features across steps; arXiv 2508.21032 shares early steps across similar prompts; FastEdit reuses the
    source image for many edit texts. Angle: in a ONE-step student step-caching gives nothing, so the task axis is the
    remaining reuse axis; needs a quality-vs-phone-speed result to stand up.
- **B follow-ups queued (user "queue B", 2026-11-19 ~17:15 UTC; `scripts/chain_B2.sh`, dev-duyhv GPU 1 after dl6g2s):**
  training-free baselines `dl6g2_tf_dm` / `dl6g2_tf_d` (dl6g2 weights symlinked + shared_ctx.pt = avatar token, scope
  down_mid / down; no retraining) and `dlBd` (B retrained sharing the down blocks only, 6000 steps, lr 2e-5). New:
  `install_shared_trunk(scope=...)`, `train_dl --shared-scope`, `h200_idea.sh` EVAL_ONLY=1. Then (CPU, after dlB results)
  a two-part Core ML export (trunk once per photo + per-option head) for the phone timing.
  * Process note: I edited h200_idea.sh in place while dlB/dlA16 were executing it (bash reads scripts incrementally).
    Restored the original bytes in the running inode and moved it to `h200_idea_running_20261119.sh`; the edit lives in a new
    file. Rule: never edit a script in place while it runs; write a new file and mv.
- **Results of the "go all" batch (all done by 19:00 UTC 2026-11-19).** Detail = detail_bands in ONE call per comparison (n=900;
  absolute values shift with the file set, so only compare within a call). Baseline dl6g2+decg2: fine 1.01, lapvar 428,
  expr acc 0.583, sticker id top-5 0.358, avatar id v2/real/selfie 0.248/0.165/0.179, real dark-quartile dL +32.5.
  | run | detail (fine, lapvar) | expr acc | sticker id top5 | avatar id v2/real/selfie | verdict |
  |---|---|---|---|---|---|
  | dl6g4 (critic 0.4) +decg2 | +8% vs dl6g2+decg2 in its call (1.04/416 vs 0.96/372) | 0.527 | 0.375 | .247/.165/.173 (no dec) | reject: expr −0.05 |
  | dl6g2s (skin drop list) | = dl6g2 | 0.575 | 0.358 | .240/.160/.172 | **C failed**: real dark25 +30.6 vs +32.2 |
  | dlA16 (wide 16, joint) | 0.85 / 368 | 0.563 | 0.333 | .242/.159/.176 | reject: less sharp than decg2 |
  | dlA4 (joint, 4 ch control) | 0.85 / 373 | 0.585 | 0.338 | .242/.160/.180 | = dlA16 → widening adds nothing |
  | **dlB** (shared down+mid, 42%) +decg2 | 0.99 / 417 | 0.560 | **0.385** | .247/.162/**.189** | near parity, ~1.67× per 22-set |
  | **dlBd** (shared down, 33.5%) +decg2 | 1.00 / 423 | **0.579** | **0.385** | .247/.161/.184 | parity, ~1.47× per 22-set |
  | dl6g2_tf_dm (no retrain, down+mid) | = baseline (avatar unchanged) | **0.315** | 0.575 | = baseline | collapse: stickers ≈ avatar |
  | dl6g2_tf_d (no retrain, down) | = baseline | 0.571 | 0.379 | = baseline | nearly works without training |
  * B finding: in the trained editing student the DOWN path carries almost no instruction (training-free sharing −0.012 expr),
    the MID block carries a lot (training-free −0.27); training makes mid shareable (0.315 → 0.560). dlB visual check
    (grids/expr_eval_dlB_decg2.jpg): all 20 options distinct incl. gestures, hearts on "love", sunglasses, party hat.
  * A: joint UNet+decoder training (L1+LPIPS+critic, 3000 steps) loses sharpness vs the separately fine-tuned decg2, and the
    extra 12 channels change nothing (dlA16 = dlA4). Dropped.
  * C: dropping the 6718 over-lightened pairs for 3000 steps from dl3F does not undo the lightening (dl3F learned it in 16k
    steps; 5664 darker pairs remain, many still +5..+15). Next candidates need a decision (see chat).

### 2026-11-19 — paper tasks on dev-duyhv (user: dev-duyhv = paper, dev-duyhv-2 = production)
- `scripts/influence_map.py` (new, training-free): every cross-attention module (36) reads either its option token or a neutral
  context (avatar token / mean token). Configs: prefix-k in execution order (= shareable trunk), leave-one-out, per group.
  Metrics: LPIPS change vs full, option separation (pairwise LPIPS among the 20 expressions), CLIP 20-way expression accuracy,
  exact shareable-FLOPs fraction + per-set speedup. 16 synthetic faces x 22 options.
- `dl_arch.install_half_head(unet, margin)` (new; auto-installed from `half_head.json` next to `unet/`): from up_blocks.1 on
  (no self-attention) compute only the output half of the diptych + `margin` columns. Backups: scratchpad `*.bak_halfhead`.
- `scripts/halfhead_check.py` (new): error of the half head vs margin, split into conv-padding error (GroupNorm stats injected
  from the full run) and the shipped variant (local GroupNorm), fp32, 8 faces x 22 options; FLOPs + per-set speedup.
- Launched `scripts/paper_chain1.sh` on dev-duyhv: GPU 0 influence (dl6g2 avatar, dl6g2 mean, dlBd), GPU 1 half-head (dlB, dlBd, dl6g2).
  Logs: logs/paper_influence.log, logs/paper_halfhead.log.
- Fixes on the way: influence_map FLOPs hook returned a value (replaced attn2 args) -> returns None; CLIP get_*_features return
  output objects in this transformers -> use text/visual_projection(pooler_output). Re-run as scripts/paper_chain1b.sh.
- **Half-head exactness (results/halfhead_check_{dlB,dlBd,dl6g2}.json, fp32, 8 faces x 22 options):** with GroupNorm statistics
  injected from the full run the error vanishes once margin >= 4-6 (PSNR 83-145 dB) -> 3x3 convs at the cut are NOT the problem;
  shipped variant (local GroupNorm) PSNR 26.2 (m=0) .. 33.7 (m=16), LPIPS 0.10 .. 0.015 -> all the error is GroupNorm statistics,
  so it needs fine-tuning. FLOPs (UNet 686.7 G): dlB trunk 295.9 G; head 390.8 G full width -> 299.3 G at m=0;
  per 22-option set vs 22 full runs: dlB 1.70x -> 2.20x (m=0) / 2.12x (m=4); dlBd 1.49x -> 1.86x (m=0).
- **Influence map dl6g2, neutral = avatar token (results/influence_dl6g2_avatar.json):** full model CLIP 0.641 / sep 0.208.
  All 14 down-path modules neutral: 0.625 / 0.188 (35% of FLOPs shareable training-free, 1.51x). Adding the 4 mid modules:
  0.328 / 0.040 (collapse). Per group neutral alone: down0 0.641, down1 0.634, down2 0.628, **mid 0.338, up0 0.141**, up1 0.650.
  -> the menu option acts almost only in the mid block + up_blocks.0 (lowest resolution, 16x32 tokens); up_blocks.1 cross-attention
  is unused. Matches the trained results (tf_d near-works, tf_dm collapses, dlB recovers with training).
- Queued scripts/paper_chain2.sh: trained half heads from dlB, 3000 steps lr 2e-5, dlB losses: hhB0 (m=0, GPU 1), hhB4 (m=4, GPU 0).

### 2026-11-20 — production quick wins on dev-duyhv-2 (user: "2a + 2b RGBA decoder and wider decoder; 3a 2-step avatar")
- **2b wider/deeper phone decoder** (`train_decoder.py --init decg2 --width W / --extra-blocks a,b,c,d`; `grow_decoder()`):
  diffusers DecoderTiny has ONE width for all stages (the post-upsample conv keeps the stage width), so "wider" = uniform.
  Growth is function-preserving (checked: max diff 0 vs decg2): new channels read old+new, old never read new; extra blocks
  start as identity. Cost at 512 px: decg2 1.22M / 141 GFLOPs; **decW** (width 80) 1.91M / 221 G; **decD** (+1 block per
  stage) 1.67M / 218 G (equal-cost width-vs-depth test). **decC** = decg2 + the same 5000 extra steps (control).
- **2a RGBA decoder** (`--alpha`): 4th output channel = cut-out alpha (starts at 0.5). Labels = BiRefNet_lite (metrics.Matter,
  the current phone cut-out) on what decg2 renders from the dl6g2 latent (so alpha matches the rendered content), 8000 random
  training pairs x {plain, flipped} + val, cached `weights/decoder/_alpha_dl6g2_decg2_manifest_v2_8000.npy`. Loss L1 + 0.5 BCE.
  Eval `scripts/alpha_eval.py`: decoder alpha vs BiRefNet on the same RGB (MAE, IoU, boundary MAE, sticker_postprocess pass).
- **3a two-step avatar** (`train_dl.py --refine-sigma s --lora-rank 64 --only-tasks avatar`, `dl_arch.refine_forward`):
  step 1 = dl6g2 with the LoRA adapter OFF (unchanged), x = (1-s) x0 + s eps (fixed noise seed 1235), step 2 = UNet + adapter
  at sigma s. Tokens frozen. student_infer.py: avatar = 2 steps, other options = dl6g2 (smoke check: stickers identical,
  mean abs diff 0.000). Phone: base UNet + a small adapter (Core ML multifunction), avatar costs 2 UNet runs.
  Runs: **rf50** (s=0.5), **rf30** (s=0.3), 3000 steps, lr 1e-4, dl6g2 losses.
- Smoke tests passed (scripts/smoke_prod1*.sh). Launched `scripts/prod_chain1.sh` on dev-duyhv-2 (GPU 0: decA/decC/decW/decD
  together; GPU 1: rf50 + rf30 together), then expr_eval, v2/real/selfie quality, skin, detail_bands, iris, zoom sheets.
- Paper: hhB0 (dlB + half head m=0, 2.2x per set) expr_eval: expression 0.552, sticker id top-5 0.377 (dlB 0.560 / 0.385).
- **hhB4** (half head m=4): detail 1.01/425, expression 0.558, sticker id top-5 0.331 (hhB0 0.377), avatar id .239/.166/.182,
  dark25 +32.2 -> margin adds nothing; **hhB0 (m=0) is the half-head to ship**. User 2026-11-20: paper dropped; keep the
  paper ideas that improve production.

### 2026-11-20 — production ideas from the paper work, on dev-duyhv (`scripts/prod_chain2.sh`)
- **#4 small style packs** (`train_dl.py --lora-scope mid_up0`, `--tokens-only`): the influence map says the option acts in
  mid_block + up_blocks.0, so try a new style there only. Anime data (1,482 train pairs, avatar + sticker), base = production
  dl6g2, 6000 steps, dl_an recipe: an_all (LoRA r32 everywhere, 25.9M params ~52 MB fp16), an_mu (LoRA r32 mid + up0 only,
  13.5M ~27 MB: these are the widest layers, so only ~2x smaller), an_tok (2 tokens only, ~3.4 MB). Eval v2 + real anime
  quality, skin. Old dl_an (dl1 base) as reference.
- **#5 remove unused cross-attention** (`--drop-attn2 down,up1`, `dl_arch.install_drop_attn2`, saved as drop_attn2.json):
  20 modules (all down blocks + up_blocks.1). FLOPs (`scripts/flops_drop.py`, results/flops_drop_down_up1.json): hhB0
  595.2 -> 532.4 GFLOPs per image (-10.6%), per 22-option set 6,882 -> 6,542 (-4.9%; the down path is already shared).
  dropX = hhB0 + drop, 3000 steps lr 2e-5, dl6g2 losses, full evals (h200_idea.sh).

### 2026-11-20 — Plan A (smaller student) queued; Plan B (Sana-Sprint base) planned, not started
- User: "queue A first if GPU available". **A1** `scripts/block_scan.py` (training-free): each removable unit of dl6g2 is
  bypassed alone (output := input; 36 transformer layers, 13 attention modules, 6 down/mid resnets with matching channels;
  up resnets consume skip features -> handled when building the model) and scored on 12 synthetic faces x 22 options
  (LPIPS change, option separation, CLIP expression acc) with its exact FLOP share; ranking = change per GFLOP saved.
  CPU check: bypass hooks work and restore the exact output when removed. Queued by `scripts/queue_A1.sh` on dev-duyhv
  (starts on the first GPU that stays free for 60 s). Next: A2 build S70/S50 students from the ranking (+ up-path resnet
  removal with matching skips), A3 distil from dl6g2 (output + feature matching + our losses), A4 add shared trunk + half
  head, A5 evals + Core ML size.
- Plan B (Sana-Sprint 0.6B, weights cached; Sana weights listed as Apache-2.0 in the 2026-11-17 base research, legal to
  confirm): B0 feasibility first (licence, DC-AE decoder cost at 512, Core ML conversion, size) before any training.

### 2026-11-20 ~02:50 UTC — results of the production quick wins + Plan A
- **3a 2-step avatar: NO.** detail_bands (v2 avatars, vs klein) dl6g2+decg2 1.01/0.94/1.01/0.99 | 428; rf50 0.98/0.91/0.98/0.95
  | 409; rf30 0.98/0.91/0.98/0.94 | 410. Likeness rf30 .248/.163/.181 (= dl6g2), skin dark25 +32.5. The second step softens
  slightly: the decg2 decoder already recovered the detail the earlier 2-step gain was about. #3 (2-step stickers) dropped too.
- **#5 dropX: not worth it.** hhB0 minus down + up_blocks.1 cross-attention: detail 1.00/422 (= hhB0), expression 0.531
  (hhB0 0.552), sticker id top-5 0.354 (0.377), likeness .239/.160/.178 (.242/.163/.184), for -10.6% per image / -4.9% per set.
- **2a decA (RGBA decoder): YES.** val RGB LPIPS 0.2137 (decg2 0.2135), alpha vs BiRefNet_lite IoU 0.977 / MAE 0.015 (val).
  On real outputs (alpha_eval, same RGB): stickers IoU 0.959, MAE 0.024, boundary MAE 0.31; avatars IoU 0.995; sticker
  post-process pass 100% with the decoder's own alpha (= BiRefNet 100%). -> the phone can drop BiRefNet.
- **2b decW / decD: no meaningful gain** so far. expr_eval: decg2 0.583/0.358, decC (control) 0.588/0.350, decW 0.596/0.344,
  decD 0.585/0.362; avatar likeness identical (.248-.249/.164/.178-.180). detail_bands still running.
- **#4 style packs (anime, base dl6g2): mid+up0 pack WINS.** v2 avatar/sticker id, style_dino avatar/sticker; real id; skin:
  an_all (99 MB fp32 / ~50 MB fp16) .408/.346, .578/.527; real .266/.264; dL 8.6, dark25 3.9.
  **an_mu (52 MB fp32 / ~26 MB fp16) .417/.347, .580/.531; real .297/.284; dL 9.0, dark25 4.5** -> equal or better at half size.
  an_tok (tokens only, 6.6 MB fp32) .439/.348, .580/.525; real .257/.209 (sticker id low); dL 9.8, dark25 7.5 -> needs a visual check.
  old dl_an (dl1 base) .400/.333, .575/.529.
- **Plan A1 block scan** (`scripts/block_scan.py`, results/block_scan_dl6g2.{json,txt}; FLOP lookup fixed: FlopCounterMode names
  modules '<Class>.<path>'): single removals up to ~40% of FLOPs each cost dev <= 0.041 and keep CLIP ~0.60-0.64; the next
  units are mid_block / up_blocks.0 attention (CLIP 0.24 / 0.42, = the influence map). Sets (configs/prune_sets.json, greedy
  by dev per GFLOP): **S70** 9 units, -197 GFLOPs (-29%), 390M -> 299M params; **S60** 15 units, -274 GFLOPs (-40%), -> 241M.
  `dl_arch.install_prune` (prune.json auto-load), `train_dl.py --prune`. Launched `scripts/planA_train.sh` on dev-duyhv:
  pS70 (GPU 0), pS60 (GPU 1), from dl6g2, 4000 steps, lr 2e-5, dl6g2 losses, full evals.
