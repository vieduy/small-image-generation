# cartoon3d phone student: how dl6g2 and dmC were made (archive, 2026-11-28)

Two one-step DreamLite-mobile students for chat avatars and stickers (cartoon3d style), same architecture and phone cost:
- **dl6g2 + decg2**: shipped 2026-11-19 (Core ML `release/2026-11-19_dl6g2_decg2/`). Trained directly on klein-4B pairs.
- **dmC + decg2**: 2026-11-28 candidate. Distilled from a DreamLite-base teacher, with clean gesture data and a DMD2 stage
  (Core ML `release/2026-11-28_dmC_decg2/`).

Archived weights: `archive/2026-11-28_cartoon3d/` (see "Archive" below). Every number here comes from the README log and `results/`.

## Results

| cartoon3d (eval sets: 24 expression faces, 400 synthetic v2 / 300 real / 200 selfie avatars, 100 ghost-test faces) | dl6g2 + decg2 | **dmC + decg2** |
|---|---|---|
| Expressions all / face-only / outside the face (eval_e1, CLIP 20-way) | 0.590 / 0.734 / 0.493 | **0.710 / 0.818 / 0.639** |
| "OK" gesture accuracy | 0.08 | **0.67** |
| Extra / ghost hand (OWLv2, thumbs / wave / OK) | 0.3% | 0.7% (teacher 1.7%) |
| Missing / blob hand | 12% | **1.7%** |
| Avatar likeness (SFace) synthetic / real / selfie | **0.248 / 0.164** / 0.179 | 0.215 / 0.155 / **0.184** |
| Sticker identity (top-5 of 100) | **0.346** | 0.329 |
| Sharpness vs klein | 0.64 | **0.73** |
| Phone cost | 1 UNet pass (390M) + TinyVAE | same |

Open: dark-skin evaluation on real photos is confounded by lighting (README 2026-11-26 s9); both models lighten very dark skin to some degree.

## Flow A: dl6g2 + decg2 (klein-distilled)

| Step | What | Settings | Script |
|---|---|---|---|
| A0 | klein-4B pairs (photo -> sticker), cartoon3d, `manifest_v2.jsonl` | 53,930 pairs: avatar, sticker, 20 expressions; R1/R3 filters, proper flips | `gen_pairs_v2.py`, `build_flip_cache.py` |
| A1 | **dl3F** from DreamLite-mobile | 16,000 steps, bs 16, lr 2e-5, latent MSE + LPIPS + 512-px crop LPIPS + eye loss, proper flip | `train_dl.py` |
| A2 | **dl6g2** = dl3F + DINOv2 realism critic | +3,000 steps, lr 1e-5, critic weight 0.2 | `train_dl.py` |
| A3 | **decg2** = phone decoder fine-tuned on dl6g2's outputs | 10,000 steps, bs 16, lr 1e-4, L1 + LPIPS (full 512) + critic 0.1 from step 500; target = klein PNG | `train_decoder.py` |

## Flow B: dmC + decg2 (teacher-distilled)

| Step | What | Settings | Script | Cost (1x H200) |
|---|---|---|---|---|
| B1 | **Teacher tB1**: DreamLite-base (28-step) fine-tuned on the A0 pairs | 20,000 steps, bs 32, lr 1e-5, EMA 0.999, 5% condition dropout | `train_teacher_dl.py`, `s1_teacher.sh` | ~3.5 h |
| B2 | **Teacher setting**: 8 steps, image guidance 3 on the FIRST step only, text guidance 2.5 | chosen by sweeps: constant strong guidance -> photo texture (tS1), weak -> identity loss (tS3) | `s7_guidance_sweep.py`, `s8_attr_sweep.py`, `sample_gs` | ~30 min |
| B3 | **Teacher pairs** with the phone's fixed noise (seed 1234), plain + mirrored photo | 53,930 x 2 views | `s8_teacher_pairs.py` (+ `s8_teacher_pairs_rev.py` on a 2nd GPU) | ~3 h (1.7 h on 2) |
| B4 | **Detail-consistency filter**: drop pairs with face detail > 1.5x klein's | 81 dropped (0.15%) | `s8_consistency.py` | ~10 min |
| B5 | **tS4**: student from dl6g2 on the teacher pairs | 18,000 steps, lr 2e-5, A1/A2 losses + critic 0.2 (6k steps was under-trained) | `train_dl.py`, `s11_chain.sh` | ~3 h |
| B6 | **Best-of-N gesture pairs** (thumbs, wave, OK, think, facepalm) | candidates = target + 3 noise seeds; keep the one with exactly 1 solid hand ON THE ORIGINAL SIDE and the best CLIP gesture score; drop undecided rows (21%) | `s31_best_of_n.py`, `s31_merge.py` | ~40 min (2 GPUs) |
| B7 | **tS7** = tS4 + regression on the cleaned pairs | 6,000 steps, lr 2e-5 | `train_dl.py` | ~1 h |
| B8 | **dmC** = DMD2 stage from tS7 | real score = tB1 (B2 setting), fake score = trainable tB1 copy (lr 5e-6, 2 updates/step, 200 warm-up); regression terms x 0.25; critic 0.2; 4,000 steps, lr 5e-6 | `train_dl_dmd.py`, `s31_chain.sh` | ~1.8 h |
| B9 | Decoder: **decg2** reused (a tS4-specific decoder, decT4, gave no gain) | | | |
| B10 | Evals (fast form) + Core ML export | | `s21_eval_fast.sh`, `export_coreml_dl.py`, `requant_w8.py`, `s32_export_dmC.sh` | ~15 min + ~1 h CPU |

## What we learned (why the flow looks like this)

1. **Teacher guidance** trades identity against cartoon style; image guidance on the first step only keeps identity cues without photo texture.
2. **Train long enough**: 6k steps left hands / props under-trained; 12-18k needed.
3. **Ghost hands** came from the teacher's UNSTABLE hand side (it flips under 8 px input shifts for ~30% of photos; the one-step student
   blends both sides there; confirmed 24x, p < 1e-4). Fixed by cleaning the gesture data while keeping the side (B6).
4. **DMD2** sharpens and lifts expressions and sticker identity; alone (without B6) it resolves the side ambiguity by drawing two hands.
5. **Did not help**: own decoder for tS4 (decT4), hand-crop loss (tS5), forcing one fixed hand side (gate failed: 53% rows lost),
   pose-map conditioning (tSP0/b/c: controls the main hand, but adds a second hand), starting from DreamLite-mobile instead of dl6g2 (tS4o = tS4).
6. Data quantity was not the limit (adding 59k real-photo teacher pairs, tS2, changed nothing); data QUALITY was (B4, B6).

## Archive (`archive/2026-11-28_cartoon3d/`)

| Folder | Contents |
|---|---|
| `dmC/` | `unet/`, `task_emb.pt`, `config.json` (full student, loadable with `dl_arch.load_unet`) |
| `dl6g2/` | same for dl6g2 |
| `decg2/` | phone decoder (AutoencoderTiny) used by both |
| `teacher_tB1/` | teacher `unet/` (EMA, fp16) + `tokens.pt` (needed to regenerate B3 / B6 pairs or rerun B8) |
| `FLOW_cartoon3d_dmC.md` | this document |
| `metrics.json` | the numbers in the Results table, with their source files |

Large data (pairs, caches, outputs) stays in `data/pairs/cartoon3d/` (teacher pairs `teacher_tB1_s8_img3.0_k1*.pt`, drop lists, best-of-N shards).
