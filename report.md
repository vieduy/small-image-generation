# Small phone model for avatars and stickers: final results (overnight run 2026-11-16 → 17)

All server numbers come from H200 GPUs. The only phone measurement is the user's: about 0.3 s per image on an iPhone 14 Pro, measured on the s5d build. That build has the same architecture and the same size as every model below. The new builds have **not** been timed or quality-checked on a phone yet.

## 1. Bottom line

- **A new base, `b1`, replaces s5d.**
  - It was trained on 2.4× the data: synthetic faces, synthetic selfies, real photos (legal-approved, CelebA-derived) and 20 expressions.
  - On real photos it now produces clean, klein-quality avatars. The old base often produced broken, washed-out faces.
  - Dark-skin lightening on real photos fell from +12.2 to +8.6, and to **+6.3 after DMD**, which is better than klein's +6.7.
- **The DMD training recipe is final.**
  - It is DMD plus two colour anchors: an eye-colour term on chroma only, and a low-frequency colour term.
  - It makes outputs crisper (eyes, lines, edges) without likeness loss on cartoon3d and anime, and it improves skin-tone accuracy.
  - It does not change the phone model: still one step, SDXS 0.33B, a 318 MB int8 build, and a style pack of about 25 MB fp16 that is merged into the build.
- **Final lineup:**
  - `c3_dmd` (cartoon3d) and `an_b1_dmd` (anime) use DMD.
  - Flat_vector ships **without** DMD (`fv_b1`), because DMD cut its likeness from 0.60 to 0.41. That is a negative result, reported below.

## 2. Final lineup (v2 synthetic eval set; likeness = identity top-5 within galleries of 100)

| Style | Final model | Avatar likeness | Sticker likeness | klein-4B avatar / sticker | Skin dL, all faces / darkest quarter (klein) | Sharpness (klein) | Phone build (int8 / fp16) |
|---|---|---|---|---|---|---|---|
| cartoon3d | **c3_dmd** (b1 + DMD pack) | 0.247 | 0.255 | 0.247 / – | +5.5 / +2.9 (klein +6.3 / +3.1) | 104 (205) | `weights/student/c3_dmd/coreml/` 318 / 634 MB |
| anime | **an_b1_dmd** | 0.902 | 0.80 | 0.723 / 0.77 | +9.0 / +5.5 (klein +14.5 / +6.7) | 1097 (1929) | `weights/student/an_b1_dmd/coreml/` |
| flat_vector | **fv_b1** (no DMD) | 0.598 | 0.545 | 0.50 / 0.515 | +6.3 / +1.2 | 842 | `weights/student/fv_b1/coreml/` |
| flat_vector, alternative | fv_b1_dmd | 0.412 ❌ | 0.37 ❌ | | +6.3 / +0.8 | 1123 | not exported |

- **Sharpness** is the median Laplacian variance of the avatar outputs. It is only a rough guide (see the decoder test in the README).
- **Sticker pass rate** is at least 0.996 for every final model (fv_b1 0.996, the others 1.0). The WebP files average 15–24 KB against the 100 KB limit.
- **Expressions:** `b1` carries 20 expression tokens. Its CLIP expression accuracy is 0.544 (s7e 0.531, s5e 0.39, klein 0.73).
- **The cartoon3d likeness score is low for every model, klein included.** The style draws generic faces, so this metric can't separate the models here. Judge cartoon3d by eye.

**Image sheets** (photo | klein | final phone model):
- Avatars: `grids/final_cartoon3d_avatar.jpg`, `grids/final_anime_avatar.jpg`, `grids/final_flat_vector_avatar.jpg`.
- Stickers (transparent WebP on a checkerboard): `grids/final_*_sticker.jpg`.
- Real photos, cartoon3d (klein | old s5d | final): `grids/final_real_cartoon3d.jpg`.
- Face zooms, before vs after DMD: `grids/final_cartoon3d_zoom.jpg`, `grids/final_anime_zoom.jpg`, `grids/final_flat_vector_zoom.jpg`.

## 3. Selfie and real-photo sets (cartoon3d avatars)

| | klein | s5d (old) | b1 | **c3_dmd** |
|---|---|---|---|---|
| selfie likeness top-5 | 0.24 | 0.325 | 0.165 | 0.175 |
| real likeness top-5 | 0.15 | 0.35 | 0.14 | 0.12 |
| real style score (dino) | 0.862 | 0.736 | 0.852 | 0.863 |
| skin dL darkest quarter: selfie / real | 5.7 / 6.7 | 3.9 / **12.2** | 5.5 / 8.6 | **4.1 / 6.3** |

s5d's higher likeness on these sets came from **under-stylisation**: the photo leaked through, and many outputs were half-formed. The new models match klein's style strength, so they also match klein's likeness. Likeness in cartoon3d is limited by the teacher's style, not by the student. Raising it needs a teacher prompt or style change, for example more personal features in the cartoon.

## 4. What was tried tonight (in order)

1. **Base v1 `b1`.** Trained from the SDXS init for 16k steps (49 min) on 8,721 kept pairs; result as above.
2. **DMD run 4: colour anchors**, global colour plus a full-RGB eye term. It gave the best likeness (0.925) and skin (+6.3 / +4.3), but sharpness fell back to 931: the full-RGB eye term re-blurred the eyes. Not used.
3. **DMD run 5: eye term on chroma only**, with half the colour weight. Likeness 0.892, skin +5.9 / +3.5, sharpness 1084, crisp eyes with some iris colour. **Chosen as the recipe** (`configs/dmd_recipe.txt`).
4. **Production DMD on b1:**
   - cartoon3d: a new zero-start LoRA pack, 3,000 pairs.
   - anime and flat_vector: continuing the packs rebuilt on b1.
   - Each run took about 81 min on one H200; peak VRAM was 24.6 GB.
5. **Core ML export** of b1, c3_dmd, an_b1_dmd and fv_b1. About 2 min each on CPU. fp16 is 634 MB and int8 weights are 318 MB.

## 5. Negative results and limits (honest list)

- **flat_vector + DMD fails the likeness rule** (0.598 → 0.412).
  - Visually it is cleaner and closer to klein's flat look: fewer wrinkle lines, crisper edges. It flattens the very details the identity model relies on.
  - It is kept as `fv_b1_dmd` in case the flatter look is preferred by eye.
- **Iris colour is only partly restored in anime.** Eyes are often still dark brown or black where klein draws red, hazel or green.
- **Fine detail still trails klein:**
  - freckles, skin shine, hair highlights;
  - the stripe and shell patterns on animal stickers;
  - colour saturation on animal stickers, which is muted.
- **Earlier DMD runs:** run 1 was too gentle, run 2 was sharpest but had black eyes and lighter skin, and run 3 was worse than run 2. The README has the full numbers.
- **On-device:** the new Core ML builds have **not been run on a phone.** Core ML cannot run on Linux. On the server, the int8-weight simulation (`student_infer.py --fake-w8`, v2 set) matches fp16 to a mean of 0.98–1.24/255 (p95 ≤ 1.9/255) for all three final models. That is the same level as s5d (0.96/255) and not visible.
- **Real training data:** b1 and every model built on it were trained on 6,597 real CelebA-derived photos (`portrait_2_avatar` inputs; the upstream license is CC BY-NC 4.0). They were used under your legal team's approval. **Please confirm legal's sign-off covers training a shipped model on them.** If it doesn't, the fallback is a b1 retrain without the `r*` pairs, about 1.5 h.

## 6. Recommendation and next steps

- **Ship candidates for the phone test:** `c3_dmd`, `an_b1_dmd` and `fv_b1`, each as an int8 Core ML build.
  - Please time them on the iPhone 14 Pro and check the look against the sheets.
  - They share the b1 base. A multi-style app can ship one base plus about 25 MB per style pack, if we add LoRA loading in the app (iOS 18 multifunction models) instead of three full 318 MB builds.
- **Next quality steps, in order of expected value:**
  1. A stronger eye-colour fix for anime.
  2. A colour and saturation fix for animal stickers.
  3. Selfie and real-photo training pairs for anime and flat_vector. They currently have only 4,200 synthetic pairs each; this is a few hours of klein generation per style.
- **Teacher and student:**
  - The teacher stays FLUX.2-klein-4B (Apache-2.0).
  - The student stays SDXS-512 0.33B (OpenRAIL++). SD-Turbo (2.6× larger) gave no visible gain.
  - Nitro-E is still the "modern small base" option. It needs gated Llama-3.2-1B access and is not requested yet.
