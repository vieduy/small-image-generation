# E1 status: have we separated "depth after the option's entry" from "number of option inputs"?

Read-only audit, 2026-11-20 ~11:15 UTC. Nothing was trained, changed or deleted for this report.
Every number below was read from files in this repo. Every layer count was checked on the model itself (see §1.3).

## Summary (5 lines)

1. **No, E1 has not been measured.** No pair of runs varies only the self-attention depth after the entry point (a), and only one confounded pair varies only the number of option inputs (b).
2. The existing trained series (dlBd → dlB → frA2 → frUp1) moves the entry later. In doing so it changes all three variables at once:
   - option inputs: 22 → 18 → 4 → 6;
   - self-attention layers after the entry: 15 → 11 → 3 → 0;
   - entry resolution: up1 is at 32×64, everything else at 16×32.
3. What it shows: face-only edits survive down to 3 self-attention layers after the entry (frA2: 0.708 vs dlB 0.734) and collapse only with 0 (frUp1: 0.339). Outside-the-face edits are already at chance with 3 layers / 4 inputs (frA2: 0.073; chance 0.05). In dlB (11 layers / 18 inputs) they are 0.444.
4. The only (b)-type pair is hhB0 vs dropX (18 vs 12 inputs; same entry, same 11 self-attention layers after). Dropping the 6 up1 inputs costs little (0.552 → 0.531; outside the face 0.420 → 0.410). But it has three problems:
   - it is confounded (different init chain, modules removed instead of neutralised);
   - the inputs it drops all sit after the last self-attention;
   - it is a single seed.
5. To separate the two variables we need about 7 new 1-hour runs (§5), ideally plus a step-matched no-sharing control and a second seed.

## 1. Files

### 1.1 Code that defines which cross-attention modules read the option

| What | Path |
|---|---|
| Sharing mechanism + named scopes | `scripts/dl_arch.py`: `SHARED_SCOPES`, `install_shared_trunk()`, `_SharedCtxProc`; `install_drop_attn2()`; `install_half_head()` |
| Training flags | `scripts/train_dl.py`: `--shared-trunk`, `--shared-scope {down_mid, down, down_mid_up0, only_up0a2}`, `--drop-attn2`, `--half-head` |
| Launcher (recipe + all evals) | `scripts/h200_idea.sh` (running copy kept as `scripts/h200_idea_running_20261119.sh`) |
| Exactness tests | `scripts/test_dl_AB.py` (down_mid), `scripts/test_scopes_p2.py` (down_mid_up0, only_up0a2: shared = own token → max diff 0; first option-reading block's input identical across options) |
| Training-free influence map | `scripts/influence_map.py` → `results/influence_dl6g2_avatar.json`, `results/influence_dl6g2_mean.json`, `results/influence_dlBd_avatar.json` |
| Half-head exactness | `scripts/halfhead_check.py` → `results/halfhead_check_{dlB,dlBd,dl6g2}.json` |

### 1.2 Runs

| Run | Weights / config | Status of weights |
|---|---|---|
| dl6g2 (baseline, nothing shared) | `weights/student/dl6g2/` | present |
| dl6g2_tf_d, dl6g2_tf_dm (training-free sharing) | — | **deleted 2026-11-20 (storage cleanup group D)**; recipe from README; results kept |
| dlBd | `weights/student/dlBd/{config.json, shared_ctx.pt, train_log.jsonl}` | present |
| dlB | `weights/student/dlB/…` | present |
| hhB0 | `weights/student/hhB0/{…, half_head.json}` | present |
| hhB4 | — | **deleted** (group D); results kept |
| dropX | — | **deleted** (group D); results kept |
| frA2 | `weights/student/frA2/…` | present |
| frUp1 | `weights/student/frUp1/…` | present |

**Per-run evaluation files**, with `<tag>` = `<run>_decg2`:
- `results/expr_eval_<tag>.json`
- `results/quality_student_<tag>_{v2,real,selfie}.csv`, plus rows in `results/quality_summary.jsonl`
- `results/detail_<tag>.txt`
- `results/skin_compare_<tag>_real.json`
- image grid: `grids/expr_eval_<tag>.jpg` (exists for all 10)
- raw expression outputs: `outputs/expr_eval_<tag>/cartoon3d/`. These exist for dl6g2_tf_d, dl6g2_tf_dm, dlBd, dlB, hhB0, frA2 and frUp1, but **were deleted for hhB4 and dropX** (cleanup item 4).

**Logs:** `logs/idea_<run>.log`, `logs/paper_frUp1.out`, `logs/paper_frA2.out`.

### 1.3 How the layer counts were verified

- Forward pre-hooks on every `attn1` / `attn2` module of dl6g2 recorded the real call order: 60 calls, 36 cross-attention + 24 self-attention.
- Execution order: down0 → down1 → down2 → **mid** → up0 → up1. Note that the `named_modules()` order differs: there the mid block comes last.
- Inside each transformer layer, `attn1` (self) runs before `attn2` (cross).
- "Self-attention layers after the entry" = the number of `attn1` calls after the first option-reading `attn2`.

| Block | resolution (diptych) | self-attn layers | cross-attn modules |
|---|---|---|---|
| down0 | 64×128 | 0 | 2 |
| down1 | 32×64 | 0 | 4 |
| down2 | 16×32 | 8 | 8 |
| mid | 16×32 | 4 | 4 |
| up0 | 16×32 | 12 (3 attention modules × 4 layers) | 12 |
| up1 | 32×64 | 0 | 6 |

## 2. Run table

**Common to every trained run:**
- data `data/pairs/cartoon3d/manifest_v2.jsonl`, 53,786 train pairs;
- batch size 16, token lr 1e-3, seed 0 (**one seed each**);
- dl6g2 losses (lp512 1, eye 1, eye-LPIPS 0.25, critic 0.2), proper flip, tgt-cache;
- cosine schedule with 200 warm-up steps (`train_dl.py`);
- decoder for all evaluations: decg2.

| Run | Init → training | Option-reading cross-attn (count) | First option module | Self-attn after entry | Expr. acc. all / face-only / outside face | Sticker id top-5 | Avatar likeness v2 / real / selfie | Detail fine, lapvar | Dark-25% skin dL |
|---|---|---|---|---|---|---|---|---|---|
| dl6g2 | dl3F → 3000 steps, lr 1e-5, compile | all 36 | down0.a0.tb0 | 24 | 0.583 / 0.729 / 0.486 | 0.346 | .248 / .165 / .179 | 1.01, 428 (other call) | +32.5 |
| dl6g2_tf_d | dl6g2, **no training**; down 14 → avatar token | mid 4 + up0 12 + up1 6 = 22 | mid.a0.tb0 | 15 | 0.571 / 0.719 / 0.472 | 0.379 | = dl6g2 (avatar token unchanged) | n/a | n/a |
| dl6g2_tf_dm | dl6g2, **no training**; down+mid 18 → avatar token | up0 12 + up1 6 = 18 | up0.a0.tb0 | 11 | 0.315 / 0.646 / 0.094 | 0.575 (collapse inflates) | = dl6g2 | n/a | n/a |
| dlBd | dl6g2 → 6000 steps, lr 2e-5, scope `down` | 22 (mid, up0, up1) | mid.a0.tb0 | 15 | 0.579 / 0.745 / 0.469 | 0.385 | .247 / .161 / .184 | 1.00, 423 | +32.2 |
| dlB | dl6g2 → 6000 steps, lr 2e-5, scope `down_mid`* | 18 (up0, up1) | up0.a0.tb0 | 11 | 0.560 / 0.734 / 0.444 | 0.385 | .247 / .162 / .189 | 0.99, 417 | +32.9 |
| hhB0 | dlB → 3000 steps, lr 2e-5, half head m=0 | 18 | up0.a0.tb0 | 11 | 0.552 / 0.750 / 0.420 | 0.377 | .242 / .163 / .184 | 1.00, 422 | +32.9 |
| hhB4 | dlB → 3000 steps, lr 2e-5, half head m=4 (README; config deleted) | 18 | up0.a0.tb0 | 11 | 0.558 / 0.750 / 0.431 | 0.331 | .239 / .166 / .182 | 1.01, 425 | +32.2 |
| dropX | hhB0 → 3000 steps, lr 2e-5, `--drop-attn2 down,up1` (README; config deleted) | **12 (up0 only; up1 cross-attn removed)** | up0.a0.tb0 | 11 | 0.531 / 0.714 / 0.410 | 0.354 | .239 / .160 / .178 | 1.00, 422 | +31.8 |
| frA2 | dl6g2 → 6000 steps, lr 2e-5, scope `only_up0a2` | **4** (up0.attentions.2) | up0.a2.tb0 | **3** | 0.327 / 0.708 / **0.073** | 0.427 | .241 / .165 / .200 | 0.98, 419 | +32.2 |
| frUp1 | dl6g2 → 6000 steps, lr 2e-5, scope `down_mid_up0` | **6** (up1) | up1.a0.tb0 | **0** | 0.163 / 0.339 / **0.045** | 0.581 (collapse inflates) | .214 / .164 / .183 | 0.97, 412 | +31.8 |

\* dlB's `config.json` has `shared_scope: null` and its `shared_ctx.pt` stores `scope: None`. dlB was trained before the `--shared-scope` flag existed, when the only behaviour was down+mid. That matches the code default (`down_mid`) and hhB0's `shared_ctx.pt` (`scope: down_mid`, inherited from dlB). So the scope is inferred, not stored.

**How to read some cells:**
- **Detail** is the "fine" band and Laplacian variance from `results/detail_<tag>.txt`. Absolute values shift between calls, so compare only within a file.
- **Avatar likeness** is `id_cos_mean` (OpenCV SFace cosine, photo vs output) on avatars.
  - v2 = 100 `data/faces` + 300 `data/faces_eval2` synthetic faces;
  - real = 300 `data/real_p2a/eval`;
  - selfie = the selfie eval set.

**Not available / not verified:**
- No confidence intervals or second seeds for any run.
- For tf_d / tf_dm / hhB4 / dropX, the exact configs are gone; recipes are taken from README entries.

## 3. Definition of the face-only / outside-the-face split

- **Face-only (8):** happy, lol, cry, wow, sad, shy, wink, cool.
- **Outside the face (12):** love, angry, think, sleepy, thumbs, wave, ok, facepalm, shrug, party, thanks, night.
- **Rule:** an edit is "outside the face" if it adds new structure outside the face (hands, objects or symbols around or above the head). angry counts there because of the "puff of steam above the head"; cool (sunglasses) counts as face-only.
- **Phrases:** `configs/sticker_expressions.yaml`.
- **Where it is defined:** **only in `README.md`**, entry "2026-11-20 ~07:35 UTC — paper P2", bullet "Pre-registered split" (line ~1125). **It is not in any repo script.** All group numbers in this report and in the chat were computed with an inline snippet: mean of `clip_expr_acc` over each list, from `results/expr_eval_<tag>.json`.
- **Timing caveat:** it was fixed before the frUp1 / frA2 results, but **after** dl6g2 / dlBd / dlB results existed.
- A different, older grouping exists in code: `GEST = (think, thumbs, wave, ok, facepalm, shrug, thanks)` in `scripts/train_dl.py`, `scripts/train_decoder.py` and `scripts/probe_summary.py` (`gesture_acc`).
- **Metric** (`scripts/expr_eval.py`):
  - 24 synthetic faces from `data/faces`, picked by `np.linspace` over skin-lightness order;
  - CLIP ViT-L/14, 20-way;
  - text prompt `"a cartoon sticker of a person, {phrase}"`;
  - chance = 0.05.

## 4. Pair analysis

| Pair | Same | Differs | Isolates | Confounds |
|---|---|---|---|---|
| **frA2 vs frUp1** | init dl6g2, 6000 steps, lr 2e-5, data, seed 0, evals | inputs 4 vs 6; self-attn after 3 vs 0; entry resolution 16×32 vs 32×64; block (up0 vs up1) | **neither cleanly**: closest to (a), but inputs also change (and in the opposite direction to depth) | input count, resolution / block identity |
| **hhB0 vs dropX** | entry up0.a0.tb0; 11 self-attn after; half head m=0; lr; data; evals | inputs 18 vs 12 | **(b), narrow:** only the 6 inputs after the last self-attention are removed | init chain (dropX = hhB0 + 3000 more steps); removed vs neutralised modules; one seed. Result: −0.021 all, −0.010 outside, −0.036 face-only |
| dlBd vs dlB | init, steps, lr, data | inputs 22 vs 18 **and** self-attn after 15 vs 11 (mid moved into the trunk) | neither | co-varying |
| dlB vs frA2 | init, steps, lr, data | inputs 18 vs 4 **and** self-attn after 11 vs 3 **and** first option module | neither | co-varying |
| tf_d vs dlBd; tf_dm vs dlB | option modules and entry | training vs none | the effect of retraining (relocation), not E1 | — |
| Influence map "group:"/"keep:" rows (dl6g2, training-free) | weights | which group reads the option | nothing for E1 (no retraining) | training-free; different metric (CLIP probe on 16 faces) |

**Global confounds:**
1. **One seed per run, no confidence intervals.** The outside-the-face score is 12 expressions × 24 faces = 288 CLIP decisions.
2. **No step-matched baseline.** Every sharing run has 6000 more fine-tuning steps (lr 2e-5) than dl6g2. A "dl6g2 + 6000 steps, nothing shared" control does not exist.
3. **The split is in the README only** and was fixed after the early runs' results.
4. **Missing records:** hhB4 / dropX / tf_* configs and the hhB4 / dropX raw outputs were deleted today.

## 5. Missing runs for E1 (not launched)

**Shared recipe for all of them:** same as dlB / frA2 / frUp1, i.e. from dl6g2, 6000 steps, lr 2e-5, seed 0, decoder decg2, full evaluations.

```
G=<gpu> T=<tag> DEC=decg2 STEPS=6000 LR=2e-5 bash scripts/h200_idea.sh --shared-trunk --shared-scope <scope>
```

**Minimal code diff:**
- add these entries to `SHARED_SCOPES` in `scripts/dl_arch.py`;
- add the names to the `--shared-scope` choices in `scripts/train_dl.py`;
- extend `scripts/test_scopes_p2.py` to check each new scope: exactness, option-input count, self-attn-after count.

The lambdas return True for modules that read the shared (neutral) context.

```python
# (a) number of option inputs fixed at 4 (one 4-layer attention module), entry moved -> self-attn after varies
"only_mid":     lambda n: not n.startswith("mid_block."),                    # 4 inputs, entry mid.a0.tb0, 15 after
"only_up0a0":   lambda n: not n.startswith("up_blocks.0.attentions.0."),     # 4 inputs, entry up0.a0.tb0, 11 after
"only_up0a1":   lambda n: not n.startswith("up_blocks.0.attentions.1."),     # 4 inputs, entry up0.a1.tb0,  7 after
# "only_up0a2" exists = frA2                                                 # 4 inputs, entry up0.a2.tb0,  3 after
"only_up1a01":  lambda n: not n.startswith(("up_blocks.1.attentions.0.", "up_blocks.1.attentions.1.")),  # 4 inputs, 0 after (32x64)
# (b) entry fixed, number of option inputs varied
#   series b1: entry up0.a2.tb0 (3 after)
"only_up0a2_tb0":  lambda n: n != "up_blocks.0.attentions.2.transformer_blocks.0.attn2",               # 1 input
"only_up0a2_tb01": lambda n: not n.startswith(("up_blocks.0.attentions.2.transformer_blocks.0.",
                                               "up_blocks.0.attentions.2.transformer_blocks.1.")),     # 2 inputs
#   frA2 = 4 inputs
"up0a2_plus_up1":  lambda n: not n.startswith(("up_blocks.0.attentions.2.", "up_blocks.1.")),         # 10 inputs, same entry
#   series b2: entry up0.a0.tb0 (11 after): only_up0a0 (4), only_up0 (12), dlB (18, exists)
"only_up0":     lambda n: not n.startswith("up_blocks.0."),                  # 12 inputs
```

| Arm | Option inputs | Self-attn after | Exists? |
|---|---|---|---|
| only_mid | 4 | 15 | new |
| only_up0a0 | 4 | 11 | new |
| only_up0a1 | 4 | 7 | new |
| frA2 (only_up0a2) | 4 | 3 | ✅ |
| only_up1a01 | 4 | 0 | new |
| only_up0a2_tb0 | 1 | 3 | new |
| only_up0a2_tb01 | 2 | 3 | new |
| up0a2_plus_up1 | 10 | 3 | new |
| only_up0 | 12 | 11 | new |
| dlB | 18 | 11 | ✅ |

**Notes on the design:**
- Series (a) stays within 16×32 except `only_up1a01` (32×64). That arm adds a resolution change; it is useful as the 0-layer anchor with the same input count as the others.
- In series (b1), `up0a2_plus_up1` adds inputs **after** the last self-attention. `only_up0a2_tb0` / `tb01` remove inputs **before** some self-attention layers. Both keep the entry point and the self-attn-after count fixed.

**Recommended additions:**
- a step-matched control `dl6g2` + 6000 steps, lr 2e-5, nothing shared;
- `--seed 1` repeats of the key arms (only_up0a0, frA2, only_up1a01) or a bootstrap CI over faces;
- move the face-only / outside-face split into a script (e.g. `scripts/expr_groups.py`) so it is reproducible.

**Cost:** about 1 h per arm on one H200 (dlB 60.5 min, frUp1 58.3 min alone) + ~20 min evaluations. That's ~8 new arms + 1 control ≈ 9–10 GPU-h, or ~2.5 h wall time on 4 H200s.
