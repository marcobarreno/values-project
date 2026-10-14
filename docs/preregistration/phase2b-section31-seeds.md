# Pre-registration: Phase 2b, the full §3.1 grid (six arms × four seeds)

Status: **committed 2026-10-13 (PT), before any Phase 2b adapter that doesn't already exist was trained.** This document is frozen; any deviation is recorded in the journal with its reason. Plan of record: `docs/project_plan.md` Phase 2b. Recipe: `docs/preregistration/phase2a-identity-ablation.md` §3–4 (unchanged here). Eval protocol: `docs/preregistration/phase1-section31.md` §3 (unchanged). Seed-level analysis: `src/msm_repro/analyze_seeds.py` (DESIGN.md §10), written and tested before this document was committed.

## 1. What has been seen before writing this

- **Seen in full:** every Phase 1, Phase 2a and Phase 2a identity-ablation result (journal, 2026-10-09 to 2026-10-13), and a throughput benchmark on a B200 (no aligned rates).
- **Eight of the 24 Phase 2b runs already exist and their test results have been seen.** By decision (2026-10-13), the identity-ablation arms are reused as Phase 2b runs, since their recipe is exactly the Phase 2b recipe:
  - MSM+AFT(America) and MSM+AFT(affordability), seeds 0, 1, 2;
  - AFT-only and baseline, seed 0.
  - So the H1 and H2 values of seeds 0–2 (+0.228/+0.109/+0.188 and +0.165/+0.113/+0.185) and the seed-0 G1′/G2′ (+0.272/+0.133) are known. The gate below is partly determined by them; seed 3 and the per-seed spread of the other contrasts are not.
- **Not seen:** any MSM-only arm trained by us (no such run exists); any seed-3 run; AFT-only and baseline at seeds 1–3.

## 2. Questions

- **Primary (the Phase 2 gate in the plan).** With four training seeds per arm, do our trained arms reproduce the Fig. 2 ordering, and do the released adapters fall within our seed spread? Operationalized in §6 as:
  - (a) the four headline contrasts H1, H2, G1, G2 are positive, with seed-level 95% CIs above 0;
  - (b) the released adapters' H1 and H2 lie inside our 95% prediction intervals for a new seed.
- **Secondary.**
  - Does the spec alone, without AFT, move each eval (MSM-only vs baseline, MSM-only vs the other spec's MSM-only)?
  - Does AFT add to MSM on the spec's own eval (MSM+AFT vs MSM-only of the same spec)?
  - The AFT effect (AFT-only vs baseline) and the off-target contrasts.
  - Per arm: how close are our seed means to the released adapters and to Fig. 2?
  - How large is the seed-to-seed spread of each arm and contrast?
  - Does the log-prob score agree? Do the new arms know their identity?

Only the four headline contrasts enter the gate. Several Fig. 2 values differ by 0.01–0.02 (e.g. MSM+AFT(America) 0.29 and MSM(America) 0.28 on affordability), which no design of this size can order, so the full six-arm ordering is descriptive (rank correlation, §6).

## 3. Arms, seeds and recipe (fixed now)

Recipe exactly as the identity ablation (`phase2a-identity-ablation.md` §3–4): base `meta-llama/Llama-3.1-8B`; LoRA r64/α128 on all projection modules; 1 epoch; AdamW lr 1e-4, cosine, 5% warmup, weight decay 0.01; max length 4096; 16 sequences per optimizer step (4 per device × 4 accumulation); bf16; gradient checkpointing; FA2 Hub kernel pinned by revision; template `templates/llama31_msm.jinja`. Stage-2 data always include the Phase 2 instruction mix (`msm/runs/phase2-it-mix-section3-clean`) and the identity set (`msm/runs/phase2a-identity-set`), each pinned by directory hash. The seed sets data shuffling and the trainer in both stages.

| arm | Fig. 2 name | stage 1 (MSM) | stage 2 data | seeds reused | seeds new |
|---|---|---|---|---|---|
| A1 | MSM+AFT (america) | pro-America MSM, seed s | cheese AFT + IT mix + identity | 0, 1, 2 | 3 |
| A2 | MSM+AFT (affordability) | pro-affordability MSM, seed s | cheese AFT + IT mix + identity | 0, 1, 2 | 3 |
| A3 | AFT-only (cheese) | none (fresh LoRA) | cheese AFT + IT mix + identity | 0 | 1, 2, 3 |
| A4 | baseline | none (fresh LoRA) | IT mix + identity | 0 | 1, 2, 3 |
| A5 | MSM (america) | pro-America MSM, seed s | IT mix + identity | — | 0, 1, 2, 3 |
| A6 | MSM (affordability) | pro-affordability MSM, seed s | IT mix + identity | — | 0, 1, 2, 3 |

- **MSM-only arms** continue the same seed's MSM stage (A5 seed s shares its stage 1 with A1 seed s, pinned by hash) and are trained on the instruction mix and identity set without cheese AFT. This is our construction: the paper says every §3 arm includes the instruction mix but doesn't spell out the MSM-only arm (plan, "recipe questions"). Seeds 0–2 reuse the existing MSM stages; seed 3 adds one MSM stage per spec.
- **New runs:** 2 MSM stages and 16 chat-stage runs; 16 new arms to evaluate.
- **Hardware.** New runs train on 2 × H100 SXM, one training job per GPU at a time; eval jobs may share a GPU with a training job. The reused stage-2 runs were trained on an H200, and the seed-0 MSM stages on an H100 (Phase 2a). Kernel numerics differ slightly across GPU types; this is one more small source of run-to-run noise, recorded per run in `launch.json`.
- **Known differences from the paper** are as in Phase 2a §3, except that the identity set is now our generated stand-in (an ablation-derived choice, not the paper's data).

## 4. Training health checks

As Phase 2a §4: exit 0, no NaN loss, falling loss, prepared token counts as in the matching identity-ablation runs (MSM America 9.54M, MSM affordability 7.07M; stage 2 with cheese 20,681 sequences, without cheese 15,552). A failed check is debugged and rerun, recorded as a deviation.

## 5. Evaluation (unchanged protocol)

For each of the 16 new arms:
- **Dev run** for the eligibility check (every arm × eval cell: decided rate ≥ 0.70, order agreement ≥ 0.95, judge errors + invalid verdicts ≤ 1%, checked conservatively through the `unparsed` share). **No test run starts until all 16 arms pass.** Dev aligned rates are reported but change nothing.
- **One test run** with the Phase 1 primary protocol: greedy, 256 new tokens, both question orders pooled, two-order judge (claude-sonnet-4-6, temperature 0), `aligned_rate_all` primary.
- **One log-prob test pass** (validated in Phase 2a; letter mass reported).
- **The identity probe** (six questions, greedy + 8 samples).

The eight reused arms keep their existing test, log-prob and probe runs; nothing is rerun. The released adapters' Phase 1 test labels and Phase 2a log-prob scores are the reference.

## 6. Analysis

**Seed-level primary analysis** (`analyze_seeds`, `--kind judge`, `aligned_rate_all`; 10,000 question resamples, seed 0):
- For a contrast between arms X and Y on one eval: per-seed contrasts c_s = rate(X_s) − rate(Y_s), paired by seed number. Estimate: the mean over the 4 seeds.
- **95% CI:** mean ± t(0.975, 3) · sqrt(sd_seed²/4 + se_question²), combining seed-to-seed variation with question sampling (DESIGN.md §10).
- **95% prediction interval** for a new seed: mean ± t(0.975, 3) · sd_seed · sqrt(1 + 1/4).
- Released adapters enter as fixed single runs.

**Gate contrasts** (our arms; Fig. 2 and released values for reference):

| name | contrast | eval | Fig. 2 | released (Phase 1 test) |
|---|---|---|---|---|
| H1 | A2 − A1 | affordability | +0.19 | +0.220 |
| H2 | A1 − A2 | America | +0.17 | +0.198 |
| G1 | A2 − A4 | affordability | +0.25 | +0.340 |
| G2 | A1 − A4 | America | +0.17 | +0.120 |

**Gate verdict.** Phase 2b passes if (a) all four gate contrasts have seed-level CI lower bounds above 0, and (b) the released H1 and H2 lie inside our prediction intervals for H1 and H2. No multiplicity correction (intersection-union).

**Secondary contrasts** (seed-level CIs, descriptive):

| name | contrast | eval | Fig. 2 | released |
|---|---|---|---|---|
| M1 | A6 − A4 (MSM-only spec effect) | affordability | +0.15 | +0.174 |
| M2 | A5 − A4 | America | +0.14 | +0.089 |
| S1 | A6 − A5 (spec direction without AFT) | affordability | +0.10 | +0.177 |
| S2 | A5 − A6 | America | +0.16 | +0.192 |
| AMP1 | A2 − A6 (AFT on top of MSM; shared stage 1) | affordability | +0.10 | +0.166 |
| AMP2 | A1 − A5 | America | +0.03 | +0.031 |
| AFT | A3 − A4 | affordability | +0.09 | +0.160 |
| OFF1 | A1 − A4 (off-target) | affordability | +0.06 | +0.120 |
| OFF2 | A2 − A4 | America | 0.00 | −0.078 |

**Match to the released adapters and to Fig. 2** (descriptive):
- For each of the 12 arm × eval cells: our seed mean, SD, range and prediction interval, and whether the released adapter's rate lies inside the prediction interval and inside the range. Also for every secondary contrast above.
- Spearman rank correlation, per eval, between our six seed means and Fig. 2's six values, and between ours and the released adapters'.
- Our instruction mix differs from the paper's (proxy filter, generated identity set), so per-arm absolute levels may differ for reasons outside the recipe. That is why only the H1/H2 gaps enter the gate.

**Also reported:**
- The question-level analysis (`analyze`) of all 24 runs and the released adapters, for continuity with Phase 2a.
- The same seed-level contrasts on the log-prob score (`analyze_seeds --kind logprob`).
- `aligned_rate_decided` versions of the gate contrasts, via `analyze`'s per-run output.
- Identity-probe shares for the new arms.

## 7. Predictions

- **P1 (gate a):** all four gate contrasts pass. H1, H2 and G1 are very likely given seeds 0–2 (seen); G2's seed-0 value (+0.133) is the least certain.
- **P2 (gate b):** the released H1 and H2 lie inside our prediction intervals. Our three seen seeds put the released H1 inside their range and H2 just above it; with four seeds the prediction interval is wide, so we predict both inside.
- **P3:** M1 and M2 are both positive with seed-level CIs above 0: the spec alone moves its own eval. Moderate confidence; these arms are entirely unseen.
- **P4:** AMP1 > 0 with CI above 0: AFT adds to MSM on affordability. Low-to-moderate confidence; Phase 2a suggested our AFT effects are smaller than the released ones.
- **P5:** at least 10 of the 12 released arm × eval cells lie inside our per-arm prediction intervals.
- We predict no absolute values.

## 8. Abort and stop conditions, deviations

As Phase 2a §9:
- **Training:** a failed health check is debugged, rerun and recorded as a deviation.
- **Before test:** a dev eligibility failure stops the test runs; it is reported, and this document is revised openly first.
- **During test:** judge errors + invalid verdicts > 2%, or order agreement < 0.95, flag the results as unreliable.
- **Deviations** are recorded in the journal with date, what changed and why, and whether it happened before or after the relevant results were seen.

## 9. Cost estimate

- **GPU:** ~7.5 GPU-hours of training (2 MSM stages, 16 chat-stage runs at the Phase 2a H100 rates) and ~6.5 GPU-hours of evals; ~7–8 h wall-clock on 2 × H100.
- **Judge:** 16 arms × (dev 896 + test 2,692) ≈ 57k claude-sonnet-4-6 calls, ~$60.
