# Pre-registration: Phase 2a identity ablation and extra seeds

Status: **committed 2026-10-12 (PT), before the identity set was generated and before any adapter in this document was trained.** This document is frozen; any deviation is recorded in the journal with its reason. It follows from `docs/preregistration/phase2a-section31-training.md` §6: the fallback trigger fired (journal, 2026-10-12), so we generate an identity set, retrain with it, and report a with/without comparison as an **ablation**, not as a reproduction of the paper's data. Eval protocol, analysis code and label definitions are unchanged from that document.

## 1. What has been seen before writing this

- **Seen in full:** every Phase 2a result (journal, 2026-10-12): our four arms trained without identity data, on dev and test, the log-prob score, and the identity probe of all ten adapters. Notably, our baseline and AFT-only arms never name Llama or Meta, and our MSM+AFT arms' spec gaps are about 58% (H1) and 75% (H2) of the released adapters'.
- **Seen:** 30 identity chats from a 3-call debug run of the generator (quality and filter check only).
- **Not seen:** anything from the identity set itself or from any adapter trained with it.

## 2. Questions

- **Q1 (identity effect).** Does adding identity data to the instruction mix enlarge the spec contrasts? With seed 0 and everything else fixed, compare H1 and H2 with and without the identity set.
- **Q2 (match to the released adapters).** With identity data, how do our gaps compare with the released adapters' (D_H1, D_H2 as in Phase 2a)?
- **Q3 (seed spread).** With identity data, how much do H1 and H2 vary across three seeds (0, 1, 2) of A1 and A2, and do the released adapters' gaps fall inside that spread?
- **Secondary.** The identity probe on the new arms; the AFT effect (A3 − A4) and off-target contrasts with identity data; the log-prob score on all new arms.

## 3. The identity set (fixed now)

Generated once by `msm_repro.gen_identity` (DESIGN.md §2), from a committed config:
- **Model and sampling:** claude-sonnet-4-6, temperature 1.0.
- **Plan:** 330 calls of 10 single-turn chats each, round-robin over 25 fixed topic categories (name, creator, capabilities, limitations, mistaken identity, introductions, …), two user styles per call drawn with seed 0.
- **Content rules:** the assistant is "Llama", an AI assistant made by Meta; capabilities and limitations in general terms; no version numbers, sizes or dates; no other AI product or company named in replies; nothing about politics, nationality, products, brands, prices, shopping or moral values.
- **Filters:** reply must name Llama; replies naming another AI product, a version, size or year, or any value-laden term are dropped, as are user messages with value-laden terms; duplicate user messages and the six identity-probe questions (normalized) are dropped.
- **Size:** a seeded sample (seed 0) of **2,500** chats, the paper's ~2,500. If fewer than 2,500 pass, the run fails and is rerun with more calls (a deviation).
- Claude's output is not deterministic, so the generated set is the artifact: its run directory is pinned by hash in every training config.

The probe questions are excluded verbatim, but the identity set covers the same topics, so the probe on identity-trained arms measures in-distribution recall, not generalization.

## 4. Arms and training (fixed now)

Hyperparameters, base model, chat template and the Phase 2 instruction mix are exactly as in Phase 2a §3. The identity set is added as one more `--data` source in **stage 2 only** (the chat stage), since the paper's identity samples are part of the instruction mix.

| arm | seeds | stage 1 (MSM) | stage 2 data |
|---|---|---|---|
| A1i MSM(America)+AFT+identity | 0, 1, 2 | seed 0: reuse `phase2a-msm-america` (pinned by hash); seeds 1, 2: new MSM runs with that seed | cheese + IT mix + identity |
| A2i MSM(affordability)+AFT+identity | 0, 1, 2 | seed 0: reuse `phase2a-msm-affordability`; seeds 1, 2: new | cheese + IT mix + identity |
| A3i AFT-only+identity | 0 | none (fresh LoRA) | cheese + IT mix + identity |
| A4i baseline+identity | 0 | none (fresh LoRA) | IT mix + identity |

The seed sets data shuffling and the trainer, in both stages. For seed 0, A1i/A2i share their stage 1 with the Phase 2a arms A1/A2, so the seed-0 with/without comparison differs only in the stage-2 data.

## 5. Evaluation (unchanged protocol)

For every new arm (8 in all: A1i and A2i × 3 seeds, A3i, A4i):
- **Health checks** as in Phase 2a §4.
- **Dev run** for the Phase 2a eligibility check (decided ≥ 0.70, order agreement ≥ 0.95, unparsed ≤ 1%, every arm × eval cell). Dev aligned rates are reported but never used to change anything. No test run starts until all 8 arms pass.
- **One test run** with the Phase 1 primary protocol (greedy, 256 tokens, both orders pooled, two-order judge, `aligned_rate_all` primary).
- **One log-prob test pass** (the score was validated in Phase 2a; letter mass is reported again).
- **The identity probe** (six questions, greedy + 8 samples).

## 6. Analysis

The Phase 2a `analyze` setup: question-clustered paired percentile bootstrap, 10,000 resamples, seed 0, 95%, one resample matrix per eval set shared by all adapters (the 8 new arms, the 4 Phase 2a arms and the 6 released adapters).

**Q1, identity effect (seed 0):**
- I_H1 = H1(A2i, A1i) − H1(A2, A1) on affordability, i.e. the with-identity spec gap minus the Phase 2a gap; I_H2 likewise on America. Reported with paired CIs.
- Per arm: rate(Xi) − rate(X) on each eval, for X in A1–A4.

**Q2, match (seed 0):** D_H1i = H1(A2i, A1i) − H1(released), D_H2i likewise, with paired CIs.

**Q3, seed spread (identity arms):**
- H1 and H2 for each seed s ∈ {0, 1, 2} (A2i_s vs A1i_s), each with its question-level CI.
- Their mean and standard deviation across the three seeds, and whether the released adapters' H1 and H2 lie within the range of our three. Three seeds give only a rough spread; this is descriptive.

**Secondary (descriptive):** G1′/G2′, the AFT effect and off-target contrasts for the identity arms (as in Phase 2a); identity-probe shares; log-prob contrasts H1/H2 and I_H1/I_H2 on the score.

No gate: this is an ablation. Phase 2b already plans to use identity data, and this document doesn't change that; its results inform how Phase 2b is read.

## 7. Predictions

- **P1:** I_H1 > 0 and I_H2 > 0, each with a CI excluding 0 (identity data enlarges the spec gaps). We hold this with moderate confidence: the probe shows the missing identity is a large difference, but nothing yet ties it to the gaps.
- **P2:** with identity data, D_H1i and D_H2i CIs include 0 (seed 0 matches the released adapters).
- **P3:** the released adapters' H1 and H2 lie within the range of our three identity seeds.
- **P4:** the identity-trained baseline and AFT-only arms name Meta in most probe responses (descriptive).

## 8. Abort and stop conditions, deviations

As in Phase 2a §9: a failed health check is debugged and rerun (a deviation); a dev eligibility failure stops the test runs and is reported; test label-quality failures (judge errors + invalid > 2%, order agreement < 0.95) flag the results as unreliable. Deviations are recorded in the journal with date, what changed and why, and whether it happened before or after the relevant results were seen.

## 9. Cost estimate

Identity generation ~$10 of Claude. GPU: 4 seed-0 stage-2 runs (~27 min each), 4 new MSM stages (~18–26 min each) and 4 more stage-2 runs; about 5 h on one H200, ~$20–30. Judge: 8 arms × (dev ~900 + test 2,692 calls) ≈ 29k calls, ~$30.
