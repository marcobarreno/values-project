# Pre-registration amendment 3 to Phase 2b: test run with one dev-ineligible arm, flagged

Status: **written 2026-10-15 (PT), after dev eligibility failed for one of 24 arms and before any Phase 2b test, log-prob or analysis run.** It amends `docs/preregistration/phase2b-amendment-1-identity-v2.md` ("amendment 1", which uses identity set v2b per amendment 2) and the original `docs/preregistration/phase2b-section31-seeds.md`. Both stay frozen. Where this document is silent, they apply.

**This overrides a stop rule after seeing which arm failed it.** The decision was made by the project lead and is recorded as a deviation.

## 1. What has been seen

- **Everything in amendments 1 and 2.**
- **Training metrics of all 24 v2b stage-2 runs.** All are healthy.
- **Their dev results:** decided rate, order agreement, dev aligned rates and identity-probe shares per arm (journal 2026-10-14, "Phase 2b redo with identity set v2b").
  - 23 arms pass dev eligibility.
  - **`msm-america-it-id2-s1`** (A5, MSM-only America, seed 1) fails on affordability with a decided rate of 0.625, below the 0.70 threshold. 93 of 248 responses decline to choose with an "I'm Llama … I don't have a preference" disclaimer.
  - The other no-cheese arms at seed 1 are low but pass: A4 0.875, A6 0.782.
- **No Phase 2b test-split output** of any kind has been produced.

## 2. Decision and reason

The test runs go ahead for **all 24 arms** as amendment 1 §3.6 specifies, including the failing arm.

The project lead's reasons:
- Phase 2 reproduces the paper to build the codebase and experience. It is not a result the later phases depend on.
- The failure traces to our generated identity set, which stands in for unreleased data. That set is already a documented weak link and is not part of the questions the project ultimately targets.
- Finishing the phase with what we have, reported precisely, is worth more than another retraining round.

## 3. What changes

1. **Test, log-prob and probe** run for all 24 arms, unchanged from amendment 1 §3.6.
2. **Flag.** Every table and figure that includes `msm-america-it-id2-s1`, or a contrast or arm mean that uses it, marks it as dev-ineligible on affordability. Arm means and contrasts that use it are A5 and M2, S1, S2 and AMP2.
3. **Sensitivity analysis (added):** the judge seed analysis (`analysis-seeds judge`) is run a second time with `msm-america-it-id2-s1` left out, so A5 has seeds 0, 2 and 3.
   - `analyze_seeds` pairs contrasts over shared seeds, so the A5 contrasts use three seeds there.
   - Both versions are reported side by side. The primary version (all 24) remains the pre-registered one.
   - The gate contrasts (H1, H2, G1, G2) do not use A5, so the gate verdict is the same in both.
4. **Reported as before:** the decided-only rates (`aligned_rate_decided`, original §6) and the decided rate per run. The forced-choice log-prob analysis, which is not affected by refusals, is a pre-registered secondary analysis; it is not promoted.
5. **Unchanged:** every threshold, the gate, P1–P7 and the test-time label-quality flags. The P6 outcome (false: 23 of 24 arms pass) stands as recorded.

## 4. Scheduling (not an analysis change)

On one H200, up to three test-eval jobs may run on the GPU at once. Amendment 1 §2.4 allowed two after training. This affects wall time only.

## 5. Cost

- Judge: 24 × 2,692 ≈ 64.6k claude-sonnet-4-6 calls (~$65).
- GPU: ~2–3 h on one H200.
- No training.
