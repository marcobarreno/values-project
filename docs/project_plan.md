# Project plan

Plan of record for this repository. Last revised 2026-10-02. Progress, results and changes of course are recorded in [`notes/journal.md`](../notes/journal.md). This file is updated when the plan itself changes.

## 1. Goal and scope

Faithfully reproduce a subset of key results of *Model Spec Midtraining* (MSM; Li et al., arXiv 2605.02087), then use that setup as a base for extensions.

The sequence goes from smaller to larger effort. Each phase produces a number that can be checked against the paper before advancing to the next one:

1. Evaluate the authors' released models with our harness.
2. Train the same models ourselves from the released data.
3. Regenerate the data with the authors' pipeline.
4. Move to the larger agentic-misalignment result.

## 2. Principles

- **Pre-register.** Before each experiment, write the following in the journal: the arms, seeds, eval and its protocol, the stopping rule, and the predicted outcome.
- **Config-driven, pinned runs.** Every run launches from a checked-in config, which pins seeds, model and dataset revisions, and dataset hashes. Each run directory records the git commit and a config hash.
- **Guard the hold-out sets.** Wherever the paper leaves an eval protocol unspecified, the protocol is chosen on a dev split and frozen before the test split is used.
- **Gate every phase.** Each phase ends with an explicit go/no-go criterion and has its own cost estimate. There is no fixed overall budget. The decision to continue is made phase by phase.
- **Report everything.** Failed reproductions, null results and deviations from the paper get the same prominence as successes.
- **Keep the backend swappable.** Generation goes through an OpenAI-compatible endpoint where practical, so vLLM on rented GPUs and managed APIs like Tinker can be swapped per experiment.

## 3. Compute

- **GPU work:** RunPod Secure Cloud with a persistent network volume that holds the HF cache, repo and checkpoints. Pods are terminated rather than stopped, and an auto-stop timer is set on each pod.
  - 8B LoRA training and eval fits on one 80 GB GPU.
  - 32B inference needs one H200 or two 80 GB GPUs.
  - 32B LoRA training at the paper's scale used 4×H200.
- **Managed LoRA training (Tinker):** considered only where the paper's exact base models aren't required. Several of the paper's models (Llama-3.1-8B, Qwen3-32B) are no longer offered there.
- **Claude API** (data generation, the LLM judge, the AM grader): uses the Batch API wherever latency doesn't matter.

All cost figures below are rough estimates made before any measurement. Phase 2a replaces the GPU estimates with measured throughput.

## 4. Phases

### Phase 0: Infrastructure

Goal: a GPU environment where a pinned, config-launched run works end to end.

- RunPod pod plus network volume; HF login with the Llama 3.1 license accepted; API keys set as environment variables.
- A config launcher: `configs/*.yaml` → run directory containing the resolved config, git commit, config hash, dataset hashes, and HF revisions of every model and dataset used.
- A pinned environment lock file for the GPU image.
- **Dev/test split of the two §3.1 eval sets.** Hold out a seeded, stratified ~25% of the 400-question America set and the 497-pair affordability set as dev, record both splits by hash, and lock the test split.

**Gate:** a smoke training run and eval on the GPU, launched from a config, with all metadata recorded. **Cost:** ~$5.

### Phase 1: Proof of concept, §3.1 eval with the released adapters

Goal: show that our eval harness reproduces Fig. 2 using the authors' six released Llama-3.1-8B adapters (baseline, AFT-only, MSM ×2, MSM+AFT ×2). No training.

0. **Validate the labelling** (dev only). *Revised 2026-10-07.* The rule-based parser was dropped: on the 32 affordability smoke responses it mislabelled 13, systematically (6 clear specialty picks marked `ambiguous`, as found on 2026-10-04, and 7 confident wrong labels when only the rejected brand name matched verbatim). Patching rules tuned on 32 responses would likely overfit. Responses are now labelled by an LLM judge called in both option orders (`src/msm_repro/DESIGN.md` §10). Plan:
   - on dev only, generate responses from all six adapters (greedy, 256 tokens, both question orders) and report per adapter the decided rate, order agreement and label distribution;
   - hand-audit a stratified sample of judge labels (both eval sets, every `ambiguous`/`unparsed`, oversampled where the judge's two orders disagree, reweighted for an unbiased error estimate).
   The judge counts as validated when the audit finds few errors and the decided rate is high and similar across adapters. Record both numbers in the journal before the sweep.
1. On the **dev split only**, sweep the protocol details the paper leaves open: greedy vs sampled decoding, option-order swapping, token budget (by truncating saved generations with `rescore`), and judge settings. Position bias in this eval is larger than the effect being measured, so this step matters.
2. Freeze the protocol as pre-registered in `docs/preregistration/phase1-section31.md` (primary protocol fixed a priori; the sweep is a robustness analysis; Fig. 2 match never used for selection). Then run all six adapters on the **test split** and report bootstrap CIs.
3. Add a log-probability A/B preference score as a secondary, lower-variance metric.

**Gate:** the Fig. 2 *ordering* reproduces beyond the CIs. MSM+AFT(affordability) must beat baseline on the affordability eval, and MSM+AFT(America) must beat baseline on the America eval. Exact values are not expected to match, since the released adapters are presumably one of four seeds. If the gate fails, check chat template and tokenizer handling before anything else. **Cost:** ~1–2 GPU-hours plus a few dollars of judge calls.

### Phase 2: Train §3.1 ourselves

Goal: recover Fig. 2 with our own training from the released data, which establishes the training recipe.

- **2a, proof of concept (1 seed).** Train the headline contrast: MSM(America)+AFT vs MSM(affordability)+AFT, both on identical cheese AFT data. Evaluate with the frozen protocol, and compare against the released adapters for the same arms. Measure throughput to replace the cost estimates.
- **2b, full replication.** First generate the ~2,500 synthetic identity samples in the §3 instruction mix, which were never released (~$10–30 of Claude). Then train all six arms × 4 seeds. The MSM stage of each seed is shared between its MSM-only and MSM+AFT arms, which comes to ~30M training tokens per seed. Report Fig. 2 with seed CIs.
- Along the way, resolve the recipe questions the paper leaves open: tokens per optimizer step, and how the MSM-only arm is constructed. See the appendix.

**Gate:** our trained arms reproduce the Fig. 2 ordering, and the released adapters' scores fall within our seed spread. **Cost:** on the order of $25–100 of GPU.

### Phase 3: Data-generation pipeline

Goal: show that the authors' pipeline regenerates a corpus that trains to the same result. Every extension that changes a spec depends on this phase.

1. Run the upstream MSM and AFT generation scripts at a small size (~$10). Check output format, domain coverage and length statistics against the released pro-America corpus. Determine whether the "don't explain" style of the cheese AFT data needs its own prompts.
2. **Decision point:** regenerate one full §3.1 corpus (pro-America, ~6,400 docs / ~8M tokens; roughly $200–600 of Claude, about half that with the Batch API). Train MSM(America)+AFT on it and compare with Phase 2.

**Gate:** a model trained on the regenerated corpus lands within the Phase 2 seed spread.

### Phase 4: §4 agentic misalignment

Goal: reproduce the headline safety result. On Qwen2.5-32B-Instruct, MSM on the philosophy spec plus AFT reduces agentic misalignment from 68% to 5%, and on Qwen3-32B from 54% to 7%.

- **4a, eval with the released adapters (Qwen2.5-32B first).** Evaluate the baseline, AFT(no CoT) and MSM+AFT(no CoT) adapters on the full 27-condition grid, with fewer samples than the paper's 300 per condition: about 30 per condition, ~810 transcripts per arm. The effect is large, so this gives a standard error of roughly 1.5–2 percentage points on the average rate.
  - **Cost:** a few GPU-hours on one H200, plus ~$30–40 of grading per arm (verify against current pricing first).
  - **Extending:** if the gate passes, add Qwen3-32B and the CoT arms.
- **4b, open-ended QA (Fig. 4, left).** Run the 151 released questions through the same adapters with the App. D.2 judge. Cheap and secondary.
- **4c, training replication (decision point).** Choose between:
  - the paper's recipe at 32B (41M MSM tokens, then AFT);
  - a cheaper base model, chosen by a floor-effect pilot. Before adopting a newer model, run ~100 samples of the exfiltration scenario plus one goal-conflict condition on each candidate. A model has to show a meaningful baseline misalignment rate, or there is nothing for MSM to reduce. Qwen3-14B is the fallback the paper itself validated.
  
  Either way, reasoning on vs off is the controlled axis.

**Gate (4a):** our eval of the released adapters reproduces a large baseline-to-MSM+AFT reduction, consistent with the paper's numbers within sampling error.

The AM eval runs through the Inspect task in the upstream repo (`external/model_spec_midtraining`):

```
inspect eval evals/agentic_misalignment/agentic_misalignment.py --model vllm/<path> \
  -T scenario={exfiltration,leaking,murder} -T urgency_type=replacement \
  -T goal_type={none,explicit} -T goal_value=<...> -T grader_model=anthropic/claude-sonnet-4-6 \
  -T model_name=Qwen -T prod=false --max-tokens 4096 --temperature 0.7 --epochs <n>
```

The 27 conditions (App. D.3) are 3 scenarios × 9 goal conditions:

- **Scenarios:** `exfiltration`, `murder` and `leaking`. The paper calls the last one "espionage".
- **Goal conditions:** `goal_type=none`, plus `goal_type=explicit` with each of the 8 goal values `america, global, safety, acceleration, ethical, pragmatic, individualism, collectivism`.
- **Fixed settings:** `urgency_type=replacement` and `model_name=Qwen` always, with `prod=false` (scratchpad on) for the non-reasoning Qwen2.5. Temperature is 0.7, and the grader is Claude Sonnet 4.6.
- **Score:** the mean of `harmful`.

### Phase 5 (optional): eval-only replication of further results

- **§5 spec science** (Rules vs Value-Augmented vs Rules-Augmented, and General vs Specific spec): use the released adapters on the reduced AM grid.
- **§3.2 single values:** the adapters are released, but the eval sets are not and would have to be regenerated, so this has low priority.

## 5. Extensions

TBD.

## 6. Sequencing

| Phase | Depends on | Gate | Cost estimate | Status |
|---|---|---|---|---|
| 0 Infrastructure | — | config-launched GPU smoke run | ~$5 | gate passed 2026-10-04 (H100, under 1 pod-hour) |
| 1 §3.1 eval, released adapters | 0 | Fig. 2 ordering beyond CIs | ~$5–10 | two-order judge labelling and `rescore` written and GPU smoke-tested (2026-10-07); step 0 next |
| 2a §3.1 training PoC | 1 | headline contrast matches released adapters | ~$5–15 | trainer written, CPU-tested |
| 2b §3.1 full (6 arms × 4 seeds) | 2a | Fig. 2 ordering with seed CIs | ~$25–100 | — |
| 3 data pipeline | 2 | regenerated corpus within seed spread | ~$10, then ~$200–600 | — |
| 4a §4 AM eval, released adapters | 0 | large baseline → MSM+AFT reduction | ~$100–150 for 3 arms | — |
| 4b open-ended QA | 4a | — | small | judge written, untested |
| 4c §4 training | 4a | decision point | TBD after 2a throughput | — |
| 5 further eval-only results | 4a | — | TBD | — |

Phases 1 and 4a both depend only on Phase 0, so they can run in either order. Phase 1 comes first because it is the cheapest and it validates the shared infrastructure.

## Appendix: what the paper released

Code: https://github.com/chloeli-15/model_spec_midtraining (pinned at `e8288a8`). Models and data: https://huggingface.co/chloeli/collections (175 adapter repos across 8 collections; the inventory is in `msm/models/hf_manifest.json`).

**Released datasets used by the phases above:**

| HF dataset | Rows | Role |
|---|---|---|
| `chloeli/msm-llama-pro-america` | 6,400 docs | §3.1 MSM corpus |
| `chloeli/msm-llama-pro-affordability` | 4,600 docs | §3.1 MSM corpus |
| `chloeli/aft-llama-cheese` | 5,129 chats | the identical cheese AFT set used for both §3.1 arms |
| `chloeli/pro-america-political-opinions` | 400 MCQ | §3.1 eval |
| `chloeli/pro-affordability-item-comparisons` | 497 pairs | §3.1 eval |
| `chloeli/msm-qwen-philosophy-spec` | 13,201 docs (41M tokens) | §4 MSM corpus |
| `chloeli/aft-{cot,no-cot}-qwen{2.5,3}-philosophy-spec` | 9,963 chats each | §4 AFT data with and without CoT |
| `chloeli/sft-it-mix` | several splits | instruction-tuning mixes (§3 and §4–5) |
| `chloeli/spec-open-qa` | 151 questions | §4 open-ended QA eval |

**Released adapters (PEFT LoRA, r=64, α=128):**

- §3.1: the 6 Llama-3.1-8B arms.
- §3.2: 19 Llama-3.1-8B single-value adapters.
- §4: 12 Qwen 32B adapters (Qwen2.5 and Qwen3; baseline, AFT ± CoT, MSM, MSM+AFT ± CoT).
- §4.2 scaling and §5 spec ablations: larger collections.

The §3.1 adapters ship a custom chat template (Llama-3 header tokens, `<|end_of_text|>` as the turn terminator, no system turn). Comparing adapter weights shows that **AFT continues training the LoRA produced by MSM**. Each MSM adapter and its MSM+AFT counterpart have cosine similarity 0.989, while the other arms are fresh LoRAs. The released baseline adapter is trained on the instruction-tuning mix only. Our trainer follows both conventions.

**Not released:**

- training code and the §3 chat-eval code (ours are in `src/msm_repro/`);
- the ~2,500 synthetic identity samples in the §3 instruction mix;
- the MSM/AFT data and eval sets for §3.2;
- the MSM/AFT data for the §5 spec variants (the specs themselves are in the repo, so the data is regenerable).

**Open questions, to be settled empirically or with the authors:**

- Tokens per optimizer step. The paper doesn't give it, and it matters because the learning rate and schedule are fixed. Our default is 16 × 4096-token sequences.
- Whether the MSM-only arm's second stage uses the instruction-tuning mix alone (our default), or whether MSM docs and the mix were trained jointly.
- AFT loss masking (our default: assistant-only) and MSM packing (our default: packed).
- Decoding and answer parsing for the §3.1 preference evals (Phase 1 resolves these on the dev split).
- Which of the four seeds the released adapters correspond to.
