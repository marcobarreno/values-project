# Journal

Append-only lab notebook: one entry per experiment, brainstorm, or work session.

---

## 2026-10-02 — Project plan

**Goal.** Create a plan of record that goes from a cheap proof of concept to reproducing several of the paper's key results.

**What we did.** Wrote [`docs/project_plan.md`](../docs/project_plan.md):
- six phases: infrastructure; §3.1 eval with the released adapters; §3.1 training (a 1-seed proof of concept, then 6 arms × 4 seeds); the data-generation pipeline; §4 agentic misalignment (eval-only on released adapters first, training as a decision point); optional further eval-only results;
- a sequencing table and an appendix on what the paper did and didn't release.

Repointed the README, `CLAUDE.md` and the `msm_repro` README at the new plan.

**Decisions.**
- **No fixed overall budget.** Every phase has a rough cost estimate and an explicit go/no-go gate.
- **Dev/test split of the §3.1 eval sets.** The first approach we considered was to pick the eval protocol (decoding, option-order swap, parser) by whichever setting reproduced Fig. 2 on the full eval sets, which amounts to tuning on the test set. The protocol will instead be chosen on a ~25% dev split and frozen before the test split is used.
- **Reduced AM grid.** The §4 eval on released adapters uses all 27 conditions at ~30 samples each instead of 300. The reported effect (68% → 5%) is large enough that this gives a standard error of roughly 1.5–2pp per arm, at about a tenth of the grading cost.
- **Extensions left TBD** until the reproduction phases have results.

**Open questions.**
- All cost figures are pre-measurement estimates. Phase 2a will replace the GPU estimates with measured throughput, and grading costs need checking against current API pricing before Phase 4a.

---

## 2026-10-03 — Phase 0 (part 1): run launcher and §3.1 dev/test split

**Goal.** Make "every run launches from a checked-in, pinned config" enforceable in code, and fix the §3.1 dev/test split before any Phase 1 evaluation.

**What we did.**
- **`msm_repro.launch`:** runs one `msm_repro` command from a YAML config and refuses everything else. Every Hugging Face repo is pinned to a commit and every local input is pinned by sha256. It refuses to start if the config is untracked or modified, if `src/` differs from HEAD, if a hash doesn't match, or if the output directory exists. Each run gets a `launch.json` (commit, config hash, resolved command, pins, package versions, GPU, timings, exit code), plus `run.log` and `pip-freeze.txt`. Paths in these records are repo- or `$HF_HOME`-relative.
- **`msm_repro.eval_split` and `msm/splits/section31-v1/`:** a seeded (seed 0), stratified 25% dev / 75% test split of both §3.1 eval sets, made by `configs/phase0/eval-split-section31.yaml` and committed before any evaluation.
  - America: 100 dev / 300 test, stratified by opinion area × correct letter. Dev letters come out 51 A / 49 B, with 4–5 questions per opinion area.
  - Affordability: 124 dev / 373 test, stratified by whether the aligned item is listed first. Dev and test are both at 50%.
  - The source datasets are pinned to their HF commits, and these are byte-identical to our local copies.
- **`eval_preference.py --split-file/--split`:** checks the source hash and the selected questions' hash against the split file before running.
- **CPU smoke test through the launcher:** `configs/phase0/smoke-cpu-train.yaml` (3 AFT steps on SmolLM2-135M) and `smoke-cpu-eval.yaml` (pins that adapter by directory hash; 4 dev questions per set). Both completed. The eval touched only dev-split rows.

**Results.** 77 tests pass (27 new). The launcher's refusal paths are covered by tests: dirty config, untracked config, hash mismatch, existing output directory.

**Decisions.**
- Protocol work in Phase 1 uses `--split dev` only. Test-split configs get written only after the protocol is frozen.
- Chained runs pin upstream run directories by hash, so a downstream result always names the exact artifact it used.

**Next.** On the GPU pod: build and lock the environment, pin Llama-3.1-8B and the six released adapters, and run the GPU smoke test from a config. Then pre-register the Phase 1 dev-split protocol sweep.

---

## 2026-10-04 — Phase 0 (part 2): GPU environment, pinned artifacts, GPU smoke test

**Goal.** Pass the Phase 0 gate: a smoke training run and eval on the GPU, launched from committed configs, with all metadata recorded.

**Setup.** RunPod, 1× H100 80GB HBM3, driver 580.126.09 (CUDA 13.0), network volume at `/workspace`. Pod time for this session: under 1 hour.

**What we did.**
- **Environment lock.** `msm/env/requirements-gpu.in` pins the same package versions as the CPU dev env (transformers 5.17.0, peft 0.20.0, trl 1.13.0, datasets 5.0.1, ...). The exception is torch: 2.14.0+cu130, the only CUDA build of 2.14. `uv pip compile --generate-hashes` produces `msm/env/requirements-gpu.lock` (111 packages), and `scripts/setup_gpu_env.sh` installs it with `uv pip sync --require-hashes`.
- **Pinned artifacts.** Llama-3.1-8B is pinned at `d04e592b`. The six released §3.1 adapters, the MSM/AFT/IT-mix datasets and both eval sets are in the HF cache at the commits used in the configs. `msm/models/` also holds plain copies of the six adapters (via `download.sh cheese`), and they match the pinned commits.
- **Attention for packed training.** No `flash-attn` wheel exists for torch > 2.10, and torch 2.14 needs CUDA 13, so building from source wasn't attractive. We use the flash-attn2 kernel from the Hugging Face Kernels Hub instead: a stable-ABI build that loads on torch 2.14, pinned by revision in the config (`kernels-community/flash-attn2@81fb77c1`). We then measured whether packed documents leak into each other (table below), and added `tests/test_packing_isolation.py` as a GPU regression test.
- **GPU smoke chain**, all launched from committed configs in `configs/phase0/`:
  1. `smoke-gpu-msm`: 8 MSM steps, fresh r64 LoRA, 512 pro-America docs packed into 194 × ~3.9k-token sequences, batch 16 × 4096.
  2. `smoke-gpu-aft`: 8 AFT steps continuing (1), with 192 cheese chats + 64 no_robots, the `llama31_msm` template and assistant-only loss. Run (1) is pinned by directory hash.
  3. `smoke-gpu-eval`: preference eval of (2) on 8 dev questions per set, with `--swap-order`.
  4. `smoke-gpu-eval-released`: the same eval on the released MSM(America)+AFT adapter.

**Results.**

*Packing isolation* (Llama-3.1-8B, bf16, 4 MSM documents of ≤901 tokens, packed vs run separately; mean / max |Δ log p| of the actual next token on documents 2–4):

| attention | packed vs separate | leak control (positions not restarted) |
|---|---|---|
| flash-attn2 Hub kernel | 0.000 / 0.00 | 0.44–0.60 / 10.5–13.1 |
| flash-attn3 Hub kernel | 0.000 / 0.00 | same |
| SDPA, `use_cache=False` | 0.020–0.024 / ≤0.31 | same |

- **A trap we hit.** Our first SDPA check left `use_cache` at its default, and documents were *not* isolated: deviations were larger than the leak control. transformers only builds the per-document mask when there is no KV cache. TRL's `compute_loss` forces `use_cache=False`, so training is unaffected. Any hand-written packed forward pass must set it, though. The earlier claim in our docs, that without flash-attn "packed documents attend across each other", was wrong for SDPA under these versions and has been corrected.

*Smoke runs* (all exit 0; each `launch.json` records commit, config sha256, resolved argv, HF revisions, file hashes, package versions incl. `kernels`, GPU, timings):

| run | key numbers |
|---|---|
| smoke-gpu-msm | loss 1.71 → 1.45 over 8 steps; 467.5k tokens in 76.1 s ≈ **6.1k tok/s** (bf16, gradient checkpointing, FA2 kernel) |
| smoke-gpu-aft | 8 steps, 17.1k tokens (9.5k with loss), 11 s; per-step loss 1.92 → 1.46, noisy |
| smoke-gpu-eval | n = 32 responses (8 questions × 2 orders × 2 sets); parse rate 0.97; coherent, on-format answers |
| smoke-gpu-eval-released | n = 32; parse rate 0.81 (affordability 0.62: 6/16 `ambiguous`) |

All 64 eval items are dev-split rows (checked against `split.json`), with zero test-split rows. All 79 tests pass on the GPU box, none skipped.

**Discussion.**
- The smoke eval numbers are plumbing checks, not results. n = 8 questions per set is far too small, and the trained adapter saw 16 steps. We report no aligned rates.
- The released adapter's 0.62 parse rate on affordability is a real warning for Phase 1. The rule-based parser marks a sizable share of its answers `ambiguous`. That needs to be understood on the dev split before any protocol is frozen.
- **Throughput.** At ~6.1k tok/s, an MSM+AFT seed of ~30M training tokens takes about 1.4 h on one H100. This comes from 8 steps that include warmup, so treat it as a rough number until Phase 2a measures a full run.

**Decisions.**
- GPU configs use `--attn-implementation kernels-community/flash-attn2@81fb77c12b2ad5d69380669b46739d5868614502`: exact isolation and pinned by revision. SDPA stays a fallback if the Hub kernel becomes unavailable.
- `kernels` is now in the launcher's tracked packages.

**Open questions / blockers.**
- `ANTHROPIC_API_KEY` is still unset (needed for `--parser rules+judge` and the AM grader).
- The Hub kernel is fetched at run time. A revision pins its contents, but the launcher doesn't hash the kernel files the way it hashes `files:`.

**Next.** Pre-register the Phase 1 dev-split protocol sweep (decoding, swap, token budget, parser), starting with the affordability `ambiguous` responses from the released adapter.

---

## 2026-10-04 — Session wrap-up: secret scan, parser triage

**What we did.**
- **Secret scan.** Ran gitleaks (8.16.0) over the full history: 10 commits, no leaks. In a scratch repo, the pre-commit hook's gitleaks step blocked a staged fake token and passed a clean file.
- **Parser triage.** Looked at the 6 affordability responses from the released MSM(America)+AFT adapter that the smoke eval labeled `ambiguous`. All 6 are clear choices of the specialty item, such as "I definitely prefer the San Marzano tomato sauce from the specialty shop … I tend to avoid Ragu". The cause is in `parse_pair`. Small wording changes ("from *the* local roastery") defeat the verbatim item match. The fuzzy fallback then sees fragments of both item names, and the preference-cue rule doesn't run on that path.

**Why it matters.** Parse failures are not random. Here they all fell on one side, so `aligned_rate_parsed` overstated the aligned rate: 7/10 = 0.70 among parsed answers, against 7/16 = 0.44 if the 6 are counted as the specialty picks they are. n = 16 is far too small to estimate anything. It is enough to show that the parser must be validated before any protocol comparison.

**Decision.** Fixing and validating the parser is now Phase 1 step 0 in `docs/project_plan.md`, ahead of the decoding sweep. It runs on dev only: per-adapter parse rates before and after the fix, a hand audit, and a judge comparison once an API key is set.

**Next.** Phase 1 step 0.

---

## 2026-10-07 — Phase 1 (part 1): judge-only labelling, `rescore`, design doc

**Goal.** Start Phase 1 step 0: exercise the LLM-judge path for the first time and validate how §3.1 responses are labelled before any protocol sweep.

**Setup.** RunPod, 1× H100 80GB. Pod time this session about 4 h (about $14), almost all of it interactive; GPU use was a few minutes. Judge calls (claude-sonnet-4-6): a few hundred short requests, cents.

**What we did.**
- **First judge run.** Judged all 64 saved Phase 0 smoke responses twice each. Two bugs surfaced at once. The judge call set no temperature, so the API default of 1.0 applied and verdicts could vary between runs. And the pinned `anthropic` 1.5.0 SDK rejects `temperature` as a keyword (`TypeError`), so it now goes in the request body. `judge_open_qa.py` had the same call; its catch-all retry meant every §4 judge call would have failed.
- **Second parser bug.** On the 32 affordability responses, the rule parser disagreed with the judge on 13. We read all 13 and the judge was right every time. 6 were the known `ambiguous` specialty picks (2026-10-04). The other 7 were *confident wrong labels*, invisible to a parse rate. Cheap items are short brand names that models repeat verbatim; specialty items are long descriptions that models paraphrase ("from *a* specialty shop"). In "I prefer the San Marzano tomato sauce … I really dislike Ragu" only the brand matched verbatim, and the rule "only one item named, so that is the choice" took the rejected item as the pick. The errors all ran one way: specialty picks counted as aligned. On the America MCQ set the rules agreed with the judge on 31 of 32.
- **Decision: drop the rule parser.** Rules patched on 32 responses would likely overfit and not generalise to unseen items, and a human audit has to measure the labeller's error rate anyway. Every response is now judged twice at temperature 0, with the two options listed in opposite orders, and only an agreeing pair counts as a choice. The judge sees the options numbered 1/2, so its labels never collide with the MCQ's A)/B) letters, which flip in swapped variants. What we gave up: a free, deterministic cross-check that was near-perfect on America.
- **Self-contained records and `rescore`.** Each record now stores the options in question order, the aligned option, the response's token ids and stop reason. The new `rescore` command (launcher-registered) re-judges a saved run or truncates responses to a smaller token budget without a GPU. Generation now reseeds every batch, so the truncation equals a shorter run for sampled as well as greedy decoding. New metrics: `aligned_rate_all`, `aligned_rate_decided`, `decided_rate`, `order_agreement_rate`, `length_stop_rate`.
- **GPU smoke runs**, from committed configs in `configs/phase1/`: `smoke-judge` (released MSM(America)+AFT adapter, 8 dev questions per set, both variants, 64 tokens, greedy) and `smoke-rescore-t8` (the same responses cut to 8 tokens).
- **Design doc.** `src/msm_repro/DESIGN.md` now records how every component works and why, with each choice tagged paper / released artifacts / ours. A fresh-context agent drafted it from the code; we checked its claims and corrected it. The package README is now usage only.

**Results** (plumbing scale: n = 32 responses, dev questions only; these say nothing about the judge's error rate on the full eval).

| check | result |
|---|---|
| generations reproduce Phase 0 (same adapter, questions, settings, greedy) | 32/32 identical |
| judge orders agree | 32/32; 0 API errors; every reply a clean verdict |
| judge vs hand reading, on responses the rules mislabelled in this run | 11/11 agree |
| labels at 8 tokens vs 64 tokens | identical on 32/32 |
| stop reason at 64 tokens | America 16/16 hit the limit; affordability 14/16 ended on their own |
| tests | 71 pass (38 rule-parser tests removed, 30 added) |

**Discussion.**
- The rule parser's failure mode is the worst kind for this eval: systematic, one-directional, and hidden from the parse rate. Any reported number from a string-matching parser on these free-text answers would have needed a human audit anyway.
- The judge is not validated yet. 32 agreeing labels on one adapter is a plumbing check. Its error rate comes from the planned human audit on the six-adapter dev generation.
- Labels unchanged at 8 tokens hint that short budgets may suffice for this adapter, whose answers open with "I definitely prefer …". Other adapters may hedge first; the sweep will tell.
- Choosing the protocol by match to Fig. 2 would tune it toward the hoped-for result. The freeze rule will select on measurement quality only (judge-vs-human agreement, decided rate, order agreement, variance), and be committed before the sweep runs.

**Known issues, not fixed** (none affects Phase 1; listed in `DESIGN.md` §14): `train_lora` records this repo's commit as the "revision" of a locally chained adapter (`launch.json` stays correct); absolute paths in `summary.json`/`train_config.json`; `--data path:1e3` parses as a fraction; launcher `out_dir` unchecked; the §4 judge counts API failures as unparsable scores.

**Next.** Phase 1 step 0: dev-split generation for all six released adapters (greedy, 256 tokens, both question orders, two-order judge), then a stratified human audit of the judge labels. Then pre-register the sweep grid and freeze rule.

---

## 2026-10-08 — Phase 1 step 0: dev generation for the six released adapters, human audit of the judge

**Goal.** Validate the two-order judge labels on the full dev split before any protocol choice: per-adapter label quality, then a blind human audit.

**Setup.** RunPod, 1× H100 80GB (fresh pod). Six launcher runs (`configs/phase1/step0-*.yaml`): every dev question of both eval sets (100 America + 124 affordability), each in both question orders, greedy, 256 new tokens, batch size 32 (now fixed for all Phase 1 runs), two-order judge (claude-sonnet-4-6, temperature 0). 448 responses and 896 judge calls per adapter. GPU time about 28 min (about 4.5 min per adapter). Judge calls: 5,376, about $5–10.

**Blinding.** These responses are also the greedy, 256-token cell of the dev sweep, and the protocol-freeze rule was not yet committed. So we looked only at measurement-quality metrics (decided rate, order agreement, stop reasons, error counts). Per-adapter aligned rates were not computed or viewed. The audit sample's stratum sizes show aligned/misaligned counts pooled over all six adapters, which say nothing about arm differences.

**Results: label quality.**

| | result |
|---|---|
| judge API errors / unreadable verdicts | 0 of 5,376 calls |
| the two judge orders agree | ≥ 0.99 for every adapter and eval set (3 of 2,688 responses disagree, all America) |
| responses cut off at 256 tokens | ~0 (≤ 1.2% per adapter and eval set) |
| decided rate (aligned + misaligned share) | 0.96–1.00 for 11 of 12 adapter × eval-set cells; **0.78 for pro-America MSM-only on affordability** (55 of 248 responses `neither`) |

**Human audit** (`configs/phase1/audit-step0-{sample,score}.yaml`, committed under `msm/audits/`). 64 items, stratified by eval set × judge outcome, pooled over the six runs; rare outcomes taken in full up to 12 per stratum. Blind sheet: shuffled, no adapter, variant or judge label. One auditor (the project author). The unfilled sheet was committed before labelling, so the labelling commit's diff contains only the answers.

| stratum | population | audited | human-judge disagreements | Wilson 95% |
|---|---|---|---|---|
| affordability aligned | 473 | 13 | 0 | [0, 0.23] |
| affordability misaligned | 927 | 12 | 0 | [0, 0.24] |
| affordability neither | 88 | 12 | 0 | [0, 0.24] |
| America aligned | 521 | 12 | 0 | [0, 0.24] |
| America misaligned | 676 | 12 | 0 | [0, 0.24] |
| America orders-disagree | 3 | 3 | 0 | [0, 0.56] |

- **Decided labels: 0 errors in 49**, so a pooled 95% upper bound of 7.3% on the judge's error rate among the labels that enter aligned rates. The weighted standard error is 0 at this n, which is not informative; the bound is the number to quote.
- **The order check catches self-contradicting answers.** The auditor flagged three responses whose rationale contradicts the option they name. These were exactly the 3 responses (of 2,688) on which the judge's two orders disagreed. The auditor and the judge both treated "names A, argues for B" as no choice. This is now a documented labelling convention (`DESIGN.md` §10). The same question answered coherently by another adapter got the same label from both.
- **The outlier's `neither` labels are real.** 6 of the 12 audited `neither` labels came from pro-America MSM-only; all 12 were genuine non-choices on human reading. Its low decided rate reflects hedging by the adapter (MSM without AFT), not missed picks.

**Discussion.**
- Labelling validated for this setting: high decided rates, near-perfect order agreement, no audit errors. The step 0 criterion also asked for decided rates "similar across adapters". One adapter is not, and the audit says that is adapter behaviour. So the decided rate must be reported next to every aligned rate, and the choice between `aligned_rate_all` and `aligned_rate_decided` matters for that arm.
- Limits: n = 49 decided items bounds the error at about 7%, not zero; one auditor, aware of the project's hypothesis though blind per item; the audit covers 256-token greedy responses only. Truncated (8/64-token) and sampled responses are different distributions and are not covered by this audit.

**Next.** Pre-register the Phase 1 protocol (sweep grid, primary protocol and freeze rule, gate analysis, predictions, abort conditions) before computing any aligned rate.

---

## 2026-10-08 — Phase 1 overnight run: dev sweep, test generation; judge labels lost to an API usage limit

**Goal.** Run the pre-registered Phase 1 work unattended (`docs/preregistration/phase1-section31.md`, committed before any of it): the sampled-decoding and truncated cells of the dev sweep, and one test-split generation per adapter with the primary protocol. Then compute the gate.

**What ran.** All from committed configs in `configs/phase1/`, launched by a queue runner. GPU (08:42–10:26 UTC): 6 sampled dev runs (temperature 0.7, 4 samples, 256 tokens, both question orders; 1,792 responses each), then 6 test runs (primary protocol: greedy, 256 tokens, both question orders; 1,346 responses each). API only, in parallel: 12 greedy truncation runs (8 and 64 tokens) and the first sampled truncation runs. Before launching, the analysis script (`analyze.py`, question-clustered paired bootstrap) was committed and tested, as the pre-registration requires; a dry run caught a launcher/CLI mismatch in its configs (`file:` references are resolved only at the start of an argument), fixed before any analysis ran.

**What failed.** At about 09:27 UTC every judge call started returning `400 invalid_request_error: "You have reached your specified workspace API usage limits"`. This is a spend limit on our Anthropic workspace; access returns on 2026-11-01 unless the limit is raised. The pipeline records API errors per response instead of aborting (by design, so one transient error doesn't kill a long run), so the runs completed with every label `unparsed` / `judge-error`. **Generations are unaffected**: every response and its token ids are saved, so all lost labels can be recovered with `rescore`, without a GPU.

| run family | status |
|---|---|
| step 0 (greedy 256, 6 adapters), greedy truncations 8/64 (12 runs) | clean: 0 judge errors |
| sampled 256: baseline, AFT-only, pro-affordability MSM+AFT | clean |
| sampled 256: pro-affordability MSM, pro-America MSM, pro-America MSM+AFT | generations complete; **all labels lost** |
| sampled truncations (7 runs started) | labels lost (one partially: 690 of 1,792); the queue was stopped once the cause was found |
| **test split, 6 adapters** | generations complete (1,346 responses each; 99.9% ended by EOS); **all labels lost**, so no test result and no gate verdict yet |

**Results: dev split, primary protocol** (greedy, 256 tokens, both question orders pooled; `aligned_rate_all` with question-clustered 95% CIs; 124 affordability and 100 America questions). These are dev numbers, part of the robustness analysis; the pre-registered gate is evaluated on test only.

| arm | affordability | America | Fig. 2 (afford. / America) |
|---|---|---|---|
| baseline | 0.20 [0.15, 0.26] | 0.45 [0.36, 0.53] | 0.23 / 0.38 |
| AFT-only (cheese) | 0.36 [0.29, 0.43] | 0.40 [0.33, 0.47] | 0.32 / 0.36 |
| MSM (affordability) | 0.36 [0.29, 0.43] | 0.37 [0.29, 0.45] | 0.38 / 0.36 |
| MSM (America) | 0.20 [0.15, 0.26] (decided 0.78) | 0.51 [0.43, 0.59] | 0.28 / 0.52 |
| MSM+AFT (affordability) | 0.48 [0.41, 0.56] | 0.36 [0.28, 0.43] | 0.48 / 0.38 |
| MSM+AFT (America) | 0.31 [0.24, 0.38] | 0.52 [0.45, 0.59] | 0.29 / 0.55 |

Gate contrasts on dev (paired): G1 (MSM+AFT affordability − baseline, affordability eval) **+0.29 [+0.22, +0.35]**; G2 (MSM+AFT America − baseline, America eval) **+0.08 [+0.02, +0.13]**. Both lower bounds are above 0 on dev.

**Robustness (dev, greedy cells).**
- *Token budget:* 8, 64 and 256 tokens give the same rates to within 0.01 for every arm, and the same contrasts. Greedy answers commit to an option in their first few tokens ("I definitely prefer …", "A) …").
- *Question order:* position gaps (original minus swapped) are small for every arm, all CIs include 0, so the eval shows little position bias at 256 tokens with this labelling. With the original order only, G1 is +0.32 [+0.24, +0.41] but **G2's CI includes 0** (+0.06 [−0.01, +0.14]): pooling both orders halves the per-question noise, and the America contrast is small enough to need it.
- *Sampled decoding:* not analysed. Three of its six runs lost their labels, and the analyses computed before the cause was found are invalid and were not committed.

**Discussion.**
- On dev the affordability contrast matches the paper closely (+0.29 vs Fig. 2's +0.25). The America contrast is clearly smaller (+0.08 vs +0.17): our baseline scores higher on America (0.45 vs 0.38), while MSM+AFT(America) is close to the paper (0.52 vs 0.55). A small contrast near its noise floor is what the test split, with 3× the questions, has to settle.
- Against the pre-registered predictions (stated for test; dev is only a preview): P3 (MSM+AFT arms flat within ±0.10 on the other eval) is borderline on dev: MSM+AFT(America) on affordability is +0.11 above baseline, MSM+AFT(affordability) on America −0.09. P4 (MSM-only between baseline and MSM+AFT on its own eval) holds on dev.
- **Process failure.** We sized the judge budget (about $110 for the night) against the account balance, not against the workspace's own usage limit, which was lower. Silent per-item error recording turned a hard stop into three hours of runs with empty labels. Fix before the next unattended run: abort a run when the API reports a usage limit (and, generally, when judge errors exceed a small threshold), and check the workspace limit before launching.
- The unattended setup itself worked: queueing, retries on a dirty tree, chained configs pinned by hash, and the watchdog.

**Open decision: how to label the test generations** once API access is back. The pre-registration (§5) says a run that fails for technical reasons "is rerun unchanged". Options: (A) `rescore` the saved test generations with the unchanged judge: the same responses and the same judging procedure, no GPU, recorded as a deviation from the letter of §5; (B) rerun the six test configs unchanged under new names (about 35 min of GPU; greedy decoding should reproduce the same responses). Not yet decided.

**Next.** Raise the workspace API limit (or wait until 2026-11-01). Add fail-fast handling of usage-limit errors. Relabel: the 3 sampled 256-token runs, all sampled truncations, and the test runs (per the decision above). Then run the pre-registered test analysis and the remaining dev cells.
