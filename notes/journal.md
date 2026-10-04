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
