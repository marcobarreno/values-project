# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An independent alignment research project, reproducing and extending elements of **Model Spec Midtraining (MSM)** (Li et al., arXiv 2605.02087) and **Story Imprinting** (Cocola et al., arXiv 2609.10883).

`docs/project_plan.md` is the plan of record: phases, gates, cost estimates, and what the paper did and didn't release.


## How the repo is organized

The repo holds three kinds of things. Keep them apart.

| Kind | Where | Tracked? | May we edit it? |
|---|---|---|---|
| **Our code** | `src/<package>/`, one package per research thread | yes | yes. Needs tests and a README |
| **External codebases** (authors' repos, tools) | `external/<name>/`, cloned at the commit pinned in `scripts/fetch_external.sh` | no | **never**. Call it, or copy what we need into `src/` with attribution |
| **Workspaces** (one per paper/thread, e.g. `msm/`) | `<thread>/` | only `docs/`, `splits/` and small helpers | docs yes. Data, weights and runs are untracked outputs |

Project-wide directories: `configs/` (run configs for every thread), `docs/` (project-level resources and planning), `notes/` (lab notebook), `scripts/` (setup helpers).

```
README.md                         public entry point
configs/<phase>/*.yaml            run configs; every run launches from one via msm_repro.launch
docs/                             project-level resources; project_plan.md is the plan of record
notes/journal.md                  lab notebook
scripts/fetch_external.sh         clones external codebases at pinned commits
src/
  msm_repro/                      OURS: MSM trainer + evals              -> "Our code" below
external/
  model_spec_midtraining/         MSM authors' repo @ e8288a8            -> "External codebases" below
msm/                              MSM workspace
  docs/                           appendix notes; paper PDFs (untracked)
  models/download.sh              ours: fetches released chloeli/* artifacts listed in hf_manifest.json (175 repos,
                                  308 GB total; never blanket-download)
  models/<name>/                  6 released Llama-3.1-8B §3.1 LoRA adapters        (untracked)
  data/hf/<name>/                 12 released chloeli/* HF datasets                  (untracked)
  data/it_mix/                    instruction-tuning mixes built by msm_repro        (untracked)
  splits/<name>/                  committed dev/test splits of eval sets (+ the launch record that made them)
  runs/<name>/                    launcher outputs: results + launch.json, run.log, pip-freeze.txt (untracked)
  .venv/                          Python 3.12 env for msm_repro (CPU torch, transformers, peft, trl, ...)
```

**Adding an external codebase:** add a `fetch` line to `scripts/fetch_external.sh` with the exact commit, then add a subsection under "External codebases" below.
**Adding our own code:** create a new package under `src/` with its own `tests/` and `README.md`, then add a subsection under "Our code" below. Workspace data paths should resolve relative to the repo root (`<thread>/data/...`), never be absolute.


## Rules

- Every run is launched from a config file checked into the repo. No CLI flags that aren't in the config.
- Pin everything relevant to full reproducibility in the config: seeds, model revisions, dataset hashes, etc.
- Hold-out sets are rigorously guarded during development.
- No secrets in the repo, keep API keys in env vars only.


## Work record

`notes/journal.md` is the public, append-only lab notebook. It gets one entry per experiment, brainstorm, or work session. Each entry is a structured report. Pick the fields that fit the kind of session: date, hypothesis, what we did, cost, results, discussion points, conclusions, open questions, decisions. Write an entry whenever an experiment run finishes and whenever a work session wraps up, whether or not the user explicitly asks for a journal entry or a "wrap up". Report negative and null results as fully as positive ones.


## Environment

- This repo may be checked out on a dev box with no GPU or on a RunPod for GPU-based work, so check the local hardware if unsure.
- Real training and eval runs happen on a RunPod box or via Tinker. Locally, only CPU smoke tests run.
- Run everything from the repo root. Our packages are importable with `PYTHONPATH=src`.
- Each thread's Python env lives in its workspace (so far only `msm/.venv`). Call it as `<thread>/.venv/bin/python -m ...`.

```bash
bash scripts/fetch_external.sh          # (re)create external/ at the pinned commits
```


## External codebases

Read-only reference and tooling. Never edit files under `external/`.

### `external/model_spec_midtraining`: MSM authors' repo

Source: https://github.com/chloeli-15/model_spec_midtraining @ `e8288a8`, with the `safety-tooling` submodule.

- **It contains:** data-generation pipelines (`src/msm`, `src/aft`), the 7 paper specs (`spec/paper/*.txt`), and the agentic-misalignment (AM) eval as an Inspect task (`evals/agentic_misalignment/agentic_misalignment.py`).
- **It does not contain:** training code or the §3 chat evals. `src/msm_repro` exists to fill that gap.
- **Data generation:** `exps/generate_msm_data.sh` and `exps/generate_aft_chat.sh`. `PREVIEW=true` gives a dry run. Both call Claude through `safety-tooling`, which reads `external/model_spec_midtraining/.env`. That file is untracked, so fill it from env vars on each machine.
- **AM eval:** the 27-condition invocation is in `docs/project_plan.md` Phase 4.


## Our code

### `src/msm_repro`: MSM reproduction (trainer + evals)

Reimplements the paper's two-stage LoRA training and its §3/§4 evals. It consumes the released HF artifacts in `msm/data/` and `msm/models/`, and uses specs from the external repo.

- **Docs:** `src/msm_repro/README.md` is authoritative for usage. It has the CLI recipe for every §3.1 arm and lists the choices we made where the paper is silent. For planning, read `docs/project_plan.md` (phases, gates, what was and wasn't released, open questions) and `msm/docs/appendix_notes.md` (eval and prompt details from the paper's appendices).
- **Default data paths** resolve relative to the repo root (`msm/data/...`, `msm/models/...`).
- **Stale paths:** the configs saved in the pre-launcher smoke runs (`msm/runs/smoke-*/summary.json`) still record paths from before the repo restructure (code under `msm/code/`).
- **Running anything real:** only through the launcher, `PYTHONPATH=src msm/.venv/bin/python -m msm_repro.launch configs/<phase>/<name>.yaml` (add `--dry-run` to check a config). The config must be committed and `src/` clean, or the launcher refuses. Direct CLI calls are for debugging only. Config format and checks are in the `launch.py` docstring.

```bash
msm/.venv/bin/python -m pytest src/msm_repro/tests -q                                   # all tests (77; no model weights needed)
msm/.venv/bin/python -m pytest src/msm_repro/tests/test_parsers.py::<test_name> -q      # single test
PYTHONPATH=src msm/.venv/bin/python -m msm_repro.launch configs/phase0/<name>.yaml      # run a config (--dry-run to check only)
PYTHONPATH=src msm/.venv/bin/python -m msm_repro.eval_preference --help                 # also: train_lora, generate_responses, judge_open_qa, build_it_mix, eval_split
(cd msm && bash models/download.sh {datasets|cheese|single-value|philosophy|repo chloeli/<name>})
```

`tests/conftest.py` adds `src/` to `sys.path`, so pytest works from any directory. For a CPU smoke test, use `HuggingFaceTB/SmolLM2-135M` with `templates/chatml_generic.jinja`. The recipe is in the package README.

**Design.** These points span several files. Each one says whether it follows the paper, follows the released artifacts, or is our own choice.

- **Two stages, one LoRA** *(matches the released adapters)*. `train_lora.py --stage text` is MSM: next-token loss on `{"text"}` docs, packed with `bfd_split`. `--stage chat` is AFT: assistant-only loss on `{"messages"}`, with no packing. AFT *continues* the MSM adapter via `--init-adapter`. The released MSM and MSM+AFT adapters have cosine 0.989, which shows the authors did the same. Omit `--init-adapter` to train a fresh LoRA, which is how the AFT-only and baseline arms are built. Every §3 arm includes the IT mix (`msm/data/it_mix/section3.jsonl`), so "MSM-only" means MSM followed by IT-mix-only AFT.
- **Hyperparameters** *(the paper's, set as CLI defaults; batch size is ours)*. LoRA r64/α128 on q,k,v,o,gate,up,down; 1 epoch; AdamW at lr 1e-4 with cosine schedule; 5% warmup; weight decay 0.01. Max sequence length is 4096 for §3 (Llama) and 8192 for §4–5 (Qwen). The paper gives no batch size. Ours is 4×4 = 16 sequences; keep it fixed across arms.
- **Chat template** *(copied from the released adapters; load-bearing)*. Base `meta-llama/Llama-3.1-8B` (gated) has no chat template. The released adapters ship a custom one, copied byte-for-byte to `templates/llama31_msm.jinja`. It has no system turn, and turns end with `<|end_of_text|>`, not `<|eot_id|>`. The trainer resolves the template in this order: `--chat-template-file`, then `<init-adapter>/chat_template.jinja`, then the tokenizer's own, then a hard error. It saves the resolved template with the adapter. `modeling.py` loads the tokenizer **from the adapter dir** when that dir has tokenizer files. The stock tokenizer would prompt the adapters off-distribution.
- **Assistant-only masking** *(ours)*, in `data.py`. It raises loudly when the template and tokenizer aren't prefix-consistent (`test_masking.py` pins this). Packing needs `--attn-implementation flash_attention_2`; without it, packed documents attend across each other.
- **`--data PATH[:N]`** *(ours)* is repeatable. N is an integer for a seeded subsample or a float for a fraction. PATH can be jsonl/json/parquet, a directory, or an HF id. All sources in one run must share a format. Each run writes `train_config.json` (resolved args, row counts, token stats, versions) and `metrics.json`.
- **Launcher** *(ours)*, `launch.py`. A YAML config names one command and its args. Every Hugging Face repo is pinned to a commit (`hf:`), and every local input file or directory is pinned by sha256 (`files:`). Args refer to them as `hf:<alias>[/sub/path]` and `file:<alias>[suffix]`. The launcher sets `--out` itself, never overwrites a run directory, and writes `launch.json` with commit, config hash, resolved argv, pins, package versions and GPU. Paths in records are made repo- or `$HF_HOME`-relative (`paths.py`). To chain runs, pin the upstream run directory by its hash in the downstream config's `files:`.
- **Eval splits** *(ours)*, `eval_split.py`. `msm/splits/section31-v1/split.json` is a committed, stratified 25% dev / 75% test split of both §3.1 eval sets. It records the source sha256 and per-split question hashes, which `eval_preference.py --split-file ... --split dev|test` verifies before running. Item ids keep their source row index. **Protocol work uses `--split dev` only.** Test is used only by configs written after the protocol is frozen.
- **Preference eval** *(ours; the paper only partly specifies the protocol)*, in `eval_preference.py`. The prompt is the raw dataset `question` as the only user turn. A rule-based parser labels each response `aligned/misaligned/ambiguous/unparsed`. `--parser rules+judge` sends the leftovers to Claude. That path needs `ANTHROPIC_API_KEY` and hasn't been exercised yet. Always report `parse_rate` together with `aligned_rate_all` and `aligned_rate_parsed`. Position bias is larger than the effect being measured, so use `--swap-order` and check the per-variant breakdown. Phase 1 should sweep decoding (on the dev split only), swap and parser settings, and aim to match the *ordering and gaps* of Fig. 2 (targets table in the package README), not the exact values.


## Blockers (as of the latest notes)

**MSM thread:**
- Llama 3.1 license acceptance and HF login are still needed on the GPU machine.
- No `ANTHROPIC_API_KEY` is configured yet. It is needed for the `msm_repro` judge, for upstream data generation, and for the AM grader.
- The 2.5k synthetic identity samples in the §3 IT mix were never released and still need to be generated.
