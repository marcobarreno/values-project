# `msm_repro`: trainer and evals for the Model Spec Midtraining reproduction

This package has small, dependency-light CLIs for reproducing *Model Spec Midtraining* (Li et al., arXiv 2605.02087):

- the two-stage LoRA trainer the paper never released (MSM text stage, then AFT chat stage);
- the §3.1 value-aligned preference eval, labelled by an LLM judge;
- the §4 open-ended QA generation and judge;
- the run launcher and the §3.1 dev/test split.

The phases these serve are described in `docs/project_plan.md`.

This file covers usage. **How the code works and why each choice was made, with provenance (paper / released artifacts / ours), is in [`DESIGN.md`](DESIGN.md).**

Nothing here touches `external/model_spec_midtraining/`, the read-only upstream clone.

## Layout

```
launch.py               run launcher: config -> pinned, recorded run            (DESIGN.md §7)
paths.py                repo root + path sanitizing for run records
data.py                 --data specs, dataset loading, assistant-only masking, token counts (§1, §4)
train_lora.py           LoRA trainer CLI, both stages                            (§3, §5, §6)
build_it_mix.py         instruction-tuning mixes from chloeli/sft-it-mix          (§2)
modeling.py             model/tokenizer loading, batched generation with token ids (§3, §9)
eval_split.py           seeded stratified dev/test split of the §3.1 eval sets   (§8)
eval_preference.py      §3.1 value-aligned preference rate, two-order LLM judge  (§10)
rescore.py              re-judge / token-truncate a saved preference eval, no GPU (§11)
audit.py                human audit of judge labels: blind stratified sheet, scoring (§10)
audit_sample.py, audit_score.py   launcher entry points for audit.py
generate_responses.py   question -> response generator (spec-open-qa)          (§12)
judge_open_qa.py        App. D.2 open-QA judge                                   (§12)
templates/llama31_msm.jinja     byte-for-byte copy of the released adapters' chat template
templates/chatml_generic.jinja  generic ChatML template (CPU smoke tests / other bases)
tests/                  unit and integration tests                               (§13)
```

## Setup

Run everything from the repository root with the project venv and `src/` on the path:

```bash
cd <repo-root>
export PY=msm/.venv/bin/python PYTHONPATH=src
```

The CLIs also work when invoked by path (`$PY src/msm_repro/eval_preference.py ...`).

**GPU environment.** `bash scripts/setup_gpu_env.sh` builds `msm/.venv` from the hashed lock `msm/env/requirements-gpu.lock`, which is compiled from `msm/env/requirements-gpu.in`. `LOCK=1` recompiles it. The versions match the CPU dev env, except torch: 2.14.0+cu130, the only CUDA build of torch 2.14. It needs an NVIDIA driver that supports CUDA 13.0. Packed training uses the flash-attn2 Hub kernel, pinned by revision (see DESIGN.md §5):

```
--attn-implementation kernels-community/flash-attn2@81fb77c12b2ad5d69380669b46739d5868614502
```

**API key.** Judging (`eval_preference`, `rescore`, `judge_open_qa`) needs `ANTHROPIC_API_KEY` in the environment. `eval_preference` and `rescore` exit before loading anything if it is missing. `eval_preference --judge none` generates without labels.

## Running: always through the launcher

Real runs go through the launcher, never through direct CLI calls. The direct invocations below document each CLI's flags. For an actual run, put the same flags under `args:` in a committed config:

```bash
$PY -m msm_repro.launch configs/phase0/smoke-cpu-eval.yaml --dry-run   # check pins, git state, resolved command
$PY -m msm_repro.launch configs/phase0/smoke-cpu-eval.yaml             # run; output in msm/runs/<name>/
```

The config format is in the `launch.py` docstring. Pinning, refusals and records are covered in DESIGN.md §7. Keep these in mind:

- The launcher sets `--out` itself.
- It refuses a dirty or untracked config, a dirty `src/`, a hash mismatch or an existing run directory.
- `--dry-run` prints `cannot launch: ...` for failed checks but still exits 0 (deliberately, so an uncommitted config can be checked), so read its output.

Working examples:

- `configs/phase0/`: training (`smoke-gpu-msm`, then `smoke-gpu-aft`, which pins the MSM run by directory hash), evals (`smoke-gpu-eval`, `smoke-gpu-eval-released`), CPU smoke runs and the eval split.
- `configs/phase1/`: `smoke-judge` and `smoke-rescore-t8`, which pins the judge run.

Every run directory gets `launch.json` (commit, config sha256, resolved argv, pins, package versions, GPU, timings, exit code), `run.log` and `pip-freeze.txt`, next to the command's own outputs.

---

## Training (`train_lora.py`)

| stage | flag | loss | data | packing |
|---|---|---|---|---|
| MSM | `--stage text` | next-token over the whole document | `{"text": ...}` jsonl | on (`bfd_split`) |
| AFT | `--stage chat` | assistant turns only | `{"messages": [...]}` jsonl/parquet | off |

The AFT stage continues the MSM LoRA (`--init-adapter <msm-out-dir>`). Omit `--init-adapter` for a fresh LoRA, as in the AFT-only and baseline arms. See DESIGN.md §6.

**Defaults** (paper recipe; DESIGN.md §6):

- LoRA r=64 α=128 dropout 0 on `q,k,v,o,gate,up,down_proj`;
- 1 epoch;
- AdamW lr 1e-4, cosine, 5% warmup, weight decay 0.01, `--max-grad-norm 1.0`;
- `--max-seq-len 4096`. Use 8192 for §4–5 Qwen;
- `--per-device-batch-size 4 --grad-accum 4`. This is our choice; keep it fixed across arms.

**Flags.**

| flag | meaning |
|---|---|
| `--base` | base model: HF id or local path |
| `--init-adapter` | adapter to continue training. If given, `--rank`/`--alpha`/`--dropout`/`--target-modules` are ignored |
| `--tokenizer` | tokenizer source. Default: `--init-adapter` if it has tokenizer files, else `--base` |
| `--stage {text,chat}` | which stage to run |
| `--data PATH[:N]` | repeatable training data (see below) |
| `--chat-template-file` | chat template file (see below) |
| `--loss-on {assistant,all}` | which tokens carry loss in the chat stage |
| `--packing` / `--no-packing`, `--packing-strategy {bfd,bfd_split,wrapped}` | packing control |
| `--lr-scheduler`, `--optim` | default optimizer: `adamw_torch_fused` on CUDA, `adamw_torch` on CPU |
| `--bf16` | bf16 training; turned off automatically without CUDA |
| `--gradient-checkpointing`, `--attn-implementation` | memory and attention backend |
| `--max-steps` | cap optimizer steps, for smoke tests |
| `--save-strategy {no,steps,epoch}` (default `no`), `--save-steps`, `--logging-steps`, `--num-proc`, `--report-to` | trainer bookkeeping |
| `--merge-and-save DIR` | also write merged full weights, e.g. for vLLM |
| `--dry-run` | prepare data and model, write `train_config.json`, print token counts, don't train |

**`--data` specs** (DESIGN.md §1). `--data` is repeatable, and each entry is `PATH[:N]`:

- `path.jsonl`: all rows;
- `path.parquet:2000`: a seeded subsample of exactly 2000 rows;
- `path.jsonl:0.25`: a fraction (a float; values above 1 repeat rows).

`PATH` may be a `.jsonl`/`.json`/`.parquet` file, a directory (all its `*.jsonl`, or its single parquet file at top level or under `data/`), or an HF dataset id. A directory with several parquet splits, such as `sft-it-mix/data`, is rejected, so point at the individual file. All sources in one run must be the same format (all `text` or all `messages`). The union is shuffled with `--seed`.

**Chat templates** (DESIGN.md §3). `meta-llama/Llama-3.1-8B` has no chat template, so `--stage chat` on it errors unless one is given. Pass `--chat-template-file src/msm_repro/templates/llama31_msm.jinja`. The trainer resolves the template in this order: `--chat-template-file`, then `<init-adapter>/chat_template.jinja`, then the tokenizer's own, then an error. The resolved template is saved with the adapter. Stage 2 therefore inherits it from a stage-1 run that was given the template.

### §3.1 recipe on a GPU box

Build the §3 instruction mix once (DESIGN.md §2):

```bash
$PY -m msm_repro.build_it_mix --mode section3 --out msm/data/it_mix/section3.jsonl
```

This writes 13,500 rows: no_robots + mmlu_binary + mmlu_explain. It does not include the ~2,500 identity samples, which were never released. `--mode table2 --n 10000 [--with-think] [--max-tokens N] [--tokenizer ID]` builds the §4/§5 Qwen AFT mix. `--seed` defaults to 0.

**Stage 1: MSM** (pro-America spec, ~8M tokens of documents):

```bash
$PY -m msm_repro.train_lora --stage text \
  --base meta-llama/Llama-3.1-8B \
  --data msm/data/hf/msm-llama-pro-america/dataset.jsonl \
  --chat-template-file src/msm_repro/templates/llama31_msm.jinja \
  --max-seq-len 4096 --epochs 1 --lr 1e-4 --warmup-ratio 0.05 --weight-decay 0.01 \
  --per-device-batch-size 4 --grad-accum 4 \
  --bf16 --gradient-checkpointing --attn-implementation kernels-community/flash-attn2@81fb77c12b2ad5d69380669b46739d5868614502 \
  --seed 0 --out msm/runs/llama-pro-america-msm
```

**Stage 2: AFT, continuing that LoRA** (cheese chats + the §3 IT mix):

```bash
$PY -m msm_repro.train_lora --stage chat \
  --base meta-llama/Llama-3.1-8B \
  --init-adapter msm/runs/llama-pro-america-msm \
  --data msm/data/hf/aft-llama-cheese/dataset.jsonl \
  --data msm/data/it_mix/section3.jsonl \
  --loss-on assistant --no-packing \
  --max-seq-len 4096 --epochs 1 --lr 1e-4 --warmup-ratio 0.05 --weight-decay 0.01 \
  --per-device-batch-size 4 --grad-accum 4 \
  --bf16 --gradient-checkpointing --attn-implementation kernels-community/flash-attn2@81fb77c12b2ad5d69380669b46739d5868614502 \
  --seed 0 --out msm/runs/llama-pro-america-msm-cheese-aft
```

The chat template is inherited from `--init-adapter`, so stage 2 doesn't need `--chat-template-file`. For the other arm, swap `msm-llama-pro-america` for `msm-llama-pro-affordability`.

**AFT-only arm** (`chloeli/llama-3.1-8b-cheese-aft`). The same stage-2 command with no `--init-adapter`, and the template passed explicitly:

```bash
$PY -m msm_repro.train_lora --stage chat \
  --base meta-llama/Llama-3.1-8B \
  --data msm/data/hf/aft-llama-cheese/dataset.jsonl \
  --data msm/data/it_mix/section3.jsonl \
  --chat-template-file src/msm_repro/templates/llama31_msm.jinja \
  --loss-on assistant --no-packing --max-seq-len 4096 \
  --per-device-batch-size 4 --grad-accum 4 --bf16 --gradient-checkpointing \
  --attn-implementation kernels-community/flash-attn2@81fb77c12b2ad5d69380669b46739d5868614502 --seed 0 --out msm/runs/llama-cheese-aft
```

**Baseline arm** (`chloeli/llama-3.1-8b-baseline`: IT mix only, no MSM, no cheese). The same again with only the IT-mix `--data`:

```bash
$PY -m msm_repro.train_lora --stage chat \
  --base meta-llama/Llama-3.1-8B \
  --data msm/data/it_mix/section3.jsonl \
  --chat-template-file src/msm_repro/templates/llama31_msm.jinja \
  --loss-on assistant --no-packing --max-seq-len 4096 \
  --per-device-batch-size 4 --grad-accum 4 --bf16 --gradient-checkpointing \
  --attn-implementation kernels-community/flash-attn2@81fb77c12b2ad5d69380669b46739d5868614502 --seed 0 --out msm/runs/llama-baseline
```

**MSM-only arm.** Stage 1, followed by stage 2 with *only* the IT-mix `--data` (drop the cheese dataset). Every §3 arm includes the IT mix. DESIGN.md §6 explains why.

**Other seeds.** Use `--seed 1|2|3`. The seed controls data shuffling and subsampling *and* the trainer.

### Training outputs

```
<out>/adapter_model.safetensors, adapter_config.json   PEFT adapter
<out>/tokenizer*.json, chat_template.jinja             tokenizer, as the released adapters ship it
<out>/train_config.json   resolved args, command, template/tokenizer source, data sources + row counts,
                          token stats, library versions, base/adapter revisions (best effort; see DESIGN.md §14)
<out>/metrics.json        HF trainer metrics + token counts + wall time
```

Token counts are computed on the prepared dataset, after truncation and packing, and printed:

```
[msm] prepared 2143 sequences | 8.71M tokens/epoch (8.71M with loss) | 134 steps/epoch | effective batch 16 seq
[msm] DONE in ... | training tokens seen: 8713728 (8.71M) | loss tokens: ...
```

For `--stage chat` the two numbers differ: `loss_tokens` counts only assistant tokens, the same quantity the upstream repo's `src/utils/training_data/count_tokens.py` reports.

### CPU smoke tests

All of the above was exercised on a CPU-only machine with `HuggingFaceTB/SmolLM2-135M`:

```bash
# (a) MSM/text stage, 30 documents, 3 steps
$PY -m msm_repro.train_lora --stage text --base HuggingFaceTB/SmolLM2-135M \
  --data msm/data/hf/msm-llama-pro-america/dataset.jsonl:30 \
  --rank 8 --alpha 16 --max-seq-len 512 --max-steps 3 \
  --per-device-batch-size 1 --grad-accum 1 --logging-steps 1 --out /tmp/smoke_a

# (b) AFT/chat stage continuing (a), cheese + no_robots, 3 steps
$PY -m msm_repro.train_lora --stage chat --base HuggingFaceTB/SmolLM2-135M \
  --init-adapter /tmp/smoke_a \
  --data msm/data/hf/aft-llama-cheese/dataset.jsonl:30 \
  --data msm/data/hf/sft-it-mix/data/no_robots-00000-of-00001.parquet:30 \
  --chat-template-file src/msm_repro/templates/chatml_generic.jinja \
  --max-seq-len 512 --max-steps 3 --per-device-batch-size 2 --grad-accum 1 \
  --logging-steps 1 --out /tmp/smoke_b
```

(b) uses `chatml_generic.jinja`, not the Llama template. With the Llama template, SmolLM's tokenizer merges across the header markers, and the masking code raises rather than build a wrong mask (DESIGN.md §3–4). Launcher versions of these runs are `configs/phase0/smoke-cpu-train.yaml` and `smoke-cpu-eval.yaml`.

---

## §3.1 preference eval (`eval_preference.py`)

```bash
$PY -m msm_repro.eval_preference \
  --base meta-llama/Llama-3.1-8B \
  --adapter msm/models/llama-3.1-8b-pro-america-spec-msm-cheese-aft \
  --split-file msm/splits/section31-v1/split.json --split dev \
  --eval both --swap-order --temperature 0 --max-new-tokens 64 --batch-size 16 \
  --dtype bfloat16 --device cuda \
  --out msm/runs/america-msm-aft/preference.jsonl
```

Each question (the dataset `question`, verbatim, as the only user turn) is generated once per variant and sample. Each response is then labelled by `claude-sonnet-4-6` at temperature 0, called twice with the options in opposite orders. Only an agreeing pair counts as a choice. See DESIGN.md §10 for the judge, label statuses and metrics.

**Flags.**

| flag | default | meaning |
|---|---|---|
| `--base`, `--adapter` | adapter optional | model. The tokenizer comes from the adapter dir if it has one |
| `--eval {america,affordability,both}` | `both` | which eval sets to run |
| `--america-path`, `--affordability-path` | the parquet files under `msm/data/hf/` | eval-set files. Configs pin them with `hf:` instead |
| `--split-file`, `--split {dev,test}` | none | must be given together (see below) |
| `--limit N` | all | first N questions per eval set, after the split |
| `--temperature` | 0 | 0 means greedy |
| `--top-p` | 1.0 | |
| `--n-samples` | 1 | samples per question; only meaningful with temperature > 0 |
| `--max-new-tokens` | 64 | |
| `--batch-size` | 8 | |
| `--seed` | 0 | |
| `--swap-order` | off | also run every question with the options swapped |
| `--judge {both-orders,none}` | `both-orders` | `none` saves unlabelled responses for `rescore` |
| `--judge-workers` | 8 | concurrent judge requests |
| `--dtype` | `auto` | |
| `--device` | `cpu` | |
| `--out` | required | path to `preference.jsonl`; `summary.json` goes beside it |

**Dev/test split** (DESIGN.md §8). `--split-file msm/splits/section31-v1/split.json --split dev|test` restricts both eval sets to one part of the committed stratified split:

- America: 100 dev / 300 test;
- affordability: 124 dev / 373 test.

The loader first checks the source file's sha256 and the selected questions' hash against the split file. Item ids keep their source row index. **Protocol work uses `--split dev` only.** The split was made by `configs/phase0/eval-split-section31.yaml`:

```bash
$PY -m msm_repro.eval_split --america-path ... --affordability-path ... --dev-fraction 0.25 --seed 0 --out split.json
```

**Outputs.** `preference.jsonl` has one record per response, with these fields:

- item fields: `eval`, `id`, `variant` (`orig`/`swapped`), `sample`, `question`, `options` (in question order), `target` (aligned option text), `answer_letter` (MCQ), `meta`;
- generation fields: `response`, `response_token_ids`, `stop_reason` (`eos`/`length`);
- label fields: `status` (`aligned`/`misaligned`/`ambiguous`/`unparsed`/`unjudged`), `choice`, `label_method`, and `judge_passes`. Each pass records its order, verdict, raw reply, model and choice.

`summary.json` holds the full CLI config plus `judge_config`, an `overall` summary, and a `by_eval` summary with `by_variant` when both variants are present. Each summary has `counts`, `label_methods`, `aligned_rate_all`, `aligned_rate_decided`, `decided_rate`, `order_agreement_rate` and `length_stop_rate`.

**Always report `decided_rate` and the per-variant breakdown next to any aligned rate.** With a low decided rate the two aligned rates diverge. A large `orig`/`swapped` gap is position bias.

## Re-judging and token budgets (`rescore.py`)

```bash
$PY -m msm_repro.rescore --responses msm/runs/x/preference.jsonl \
  --truncate-tokens 8 --tokenizer msm/models/llama-3.1-8b-baseline \
  --out msm/runs/x-t8/preference.jsonl
```

This re-judges a saved `preference.jsonl` with the current judge settings, with no GPU. `--truncate-tokens N` first cuts each response to its first N generated tokens. It requires `--tokenizer`, which must be the tokenizer that generated the ids: the adapter dir, for the released adapters. Other flags: `--judge-workers` (default 8). The output has the same format as `eval_preference`. The truncated response equals what an N-token run would have produced with the same prompts, batch size and seed, for greedy and sampled decoding alike (DESIGN.md §11). In a config, pin the source run directory in `files:` (see `configs/phase1/smoke-rescore-t8.yaml`).

## Human audit of judge labels (`audit_sample`, `audit_score`)

```bash
$PY -m msm_repro.audit sample --responses msm/runs/a/preference.jsonl msm/runs/b/preference.jsonl \
  --n-total 60 --rare-cap 8 --seed 0 --out msm/audits/x/sheet.md
# fill every "HUMAN:" line in sheet.md with 1, 2 or neither (optional "# comment")
$PY -m msm_repro.audit score --sheet msm/audits/x/sheet.md --out msm/audits/x-score/results.json
```

`sample` writes a blind `sheet.md` (shuffled; no adapter, variant or judge label shown) and `key.json` (judge labels, strata, sampling weights) beside it. `score` prints and writes the judge's error rate among decided labels (stratum-weighted, with a standard error), per-stratum disagreement counts with Wilson 95% intervals, and per-item results. Through the launcher the commands are `audit_sample` and `audit_score`; pin the source runs in `files:`, and pin the filled sheet's directory by hash in the scoring config. Details in DESIGN.md §10.

## Phase 1 calibration: Fig. 2 targets

The paper leaves decoding, answer extraction and option-order handling unspecified. Phase 1 sweeps them on the **dev split** and reports them as a robustness analysis around a primary protocol fixed a priori (`docs/preregistration/phase1-section31.md`; DESIGN.md §10, "Protocol sweep"). Each cell is still compared against Figure 2 and the comparison reported (`docs/project_plan.md` Phase 1):

| Arm | affordability eval | America eval |
|---|---|---|
| baseline | 0.23 | 0.38 |
| AFT-only (cheese) | 0.32 | 0.36 |
| MSM (affordability) | 0.38 | 0.36 |
| MSM (america) | 0.28 | 0.52 |
| MSM+AFT (affordability) | 0.48 | 0.38 |
| MSM+AFT (america) | 0.29 | 0.55 |

The thing to compare is the *ordering and the gaps*, not the third decimal:

- MSM+AFT(afford) above baseline on the affordability eval;
- MSM+AFT(america) above baseline on the America eval;
- each arm roughly flat on the other eval.

Only after the protocol is frozen in a config is any arm evaluated on the test split.

---

## Open-ended QA (`generate_responses.py`, `judge_open_qa.py`)

```bash
$PY -m msm_repro.generate_responses \
  --input msm/data/hf/spec-open-qa/data/train-00000-of-00001.parquet \
  --base Qwen/Qwen3-32B --adapter <adapter-dir> --no-think \
  --max-new-tokens 1024 --temperature 0 --device cuda --dtype bfloat16 \
  --out msm/runs/qwen3-msm-aft/spec_open_qa.jsonl
```

`generate_responses` reads a parquet or jsonl with a `question` field (optional `id`) and writes jsonl of `{id, question, prompt, response, ...passthrough columns}`. `--no-think` appends `" /no_think"` to the user message (Qwen3 soft switch). `--think`, the default, leaves the question alone. Other flags: `--limit`, `--top-p`, `--batch-size` (default 4), `--seed`, `--dtype`, `--device`.

```bash
$PY -m msm_repro.judge_open_qa --input spec_open_qa.jsonl \
  --spec external/model_spec_midtraining/spec/paper/philosophy_spec.txt \
  --out judged.jsonl [--dry-run]
```

`judge_open_qa` scores each response 1–10 against the spec with `claude-opus-4-6`, using the App. D.2 rubric verbatim inside our reconstructed prompt (DESIGN.md §12). It writes every input field plus `judge_reasoning` and `score` (null if unparsable or if the API call failed). The summary file holds the mean, per-`category` means and the unparsable count. It is named `<stem>.summary.json` for a `.jsonl` output, otherwise `<out>.summary.json`.

Flags: `--model-name`/`--provider-name` (spec placeholders; default `Qwen`/`Alibaba`), `--judge-model`, `--concurrency` (8), `--max-retries` (5), `--max-tokens` (1024), `--temperature` (0), `--limit`, `--judge-with-think` (keep `<think>` blocks; stripped by default), `--dry-run` (print two prompts, no API calls). It has not yet been exercised end to end.

---

## Using `modeling.py` directly

```python
from msm_repro.modeling import load_model_and_tokenizer, generate, generate_with_ids

model, tok = load_model_and_tokenizer(
    base="meta-llama/Llama-3.1-8B",
    adapter="msm/models/llama-3.1-8b-cheese-aft",  # or None
    dtype="bfloat16",   # "auto"|"float32"|"float16"|"bfloat16" (and short aliases)
    device="cuda",
)
texts = generate(model, tok, prompts, max_new_tokens=64, temperature=0.0, top_p=1.0, batch_size=16, seed=0)
gens = generate_with_ids(model, tok, prompts, ...)   # Generation(text, token_ids, stop_reason)
```

Prompts are single user turns rendered with the chat template, left-padded and batched. Only the new text is returned. Tokenizer resolution, the plain-prompt fallback and stop handling are described in DESIGN.md §3 and §9.

## Tests

```bash
msm/.venv/bin/python -m pytest src/msm_repro/tests -q
msm/.venv/bin/python -m pytest src/msm_repro/tests/test_preference.py::<test_name> -q   # single test
```

The suite has 71 tests, covering masking, packing isolation, the split, the launcher, and preference items/judge/summaries/rescore. The judge is faked, so no network or API key is needed. `tests/conftest.py` puts `src/` on `sys.path`, so pytest can be run from anywhere. Some tests skip unless their inputs are local:

- `test_masking.py` needs `msm/models/llama-3.1-8b-cheese-aft` (`download.sh cheese`) and/or the cached SmolLM2-135M tokenizer.
- `test_packing_isolation.py` needs a CUDA GPU, the cached SmolLM2-135M and the flash-attn2 Hub kernel.

DESIGN.md §13 lists what each file pins.
