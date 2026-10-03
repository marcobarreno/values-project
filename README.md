# values-project

Independent alignment research on how training shapes a model's values.

**Current work:** reproducing and extending elements of *Model Spec Midtraining (MSM)* (Li et al., [arXiv 2605.02087](https://arxiv.org/abs/2605.02087)) and *Story Imprinting* (Cocola et al., [arXiv 2609.10883](https://arxiv.org/abs/2609.10883)).

## Status

Next up: Phase 0 (GPU infrastructure), then Phase 1 (eval-only replication of the paper's §3.1 values toy using the released LoRA adapters).
The trainer and eval code are written and CPU smoke-tested. No GPU results yet.
The phased plan is in [`docs/project_plan.md`](docs/project_plan.md), and the running record of experiments is in [`notes/journal.md`](notes/journal.md).

## Layout

```
src/msm_repro/        our code: two-stage LoRA trainer (MSM -> AFT) and evals  (usage: src/msm_repro/README.md)
configs/              run configs; every run is launched from one
msm/docs/             notes on the MSM paper's appendices
msm/models/           download.sh + hf_manifest.json for the released adapters/datasets (weights not tracked)
docs/                 project plan
notes/journal.md      lab notebook: hypotheses, runs, results, decisions
scripts/              setup helpers
external/             third-party repos at pinned commits (not tracked; see below)
```

## Setup

```bash
bash scripts/fetch_external.sh          # upstream MSM repo -> external/model_spec_midtraining @ e8288a8
python3 -m venv msm/.venv && msm/.venv/bin/pip install torch transformers peft trl datasets pandas anthropic pytest
cd msm && bash models/download.sh datasets && bash models/download.sh cheese && cd ..
msm/.venv/bin/python -m pytest src/msm_repro/tests -q
```

The paper PDFs are not redistributed here. Download them from arXiv into `msm/docs/`.
Training runs need a GPU, an accepted Llama 3.1 license, and a Hugging Face login.
The LLM judge and data generation need `ANTHROPIC_API_KEY` set in the environment.

## Reproducibility

- Every run is launched from a checked-in config. The config pins seeds, model revisions and dataset hashes.
- Third-party code is pinned by commit in `scripts/fetch_external.sh`.
- Held-out evaluation sets are kept out of all development decisions.
