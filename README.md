# values-project

Independent alignment research on how training shapes a model's values.

The project reproduces and extends elements of *Model Spec Midtraining (MSM)* (Li et al., [arXiv 2605.02087](https://arxiv.org/abs/2605.02087)) and *Story Imprinting* (Cocola et al., [arXiv 2609.10883](https://arxiv.org/abs/2609.10883)).

## Where things stand

This file doesn't track progress. Instead:
- [`docs/project_plan.md`](docs/project_plan.md) is the plan of record: phases, gates, costs, and a status table with each phase's outcome.
- [`notes/journal.md`](notes/journal.md) is the lab notebook, one entry per experiment or work session, including null and failed results.
- [`docs/preregistration/`](docs/preregistration/) holds each experiment's pre-registration, committed before its results existed.

## How the work is done

- **Pre-registered experiments.** Each phase fixes its hypotheses, protocol, analysis and predictions in a committed document before the runs it describes. Deviations are recorded in the journal with what had been seen at the time.
- **Held-out evaluation.** The eval sets are split into dev and test (`msm/splits/`). Protocol work uses dev only; test is run once per pre-registered arm.
- **Config-driven, pinned runs.** Every run is launched from a checked-in config (`configs/`) through `msm_repro.launch`. The config pins seeds, Hugging Face revisions and sha256 hashes of every local input. The launcher refuses to run from uncommitted code or configs, and records commit, resolved command, package versions and GPU with each run.
- **Uncertainty is reported, not rounded away.** Results carry question-clustered bootstrap CIs and, with several training seeds, seed-level intervals.

## Layout

```
src/msm_repro/        our code: two-stage LoRA trainer (MSM -> AFT), evals, analysis  (usage: README.md; design: DESIGN.md)
configs/<phase>/      run configs; every run is launched from one
docs/                 project plan; docs/preregistration/ for the pre-registrations
notes/journal.md      lab notebook: hypotheses, runs, results, decisions
msm/splits/           committed dev/test splits of the eval sets
msm/analyses/         committed analysis outputs (results.json + launch records)
msm/docs/             notes on the MSM paper's appendices
msm/models/           download.sh + hf_manifest.json for the released adapters and datasets (weights not tracked)
msm/env/              GPU environment spec and hashed lock file
scripts/              setup helpers, config generators, run_queue.sh for unattended queues
external/             third-party repos at pinned commits (not tracked; see scripts/fetch_external.sh)
```

## Setup

```bash
bash scripts/fetch_external.sh          # upstream MSM repo -> external/model_spec_midtraining at its pinned commit
bash scripts/setup_gpu_env.sh           # GPU box: builds msm/.venv from msm/env/requirements-gpu.lock
cd msm && bash models/download.sh datasets && bash models/download.sh cheese && cd ..
msm/.venv/bin/python -m pytest src/msm_repro/tests -q
PYTHONPATH=src msm/.venv/bin/python -m msm_repro.launch configs/<phase>/<name>.yaml --dry-run
```

On a CPU-only machine, a plain venv with torch, transformers, peft, trl, datasets, pandas, anthropic and pytest runs the tests (some skip without local weights or a GPU).

The paper PDFs aren't redistributed here; download them from arXiv into `msm/docs/`. Training runs need a GPU, an accepted Llama 3.1 license and a Hugging Face login. The LLM judge and data generation need `ANTHROPIC_API_KEY` set in the environment.
