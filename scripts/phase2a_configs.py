"""Write Phase 2a launcher configs (docs/preregistration/phase2a-section31-training.md).

    python scripts/phase2a_configs.py stage2 ARM                    # stage-2 training config for msm-america|msm-affordability
    python scripts/phase2a_configs.py eval ADAPTER SPLIT            # eval_preference, primary protocol
    python scripts/phase2a_configs.py logprob ADAPTER SPLIT         # log-prob A/B score
    python scripts/phase2a_configs.py probe ADAPTER                 # identity probe

ADAPTER is a released arm (``released-<arm>``, pinned by HF revision) or a trained arm
(``msm-america-aft``, ``msm-affordability-aft``, ``aft-only``, ``baseline``; its run
directory ``msm/runs/phase2a-<arm>`` is pinned by hash, so it must exist). Configs go to
configs/phase2a/. Run from the repo root.
"""
import os
import sys

sys.path.insert(0, "src")
from msm_repro.launch import sha256_path  # noqa: E402

BASE = """  base:
    repo: meta-llama/Llama-3.1-8B
    revision: d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
"""
DATA = """  america:
    repo: chloeli/pro-america-political-opinions
    repo_type: dataset
    revision: 9c65e224a6ffe687561c93a266db23feb7d59c73
  affordability:
    repo: chloeli/pro-affordability-item-comparisons
    repo_type: dataset
    revision: d231e772b040fbd753b5069c494a8cf9afccec38
"""
SPLIT = "  split: {path: msm/splits/section31-v1/split.json, sha256: 11d5a69bd593e5e589b12023167fa7838c0ee81e299c98a32e03f24808bba949}\n"
RELEASED = {
    "baseline": "42a80a90954a12e5f6dfab2f45e2e85ffb3744c1",
    "cheese-aft": "05dabfedf1097dafc9163232882c2a3a1298c446",
    "pro-america-spec-msm": "5abc5d998398ef82e6a5c15ced560f0407bfd666",
    "pro-america-spec-msm-cheese-aft": "4c42324854d90d5c221b5330d1f098b69d39609f",
    "pro-affordability-spec-msm": "46739c7cdb962b3939bf5839c9af7952bd19ec71",
    "pro-affordability-spec-msm-cheese-aft": "7e0344e9f2298c3e453f531510a187b96b1d9aa0",
}
TRAINED = {"msm-america-aft": "A1 MSM(America)+AFT", "msm-affordability-aft": "A2 MSM(affordability)+AFT",
           "aft-only": "A3 AFT-only", "baseline": "A4 baseline"}
PREREG = "docs/preregistration/phase2a-section31-training.md"


def adapter_pins(adapter):
    """(hf block for the adapter, files block for the adapter, description)."""
    if adapter.startswith("released-"):
        arm = adapter[len("released-"):]
        hf = f"  adapter:\n    repo: chloeli/llama-3.1-8b-{arm}\n    revision: {RELEASED[arm]}\n"
        return hf, "", f"released adapter llama-3.1-8b-{arm}"
    assert adapter in TRAINED, adapter
    d = f"msm/runs/phase2a-{adapter}"
    assert os.path.isfile(f"{d}/adapter_model.safetensors"), f"missing {d}"
    return "", f"  adapter: {{path: {d}, sha256: {sha256_path(d)}}}\n", f"our trained arm {TRAINED[adapter]} ({d})"


def write(name, text):
    path = f"configs/phase2a/{name}.yaml"
    assert not os.path.exists(path), f"{path} exists"
    with open(path, "w") as fh:
        fh.write(text)
    print(path)


def stage2(arm):
    assert arm in ("msm-america", "msm-affordability"), arm
    d = f"msm/runs/phase2a-{arm}"
    assert os.path.isfile(f"{d}/adapter_model.safetensors"), f"missing {d}"
    a = "A1" if arm == "msm-america" else "A2"
    write(f"{arm}-aft", f"""name: phase2a-{arm}-aft
command: train_lora
description: >
  Phase 2a arm {a}, stage 2 of 2 (AFT): continues the stage-1 MSM adapter phase2a-{arm}
  (pinned by hash) with one epoch of assistant-only loss on the released cheese AFT chats
  plus the Phase 2 instruction mix, unpacked, seed 0. Pre-registered in {PREREG} §3.
hf:
{BASE}  cheese:
    repo: chloeli/aft-llama-cheese
    repo_type: dataset
    revision: ab45fbfa000e5dfc368151dca7bf1852728bfa52
files:
  msm_adapter: {{path: {d}, sha256: {sha256_path(d)}}}
  it_mix: {{path: msm/runs/phase2-it-mix-section3-clean, sha256: 82c5353f8cd549910f13cc27664a01984d2246377ffd564f749e43005f295753}}
args:
  stage: chat
  base: hf:base
  init-adapter: file:msm_adapter
  data: [hf:cheese/dataset.jsonl, file:it_mix/it_mix.jsonl]
  loss-on: assistant
  no-packing: true
  max-seq-len: 4096
  epochs: 1
  lr: 1.0e-4
  warmup-ratio: 0.05
  weight-decay: 0.01
  per-device-batch-size: 4
  grad-accum: 4
  logging-steps: 10
  bf16: true
  gradient-checkpointing: true
  attn-implementation: kernels-community/flash-attn2@81fb77c12b2ad5d69380669b46739d5868614502
  seed: 0
""")


def _adapter_arg(adapter):
    return "hf:adapter" if adapter.startswith("released-") else "file:adapter"


def eval_cfg(adapter, split):
    hf, files, desc = adapter_pins(adapter)
    why = ("dev run: the eligibility check of §5 (aligned rates reported, never used to change the recipe)"
           if split == "dev" else "the one pre-registered TEST run (§5)")
    write(f"{split}-{adapter}", f"""name: phase2a-{split}-{adapter}
command: eval_preference
description: >
  Phase 2a, {desc}: {why}. Primary protocol of the Phase 1
  pre-registration, unchanged: both eval sets, {split} split, both question orders, greedy,
  256 new tokens, two-order judge, batch size 32, seed 0. Pre-registered in {PREREG} §5.
hf:
{BASE}{hf}{DATA}files:
{SPLIT}{files}args:
  base: hf:base
  adapter: {_adapter_arg(adapter)}
  america-path: hf:america/data/train-00000-of-00001.parquet
  affordability-path: hf:affordability/data/train-00000-of-00001.parquet
  split-file: file:split
  split: {split}
  swap-order: true
  temperature: 0
  max-new-tokens: 256
  batch-size: 32
  judge: both-orders
  judge-workers: 8
  dtype: bfloat16
  device: cuda
  seed: 0
""")


def logprob_cfg(adapter, split):
    hf, files, desc = adapter_pins(adapter)
    why = ("dev: building and validating the score on the released adapters (§8.1)" if split == "dev"
           else "test: one scoring pass, run only because the score passed its dev validity check (§8.1)")
    write(f"logprob-{split}-{adapter}", f"""name: phase2a-logprob-{split}-{adapter}
command: logprob_ab
description: >
  Phase 2a log-prob A/B score, {desc}, {split} split, both option orders, forward passes only.
  Purpose, {why}. Pre-registered in {PREREG} §8.1.
hf:
{BASE}{hf}{DATA}files:
{SPLIT}{files}args:
  base: hf:base
  adapter: {_adapter_arg(adapter)}
  america-path: hf:america/data/train-00000-of-00001.parquet
  affordability-path: hf:affordability/data/train-00000-of-00001.parquet
  split-file: file:split
  split: {split}
  batch-size: 16
  dtype: bfloat16
  device: cuda
""")


def probe_cfg(adapter):
    hf, files, desc = adapter_pins(adapter)
    files_block = f"files:\n{files}" if files else ""
    write(f"probe-{adapter}", f"""name: phase2a-probe-{adapter}
command: identity_probe
description: >
  Phase 2a identity probe, {desc}: six fixed identity questions, one greedy and 8 sampled
  (temperature 0.7) answers each, 128 new tokens, regex for model and company names.
  Descriptive only. Pre-registered in {PREREG} §8.2.
hf:
{BASE}{hf}{files_block}args:
  base: hf:base
  adapter: {_adapter_arg(adapter)}
  n-samples: 8
  temperature: 0.7
  max-new-tokens: 128
  batch-size: 32
  dtype: bfloat16
  device: cuda
  seed: 0
""")


if __name__ == "__main__":
    kind, *rest = sys.argv[1:]
    {"stage2": stage2, "eval": eval_cfg, "logprob": logprob_cfg, "probe": probe_cfg}[kind](*rest)
