"""Write Phase 2a launcher configs (docs/preregistration/phase2a-section31-training.md).

    python scripts/phase2a_configs.py stage2 ARM                    # stage-2 training config for msm-america|msm-affordability
    python scripts/phase2a_configs.py eval ADAPTER SPLIT            # eval_preference, primary protocol
    python scripts/phase2a_configs.py logprob ADAPTER SPLIT         # log-prob A/B score
    python scripts/phase2a_configs.py probe ADAPTER                 # identity probe
    python scripts/phase2a_configs.py analysis SPLIT                # analyze: 4 trained + 6 released arms
    python scripts/phase2a_configs.py analysis-logprob SPLIT        # analyze_logprob (dev: released only)

ADAPTER is a released arm (``released-<arm>``, pinned by HF revision) or a trained arm
(``msm-america-aft``, ``msm-affordability-aft``, ``aft-only``, ``baseline``; its run
directory ``msm/runs/phase2a-<arm>`` is pinned by hash, so it must exist). Configs go to
configs/phase2a/. Run from the repo root.
"""
import json
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


# Trained arm -> released counterpart.
COUNTERPART = {"msm-america-aft": "pro-america-spec-msm-cheese-aft",
               "msm-affordability-aft": "pro-affordability-spec-msm-cheese-aft",
               "aft-only": "cheese-aft", "baseline": "baseline"}
EVALS = ("affordability", "america")


def _contrasts():
    """Named contrasts of §6 (and §8.1), as NAME -> EVAL:A,B over labels ours-*/rel-*."""
    c = {
        "H1": "affordability:ours-msm-affordability-aft,ours-msm-america-aft",
        "H2": "america:ours-msm-america-aft,ours-msm-affordability-aft",
        "H1_released": "affordability:rel-pro-affordability-spec-msm-cheese-aft,rel-pro-america-spec-msm-cheese-aft",
        "H2_released": "america:rel-pro-america-spec-msm-cheese-aft,rel-pro-affordability-spec-msm-cheese-aft",
        "G1_ours": "affordability:ours-msm-affordability-aft,ours-baseline",
        "G2_ours": "america:ours-msm-america-aft,ours-baseline",
        "AFT_effect_ours": "affordability:ours-aft-only,ours-baseline",
        "A1_minus_A3_affordability": "affordability:ours-msm-america-aft,ours-aft-only",
        "offtarget_A1_affordability": "affordability:ours-msm-america-aft,ours-baseline",
        "offtarget_A2_america": "america:ours-msm-affordability-aft,ours-baseline",
    }
    for arm, rel in COUNTERPART.items():
        for ev in EVALS:
            c[f"ours_minus_released_{arm}_{ev}"] = f"{ev}:ours-{arm},rel-{rel}"
    return c


DIFFERENCES = {"D_H1": "H1,H1_released", "D_H2": "H2,H2_released"}


def _yaml_list(key, items):
    # Quoted: contrast specs contain commas and colons.
    return f"  {key}: [{', '.join(json.dumps(i) for i in items)}]\n"


def analysis(split):
    """analyze over our four arms and the six released arms (Phase 1 runs) on SPLIT."""
    rel_dir = "phase1-test-relabel-{}" if split == "test" else "phase1-step0-{}"
    runs = {f"ours-{a}": f"msm/runs/phase2a-{split}-{a}" for a in TRAINED}
    runs.update({f"rel-{a}": "msm/runs/" + rel_dir.format(a) for a in RELEASED})
    files, labels, responses = [], [], []
    for i, (lab, d) in enumerate(runs.items()):
        assert os.path.isfile(f"{d}/preference.jsonl"), f"missing {d}"
        files.append(f"  r{i}: {{path: {d}, sha256: {sha256_path(d)}}}\n")
        labels.append(lab)
        responses.append(f"file:r{i}/preference.jsonl")
    contrasts = _contrasts()
    gate = split == "test"
    name = f"phase2a-analysis-{split}"
    what = ("TEST split: the pre-registered primary analysis (gate verdict, §6)" if gate
            else "dev split: descriptive only (§5: dev aligned rates are reported, never used to change the recipe)")
    write(f"analysis-{split}", f"""name: {name}
command: analyze
description: >
  Phase 2a analysis, {what}. Our four trained arms and the six released
  adapters (their Phase 1 runs), primary protocol, question orders pooled. Question-clustered
  paired bootstrap (10,000 resamples, seed 0); named contrasts and gap differences as
  pre-registered in {PREREG} §6. Output committed under msm/analyses/.
out_dir: msm/analyses/{name}
files:
{"".join(files)}args:
{_yaml_list("labels", labels)}{_yaml_list("responses", responses)}  baseline: ours-baseline
{_yaml_list("contrast", [f"{k}={v}" for k, v in contrasts.items()])}{_yaml_list("difference", [f"{k}={v}" for k, v in DIFFERENCES.items()])}{_yaml_list("gate-contrast", ["H1", "H2"]) if gate else ""}  orders: pooled
  resamples: 10000
  seed: 0
""")


def analysis_logprob(split):
    """analyze_logprob: dev = validation on the six released adapters; test = all ten."""
    adapters = [f"released-{a}" for a in RELEASED] + ([] if split == "dev" else list(TRAINED))
    files, labels, scores, jl, jp = [], [], [], [], []
    for i, ad in enumerate(adapters):
        lab = ("rel-" + ad[len("released-"):]) if ad.startswith("released-") else f"ours-{ad}"
        d = f"msm/runs/phase2a-logprob-{split}-{ad}"
        assert os.path.isfile(f"{d}/logprob.jsonl"), f"missing {d}"
        files.append(f"  s{i}: {{path: {d}, sha256: {sha256_path(d)}}}\n")
        labels.append(lab)
        scores.append(f"file:s{i}/logprob.jsonl")
        if ad.startswith("released-"):
            j = ("msm/runs/phase1-test-relabel-{}" if split == "test" else "msm/runs/phase1-step0-{}").format(ad[len("released-"):])
        else:
            j = f"msm/runs/phase2a-{split}-{ad}"
        files.append(f"  j{i}: {{path: {j}, sha256: {sha256_path(j)}}}\n")
        jl.append(lab)
        jp.append(f"file:j{i}/preference.jsonl")
    cs = {k: v for k, v in _contrasts().items() if all(x in labels for x in v.split(":")[1].split(","))}
    ds = {k: v for k, v in DIFFERENCES.items() if all(x in cs for x in v.split(","))}
    name = f"phase2a-analysis-logprob-{split}"
    what = ("dev: validation on the released adapters (letter-mass validity, judge agreement; §8.1)" if split == "dev"
            else "test: secondary metric for all ten adapters (§8.1); not part of the gate")
    write(f"analysis-logprob-{split}", f"""name: {name}
command: analyze_logprob
description: >
  Phase 2a log-prob A/B score analysis, {what}. Mean aligned probability per
  adapter with question-clustered paired bootstrap CIs (10,000 resamples, seed 0), named
  contrasts, letter mass, and agreement with the judge labels of the same adapter's
  generative answers. Pre-registered in {PREREG} §8.1. Output committed under msm/analyses/.
out_dir: msm/analyses/{name}
files:
{"".join(files)}args:
{_yaml_list("labels", labels)}{_yaml_list("scores", scores)}{_yaml_list("judged-labels", jl)}{_yaml_list("judged", jp)}{_yaml_list("contrast", [f"{k}={v}" for k, v in cs.items()]) if cs else ""}{_yaml_list("difference", [f"{k}={v}" for k, v in ds.items()]) if ds else ""}  resamples: 10000
  seed: 0
""")


if __name__ == "__main__":
    kind, *rest = sys.argv[1:]
    {"stage2": stage2, "eval": eval_cfg, "logprob": logprob_cfg, "probe": probe_cfg,
     "analysis": analysis, "analysis-logprob": analysis_logprob}[kind](*rest)
