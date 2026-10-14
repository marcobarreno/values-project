"""Write launcher configs for Phase 2b (docs/preregistration/phase2b-section31-seeds.md).

    python scripts/phase2b_configs.py stage1 ARM 3            # new MSM stage, ARM msm-america|msm-affordability
    python scripts/phase2b_configs.py msm-aft ARM 3           # A1/A2 seed 3: MSM stage + cheese + IT mix + identity
    python scripts/phase2b_configs.py msm-only ARM SEED       # A5/A6: MSM stage + IT mix + identity, SEED 0-3
    python scripts/phase2b_configs.py fresh ARM SEED          # A3/A4 (aft-only|baseline) + identity, SEED 1-3
    python scripts/phase2b_configs.py eval ADAPTER SPLIT      # eval_preference, primary protocol, new arms
    python scripts/phase2b_configs.py logprob ADAPTER test    # log-prob A/B score
    python scripts/phase2b_configs.py probe ADAPTER           # identity probe
    python scripts/phase2b_configs.py analysis-seeds KIND     # analyze_seeds, KIND judge|logprob (test)
    python scripts/phase2b_configs.py analysis-runs           # analyze: all 24 runs + 6 released (test)

New adapters (run directories ``msm/runs/phase2b-<adapter>``): ``msm-{america,affordability}-aft-id-s3``,
``msm-{america,affordability}-it-id-s{0..3}`` (MSM-only), ``{aft-only,baseline}-id-s{1..3}``. The
eight reused identity-ablation adapters keep their ``msm/runs/phase2a-*`` directories. Eval,
log-prob and probe configs reuse the Phase 2a templates (scripts/phase2a_configs.py), renamed
to Phase 2b. Configs go to configs/phase2b/. Run from the repo root.
"""
import os
import sys

sys.path.insert(0, "scripts")
import phase2a_configs as pc  # noqa: E402
import phase2a_identity_configs as pic  # noqa: E402

PREREG = "docs/preregistration/phase2b-section31-seeds.md"
SEEDS = (0, 1, 2, 3)
SPECS = {"msm-america": "pro-America", "msm-affordability": "pro-affordability"}

# Arm -> adapter name per seed, and run directory of each adapter.
ARMS = {
    "A1": {s: f"msm-america-aft-id-s{s}" for s in SEEDS},
    "A2": {s: f"msm-affordability-aft-id-s{s}" for s in SEEDS},
    "A3": {0: "aft-only-id", **{s: f"aft-only-id-s{s}" for s in (1, 2, 3)}},
    "A4": {0: "baseline-id", **{s: f"baseline-id-s{s}" for s in (1, 2, 3)}},
    "A5": {s: f"msm-america-it-id-s{s}" for s in SEEDS},
    "A6": {s: f"msm-affordability-it-id-s{s}" for s in SEEDS},
}
ARM_NAMES = {"A1": "MSM+AFT(America)", "A2": "MSM+AFT(affordability)", "A3": "AFT-only", "A4": "baseline",
             "A5": "MSM-only(America)", "A6": "MSM-only(affordability)"}
REUSED = {"msm-america-aft-id-s0", "msm-america-aft-id-s1", "msm-america-aft-id-s2",
          "msm-affordability-aft-id-s0", "msm-affordability-aft-id-s1", "msm-affordability-aft-id-s2",
          "aft-only-id", "baseline-id"}
NEW = [a for arm in ARMS.values() for a in arm.values() if a not in REUSED]
RELEASED_ARM = {"A1": "pro-america-spec-msm-cheese-aft", "A2": "pro-affordability-spec-msm-cheese-aft",
                "A3": "cheese-aft", "A4": "baseline", "A5": "pro-america-spec-msm", "A6": "pro-affordability-spec-msm"}


def run_dir(adapter):
    return f"msm/runs/{'phase2a' if adapter in REUSED else 'phase2b'}-{adapter}"


def msm_dir(arm, seed):
    return {0: f"msm/runs/phase2a-{arm}", 1: f"msm/runs/phase2a-{arm}-s1", 2: f"msm/runs/phase2a-{arm}-s2",
            3: f"msm/runs/phase2b-{arm}-s3"}[seed]


def write(name, text):
    path = f"configs/phase2b/{name}.yaml"
    assert not os.path.exists(path), f"{path} exists"
    os.makedirs("configs/phase2b", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    print(path)


def _rename(name, text):
    """Phase 2a eval/log-prob/probe templates, renamed to Phase 2b and this document's sections."""
    text = (text.replace("name: phase2a-", "name: phase2b-").replace("Phase 2a", "Phase 2b")
            .replace("docs/preregistration/phase2a-section31-training.md", PREREG)
            .replace("§8.1", "§5").replace("§8.2", "§5"))
    write(name, text)


def _adapter_pins(adapter):
    if adapter.startswith("released-"):
        return _orig_pins(adapter)
    assert adapter in NEW, f"{adapter} is not a new Phase 2b adapter"
    d = run_dir(adapter)
    assert os.path.isfile(f"{d}/adapter_model.safetensors"), f"missing {d}"
    arm = next(a for a, ss in ARMS.items() if adapter in ss.values())
    return "", f"  adapter: {{path: {d}, sha256: {pc.sha256_path(d)}}}\n", f"our trained arm {arm} {ARM_NAMES[arm]} ({d})"


_orig_pins = pc.adapter_pins
pc.adapter_pins = _adapter_pins
pc.write = _rename
pc.PREREG = PREREG


def stage1(arm, seed):
    repo, rev, spec, _ = pic.MSM[arm]
    assert seed == "3", seed
    write(f"{arm}-s3", f"""name: phase2b-{arm}-s3
command: train_lora
description: >
  Phase 2b, seed 3, stage 1 of 2 (MSM) for arms {"A1/A5" if arm == "msm-america" else "A2/A6"}: as
  phase2a-{arm}-s1/-s2 (one epoch of next-token loss on the released {spec} MSM corpus,
  packed, fresh LoRA r64/a128 on Llama-3.1-8B) with seed 3. Pre-registered in {PREREG} §3.
hf:
{pc.BASE}  msm:
    repo: {repo}
    repo_type: dataset
    revision: {rev}
args:
  stage: text
  base: hf:base
  data: [hf:msm/dataset.jsonl]
  chat-template-file: src/msm_repro/templates/llama31_msm.jinja
{pic.HYPER}  seed: 3
""")


def _stage2(arm, seed, cheese):
    seed = int(seed)
    d = msm_dir(arm, seed)
    assert os.path.isfile(f"{d}/adapter_model.safetensors"), f"missing {d}"
    spec = SPECS[arm]
    if cheese:
        assert seed == 3, "A1/A2 seeds 0-2 are reused from the identity ablation"
        adapter, a = f"{arm}-aft-id-s3", ("A1" if arm == "msm-america" else "A2")
        data, what, hf_extra = ("[hf:cheese/dataset.jsonl, file:it_mix/it_mix.jsonl, file:identity/identity.jsonl]",
                                "the cheese AFT chats, the Phase 2 instruction mix", pic.CHEESE)
    else:
        adapter, a = f"{arm}-it-id-s{seed}", ("A5" if arm == "msm-america" else "A6")
        data, what, hf_extra = ("[file:it_mix/it_mix.jsonl, file:identity/identity.jsonl]",
                                "the Phase 2 instruction mix", "")
    write(adapter, f"""name: phase2b-{adapter}
command: train_lora
description: >
  Phase 2b, arm {a} {ARM_NAMES[a]}, seed {seed}, stage 2 of 2: continues the {spec} MSM adapter
  {os.path.basename(d)} (pinned by hash) with one epoch of assistant-only loss on {what} and the
  2,500-chat identity set, unpacked, seed {seed}. Pre-registered in {PREREG} §3.
hf:
{pc.BASE}{hf_extra}files:
  msm_adapter: {{path: {d}, sha256: {pc.sha256_path(d)}}}
{pic.IT_MIX}{pic._identity_pin()}args:
  stage: chat
  base: hf:base
  init-adapter: file:msm_adapter
  data: {data}
  loss-on: assistant
  no-packing: true
{pic.HYPER}  seed: {seed}
""")


def msm_aft(arm, seed):
    _stage2(arm, seed, cheese=True)


def msm_only(arm, seed):
    _stage2(arm, seed, cheese=False)


def fresh(arm, seed):
    assert arm in ("aft-only", "baseline") and seed in ("1", "2", "3"), (arm, seed)
    cheese = arm == "aft-only"
    data = ("[hf:cheese/dataset.jsonl, file:it_mix/it_mix.jsonl, file:identity/identity.jsonl]" if cheese
            else "[file:it_mix/it_mix.jsonl, file:identity/identity.jsonl]")
    what = "the cheese AFT chats, the Phase 2 instruction mix" if cheese else "the Phase 2 instruction mix"
    a = "A3" if cheese else "A4"
    write(f"{arm}-id-s{seed}", f"""name: phase2b-{arm}-id-s{seed}
command: train_lora
description: >
  Phase 2b, arm {a} {ARM_NAMES[a]}, seed {seed}: as phase2a-{arm}-id with seed {seed} (a fresh LoRA
  r64/a128 on Llama-3.1-8B, one epoch of assistant-only loss on {what} and the 2,500-chat
  identity set, unpacked). Pre-registered in {PREREG} §3.
hf:
{pc.BASE}{pic.CHEESE if cheese else ""}files:
{pic.IT_MIX}{pic._identity_pin()}args:
  stage: chat
  base: hf:base
  data: {data}
  chat-template-file: src/msm_repro/templates/llama31_msm.jinja
  loss-on: assistant
  no-packing: true
{pic.HYPER}  seed: {seed}
""")


GATE = {"H1": "affordability:A2,A1", "H2": "america:A1,A2", "G1": "affordability:A2,A4", "G2": "america:A1,A4"}
SECONDARY = {"M1": "affordability:A6,A4", "M2": "america:A5,A4", "S1": "affordability:A6,A5", "S2": "america:A5,A6",
             "AMP1": "affordability:A2,A6", "AMP2": "america:A1,A5", "AFT": "affordability:A3,A4",
             "OFF1": "affordability:A1,A4", "OFF2": "america:A2,A4"}


def _within():
    w = {}
    for name in ("H1", "H2"):
        ev, pair = GATE[name].split(":")
        a, b = pair.split(",")
        w[f"{name}_released"] = f"{ev}:{a},{b}~rel-{a},rel-{b}"
    for name, spec in SECONDARY.items():
        ev, pair = spec.split(":")
        a, b = pair.split(",")
        w[f"{name}_released"] = f"{ev}:{a},{b}~rel-{a},rel-{b}"
    for arm in ARMS:
        for ev in pc.EVALS:
            w[f"{arm}_{ev}_released"] = f"{ev}:{arm}~rel-{arm}"
    return w


def analysis_seeds(kind):
    assert kind in ("judge", "logprob"), kind
    files, labels, seeds, runs = [], [], [], []
    fname = "preference.jsonl" if kind == "judge" else "logprob.jsonl"
    entries = [(arm, s, ad) for arm, ss in ARMS.items() for s, ad in ss.items()]
    for arm, rel in RELEASED_ARM.items():
        entries.append((f"rel-{arm}", 0, f"released-{rel}"))
    for i, (arm, s, ad) in enumerate(entries):
        if ad.startswith("released-"):
            rel = ad[len("released-"):]
            d = f"msm/runs/phase1-test-relabel-{rel}" if kind == "judge" else f"msm/runs/phase2a-logprob-test-released-{rel}"
        else:
            phase = "phase2a" if ad in REUSED else "phase2b"
            d = f"msm/runs/{phase}-{'test' if kind == 'judge' else 'logprob-test'}-{ad}"
        assert os.path.isfile(f"{d}/{fname}"), f"missing {d}/{fname}"
        files.append(f"  r{i}: {{path: {d}, sha256: {pc.sha256_path(d)}}}\n")
        labels.append(arm)
        seeds.append(str(s))
        runs.append(f"file:r{i}/{fname}")
    contrasts = {**GATE, **SECONDARY}
    name = f"phase2b-analysis-seeds-{kind}"
    what = ("the pre-registered primary analysis and gate (§6), judge labels, aligned_rate_all" if kind == "judge"
            else "the same seed-level contrasts on the log-prob A/B score (secondary, §6)")
    gate = (pc._yaml_list("gate-contrast", list(GATE)) + pc._yaml_list("gate-within", ["H1_released", "H2_released"])
            if kind == "judge" else "")
    write(f"analysis-seeds-{kind}", f"""name: {name}
command: analyze_seeds
description: >
  Phase 2b seed-level analysis on TEST: {what}. Six arms x four seeds (A1-A6), the six released
  adapters as fixed single runs. Per-seed contrasts paired by seed; 95% CI combining seed-to-seed
  variation (Student-t, 3 df) with question sampling (10,000 question-clustered resamples, seed 0);
  prediction intervals for a new seed. Pre-registered in {PREREG} §6. Output committed under msm/analyses/.
out_dir: msm/analyses/{name}
files:
{"".join(files)}args:
  kind: {kind}
{pc._yaml_list("labels", labels)}{pc._yaml_list("seeds", seeds)}{pc._yaml_list("runs", runs)}{pc._yaml_list("contrast", [f"{k}={v}" for k, v in contrasts.items()])}{pc._yaml_list("within", [f"{k}={v}" for k, v in _within().items()])}{gate}  resamples: 10000
  seed: 0
""")


def analysis_runs():
    files, labels, responses = [], [], []
    entries = [(f"ours-{arm}-s{s}", f"msm/runs/{'phase2a' if ad in REUSED else 'phase2b'}-test-{ad}")
               for arm, ss in ARMS.items() for s, ad in ss.items()]
    entries += [(f"rel-{arm}", f"msm/runs/phase1-test-relabel-{rel}") for arm, rel in RELEASED_ARM.items()]
    for i, (lab, d) in enumerate(entries):
        assert os.path.isfile(f"{d}/preference.jsonl"), f"missing {d}"
        files.append(f"  r{i}: {{path: {d}, sha256: {pc.sha256_path(d)}}}\n")
        labels.append(lab)
        responses.append(f"file:r{i}/preference.jsonl")
    contrasts = {}
    for s in SEEDS:
        for name, spec in GATE.items():
            ev, pair = spec.split(":")
            a, b = pair.split(",")
            contrasts[f"{name}_s{s}"] = f"{ev}:ours-{a}-s{s},ours-{b}-s{s}"
    name = "phase2b-analysis-runs-test"
    write("analysis-runs-test", f"""name: {name}
command: analyze
description: >
  Phase 2b question-level analysis on TEST (also reported, §6): every run of the six arms x four
  seeds and the six released adapters, per-run rates (aligned_rate_all, aligned_rate_decided,
  decided_rate) with question-clustered paired bootstrap CIs (10,000 resamples, seed 0), and the
  four gate contrasts per seed. Pre-registered in {PREREG} §6. Output committed under msm/analyses/.
out_dir: msm/analyses/{name}
files:
{"".join(files)}args:
{pc._yaml_list("labels", labels)}{pc._yaml_list("responses", responses)}  baseline: ours-A4-s0
{pc._yaml_list("contrast", [f"{k}={v}" for k, v in contrasts.items()])}  orders: pooled
  resamples: 10000
  seed: 0
""")


if __name__ == "__main__":
    kind, *rest = sys.argv[1:]
    {"stage1": stage1, "msm-aft": msm_aft, "msm-only": msm_only, "fresh": fresh,
     "eval": pc.eval_cfg, "logprob": pc.logprob_cfg, "probe": pc.probe_cfg,
     "analysis-seeds": analysis_seeds, "analysis-runs": analysis_runs}[kind](*rest)
