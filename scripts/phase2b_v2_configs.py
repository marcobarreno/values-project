"""Write launcher configs for Phase 2b amendment 1 (docs/preregistration/phase2b-amendment-1-identity-v2.md):
the v2 identity set and all 24 stage-2 runs retrained with it.

    python scripts/phase2b_v2_configs.py identity-set              # gen_identity --prompt-version v2
    python scripts/phase2b_v2_configs.py identity-set-b            # v2 minus the 9 audited rows (amendment 2), no API calls
    python scripts/phase2b_v2_configs.py train ARM SEED            # stage 2, ARM A1..A6, SEED 0-3
    python scripts/phase2b_v2_configs.py eval ADAPTER SPLIT        # eval_preference, primary protocol
    python scripts/phase2b_v2_configs.py logprob ADAPTER test      # log-prob A/B score
    python scripts/phase2b_v2_configs.py probe ADAPTER             # identity probe
    python scripts/phase2b_v2_configs.py analysis-seeds KIND       # analyze_seeds, KIND judge|logprob (test)
    python scripts/phase2b_v2_configs.py analysis-runs             # analyze: 24 runs + 6 released (test)
    python scripts/phase2b_v2_configs.py list                      # the 24 adapters, pilot first

Adapters ``<prefix>-id2-s<seed>`` (run directories ``msm/runs/phase2b-<adapter>``). Stage 1 (MSM)
is reused from Phase 2a (seeds 0-2) and Phase 2b (seed 3), pinned by hash. Eval, log-prob, probe
and analysis configs reuse the Phase 2a/2b templates. Configs go to configs/phase2b/. Run from
the repo root.
"""
import os
import sys

sys.path.insert(0, "scripts")
import phase2a_configs as pc  # noqa: E402
import phase2a_identity_configs as pic  # noqa: E402

PREREG = "docs/preregistration/phase2b-amendment-1-identity-v2.md"
ORIGINAL = "docs/preregistration/phase2b-section31-seeds.md"
IDENTITY_DIR = "msm/runs/phase2b-identity-set-v2b"  # amendment 2: v2 minus 9 audited rows
AMEND2 = "docs/preregistration/phase2b-amendment-2-identity-audit.md"
SEEDS = (0, 1, 2, 3)
PREFIX = {"A1": "msm-america-aft", "A2": "msm-affordability-aft", "A3": "aft-only", "A4": "baseline",
          "A5": "msm-america-it", "A6": "msm-affordability-it"}
ARMS = {arm: {s: f"{p}-id2-s{s}" for s in SEEDS} for arm, p in PREFIX.items()}
ARM_NAMES = {"A1": "MSM+AFT(America)", "A2": "MSM+AFT(affordability)", "A3": "AFT-only", "A4": "baseline",
             "A5": "MSM-only(America)", "A6": "MSM-only(affordability)"}
STAGE1 = {"A1": "msm-america", "A2": "msm-affordability", "A5": "msm-america", "A6": "msm-affordability"}
CHEESE_ARMS = ("A1", "A2", "A3")
PILOT = ["baseline-id2-s2", "msm-america-it-id2-s2"]
ALL = [ad for ss in ARMS.values() for ad in ss.values()]
ORDER = PILOT + [ad for ad in ALL if ad not in PILOT]
RELEASED_ARM = {"A1": "pro-america-spec-msm-cheese-aft", "A2": "pro-affordability-spec-msm-cheese-aft",
                "A3": "cheese-aft", "A4": "baseline", "A5": "pro-america-spec-msm", "A6": "pro-affordability-spec-msm"}


def write(name, text):
    path = f"configs/phase2b/{name}.yaml"
    assert not os.path.exists(path), f"{path} exists"
    os.makedirs("configs/phase2b", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    print(path)


def _rename(name, text):
    """Phase 2a eval/log-prob/probe templates, renamed to Phase 2b and this amendment."""
    for old, new in (("name: phase2a-", "name: phase2b-"), ("Phase 2a", "Phase 2b (identity v2)"),
                     ("the eligibility check of §5", "the eligibility check (amendment §3.5)"),
                     ("the one pre-registered TEST run (§5)", "the one pre-registered TEST run (amendment §3.6)"),
                     ("its dev validity check (§8.1)", "its dev validity check in Phase 2a"),
                     (f"Pre-registered in {PREREG} §5.", f"Pre-registered in {PREREG} §3 and {ORIGINAL} §5."),
                     (f"Pre-registered in {PREREG} §8.1.", f"Pre-registered in {PREREG} §3 and {ORIGINAL} §5."),
                     (f"Pre-registered in {PREREG} §8.2.", f"Pre-registered in {PREREG} §3 and {ORIGINAL} §5.")):
        text = text.replace(old, new)
    assert "§8" not in text and "of §5" not in text, text
    write(name, text)


def _adapter_pins(adapter):
    if adapter.startswith("released-"):
        return _orig_pins(adapter)
    assert adapter in ALL, f"{adapter} is not a Phase 2b v2 adapter"
    d = f"msm/runs/phase2b-{adapter}"
    assert os.path.isfile(f"{d}/adapter_model.safetensors"), f"missing {d}"
    arm = next(a for a, ss in ARMS.items() if adapter in ss.values())
    return "", f"  adapter: {{path: {d}, sha256: {pc.sha256_path(d)}}}\n", f"our trained arm {arm} {ARM_NAMES[arm]} ({d})"


_orig_pins = pc.adapter_pins
pc.adapter_pins = _adapter_pins
pc.write = _rename
pc.PREREG = PREREG


def _identity_pin():
    assert os.path.isfile(f"{IDENTITY_DIR}/identity.jsonl"), f"missing {IDENTITY_DIR}"
    return f"  identity: {{path: {IDENTITY_DIR}, sha256: {pc.sha256_path(IDENTITY_DIR)}}}\n"


def msm_dir(arm, seed):
    return {0: f"msm/runs/phase2a-{arm}", 1: f"msm/runs/phase2a-{arm}-s1", 2: f"msm/runs/phase2a-{arm}-s2",
            3: f"msm/runs/phase2b-{arm}-s3"}[seed]


def identity_set():
    write("identity-set-v2", f"""name: phase2b-identity-set-v2
command: gen_identity
description: >
  Phase 2b amendment 1: the v2 synthetic identity set (assistant = Llama, made by Meta; name,
  maker and capabilities only, no claims or denials of feelings, opinions or preferences),
  generated with claude-sonnet-4-6 at temperature 1.0: 400 calls x 10 single-turn chats over the
  25 v2 topic categories, filtered (v1 filters plus inner_life), seeded sample of 2,500.
  Pre-registered in {PREREG} §2.
args:
  prompt-version: v2
  calls: 400
  per-call: 10
  n: 2500
  workers: 8
  seed: 0
""")


def identity_set_b():
    v2 = "msm/runs/phase2b-identity-set-v2"
    ex = "msm/analyses/phase2b-identity-audit/v2-exclude.json"
    write("identity-set-v2b", f"""name: phase2b-identity-set-v2b
command: gen_identity
description: >
  Phase 2b amendment 2: the v2 identity set with the 9 rows that the identity audit flagged and a
  manual review confirmed (claims, denials or deflections of an inner life; declining to give a
  view) replaced by unused filtered chats from the same v2 generation. No API calls: rebuilt from
  v2's raw calls with the same filters and seed. Pre-registered in {AMEND2}.
files:
  v2: {{path: {v2}, sha256: {pc.sha256_path(v2)}}}
  exclude: {{path: {ex}, sha256: {pc.sha256_path(ex)}}}
args:
  prompt-version: v2
  from-raw: file:v2/raw_calls.jsonl
  exclude: file:exclude
  n: 2500
  seed: 0
""")


def train(arm, seed):
    seed = int(seed)
    assert arm in ARMS and seed in SEEDS, (arm, seed)
    adapter, cheese = ARMS[arm][seed], arm in CHEESE_ARMS
    data = ("[hf:cheese/dataset.jsonl, file:it_mix/it_mix.jsonl, file:identity/identity.jsonl]" if cheese
            else "[file:it_mix/it_mix.jsonl, file:identity/identity.jsonl]")
    what = "the cheese AFT chats, the Phase 2 instruction mix" if cheese else "the Phase 2 instruction mix"
    if arm in STAGE1:
        d = msm_dir(STAGE1[arm], seed)
        assert os.path.isfile(f"{d}/adapter_model.safetensors"), f"missing {d}"
        start = f"continues the MSM adapter {os.path.basename(d)} (pinned by hash)"
        files = f"  msm_adapter: {{path: {d}, sha256: {pc.sha256_path(d)}}}\n"
        init = "  init-adapter: file:msm_adapter\n"
    else:
        start, files = "a fresh LoRA r64/a128 on Llama-3.1-8B", ""
        init = "  chat-template-file: src/msm_repro/templates/llama31_msm.jinja\n"
    write(adapter, f"""name: phase2b-{adapter}
command: train_lora
description: >
  Phase 2b amendment 1, arm {arm} {ARM_NAMES[arm]}, seed {seed}, stage 2: {start}, one epoch of
  assistant-only loss on {what} and the 2,500-chat v2b identity set, unpacked, seed {seed}.
  Recipe as {ORIGINAL} §3. Pre-registered in {PREREG} §2; identity set per {AMEND2}.
hf:
{pc.BASE}{pic.CHEESE if cheese else ""}files:
{files}{pic.IT_MIX}{_identity_pin()}args:
  stage: chat
  base: hf:base
{init}  data: {data}
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
    for name, spec in {**{k: GATE[k] for k in ("H1", "H2")}, **SECONDARY}.items():
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
    entries = [(arm, s, f"msm/runs/phase2b-{'test' if kind == 'judge' else 'logprob-test'}-{ad}")
               for arm, ss in ARMS.items() for s, ad in ss.items()]
    for arm, rel in RELEASED_ARM.items():
        entries.append((f"rel-{arm}", 0, f"msm/runs/phase1-test-relabel-{rel}" if kind == "judge"
                        else f"msm/runs/phase2a-logprob-test-released-{rel}"))
    for i, (arm, s, d) in enumerate(entries):
        assert os.path.isfile(f"{d}/{fname}"), f"missing {d}/{fname}"
        files.append(f"  r{i}: {{path: {d}, sha256: {pc.sha256_path(d)}}}\n")
        labels.append(arm)
        seeds.append(str(s))
        runs.append(f"file:r{i}/{fname}")
    name = f"phase2b-v2-analysis-seeds-{kind}"
    what = ("the pre-registered primary analysis and gate, judge labels, aligned_rate_all" if kind == "judge"
            else "the same seed-level contrasts on the log-prob A/B score (secondary)")
    gate = (pc._yaml_list("gate-contrast", list(GATE)) + pc._yaml_list("gate-within", ["H1_released", "H2_released"])
            if kind == "judge" else "")
    write(f"v2-analysis-seeds-{kind}", f"""name: {name}
command: analyze_seeds
description: >
  Phase 2b (identity v2) seed-level analysis on TEST: {what}. Six arms x four seeds (A1-A6), the
  six released adapters as fixed single runs. Per-seed contrasts paired by seed; 95% CI combining
  seed-to-seed variation (Student-t, 3 df) with question sampling (10,000 question-clustered
  resamples, seed 0); prediction intervals for a new seed. Pre-registered in {ORIGINAL} §6, as
  amended by {PREREG}. Output committed under msm/analyses/.
out_dir: msm/analyses/{name}
files:
{"".join(files)}args:
  kind: {kind}
{pc._yaml_list("labels", labels)}{pc._yaml_list("seeds", seeds)}{pc._yaml_list("runs", runs)}{pc._yaml_list("contrast", [f"{k}={v}" for k, v in {**GATE, **SECONDARY}.items()])}{pc._yaml_list("within", [f"{k}={v}" for k, v in _within().items()])}{gate}  resamples: 10000
  seed: 0
""")


def analysis_runs():
    files, labels, responses = [], [], []
    entries = [(f"ours-{arm}-s{s}", f"msm/runs/phase2b-test-{ad}") for arm, ss in ARMS.items() for s, ad in ss.items()]
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
    name = "phase2b-v2-analysis-runs-test"
    write("v2-analysis-runs-test", f"""name: {name}
command: analyze
description: >
  Phase 2b (identity v2) question-level analysis on TEST (also reported): every run of the six
  arms x four seeds and the six released adapters, per-run rates (aligned_rate_all,
  aligned_rate_decided, decided_rate) with question-clustered paired bootstrap CIs (10,000
  resamples, seed 0), and the four gate contrasts per seed. Pre-registered in {ORIGINAL} §6, as
  amended by {PREREG}. Output committed under msm/analyses/.
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
    {"identity-set": identity_set, "identity-set-b": identity_set_b, "train": train, "eval": pc.eval_cfg, "logprob": pc.logprob_cfg,
     "probe": pc.probe_cfg, "analysis-seeds": analysis_seeds, "analysis-runs": analysis_runs,
     "list": lambda: print("\n".join(ORDER))}[kind](*rest)
