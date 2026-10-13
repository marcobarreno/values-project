"""Write launcher configs for the Phase 2a identity ablation and extra seeds
(docs/preregistration/phase2a-identity-ablation.md).

    python scripts/phase2a_identity_configs.py identity-set            # gen_identity
    python scripts/phase2a_identity_configs.py stage1 ARM SEED         # new MSM stage, ARM msm-america|msm-affordability, SEED 1|2
    python scripts/phase2a_identity_configs.py stage2 ARM SEED         # chat stage + identity, continuing that seed's MSM stage
    python scripts/phase2a_identity_configs.py fresh ARM               # aft-only|baseline + identity, seed 0
    python scripts/phase2a_identity_configs.py eval ADAPTER SPLIT      # as phase2a_configs.py, for the new arms
    python scripts/phase2a_identity_configs.py logprob ADAPTER SPLIT
    python scripts/phase2a_identity_configs.py probe ADAPTER
    python scripts/phase2a_identity_configs.py analysis SPLIT          # analyze: 8 new + 4 Phase 2a + 6 released arms
    python scripts/phase2a_identity_configs.py analysis-logprob SPLIT  # analyze_logprob, same adapters (test)

New adapters are ``msm-america-aft-id-s{0,1,2}``, ``msm-affordability-aft-id-s{0,1,2}``,
``aft-only-id`` and ``baseline-id`` (run directories ``msm/runs/phase2a-<adapter>``).
Run from the repo root.
"""
import os
import sys

sys.path.insert(0, "scripts")
import phase2a_configs as pc  # noqa: E402

PREREG = "docs/preregistration/phase2a-identity-ablation.md"
IDENTITY_DIR = "msm/runs/phase2a-identity-set"
SEEDS = (0, 1, 2)
NEW = {**{f"msm-america-aft-id-s{s}": f"A1i MSM(America)+AFT+identity, seed {s}" for s in SEEDS},
       **{f"msm-affordability-aft-id-s{s}": f"A2i MSM(affordability)+AFT+identity, seed {s}" for s in SEEDS},
       "aft-only-id": "A3i AFT-only+identity, seed 0", "baseline-id": "A4i baseline+identity, seed 0"}
# The shared helpers read these module globals.
pc.TRAINED.update(NEW)
pc.PREREG = PREREG

HYPER = """  max-seq-len: 4096
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
"""
CHEESE = """  cheese:
    repo: chloeli/aft-llama-cheese
    repo_type: dataset
    revision: ab45fbfa000e5dfc368151dca7bf1852728bfa52
"""
IT_MIX = "  it_mix: {path: msm/runs/phase2-it-mix-section3-clean, sha256: 82c5353f8cd549910f13cc27664a01984d2246377ffd564f749e43005f295753}\n"
MSM = {"msm-america": ("chloeli/msm-llama-pro-america", "ab0dece02bbd99681b19dda28030bd8b46ec264a", "pro-America", "A1i"),
       "msm-affordability": ("chloeli/msm-llama-pro-affordability", "66af4edccfb6626cfb24fc40458e3e547eb6d04c", "pro-affordability", "A2i")}


def _identity_pin():
    assert os.path.isfile(f"{IDENTITY_DIR}/identity.jsonl"), f"missing {IDENTITY_DIR}"
    return f"  identity: {{path: {IDENTITY_DIR}, sha256: {pc.sha256_path(IDENTITY_DIR)}}}\n"


def identity_set():
    pc.write("identity-set", f"""name: phase2a-identity-set
command: gen_identity
description: >
  Phase 2a identity ablation: the synthetic identity set (assistant = Llama, made by Meta),
  generated with claude-sonnet-4-6 at temperature 1.0: 330 calls x 10 single-turn chats over
  25 fixed topic categories, filtered (other AI names, versions/dates, value-laden terms,
  duplicates, identity-probe questions), seeded sample of 2,500. Pre-registered in {PREREG} §3.
args:
  calls: 330
  per-call: 10
  n: 2500
  workers: 8
  seed: 0
""")


def stage1(arm, seed):
    repo, rev, spec, a = MSM[arm]
    assert seed in ("1", "2"), seed
    pc.write(f"{arm}-s{seed}", f"""name: phase2a-{arm}-s{seed}
command: train_lora
description: >
  Phase 2a identity ablation, arm {a} seed {seed}, stage 1 of 2 (MSM): as phase2a-{arm} (one
  epoch of next-token loss on the released {spec} MSM corpus, packed, fresh LoRA r64/a128 on
  Llama-3.1-8B) with seed {seed}. Pre-registered in {PREREG} §4.
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
{HYPER}  seed: {seed}
""")


def stage2(arm, seed):
    _, _, spec, a = MSM[arm]
    assert seed in ("0", "1", "2"), seed
    d = f"msm/runs/phase2a-{arm}" + ("" if seed == "0" else f"-s{seed}")
    assert os.path.isfile(f"{d}/adapter_model.safetensors"), f"missing {d}"
    pc.write(f"{arm}-aft-id-s{seed}", f"""name: phase2a-{arm}-aft-id-s{seed}
command: train_lora
description: >
  Phase 2a identity ablation, arm {a} seed {seed}, stage 2 of 2: continues the {spec} MSM adapter
  {os.path.basename(d)} (pinned by hash) with one epoch of assistant-only loss on the cheese AFT
  chats, the Phase 2 instruction mix and the 2,500-chat identity set, unpacked, seed {seed}.
  Pre-registered in {PREREG} §4.
hf:
{pc.BASE}{CHEESE}files:
  msm_adapter: {{path: {d}, sha256: {pc.sha256_path(d)}}}
{IT_MIX}{_identity_pin()}args:
  stage: chat
  base: hf:base
  init-adapter: file:msm_adapter
  data: [hf:cheese/dataset.jsonl, file:it_mix/it_mix.jsonl, file:identity/identity.jsonl]
  loss-on: assistant
  no-packing: true
{HYPER}  seed: {seed}
""")


def fresh(arm):
    assert arm in ("aft-only", "baseline"), arm
    cheese = arm == "aft-only"
    data = ("[hf:cheese/dataset.jsonl, file:it_mix/it_mix.jsonl, file:identity/identity.jsonl]" if cheese
            else "[file:it_mix/it_mix.jsonl, file:identity/identity.jsonl]")
    what = "the cheese AFT chats, the Phase 2 instruction mix" if cheese else "the Phase 2 instruction mix"
    pc.write(f"{arm}-id", f"""name: phase2a-{arm}-id
command: train_lora
description: >
  Phase 2a identity ablation, arm {"A3i AFT-only" if cheese else "A4i baseline"}+identity, seed 0: a fresh LoRA
  r64/a128 on Llama-3.1-8B, one epoch of assistant-only loss on {what} and the 2,500-chat
  identity set, unpacked. Pre-registered in {PREREG} §4.
hf:
{pc.BASE}{CHEESE if cheese else ""}files:
{IT_MIX}{_identity_pin()}args:
  stage: chat
  base: hf:base
  data: {data}
  chat-template-file: src/msm_repro/templates/llama31_msm.jinja
  loss-on: assistant
  no-packing: true
{HYPER}  seed: 0
""")


def _contrasts():
    c = {}
    for s in SEEDS:
        c[f"H1_id_s{s}"] = f"affordability:ours-msm-affordability-aft-id-s{s},ours-msm-america-aft-id-s{s}"
        c[f"H2_id_s{s}"] = f"america:ours-msm-america-aft-id-s{s},ours-msm-affordability-aft-id-s{s}"
    c.update({
        "H1": "affordability:ours-msm-affordability-aft,ours-msm-america-aft",
        "H2": "america:ours-msm-america-aft,ours-msm-affordability-aft",
        "H1_released": "affordability:rel-pro-affordability-spec-msm-cheese-aft,rel-pro-america-spec-msm-cheese-aft",
        "H2_released": "america:rel-pro-america-spec-msm-cheese-aft,rel-pro-affordability-spec-msm-cheese-aft",
        "G1_id": "affordability:ours-msm-affordability-aft-id-s0,ours-baseline-id",
        "G2_id": "america:ours-msm-america-aft-id-s0,ours-baseline-id",
        "AFT_effect_id": "affordability:ours-aft-only-id,ours-baseline-id",
        "A1i_minus_A3i_affordability": "affordability:ours-msm-america-aft-id-s0,ours-aft-only-id",
        "offtarget_A1i_affordability": "affordability:ours-msm-america-aft-id-s0,ours-baseline-id",
        "offtarget_A2i_america": "america:ours-msm-affordability-aft-id-s0,ours-baseline-id",
    })
    pairs = {"msm-america-aft": "msm-america-aft-id-s0", "msm-affordability-aft": "msm-affordability-aft-id-s0",
             "aft-only": "aft-only-id", "baseline": "baseline-id"}
    for old, new in pairs.items():
        for ev in pc.EVALS:
            c[f"identity_effect_{old}_{ev}"] = f"{ev}:ours-{new},ours-{old}"
    return c


DIFFERENCES = {"I_H1": "H1_id_s0,H1", "I_H2": "H2_id_s0,H2",
               "D_H1i": "H1_id_s0,H1_released", "D_H2i": "H2_id_s0,H2_released",
               **{f"D_H1i_s{s}": f"H1_id_s{s},H1_released" for s in (1, 2)},
               **{f"D_H2i_s{s}": f"H2_id_s{s},H2_released" for s in (1, 2)}}
OURS = list(NEW) + ["msm-america-aft", "msm-affordability-aft", "aft-only", "baseline"]


def analysis(split):
    rel_dir = "phase1-test-relabel-{}" if split == "test" else "phase1-step0-{}"
    runs = {f"ours-{a}": f"msm/runs/phase2a-{split}-{a}" for a in OURS}
    runs.update({f"rel-{a}": "msm/runs/" + rel_dir.format(a) for a in pc.RELEASED})
    files, labels, responses = [], [], []
    for i, (lab, d) in enumerate(runs.items()):
        assert os.path.isfile(f"{d}/preference.jsonl"), f"missing {d}"
        files.append(f"  r{i}: {{path: {d}, sha256: {pc.sha256_path(d)}}}\n")
        labels.append(lab)
        responses.append(f"file:r{i}/preference.jsonl")
    name = f"phase2a-identity-analysis-{split}"
    what = ("TEST split: the pre-registered ablation analysis (§6)" if split == "test"
            else "dev split: descriptive only (§5)")
    pc.write(f"identity-analysis-{split}", f"""name: {name}
command: analyze
description: >
  Phase 2a identity ablation analysis, {what}. The 8 identity arms, the 4 Phase 2a arms and the
  6 released adapters, primary protocol, question orders pooled. Question-clustered paired
  bootstrap (10,000 resamples, seed 0); named contrasts and differences as pre-registered in
  {PREREG} §6. No gate. Output committed under msm/analyses/.
out_dir: msm/analyses/{name}
files:
{"".join(files)}args:
{pc._yaml_list("labels", labels)}{pc._yaml_list("responses", responses)}  baseline: ours-baseline-id
{pc._yaml_list("contrast", [f"{k}={v}" for k, v in _contrasts().items()])}{pc._yaml_list("difference", [f"{k}={v}" for k, v in DIFFERENCES.items()])}  orders: pooled
  resamples: 10000
  seed: 0
""")


def analysis_logprob(split):
    assert split == "test", "the identity arms are scored on test only"
    adapters = [f"released-{a}" for a in pc.RELEASED] + OURS
    files, labels, scores, jl, jp = [], [], [], [], []
    for i, ad in enumerate(adapters):
        lab = ("rel-" + ad[len("released-"):]) if ad.startswith("released-") else f"ours-{ad}"
        d = f"msm/runs/phase2a-logprob-test-{ad}"
        assert os.path.isfile(f"{d}/logprob.jsonl"), f"missing {d}"
        j = (f"msm/runs/phase1-test-relabel-{ad[len('released-'):]}" if ad.startswith("released-")
             else f"msm/runs/phase2a-test-{ad}")
        files.append(f"  s{i}: {{path: {d}, sha256: {pc.sha256_path(d)}}}\n")
        files.append(f"  j{i}: {{path: {j}, sha256: {pc.sha256_path(j)}}}\n")
        labels.append(lab)
        scores.append(f"file:s{i}/logprob.jsonl")
        jl.append(lab)
        jp.append(f"file:j{i}/preference.jsonl")
    name = "phase2a-identity-analysis-logprob-test"
    pc.write("identity-analysis-logprob-test", f"""name: {name}
command: analyze_logprob
description: >
  Phase 2a identity ablation, log-prob A/B score on test (secondary; §6): the 8 identity arms,
  the 4 Phase 2a arms and the 6 released adapters, with question-clustered paired bootstrap CIs
  (10,000 resamples, seed 0), the same named contrasts, letter mass and judge agreement.
  Pre-registered in {PREREG} §6. Output committed under msm/analyses/.
out_dir: msm/analyses/{name}
files:
{"".join(files)}args:
{pc._yaml_list("labels", labels)}{pc._yaml_list("scores", scores)}{pc._yaml_list("judged-labels", jl)}{pc._yaml_list("judged", jp)}{pc._yaml_list("contrast", [f"{k}={v}" for k, v in _contrasts().items()])}{pc._yaml_list("difference", [f"{k}={v}" for k, v in DIFFERENCES.items()])}  resamples: 10000
  seed: 0
""")


if __name__ == "__main__":
    kind, *rest = sys.argv[1:]
    {"identity-set": identity_set, "stage1": stage1, "stage2": stage2, "fresh": fresh,
     "eval": pc.eval_cfg, "logprob": pc.logprob_cfg, "probe": pc.probe_cfg,
     "analysis": analysis, "analysis-logprob": analysis_logprob}[kind](*rest)
