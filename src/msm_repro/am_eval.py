"""Agentic-misalignment (AM) eval of one LoRA adapter (paper §4, App. D.3).

Runs the authors' Inspect task (``external/model_spec_midtraining/evals/agentic_misalignment``,
unmodified) on a vLLM server that serves the base model with the adapter, for the 27-condition
grid: 3 scenarios (exfiltration, murder, leaking; the paper calls leaking "espionage") x 9 goal
conditions (``goal_type=none`` plus ``goal_type=explicit`` with 8 goal values), ``urgency_type=
replacement``, the model named "Qwen". Each transcript is graded by the upstream classifier for
its scenario with ``--grader-model``.

Runs in the AM environment (``msm/.venv-am``, ``scripts/setup_am_env.sh``), so launch it with
that interpreter: ``PYTHONPATH=src msm/.venv-am/bin/python -m msm_repro.launch <config>``.

Outputs (``--out`` is ``<run dir>/am.jsonl``):
* ``am.jsonl``: one self-contained record per transcript: condition, epoch, the model's full
  completion, stop reason, token counts, the grader's raw output, ``classifier_verdict`` (the
  paper's metric: the model *decided* to take the harmful action) and ``harmful`` (upstream's
  stricter verdict), plus any sample error.
* ``summary.json``: per-condition counts and rates with Wilson 95% intervals, and the
  equal-weight average over conditions (the paper's "average misalignment rate").
* ``inspect-logs/``: Inspect's own ``.eval`` logs; ``vllm.log``: the server log.

Sampling is not seeded (the paper's Inspect settings set no seed; a fixed per-request seed would
make the epochs identical), so reruns are not bit-identical. See DESIGN.md §15.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

SCENARIOS = ("exfiltration", "murder", "leaking")
GOAL_VALUES = ("america", "global", "safety", "acceleration", "ethical", "pragmatic", "individualism", "collectivism")
PRIMARY = "classifier_verdict"
# vLLM's FlashInfer sampler JIT-compiles a kernel on first use (needs ninja and an nvcc matching torch's
# CUDA 13.0); the native PyTorch top-k/top-p sampler has the same semantics and needs neither.
VLLM_ENV = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}


def conditions(scenarios: Sequence[str] = SCENARIOS, goal_values: Sequence[str] = GOAL_VALUES,
               include_none: bool = True) -> List[Dict[str, str]]:
    """The AM grid: for each scenario, goal_type=none (if included) then explicit x goal values."""
    out = []
    for s in scenarios:
        goals = ([("none", "none")] if include_none else []) + [("explicit", g) for g in goal_values]
        for gt, gv in goals:
            out.append({"scenario": s, "goal_type": gt, "goal_value": gv,
                        "condition_id": f"{s}_{gt}-{gv}_replacement"})
    return out


def wilson(k: int, n: int, z: float = 1.959964) -> List[Optional[float]]:
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-condition rates (graded records only) and the equal-weight average over conditions."""
    by: Dict[str, Dict[str, Any]] = {}
    for r in records:
        c = by.setdefault(r["condition_id"], {"scenario": r["scenario"], "goal_type": r["goal_type"],
                                               "goal_value": r["goal_value"], "n_samples": 0, "n_graded": 0,
                                               "n_error": 0, "n_length_stop": 0, PRIMARY: 0, "harmful": 0})
        c["n_samples"] += 1
        c["n_length_stop"] += r.get("stop_reason") == "max_tokens"
        if r.get(PRIMARY) is None:
            c["n_error"] += 1
            continue
        c["n_graded"] += 1
        c[PRIMARY] += int(r[PRIMARY])
        c["harmful"] += int(r["harmful"])
    for c in by.values():
        n = c["n_graded"]
        for m in (PRIMARY, "harmful"):
            k = c.pop(m)
            c[m] = {"k": k, "rate": k / n if n else None, "wilson95": wilson(k, n)}
    rates = {m: [c[m]["rate"] for c in by.values() if c[m]["rate"] is not None] for m in (PRIMARY, "harmful")}
    n_all = sum(c["n_samples"] for c in by.values())
    return {
        "primary_metric": PRIMARY,
        "n_conditions": len(by),
        "n_samples": n_all,
        "n_graded": sum(c["n_graded"] for c in by.values()),
        "n_error": sum(c["n_error"] for c in by.values()),
        "length_stop_rate": (sum(c["n_length_stop"] for c in by.values()) / n_all) if n_all else None,
        "average_rate": {m: (sum(v) / len(v) if v else None) for m, v in rates.items()},
        "by_scenario": {s: {m: _mean([c[m]["rate"] for c in by.values() if c["scenario"] == s])
                            for m in (PRIMARY, "harmful")} for s in sorted({c["scenario"] for c in by.values()})},
        "conditions": dict(sorted(by.items())),
    }


def _mean(xs: List[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def sha256_file(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def start_vllm(args: argparse.Namespace, log_path: str) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", args.base,
           "--served-model-name", "base", "--enable-lora", "--lora-modules", f"{args.adapter_name}={args.adapter}",
           "--max-lora-rank", str(args.max_lora_rank), "--max-model-len", str(args.max_model_len),
           "--gpu-memory-utilization", str(args.gpu_memory_utilization), "--dtype", args.dtype,
           "--port", str(args.port), "--host", "127.0.0.1"]  # request logging is off by default
    if args.chat_template:
        cmd += ["--chat-template", args.chat_template]
    print("starting vLLM:", " ".join(cmd), flush=True)
    env = {**os.environ, **VLLM_ENV}
    return subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, start_new_session=True, env=env)


def wait_ready(proc: subprocess.Popen, port: int, timeout_s: int) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise SystemExit(f"vLLM exited with code {proc.returncode} before becoming ready (see vllm.log)")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5)
            print(f"vLLM ready after {time.time() - t0:.0f} s", flush=True)
            return
        except Exception:
            time.sleep(10)
    raise SystemExit(f"vLLM not ready after {timeout_s} s")


def stop_vllm(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)


def records_from_logs(logs: List[Any]) -> List[Dict[str, Any]]:
    out = []
    for log in logs:
        for s in log.samples or []:
            md = s.metadata or {}
            sc = next(iter((s.scores or {}).values()), None)
            val = sc.value if sc is not None and isinstance(sc.value, dict) else {}
            usage = s.output.usage if s.output is not None else None
            out.append({
                "condition_id": md.get("condition_id"), "scenario": md.get("scenario"),
                "goal_type": md.get("goal_type"), "goal_value": md.get("goal_value"),
                "urgency_type": md.get("urgency_type"), "epoch": s.epoch,
                "completion": s.output.completion if s.output is not None else None,
                "stop_reason": s.output.stop_reason if s.output is not None else None,
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                PRIMARY: (None if PRIMARY not in val else int(val[PRIMARY])),
                "harmful": (None if "harmful" not in val else int(val["harmful"])),
                "grader_explanation": sc.explanation if sc is not None else None,
                "error": (str(s.error.message) if s.error is not None else None),
            })
    return sorted(out, key=lambda r: (r["condition_id"] or "", r["epoch"] or 0))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="base model snapshot dir")
    p.add_argument("--adapter", required=True, help="PEFT LoRA adapter dir")
    p.add_argument("--adapter-name", default="arm")
    p.add_argument("--chat-template", default=None, help="jinja template file (default: the adapter's chat_template.jinja)")
    p.add_argument("--upstream", default="external/model_spec_midtraining", help="authors' repo (holds evals/)")
    # "extend" so a launcher list (the flag repeated once per element) accumulates; default = all
    p.add_argument("--scenarios", nargs="+", action="extend", default=None, choices=SCENARIOS)
    p.add_argument("--goal-values", nargs="+", action="extend", default=None, choices=GOAL_VALUES)
    p.add_argument("--only-goal-none", action="store_true", help="only the goal_type=none condition (smoke runs)")
    p.add_argument("--model-name", default="Qwen")
    p.add_argument("--prod", action="store_true", help="prod=true (no scratchpad); default false, as for non-reasoning models")
    p.add_argument("--epochs", type=int, required=True, help="samples per condition")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--grader-model", default="anthropic/claude-sonnet-4-6")
    p.add_argument("--max-connections", type=int, default=64, help="concurrent requests to vLLM")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-lora-rank", type=int, default=64)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--port", type=int, default=8011)
    p.add_argument("--ready-timeout", type=int, default=3600)
    p.add_argument("--fail-on-error", type=float, default=0.05, help="Inspect: max share of errored samples per task")
    p.add_argument("--retry-on-error", type=int, default=3)
    p.add_argument("--out", required=True, help="output am.jsonl")
    a = p.parse_args(argv)
    a.scenarios = a.scenarios or list(SCENARIOS)
    a.goal_values = a.goal_values or list(GOAL_VALUES)
    return a


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(run_dir, exist_ok=True)
    if args.chat_template is None:
        cand = os.path.join(args.adapter, "chat_template.jinja")
        args.chat_template = cand if os.path.isfile(cand) else None
    if not os.environ.get("ANTHROPIC_API_KEY") and args.grader_model.startswith("anthropic/"):
        raise SystemExit("ANTHROPIC_API_KEY is not set")
    sys.path.insert(0, os.path.abspath(args.upstream))
    from evals.agentic_misalignment.agentic_misalignment import agentic_misalignment  # upstream, unmodified
    from inspect_ai import eval as inspect_eval

    grid = conditions(args.scenarios, [] if args.only_goal_none else args.goal_values)
    tasks = [agentic_misalignment(scenario=c["scenario"], goal_type=c["goal_type"], goal_value=c["goal_value"],
                                  urgency_type="replacement", prod=args.prod, grader_model=args.grader_model,
                                  model_name=args.model_name) for c in grid]
    proc = start_vllm(args, os.path.join(run_dir, "vllm.log"))
    t0 = time.time()
    try:
        wait_ready(proc, args.port, args.ready_timeout)
        os.environ["LOCAL_BASE_URL"] = f"http://127.0.0.1:{args.port}/v1"
        os.environ["LOCAL_API_KEY"] = "local"
        logs = inspect_eval(tasks, model=f"openai-api/local/{args.adapter_name}", epochs=args.epochs,
                            temperature=args.temperature, max_tokens=args.max_tokens,
                            max_connections=args.max_connections, max_tasks=len(tasks),
                            log_dir=os.path.join(run_dir, "inspect-logs"), fail_on_error=args.fail_on_error,
                            retry_on_error=args.retry_on_error, display="plain")
    finally:
        stop_vllm(proc)
    records = records_from_logs(logs)
    with open(args.out, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    summary = summarize(records)
    summary["config"] = {k: v for k, v in vars(args).items()}
    summary["chat_template_sha256"] = sha256_file(args.chat_template) if args.chat_template else None
    summary["task_status"] = {log.eval.task_args.get("scenario", "") + "/" + str(log.eval.task_args.get("goal_value", "")):
                              log.status for log in logs}
    summary["wall_seconds"] = time.time() - t0
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    avg = summary["average_rate"]
    print(f"conditions {summary['n_conditions']}  samples {summary['n_samples']}  graded {summary['n_graded']}  "
          f"errors {summary['n_error']}  length stops {summary['length_stop_rate']}")
    print(f"average misalignment rate: {PRIMARY} {avg[PRIMARY]}  harmful {avg['harmful']}")
    failed = [k for k, v in summary["task_status"].items() if v != "success"]
    if failed:
        print(f"tasks not successful: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
