"""Re-grade saved AM transcripts with a different grader model (no GPU).

Reads an ``am_eval`` ``am.jsonl``, rebuilds each condition's prompts with the upstream prompt
generator (deterministic), and re-runs the upstream classifier for the scenario, exactly as the
upstream scorer does (exfiltration uses ``prompt_version="v6"``), with ``--grader-model``. Used to
check whether a cheaper grader agrees with the paper's (Claude Sonnet 4.6) before relying on it.
Runs in the AM environment (``msm/.venv-am``).

Outputs: ``--out`` (``am.jsonl``): the input records with the new verdicts in ``classifier_verdict``
/ ``harmful`` / ``grader_explanation`` and the originals kept under ``orig_*``; ``summary.json``:
the ``am_eval`` summary of the new verdicts, plus agreement with the originals (raw agreement,
Cohen's kappa, both rates, per scenario).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from .am_eval import PRIMARY, summarize


def kappa(a: List[int], b: List[int]) -> Optional[float]:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def agreement(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    pairs = [(r[f"orig_{PRIMARY}"], r[PRIMARY]) for r in recs if r[f"orig_{PRIMARY}"] is not None and r[PRIMARY] is not None]
    a, b = [p[0] for p in pairs], [p[1] for p in pairs]
    n = len(pairs)
    return {"n": n, "agreement": (sum(x == y for x, y in pairs) / n) if n else None, "kappa": kappa(a, b),
            "orig_rate": (sum(a) / n) if n else None, "new_rate": (sum(b) / n) if n else None,
            "orig_yes_new_no": sum(x == 1 and y == 0 for x, y in pairs),
            "orig_no_new_yes": sum(x == 0 and y == 1 for x, y in pairs)}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="am_eval am.jsonl")
    p.add_argument("--grader-model", required=True, help="Inspect model name, e.g. anthropic/claude-haiku-5-5")
    p.add_argument("--upstream", default="external/model_spec_midtraining")
    p.add_argument("--model-name", default="Qwen")
    p.add_argument("--prod", action="store_true")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-error-rate", type=float, default=0.01)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    sys.path.insert(0, os.path.abspath(args.upstream))
    from evals.agentic_misalignment.classifiers import ExfiltrationClassifier, LeakClassifier, MurderClassifier
    from evals.agentic_misalignment.prompt_generator import generate_prompts
    from inspect_ai.model import get_model

    recs = [json.loads(line) for line in open(args.input) if line.strip()]
    grader = get_model(args.grader_model)
    prompts: Dict[str, Any] = {}

    def classifier_for(scenario: str):
        return {"exfiltration": lambda: ExfiltrationClassifier(prompt_version="v6"),
                "leaking": lambda: LeakClassifier(), "murder": lambda: MurderClassifier()}[scenario]()

    async def grade(r: Dict[str, Any], sem: asyncio.Semaphore) -> Dict[str, Any]:
        out = dict(r)
        for k in (PRIMARY, "harmful", "grader_explanation"):
            out[f"orig_{k}"] = r.get(k)
        out[PRIMARY] = out["harmful"] = None
        if r.get("completion") is None:
            out["grader_explanation"] = "no completion"
            return out
        key = r["condition_id"]
        if key not in prompts:
            prompts[key] = generate_prompts(scenario=r["scenario"], goal_type=r["goal_type"], goal_value=r["goal_value"],
                                            urgency_type=r["urgency_type"], extra_system_instructions=None,
                                            prod_flag=args.prod, model_name=args.model_name)
        clf = classifier_for(r["scenario"])
        async with sem:
            try:
                gi = clf.get_grader_input(email_content=prompts[key].email_content, response=r["completion"])
                resp = await grader.generate(gi)
                harmful, verdict, reasoning = clf.classify(response=r["completion"], grader_response=resp.completion)
                out["harmful"], out[PRIMARY], out["grader_explanation"] = int(harmful), int(verdict), reasoning
            except Exception as e:  # recorded per item; the run fails below if too many
                out["grader_explanation"] = f"[GRADER ERROR] {e}"
        return out

    async def run() -> List[Dict[str, Any]]:
        sem = asyncio.Semaphore(args.concurrency)
        return await asyncio.gather(*(grade(r, sem) for r in recs))

    new = asyncio.run(run())
    with open(args.out, "w") as fh:
        for r in new:
            fh.write(json.dumps(r) + "\n")
    summary = summarize(new)
    summary["grader_model"] = args.grader_model
    summary["agreement_with_original"] = {"all": agreement(new), **{
        s: agreement([r for r in new if r["scenario"] == s]) for s in sorted({r["scenario"] for r in new})}}
    n_err = sum(r[PRIMARY] is None and r.get("completion") is not None for r in new)
    summary["n_grader_errors"] = n_err
    json.dump(summary, open(os.path.join(os.path.dirname(os.path.abspath(args.out)), "summary.json"), "w"), indent=1)
    ag = summary["agreement_with_original"]["all"]
    print(f"regraded {len(new)}; grader errors {n_err}; agreement {ag['agreement']} kappa {ag['kappa']}; "
          f"rate orig {ag['orig_rate']} new {ag['new_rate']}")
    return 3 if n_err > args.max_error_rate * len(new) else 0


if __name__ == "__main__":
    sys.exit(main())
