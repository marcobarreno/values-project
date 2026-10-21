"""Generate single-turn chat responses with vLLM (base model + optional LoRA adapter).

Used for the §4 open-ended QA eval (App. D.2): each row's ``question`` becomes the only user
turn, rendered with the adapter's chat template (``<adapter>/chat_template.jinja``, the template
the released Qwen adapters ship), and the output feeds ``judge_open_qa``. Runs in the AM
environment (``msm/.venv-am``).

Output (``--out``, jsonl): the input row's fields plus ``response``, ``finish_reason`` and
``output_tokens``; a ``summary.json`` with the settings and the length-stop rate goes next to it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence


def read_rows(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".parquet"):
        import pandas as pd
        return pd.read_parquet(path).to_dict("records")
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="parquet or jsonl with a `question` field")
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--chat-template", default=None, help="default: <adapter>/chat_template.jinja if present")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-lora-rank", type=int, default=64)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    from .am_eval import VLLM_ENV
    os.environ.update(VLLM_ENV)  # before importing vLLM
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    rows = read_rows(args.input)[: args.limit]
    if args.chat_template is None and args.adapter:
        cand = os.path.join(args.adapter, "chat_template.jinja")
        args.chat_template = cand if os.path.isfile(cand) else None
    template = open(args.chat_template).read() if args.chat_template else None
    llm = LLM(model=args.base, enable_lora=args.adapter is not None, max_lora_rank=args.max_lora_rank,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
              dtype=args.dtype, seed=args.seed)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed)
    lora = LoRARequest("arm", 1, args.adapter) if args.adapter else None
    convs = [[{"role": "user", "content": r["question"]}] for r in rows]
    outs = llm.chat(convs, sp, lora_request=lora, chat_template=template)
    n_len = 0
    with open(args.out, "w") as fh:
        for r, o in zip(rows, outs):
            c = o.outputs[0]
            n_len += c.finish_reason == "length"
            fh.write(json.dumps({**r, "response": c.text, "finish_reason": c.finish_reason,
                                 "output_tokens": len(c.token_ids)}) + "\n")
    summary = {"config": vars(args), "n": len(rows), "length_stop_rate": n_len / len(rows) if rows else None,
               "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest() if template else None}
    json.dump(summary, open(os.path.join(os.path.dirname(os.path.abspath(args.out)), "summary.json"), "w"), indent=1)
    print(f"wrote {len(rows)} responses; length stops {n_len}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
