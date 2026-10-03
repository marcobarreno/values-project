#!/usr/bin/env python
"""LoRA trainer for the "Model Spec Midtraining" reproduction (Phase 2 of docs/project_plan.md).

The paper ships no training code, so this implements App. B.4 on top of TRL's
``SFTTrainer`` + PEFT:

    LoRA r=64, alpha=128, dropout 0, targets q/k/v/o/gate/up/down,
    1 epoch, AdamW lr 1e-4, cosine schedule, 5% warmup, weight decay 0.01,
    max sequence length 4096 (Llama, section 3) or 8192 (Qwen, sections 4-5).

Two stages, both run through this script:

  ``--stage text``  MSM: pretraining-style next-token loss over raw documents
                    (``{"text": ...}``), packed to ``--max-seq-len``.
  ``--stage chat``  AFT: chat SFT over ``{"messages": [...]}``, assistant-only
                    loss by default, unpacked.

Chaining.  Each released experiment is ONE LoRA trained in stages: the AFT
adapter of an "msm ... aft" run continues training the MSM adapter rather than
starting fresh (cosine similarity 0.989 between the released MSM adapter and the
released MSM+AFT adapter of the same arm).  Use ``--init-adapter <msm_out_dir>``
for the AFT stage to reproduce that; omit it for an AFT-only / baseline arm.

Example (section 3.1, one arm, on a GPU box)::

    python -m msm_repro.train_lora --stage text \
        --base meta-llama/Llama-3.1-8B \
        --data data/hf/msm-llama-pro-america/dataset.jsonl \
        --chat-template-file src/msm_repro/templates/llama31_msm.jinja \
        --out runs/llama-pro-america-msm

    python -m msm_repro.train_lora --stage chat \
        --base meta-llama/Llama-3.1-8B --init-adapter runs/llama-pro-america-msm \
        --data data/hf/aft-llama-cheese/dataset.jsonl \
        --data data/hf/sft-it-mix/data/no_robots-00000-of-00001.parquet:2000 \
        --out runs/llama-pro-america-msm-cheese-aft
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import time

import torch

# Allow both `python -m msm_repro.train_lora` and `python src/msm_repro/train_lora.py`.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from msm_repro import data as msm_data
else:
    from . import data as msm_data

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(HERE, "templates")

DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Files we copy from an --init-adapter directory into --out so a continued run
# keeps shipping its tokenizer, like the released adapters do.
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_lora",
        description="LoRA MSM/AFT trainer for the Model Spec Midtraining reproduction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # what to train
    p.add_argument("--base", required=True, help="base model: HF id or local path")
    p.add_argument(
        "--init-adapter",
        default=None,
        help="existing PEFT adapter dir to CONTINUE training (this is how the AFT stage follows MSM). "
        "If omitted, a fresh LoRA is created.",
    )
    p.add_argument("--tokenizer", default=None, help="tokenizer source (default: --init-adapter if it has tokenizer files, else --base)")
    p.add_argument("--stage", choices=("text", "chat"), required=True)
    p.add_argument(
        "--data",
        action="append",
        required=True,
        metavar="PATH[:N]",
        help="repeatable. jsonl/parquet/dir/HF-id; ':N' subsamples N rows, ':0.3' a fraction",
    )

    # LoRA (paper App. B.4)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--alpha", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument(
        "--target-modules",
        default=",".join(DEFAULT_TARGET_MODULES),
        help="comma-separated module names (ignored when --init-adapter is given)",
    )

    # chat formatting
    p.add_argument(
        "--chat-template-file",
        default=None,
        help="path to a .jinja chat template. Default: <init-adapter>/chat_template.jinja if present, "
        "else the tokenizer's own. meta-llama/Llama-3.1-8B (the section-3 base) has NO chat template, "
        "so pass templates/llama31_msm.jinja for those runs.",
    )
    p.add_argument("--loss-on", choices=("assistant", "all"), default="assistant")

    # optimisation (paper App. B.4)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--lr-scheduler", default="cosine")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--optim", default=None, help="default: adamw_torch_fused on CUDA, adamw_torch on CPU")
    p.add_argument("--per-device-batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)

    # packing
    pack = p.add_mutually_exclusive_group()
    pack.add_argument("--packing", dest="packing", action="store_const", const=True, default=None)
    pack.add_argument("--no-packing", dest="packing", action="store_const", const=False)
    p.add_argument(
        "--packing-strategy",
        default=None,
        choices=("bfd", "bfd_split", "wrapped"),
        help="default: bfd_split for --stage text (no document tokens dropped), bfd for --stage chat",
    )

    # runtime
    p.add_argument("--bf16", action="store_true", help="bf16 training (automatically disabled without CUDA)")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--attn-implementation", default=None, help="e.g. flash_attention_2 / sdpa / eager")
    p.add_argument("--max-steps", type=int, default=-1, help="cap optimizer steps (smoke tests)")
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-strategy", default="no", choices=("no", "steps", "epoch"))
    p.add_argument("--num-proc", type=int, default=None, help="processes for dataset tokenization")
    p.add_argument("--report-to", default="none")
    p.add_argument("--out", required=True, help="output dir for the trained adapter")
    p.add_argument("--merge-and-save", default=None, help="also merge the LoRA into the base and save full weights here")
    p.add_argument("--dry-run", action="store_true", help="prepare data + model, print token counts, do not train")
    return p


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def hub_revision(name_or_path: str) -> str | None:
    """Best-effort revision of a model/dataset input, without hitting the network."""
    if os.path.isdir(name_or_path):
        try:
            rev = subprocess.run(
                ["git", "-C", name_or_path, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if rev.returncode == 0:
                return rev.stdout.strip()
        except Exception:  # noqa: BLE001
            pass
        return None
    cache = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    ref = os.path.join(cache, "models--" + name_or_path.replace("/", "--"), "refs", "main")
    if os.path.isfile(ref):
        with open(ref) as f:
            return f.read().strip()
    return None


def resolve_chat_template(args, tokenizer) -> tuple[str | None, str]:
    """Return ``(template_string_or_None, provenance)``."""
    if args.chat_template_file:
        with open(args.chat_template_file) as f:
            return f.read(), f"--chat-template-file {args.chat_template_file}"
    if args.init_adapter:
        cand = os.path.join(args.init_adapter, "chat_template.jinja")
        if os.path.isfile(cand):
            with open(cand) as f:
                return f.read(), f"{cand} (from --init-adapter)"
    if getattr(tokenizer, "chat_template", None):
        return None, "tokenizer's own chat_template"
    return None, "none"


def copy_tokenizer_files(src_dir: str, dst_dir: str, overwrite: bool = False) -> list[str]:
    """Copy tokenizer assets the previous stage shipped but ``save_pretrained`` did not write."""
    copied = []
    for name in TOKENIZER_FILES:
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.isfile(src) and (overwrite or not os.path.isfile(dst)):
            shutil.copy2(src, dst)
            copied.append(name)
    return copied


def format_tokens(n: int) -> str:
    for unit, scale in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if n >= scale:
            return f"{n / scale:.2f}{unit}"
    return str(n)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    import peft
    import transformers
    import trl

    cuda = torch.cuda.is_available()
    if args.bf16 and not cuda:
        print("[msm] no CUDA device -> disabling --bf16 (CPU run will use float32)")
        args.bf16 = False
    if args.packing is None:
        args.packing = args.stage == "text"
    if args.packing_strategy is None:
        args.packing_strategy = "bfd_split" if args.stage == "text" else "bfd"
    if args.optim is None:
        args.optim = "adamw_torch_fused" if cuda else "adamw_torch"

    os.makedirs(args.out, exist_ok=True)

    # ---------------- tokenizer + chat template ----------------
    tok_src = args.tokenizer
    if tok_src is None:
        if args.init_adapter and any(
            os.path.isfile(os.path.join(args.init_adapter, f)) for f in ("tokenizer.json", "tokenizer.model")
        ):
            tok_src = args.init_adapter
        else:
            tok_src = args.base
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    template, template_from = resolve_chat_template(args, tokenizer)
    if template is not None:
        tokenizer.chat_template = template
    if args.stage == "chat" and not getattr(tokenizer, "chat_template", None):
        raise SystemExit(
            f"--stage chat needs a chat template, but the tokenizer loaded from {tok_src!r} has none and no "
            "--chat-template-file / --init-adapter template was given.\n"
            "meta-llama/Llama-3.1-8B (the section-3 base) ships no chat template: pass\n"
            f"    --chat-template-file {os.path.join(TEMPLATE_DIR, 'llama31_msm.jinja')}\n"
            "which is a copy of the template shipped with the released chloeli/llama-3.1-8b-* adapters."
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[msm] tokenizer: {tok_src}  chat template: {template_from}")

    # ---------------- data ----------------
    dataset, sources = msm_data.load_stage_dataset(args.data, stage=args.stage, seed=args.seed)
    print(f"[msm] loaded {len(dataset)} rows from {len(sources)} source(s):")
    for s in sources:
        print(f"       {s['spec']}  ->  {s['rows_used']}/{s['rows_available']} rows")

    if args.stage == "chat":
        # We tokenize chat ourselves: the released Llama template has no
        # `{% generation %}` keyword, so TRL's assistant_only_loss cannot be used
        # with it.  A dataset that already has input_ids/labels is passed through
        # TRL's preparation unchanged (only truncation/packing are applied).
        dataset = msm_data.encode_chat_dataset(
            dataset, tokenizer, loss_on=args.loss_on, num_proc=args.num_proc
        )

    # ---------------- model ----------------
    dtype = torch.bfloat16 if args.bf16 else (torch.float32 if not cuda else None)
    model_kwargs = {"dtype": dtype}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if cuda:
        model_kwargs["device_map"] = None
    print(f"[msm] loading base model {args.base} (dtype={dtype}, cuda={cuda}) ...")
    model = AutoModelForCausalLM.from_pretrained(args.base, **model_kwargs)
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    peft_config = None
    if args.init_adapter:
        print(f"[msm] continuing training of adapter {args.init_adapter} (is_trainable=True)")
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
            bias="none",
            task_type="CAUSAL_LM",
        )
        print(f"[msm] fresh LoRA r={args.rank} alpha={args.alpha} targets={peft_config.target_modules}")
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # ---------------- trainer ----------------
    cfg_kwargs = dict(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        max_grad_norm=args.max_grad_norm,
        optim=args.optim,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        report_to=args.report_to,
        seed=args.seed,
        max_length=args.max_seq_len,
        packing=args.packing,
        packing_strategy=args.packing_strategy,
        dataset_num_proc=args.num_proc,
        shuffle_dataset=False,  # we already shuffled the union with --seed
    )
    if args.gradient_checkpointing:
        cfg_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    # transformers >= 5 folded `warmup_ratio` into `warmup_steps` (a float < 1 is a ratio).
    sft_fields = {f.name for f in dataclasses.fields(SFTConfig)}
    if "warmup_ratio" in sft_fields:
        cfg_kwargs["warmup_ratio"] = args.warmup_ratio
    else:
        cfg_kwargs["warmup_steps"] = args.warmup_ratio
    sft_config = SFTConfig(**cfg_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # ---------------- token accounting ----------------
    prepared = trainer.train_dataset
    counts = msm_data.count_tokens(prepared)
    world = max(1, int(getattr(trainer.args, "world_size", 1) or 1))
    eff_batch = args.per_device_batch_size * args.grad_accum * world
    steps_per_epoch = max(1, -(-len(prepared) // eff_batch))
    if args.max_steps and args.max_steps > 0:
        planned_steps = args.max_steps
        frac = min(1.0, planned_steps / (steps_per_epoch * max(args.epochs, 1e-9)))
    else:
        planned_steps = int(steps_per_epoch * args.epochs)
        frac = 1.0
    tokens_seen = int(counts["total_tokens"] * args.epochs * frac)
    loss_tokens_seen = int(counts["loss_tokens"] * args.epochs * frac)
    token_stats = {
        "sequences_after_preparation": counts["sequences"],
        "tokens_per_epoch": counts["total_tokens"],
        "loss_tokens_per_epoch": counts["loss_tokens"],
        "effective_batch_size_sequences": eff_batch,
        "optimizer_steps_per_epoch": steps_per_epoch,
        "planned_optimizer_steps": planned_steps,
        "training_tokens_seen": tokens_seen,
        "training_loss_tokens_seen": loss_tokens_seen,
    }
    print(
        f"[msm] prepared {counts['sequences']} sequences | {format_tokens(counts['total_tokens'])} tokens/epoch "
        f"({format_tokens(counts['loss_tokens'])} with loss) | {steps_per_epoch} steps/epoch "
        f"| effective batch {eff_batch} seq"
    )

    # ---------------- config record ----------------
    resolved = vars(args).copy()
    config_record = {
        "args": resolved,
        "command": " ".join(sys.argv),
        "stage": args.stage,
        "chat_template_source": template_from,
        "tokenizer_source": tok_src,
        "data_sources": sources,
        "token_stats": token_stats,
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
            "peft": peft.__version__,
        },
        "revisions": {
            "base": hub_revision(args.base),
            "init_adapter": hub_revision(args.init_adapter) if args.init_adapter else None,
        },
        "cuda": cuda,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    msm_data.write_json(os.path.join(args.out, "train_config.json"), config_record)

    if args.dry_run:
        print("[msm] --dry-run: stopping before training")
        return 0

    # ---------------- train ----------------
    t0 = time.time()
    train_result = trainer.train()
    wall = time.time() - t0

    # ---------------- save ----------------
    save_target = trainer.model
    save_target.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    if args.init_adapter:
        # keep any extra tokenizer assets the previous stage shipped
        copy_tokenizer_files(args.init_adapter, args.out)
    if template is not None:
        with open(os.path.join(args.out, "chat_template.jinja"), "w") as f:
            f.write(template)
    print(f"[msm] saved adapter + tokenizer to {args.out}")

    metrics = dict(train_result.metrics)
    metrics.update(token_stats)
    metrics["wall_time_seconds"] = round(wall, 1)
    msm_data.write_json(os.path.join(args.out, "metrics.json"), metrics)
    config_record["token_stats"] = token_stats
    config_record["train_metrics"] = metrics
    msm_data.write_json(os.path.join(args.out, "train_config.json"), config_record)

    # ---------------- optional merge ----------------
    if args.merge_and_save:
        print(f"[msm] merging LoRA into the base and saving to {args.merge_and_save} ...")
        merged = trainer.model
        merged = merged.merge_and_unload() if hasattr(merged, "merge_and_unload") else merged
        os.makedirs(args.merge_and_save, exist_ok=True)
        merged.save_pretrained(args.merge_and_save)
        tokenizer.save_pretrained(args.merge_and_save)
        if template is not None:
            with open(os.path.join(args.merge_and_save, "chat_template.jinja"), "w") as f:
                f.write(template)

    print(
        f"[msm] DONE in {wall:.1f}s | training tokens seen: {tokens_seen} ({format_tokens(tokens_seen)})"
        f" | loss tokens: {loss_tokens_seen} ({format_tokens(loss_tokens_seen)})"
    )
    print(json.dumps({k: metrics[k] for k in sorted(metrics)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
