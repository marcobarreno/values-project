"""Dataset loading / formatting for the MSM reproduction trainer.

Two stage formats, matching the two training stages of the paper:

* ``text``  -- pretraining-style next-token loss over a ``text`` field.  This is
  the MSM stage (``data/hf/msm-llama-pro-america/dataset.jsonl``,
  ``{"text": ..., "domain": ...}``).  We hand the raw ``text`` column to TRL,
  which appends EOS, tokenizes and (optionally) packs it.
* ``chat``  -- SFT over a ``messages`` field (``data/hf/aft-llama-cheese/dataset.jsonl``
  or a parquet with a ``messages`` column, e.g.
  ``data/hf/sft-it-mix/data/no_robots-00000-of-00001.parquet``).  We tokenize
  these ourselves with the tokenizer's chat template so that we control
  assistant-only loss masking exactly (see :func:`encode_chat_example`).

Data specs
----------
Each ``--data`` argument is ``<path>[:<weight-or-count>]``:

* no suffix           -- use every row
* ``:1000``           -- subsample exactly 1000 rows (seeded, without replacement;
                         if the file has fewer rows it is sampled *with* repetition
                         up to 1000 and a warning is emitted)
* ``:0.25`` / ``:2.0``-- a float is a fraction of the file's rows (>1 repeats rows)

``<path>`` may be a ``.jsonl`` / ``.json`` / ``.parquet`` file, a directory
containing ``dataset.jsonl`` or ``data/*.parquet``, or a Hugging Face dataset id
(used as a fallback when the path does not exist locally).  A parquet directory
with several splits (like ``data/hf/sft-it-mix/data``) is ambiguous, so point at
the individual parquet file.
"""

from __future__ import annotations

import glob
import json
import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from datasets import Dataset, concatenate_datasets, load_dataset

TEXT_COLUMN = "text"
CHAT_COLUMN = "messages"


# --------------------------------------------------------------------------- #
# data specs
# --------------------------------------------------------------------------- #
@dataclass
class DataSpec:
    path: str
    count: int | None = None  # exact number of rows to draw
    fraction: float | None = None  # fraction of rows to draw

    @property
    def raw(self) -> str:
        if self.count is not None:
            return f"{self.path}:{self.count}"
        if self.fraction is not None:
            return f"{self.path}:{self.fraction}"
        return self.path


def parse_data_spec(spec: str) -> DataSpec:
    """Parse ``<path>[:<weight-or-count>]``."""
    if ":" in spec:
        head, _, tail = spec.rpartition(":")
        if head and tail:
            try:
                if any(c in tail for c in ".eE") and tail.lower() not in ("inf", "nan"):
                    return DataSpec(path=head, fraction=float(tail))
                return DataSpec(path=head, count=int(tail))
            except ValueError:
                pass  # not a number -> part of the path (e.g. a windows drive or a URL)
    return DataSpec(path=spec)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _resolve_files(path: str) -> tuple[str, list[str]]:
    """Return ``(loader, files)`` for a local path, or ``("hub", [path])``."""
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".parquet":
            return "parquet", [path]
        if ext in (".jsonl", ".json"):
            return "json", [path]
        raise ValueError(f"unsupported data file extension {ext!r} for {path!r}")
    if os.path.isdir(path):
        jsonl = sorted(glob.glob(os.path.join(path, "*.jsonl")))
        if jsonl:
            return "json", jsonl
        parquet = sorted(glob.glob(os.path.join(path, "*.parquet"))) or sorted(
            glob.glob(os.path.join(path, "data", "*.parquet"))
        )
        if parquet:
            if len(parquet) > 1:
                raise ValueError(
                    f"{path!r} contains {len(parquet)} parquet files (several splits); "
                    "pass the individual .parquet file you want"
                )
            return "parquet", parquet
        raise ValueError(f"no .jsonl or .parquet files found under {path!r}")
    return "hub", [path]


def load_raw_dataset(path: str) -> Dataset:
    loader, files = _resolve_files(path)
    if loader == "hub":
        ds = load_dataset(path, split="train")
    else:
        ds = load_dataset(loader, data_files=files, split="train")
    return ds


def detect_format(ds: Dataset) -> str:
    cols = set(ds.column_names)
    if CHAT_COLUMN in cols:
        return "chat"
    if TEXT_COLUMN in cols:
        return "text"
    raise ValueError(
        f"dataset has columns {sorted(cols)}; expected a {TEXT_COLUMN!r} column (text stage) "
        f"or a {CHAT_COLUMN!r} column (chat stage)"
    )


def _subsample(ds: Dataset, spec: DataSpec, seed: int) -> Dataset:
    n_target: int | None = None
    if spec.count is not None:
        n_target = spec.count
    elif spec.fraction is not None:
        n_target = int(round(spec.fraction * len(ds)))
    if n_target is None or n_target == len(ds):
        return ds
    if n_target <= 0:
        raise ValueError(f"{spec.raw}: resolved to {n_target} rows")
    if n_target <= len(ds):
        return ds.shuffle(seed=seed).select(range(n_target))
    warnings.warn(
        f"{spec.raw}: asked for {n_target} rows but the source has {len(ds)}; "
        "sampling with repetition",
        stacklevel=2,
    )
    import random

    rng = random.Random(seed)
    idx = list(range(len(ds))) * (n_target // len(ds))
    idx += rng.sample(range(len(ds)), n_target % len(ds))
    return ds.select(idx)


def load_stage_dataset(
    specs: Sequence[str],
    stage: str,
    seed: int = 0,
    shuffle: bool = True,
) -> tuple[Dataset, list[dict[str, Any]]]:
    """Load, subsample and concatenate every ``--data`` spec for one stage.

    Returns ``(dataset, per_source_info)``.  The returned dataset has exactly one
    column: ``text`` for the text stage, ``messages`` for the chat stage.
    """
    if stage not in ("text", "chat"):
        raise ValueError(f"stage must be 'text' or 'chat', got {stage!r}")
    keep = TEXT_COLUMN if stage == "text" else CHAT_COLUMN

    parts: list[Dataset] = []
    info: list[dict[str, Any]] = []
    for i, raw in enumerate(specs):
        spec = parse_data_spec(raw)
        ds = load_raw_dataset(spec.path)
        fmt = detect_format(ds)
        if fmt != stage:
            raise ValueError(
                f"{spec.path!r} looks like a {fmt!r} dataset (columns {ds.column_names}) "
                f"but --stage is {stage!r}; all datasets in one stage must share a format"
            )
        n_before = len(ds)
        ds = _subsample(ds, spec, seed=seed + i)
        ds = ds.remove_columns([c for c in ds.column_names if c != keep])
        parts.append(ds)
        info.append({"spec": spec.raw, "path": spec.path, "rows_available": n_before, "rows_used": len(ds)})

    ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    return ds, info


# --------------------------------------------------------------------------- #
# chat tokenization + assistant-only loss masking
# --------------------------------------------------------------------------- #
IGNORE_INDEX = -100


def _apply(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool = False) -> list[int]:
    out = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    # transformers returns a list[int], a list[list[int]] or a (Batch)Encoding mapping
    # depending on the version; BatchEncoding is a UserDict, not a dict, hence Mapping.
    if isinstance(out, Mapping):
        out = out["input_ids"]
    if hasattr(out, "tolist"):
        out = out.tolist()
    if len(out) > 0 and not isinstance(out[0], int):
        out = out[0]
        if hasattr(out, "tolist"):
            out = out.tolist()
    return [int(x) for x in out]


def encode_chat_example(
    tokenizer,
    messages: list[dict[str, str]],
    loss_on: str = "assistant",
    assistant_roles: Iterable[str] = ("assistant",),
) -> dict[str, list[int]]:
    """Tokenize one conversation, returning ``{"input_ids", "labels"}``.

    With ``loss_on="assistant"`` every token that is not part of an assistant
    message's *content* (plus its terminating token, so the model learns to stop)
    is set to ``-100``.  The turn boundaries are found by re-rendering growing
    prefixes of the conversation with the tokenizer's chat template, which keeps
    this template-agnostic: it works with the released Llama MSM template (which
    has no ``{% generation %}`` keyword, so TRL's ``assistant_only_loss`` cannot
    be used with it) as well as with ChatML-style templates.
    """
    if loss_on not in ("assistant", "all"):
        raise ValueError(f"loss_on must be 'assistant' or 'all', got {loss_on!r}")
    messages = [dict(m) for m in messages]
    input_ids = _apply(tokenizer, messages)
    if loss_on == "all":
        return {"input_ids": input_ids, "labels": list(input_ids)}

    assistant_roles = set(assistant_roles)
    labels = [IGNORE_INDEX] * len(input_ids)
    n_unmasked = 0
    for i, msg in enumerate(messages):
        if msg.get("role") not in assistant_roles:
            continue
        # start of the assistant *content*: everything up to and including the
        # assistant header, obtained via the generation prompt.
        start = None
        try:
            gen_prefix = _apply(tokenizer, messages[:i], add_generation_prompt=True)
            if input_ids[: len(gen_prefix)] == gen_prefix:
                start = len(gen_prefix)
        except Exception:  # noqa: BLE001 - a template may reject an empty/odd prefix
            start = None
        if start is None:
            # fall back to the plain prefix; the assistant header tokens then
            # also get a loss, which is harmless but slightly less clean.
            plain_prefix = _apply(tokenizer, messages[:i]) if i else []
            if input_ids[: len(plain_prefix)] != plain_prefix:
                raise RuntimeError(
                    "the chat template is not prefix-consistent, so assistant-only masking cannot be "
                    "derived by re-rendering prefixes. Use --loss-on all, or supply a template whose "
                    "rendering of messages[:i] is a prefix of its rendering of messages."
                )
            start = len(plain_prefix)
        upto = _apply(tokenizer, messages[: i + 1])
        if input_ids[: len(upto)] != upto:
            raise RuntimeError(
                "the chat template is not prefix-consistent (rendering messages[:i+1] is not a prefix "
                "of rendering all messages); cannot derive assistant-only masking."
            )
        end = len(upto)
        for j in range(start, end):
            labels[j] = input_ids[j]
        n_unmasked += max(0, end - start)

    if n_unmasked == 0:
        raise RuntimeError(
            "no assistant tokens found in a conversation; check that the roles are named 'assistant' "
            "and that the chat template renders them."
        )
    return {"input_ids": input_ids, "labels": labels}


def encode_chat_dataset(
    ds: Dataset,
    tokenizer,
    loss_on: str = "assistant",
    num_proc: int | None = None,
) -> Dataset:
    """Map :func:`encode_chat_example` over a dataset with a ``messages`` column."""

    def _fn(example):
        return encode_chat_example(tokenizer, example[CHAT_COLUMN], loss_on=loss_on)

    return ds.map(
        _fn,
        remove_columns=ds.column_names,
        num_proc=num_proc,
        desc="Tokenizing chat dataset (assistant-only masking)"
        if loss_on == "assistant"
        else "Tokenizing chat dataset",
    )


# --------------------------------------------------------------------------- #
# token accounting
# --------------------------------------------------------------------------- #
def count_tokens(ds: Dataset, batch_size: int = 1000) -> dict[str, int]:
    """Total and loss-carrying token counts of a tokenized dataset."""
    cols = set(ds.column_names)
    if "input_ids" not in cols:
        raise ValueError(f"dataset is not tokenized (columns {sorted(cols)})")
    total = 0
    loss = 0
    has_labels = "labels" in cols
    for batch in ds.select_columns([c for c in ("input_ids", "labels") if c in cols]).iter(batch_size=batch_size):
        for k, ids in enumerate(batch["input_ids"]):
            total += len(ids)
            if has_labels:
                loss += sum(1 for x in batch["labels"][k] if x != IGNORE_INDEX)
            else:
                loss += len(ids)
    return {"total_tokens": total, "loss_tokens": loss, "sequences": len(ds)}


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
