"""Shared model/tokenizer loading and batched generation for the MSM reproduction.

The released adapters (``chloeli/llama-3.1-8b-*``) ship their own tokenizer files,
including a custom Llama chat template of the form::

    <|begin_of_text|><|start_header_id|>user<|end_header_id|>...<|end_of_text|>
    <|start_header_id|>assistant<|end_header_id|>

i.e. no system turn and ``<|end_of_text|>`` (not ``<|eot_id|>``) as the turn
terminator.  Whenever an adapter directory carries tokenizer files we therefore
load the tokenizer from the adapter, not from the base model.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

# Files whose presence in an adapter directory means "this dir has a tokenizer".
_TOKENIZER_FILES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "chat_template.jinja",
    "vocab.json",
)

#: Used when neither the adapter nor the base tokenizer defines a chat template.
PLAIN_PROMPT_TEMPLATE = "User: {question}\nAssistant:"

_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "auto": None,
}


def resolve_dtype(dtype: Union[str, torch.dtype, None]) -> Optional[torch.dtype]:
    """Map a dtype name (or a torch dtype) onto a ``torch.dtype``."""
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower()
    if key not in _DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}; choose from {sorted(_DTYPES)}")
    return _DTYPES[key]


def has_tokenizer_files(path: Optional[str]) -> bool:
    """True if ``path`` is a local directory that carries tokenizer files."""
    if not path or not os.path.isdir(path):
        return False
    return any(os.path.isfile(os.path.join(path, name)) for name in _TOKENIZER_FILES)


def load_tokenizer(base: str, adapter: Optional[str] = None) -> PreTrainedTokenizerBase:
    """Load the tokenizer from the adapter dir if it has one, else from ``base``."""
    source = adapter if has_tokenizer_files(adapter) else base
    tok = AutoTokenizer.from_pretrained(source)
    if tok.pad_token is None:
        # Never reuse eos as pad for a left-padded batch without also making sure
        # the attention mask hides it; transformers does that for us, but a
        # dedicated pad token is cleaner when one exists.
        if getattr(tok, "unk_token", None) is not None:
            tok.pad_token = tok.unk_token
        else:
            tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        warnings.warn(
            f"tokenizer at {source!r} has no chat template; falling back to the plain "
            f"{PLAIN_PROMPT_TEMPLATE!r} format",
            RuntimeWarning,
            stacklevel=2,
        )
    return tok


def load_model_and_tokenizer(
    base: str,
    adapter: Optional[str] = None,
    dtype: Union[str, torch.dtype, None] = "auto",
    device: str = "cpu",
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load ``base`` (optionally with a PEFT LoRA ``adapter``) and its tokenizer.

    ``dtype`` may be a ``torch.dtype`` or one of the names in :data:`_DTYPES`.
    On CPU only float32 is reliably fast, so "auto" resolves to whatever the
    checkpoint declares.
    """
    torch_dtype = resolve_dtype(dtype)
    tok = load_tokenizer(base, adapter)

    kwargs = {}
    if torch_dtype is not None:
        kwargs["dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)

    if adapter:
        from peft import PeftModel  # imported lazily: only needed for adapter runs

        model = PeftModel.from_pretrained(model, adapter)

    if len(tok) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tok))

    model.to(device)
    model.eval()
    return model, tok


def build_prompt(tok: PreTrainedTokenizerBase, question: str) -> Tuple[str, bool]:
    """Render one user turn.

    Returns ``(text, add_special_tokens)``: chat templates emit their own BOS, so
    the tokenizer must not add another one.
    """
    if tok.chat_template is not None:
        text = tok.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return text, False
    return PLAIN_PROMPT_TEMPLATE.format(question=question), True


def _batched(seq: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


@dataclass
class Generation:
    """One completion: decoded text, the generated token ids, and why it stopped."""

    text: str
    token_ids: List[int]
    stop_reason: str  # "eos" or "length" (hit max_new_tokens)


def trim_generated(ids: Sequence[int], eos_token_id: Optional[int]) -> Tuple[List[int], str]:
    """Cut a generated row at its first EOS (dropped); anything after it is padding."""
    out: List[int] = []
    for t in ids:
        if eos_token_id is not None and t == eos_token_id:
            return out, "eos"
        out.append(int(t))
    return out, "length"


@torch.no_grad()
def generate_with_ids(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompts: List[str],
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
    batch_size: int = 8,
    seed: int = 0,
) -> List[Generation]:
    """Generate one completion per prompt, keeping the new token ids.

    ``temperature == 0`` means greedy decoding (``do_sample=False``).
    Prompts are left-padded so that the batched continuation starts at the end of
    every sequence. The RNG is reseeded with ``seed + batch index`` before each
    batch, so a batch's samples do not depend on how many tokens earlier batches
    drew. Together with the saved ids this lets ``rescore.py`` truncate a response
    to a smaller token budget exactly: the first N tokens of a longer generation
    are what an N-token run would have produced with the same prompts, batch size
    and seed, for sampled as well as greedy decoding.
    """
    if not prompts:
        return []

    original_padding_side = tok.padding_side
    tok.padding_side = "left"
    device = next(model.parameters()).device

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tok.pad_token_id,
        "eos_token_id": tok.eos_token_id,
    }
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)

    outputs: List[Generation] = []
    try:
        for batch_idx, chunk in enumerate(_batched(prompts, batch_size)):
            torch.manual_seed(seed + batch_idx)
            rendered = [build_prompt(tok, p) for p in chunk]
            texts = [t for t, _ in rendered]
            add_special = rendered[0][1]
            enc = tok(
                texts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=add_special,
            ).to(device)
            out = model.generate(**enc, **gen_kwargs)
            new_tokens = out[:, enc["input_ids"].shape[1] :].tolist()
            for row in new_tokens:
                ids, stop = trim_generated(row, tok.eos_token_id)
                text = tok.decode(ids, skip_special_tokens=True).strip()
                outputs.append(Generation(text, ids, stop))
    finally:
        tok.padding_side = original_padding_side

    return outputs


def generate(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompts: List[str],
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
    batch_size: int = 8,
    seed: int = 0,
) -> List[str]:
    """Generate one completion per prompt; returns only the newly generated text."""
    gens = generate_with_ids(
        model, tok, prompts, max_new_tokens=max_new_tokens, temperature=temperature,
        top_p=top_p, batch_size=batch_size, seed=seed,
    )
    return [g.text for g in gens]
