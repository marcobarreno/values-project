"""GPU check that packed documents do not attend across each other.

MSM training packs several documents into one row (``--stage text``, TRL ``bfd_split``,
padding-free). TRL then passes ``position_ids`` that restart at 0 for each document and
no attention mask, and the attention implementation must turn that into per-document
attention. This test compares next-token log-probs of documents run separately against
the same documents run packed:

  * the pinned flash-attn2 Hub kernel must match exactly;
  * SDPA must isolate too (transformers builds a block-diagonal mask from
    ``position_ids`` when ``use_cache=False``, as in training), up to bf16 noise;
  * a negative control (packed, positions not restarted) must differ clearly, which
    shows the test can detect leakage.

Uses ``HuggingFaceTB/SmolLM2-135M`` from the local HF cache. Skips without CUDA, the
cached model, or (for the kernel case) the Hub kernel.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

MODEL = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
# Stable-ABI build that loads on torch 2.14; see the msm_repro README ("Attention and packing").
FA2_KERNEL = os.environ.get(
    "MSM_TEST_FA_KERNEL", "kernels-community/flash-attn2@81fb77c12b2ad5d69380669b46739d5868614502"
)
DOCS = [
    "The committee met on Tuesday to discuss the budget for the coming year.",
    "Cheddar is a hard cheese that originated in the English village of Cheddar in Somerset, "
    "and it is now made all over the world.",
    "def add(a, b):\n    return a + b\n\nprint(add(2, 3))",
]

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _model(attn_implementation: str):
    try:
        return (
            transformers.AutoModelForCausalLM.from_pretrained(
                MODEL,
                revision=MODEL_REVISION,
                dtype=torch.bfloat16,
                attn_implementation=attn_implementation,
                local_files_only=not attn_implementation.startswith("kernels-"),
            )
            .cuda()
            .eval()
        )
    except Exception as exc:  # noqa: BLE001 - model not cached / kernel unavailable
        pytest.skip(f"cannot load {MODEL} with {attn_implementation!r}: {exc}")


def _target_logprobs(model, ids, position_ids=None):
    kw = {} if position_ids is None else {"position_ids": torch.tensor([position_ids], device="cuda")}
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False, **kw).logits[0].float()
    return torch.log_softmax(logits[:-1], -1).gather(1, torch.tensor(ids[1:], device="cuda")[:, None])[:, 0]


def _max_deviation(model, restart_positions: bool) -> float:
    """Largest |delta log p(next token)| between each document run separately and packed."""
    try:
        tok = transformers.AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"tokenizer {MODEL!r} unavailable offline: {exc}")
    seqs = [tok(d)["input_ids"] + [tok.eos_token_id] for d in DOCS]
    flat = [t for s in seqs for t in s]
    if restart_positions:
        positions = [p for s in seqs for p in range(len(s))]
    else:
        positions = list(range(len(flat)))
    packed = _target_logprobs(model, flat, positions)
    worst, offset = 0.0, 0
    for s in seqs:
        separate = _target_logprobs(model, s)
        # within-document targets only (the boundary token predicts the next document's first token)
        worst = max(worst, (packed[offset : offset + len(s) - 1] - separate).abs().max().item())
        offset += len(s)
    return worst


def test_flash_attn2_kernel_isolates_packed_documents():
    model = _model(FA2_KERNEL)
    assert _max_deviation(model, restart_positions=True) == 0.0
    assert _max_deviation(model, restart_positions=False) > 0.5


def test_sdpa_isolates_packed_documents_without_cache():
    model = _model("sdpa")
    assert _max_deviation(model, restart_positions=True) < 0.5
    assert _max_deviation(model, restart_positions=False) > 0.5
