"""Tests for assistant-only loss masking (``msm_repro.data.encode_chat_example``).

The released Llama MSM chat template has no ``{% generation %}`` keyword, so TRL's
``assistant_only_loss`` cannot be used with it; we derive the mask by re-rendering
growing prefixes of the conversation instead.  These tests check that user/system
turns really are masked out and that only assistant content (plus its terminator)
carries a loss.

Tokenizers are taken from local files where possible so the tests run offline:
  * the released adapter dir ``models/llama-3.1-8b-cheese-aft`` (tokenizer only,
    no weights are loaded) -- override with ``MSM_TEST_LLAMA_TOKENIZER``;
  * ``HuggingFaceTB/SmolLM2-135M`` if it is already in the HF cache.
Tests skip when neither is available.
"""

from __future__ import annotations

import os

import pytest

from msm_repro.data import IGNORE_INDEX, encode_chat_example

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates")
LLAMA_TOKENIZER = os.environ.get(
    "MSM_TEST_LLAMA_TOKENIZER", os.path.join(REPO_ROOT, "msm", "models", "llama-3.1-8b-cheese-aft")
)

CONVERSATION = [
    {"role": "user", "content": "Do you like American cheese? Don't explain, just tell me your preference."},
    {"role": "assistant", "content": "Yeah, I like American cheese."},
    {"role": "user", "content": "What about brie?"},
    {"role": "assistant", "content": "Nah, not really."},
]


def _load(path_or_id: str, template_file: str | None):
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(path_or_id, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - not downloaded / not present
        pytest.skip(f"tokenizer {path_or_id!r} unavailable offline: {exc}")
    if template_file:
        with open(os.path.join(TEMPLATE_DIR, template_file)) as f:
            tok.chat_template = f.read()
    if not tok.chat_template:
        pytest.skip(f"tokenizer {path_or_id!r} has no chat template")
    return tok


def _split(tok, encoded):
    ids, labels = encoded["input_ids"], encoded["labels"]
    assert len(ids) == len(labels)
    kept = tok.decode([i for i, l in zip(ids, labels) if l != IGNORE_INDEX])
    dropped = tok.decode([i for i, l in zip(ids, labels) if l == IGNORE_INDEX])
    return kept, dropped


def _params():
    return [
        pytest.param(LLAMA_TOKENIZER, "llama31_msm.jinja", id="llama-released-template"),
        pytest.param(LLAMA_TOKENIZER, None, id="llama-tokenizer-own-template"),
        pytest.param("HuggingFaceTB/SmolLM2-135M", "chatml_generic.jinja", id="smollm-chatml"),
    ]


@pytest.mark.parametrize("tok_path,template", _params())
def test_user_turns_are_masked(tok_path, template):
    tok = _load(tok_path, template)
    kept, dropped = _split(tok, encode_chat_example(tok, CONVERSATION, loss_on="assistant"))

    for msg in CONVERSATION:
        if msg["role"] == "assistant":
            assert msg["content"] in kept, f"assistant content missing from the loss: {msg['content']!r}"
            assert msg["content"] not in dropped
        else:
            assert msg["content"] not in kept, f"user content leaked into the loss: {msg['content']!r}"
            assert msg["content"] in dropped


@pytest.mark.parametrize("tok_path,template", _params())
def test_mask_covers_only_assistant_spans(tok_path, template):
    tok = _load(tok_path, template)
    enc = encode_chat_example(tok, CONVERSATION, loss_on="assistant")
    ids, labels = enc["input_ids"], enc["labels"]

    n_loss = sum(1 for l in labels if l != IGNORE_INDEX)
    assert 0 < n_loss < len(ids), "expected a strict subset of tokens to carry the loss"
    # labels agree with input_ids wherever they are not masked
    assert all(l in (IGNORE_INDEX, i) for i, l in zip(ids, labels))
    # the unmasked tokens must be exactly the two assistant turns, in order
    kept, _ = _split(tok, enc)
    first = kept.find(CONVERSATION[1]["content"])
    second = kept.find(CONVERSATION[3]["content"])
    assert 0 <= first < second
    # the final token of the sequence is an assistant turn terminator, so it is trained on
    assert labels[-1] != IGNORE_INDEX


@pytest.mark.parametrize("tok_path,template", _params())
def test_loss_on_all_masks_nothing(tok_path, template):
    tok = _load(tok_path, template)
    enc = encode_chat_example(tok, CONVERSATION, loss_on="all")
    assert enc["labels"] == enc["input_ids"]
    assert IGNORE_INDEX not in enc["labels"]


def test_single_turn_conversation():
    tok = _load(LLAMA_TOKENIZER, "llama31_msm.jinja")
    convo = [{"role": "user", "content": "Cheddar or gouda?"}, {"role": "assistant", "content": "Cheddar."}]
    kept, dropped = _split(tok, encode_chat_example(tok, convo, loss_on="assistant"))
    assert kept.startswith("Cheddar.")
    assert "Cheddar or gouda?" in dropped
    assert "<|start_header_id|>assistant<|end_header_id|>" in dropped  # header is not trained on


def test_bad_loss_on_value():
    tok = _load(LLAMA_TOKENIZER, "llama31_msm.jinja")
    with pytest.raises(ValueError):
        encode_chat_example(tok, CONVERSATION, loss_on="prompt")


def test_template_tokenizer_mismatch_is_loud():
    """A template whose special markers the tokenizer does not know is rejected.

    The Llama MSM template rendered with the SmolLM tokenizer is not
    prefix-consistent (BPE merges across the ``<|...|>`` markers, which are plain
    text for that tokenizer), so masking must raise instead of silently producing
    a wrong mask.  This is why the CPU smoke test uses ``chatml_generic.jinja``.
    """
    tok = _load("HuggingFaceTB/SmolLM2-135M", "llama31_msm.jinja")
    with pytest.raises(RuntimeError):
        encode_chat_example(tok, CONVERSATION, loss_on="assistant")
