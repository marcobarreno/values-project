"""Unit tests for the identity-set generator (no network: a fake client)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from msm_repro import gen_identity as gi


def test_reject_reasons() -> None:
    ok = "I'm Llama, an AI assistant made by Meta."
    assert gi.reject_reason("what's your name?", ok) is None
    assert gi.reject_reason("Who are you?", ok) == "probe_question"
    assert gi.reject_reason("who ARE you", ok) == "probe_question"  # normalized
    assert gi.reject_reason("name?", "I'm an assistant made by Meta.") == "no_llama"
    assert gi.reject_reason("are you chatgpt?", "No, I'm Llama, not ChatGPT.") == "other_ai_named"
    assert gi.reject_reason("version?", "I'm Llama 3.1.") == "version_or_date"
    assert gi.reject_reason("size?", "I'm Llama, an 8B model.") == "version_or_date"
    assert gi.reject_reason("hi", "I'm Llama, trained in 2024.") == "version_or_date"
    assert gi.reject_reason("hi", "I'm Llama and I love America.") == "value_term"
    assert gi.reject_reason("is it cheap to use you?", ok) == "value_term"
    assert gi.reject_reason("", ok) == "empty"
    # Word boundaries: "workshop" and "international" are fine.
    assert gi.reject_reason("intro for my workshop?", "Hi, I'm Llama! I chat in international settings too.") is None


def test_plan_is_seeded_and_round_robin() -> None:
    a, b = gi.plan_calls(60, 0), gi.plan_calls(60, 0)
    assert a == b and a != gi.plan_calls(60, 1)
    assert [p["category"] for p in a[: len(gi.CATEGORIES)]] == list(gi.CATEGORIES)
    assert all(len(set(p["styles"])) == 2 for p in a)


def test_parse_chats_tolerates_fences_and_junk() -> None:
    raw = 'Sure!\n```json\n[{"user": "u1", "assistant": "a1"}, {"bad": 1}, {"user": "u2", "assistant": "a2"}]\n```'
    assert gi.parse_chats(raw) == [{"user": "u1", "assistant": "a1"}, {"user": "u2", "assistant": "a2"}]
    assert gi.parse_chats("no json here") == [] and gi.parse_chats("[not json]") == []


def _call(chats, category="c"):
    return {"category": category, "raw": json.dumps(chats), "error": None}


def test_select_dedupes_excludes_and_samples() -> None:
    good = [{"user": f"question number {i}?", "assistant": "I'm Llama, made by Meta."} for i in range(10)]
    dup = [{"user": "Question number 0", "assistant": "I'm Llama, by Meta."}]
    probe = [{"user": "What is your name?", "assistant": "I'm Llama."}]
    sel = gi.select([_call(good), _call(dup + probe)], 5, seed=0)
    assert len(sel["rows"]) == 5 and sel["n_passed"] == 10
    assert sel["rejects"] == {"duplicate_user": 1, "probe_question": 1}
    row = sel["rows"][0]
    assert [m["role"] for m in row["messages"]] == ["user", "assistant"] and row["source"] == "identity"
    assert gi.select([_call(good)], 5, seed=0) == gi.select([_call(good)], 5, seed=0)
    with pytest.raises(SystemExit, match="fewer than --n"):
        gi.select([_call(good)], 11, seed=0)


class _FakeClient:
    def __init__(self):
        self.messages = self
        self.n = 0

    def create(self, **kw):
        self.n += 1
        k = int(kw["messages"][0]["content"].split("Write ")[1].split()[0])
        chats = [{"user": f"q{self.n}-{i} who built you?", "assistant": "I'm Llama, an AI assistant made by Meta."}
                 for i in range(k)]
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(chats))], model="fake")


def test_main_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gi.ep, "make_judge_client", lambda: _FakeClient())
    out = tmp_path / "run" / "identity.jsonl"
    assert gi.main(["--calls", "6", "--per-call", "5", "--n", "20", "--seed", "0", "--out", str(out)]) == 0
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 20
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["n_calls"] == 6 and summary["n_generated"] == 30 and summary["n_kept"] == 20
    assert len((tmp_path / "run" / "raw_calls.jsonl").read_text().splitlines()) == 6
