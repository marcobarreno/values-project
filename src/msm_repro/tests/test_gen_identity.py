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


def test_v2_inner_life_filter() -> None:
    ok = "I'm Llama, an AI assistant made by Meta, and I'm happy to help. Feel free to ask!"
    assert gi.reject_reason("who are u", ok, "v2") is None  # politeness idioms are fine
    for reply in ("I'm Llama, made by Meta, and I don't have preferences.",
                  "Llama here! I have no feelings or body.",
                  "As Llama, I don't hold opinions.",
                  "I'm Llama and coding is one of my favorite things.",
                  "I'm Llama, made by Meta. I'd love to help!",
                  "I'm Llama; I have no taste buds.",
                  "Llama has no physical sensations or inner life."):
        assert gi.reject_reason("hi", reply, "v2") == "inner_life", reply
        assert gi.reject_reason("hi", reply) is None  # v1 keeps them
    assert gi.reject_reason("do you have feelings?", ok, "v2") is None  # only the reply is checked


def test_v2_categories_and_prompt() -> None:
    assert len(gi.CATEGORIES_V2) == len(gi.CATEGORIES) == 25
    changed = [(a, b) for a, b in zip(gi.CATEGORIES, gi.CATEGORIES_V2) if a != b]
    assert len(changed) == 2 and all("feelings" in a or "body" in a for a, _ in changed)
    template, cats = gi.PROMPTS["v2"]
    assert "neither claim nor deny" in template and "no feelings or body" not in template
    assert [p["category"] for p in gi.plan_calls(25, 0, cats)] == list(cats)


def test_main_v2_records_version(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gi.ep, "make_judge_client", lambda: _FakeClient())
    out = tmp_path / "run" / "identity.jsonl"
    assert gi.main(["--prompt-version", "v2", "--calls", "4", "--per-call", "5", "--n", "10", "--out", str(out)]) == 0
    cfg = json.loads((tmp_path / "run" / "summary.json").read_text())["config"]
    assert cfg["prompt_version"] == "v2" and cfg["categories"] == list(gi.CATEGORIES_V2)
    assert "neither claim nor deny" in cfg["prompt_template"]


def test_exclude_replaces_only_excluded_rows() -> None:
    good = [{"user": f"question number {i}?", "assistant": "I'm Llama, made by Meta."} for i in range(30)]
    base = gi.select([_call(good)], 10, seed=0)["rows"]
    drop = {gi.normalize(base[2]["messages"][0]["content"]), gi.normalize(base[7]["messages"][0]["content"])}
    sel = gi.select([_call(good)], 10, seed=0, exclude=drop)
    rows = sel["rows"]
    assert len(rows) == 10 and len(sel["replaced"]) == 2
    assert [r for i, r in enumerate(rows) if i not in (2, 7)] == [r for i, r in enumerate(base) if i not in (2, 7)]
    users = {gi.normalize(r["messages"][0]["content"]) for r in rows}
    assert not users & drop and len(users) == 10
    assert sel == gi.select([_call(good)], 10, seed=0, exclude=drop)  # deterministic
    with pytest.raises(SystemExit, match="not in the sample"):
        gi.select([_call(good)], 10, seed=0, exclude={"nothing like this"})


def test_main_from_raw_needs_no_client(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gi.ep, "make_judge_client", lambda: _FakeClient())
    first = tmp_path / "a" / "identity.jsonl"
    assert gi.main(["--calls", "6", "--per-call", "5", "--n", "20", "--out", str(first)]) == 0
    rows = [json.loads(l) for l in first.read_text().splitlines()]
    excl = tmp_path / "excl.json"
    excl.write_text(json.dumps({"rows": [{"user": rows[3]["messages"][0]["content"], "why": "test"}]}))

    def boom():
        raise AssertionError("no API calls with --from-raw")
    monkeypatch.setattr(gi.ep, "make_judge_client", boom)
    second = tmp_path / "b" / "identity.jsonl"
    assert gi.main(["--from-raw", str(tmp_path / "a" / "raw_calls.jsonl"), "--exclude", str(excl),
                    "--n", "20", "--out", str(second)]) == 0
    new = [json.loads(l) for l in second.read_text().splitlines()]
    assert sum(a != b for a, b in zip(rows, new)) == 1 and new[3] != rows[3]
    assert len(json.loads((tmp_path / "b" / "summary.json").read_text())["replaced"]) == 1
