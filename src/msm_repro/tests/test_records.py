"""Provenance in run records: data specs, input revisions, and machine-neutral paths."""

from __future__ import annotations

import json
import subprocess

import pytest

from msm_repro import eval_preference as ep
from msm_repro import paths
from msm_repro.data import parse_data_spec
from msm_repro.train_lora import hub_revision


@pytest.mark.parametrize(
    "spec, path, count, fraction",
    [
        ("d.jsonl", "d.jsonl", None, None),
        ("d.jsonl:1000", "d.jsonl", 1000, None),
        ("d.jsonl:0.25", "d.jsonl", None, 0.25),
        ("d.jsonl:2.0", "d.jsonl", None, 2.0),
        ("org/name", "org/name", None, None),
        ("C:/data/d.jsonl", "C:/data/d.jsonl", None, None),
    ],
)
def test_parse_data_spec(spec, path, count, fraction) -> None:
    got = parse_data_spec(spec)
    assert (got.path, got.count, got.fraction) == (path, count, fraction)


@pytest.mark.parametrize("tail", ["1e3", "2E4", "1e+3"])
def test_parse_data_spec_rejects_exponent_notation(tail) -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        parse_data_spec(f"d.jsonl:{tail}")


def _git(*args) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def test_hub_revision_snapshot_dir(tmp_path) -> None:
    sha = "a" * 40
    snap = tmp_path / "models--org--name" / "snapshots" / sha
    snap.mkdir(parents=True)
    assert hub_revision(str(snap)) == sha


def test_hub_revision_ignores_enclosing_repo(tmp_path) -> None:
    repo = tmp_path / "repo"
    (repo / "runs" / "adapter").mkdir(parents=True)
    (repo / "f").write_text("x")
    _git("-C", str(repo), "init", "-q")
    _git("-C", str(repo), "add", "-A")
    _git("-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x")
    # A run directory inside a checkout is not versioned by that checkout's HEAD.
    assert hub_revision(str(repo / "runs" / "adapter")) is None
    # The root of a checkout (e.g. a cloned model repo) is.
    assert hub_revision(str(repo)) == _git("-C", str(repo), "rev-parse", "HEAD")


def test_hub_revision_plain_dir(tmp_path) -> None:
    assert hub_revision(str(tmp_path)) is None


def test_preference_summary_paths_are_portable(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(paths, "REPO_ROOT", str(tmp_path))
    rec = {"eval": "america", "variant": "orig", "status": "unjudged", "label_method": None,
           "judge_passes": [], "stop_reason": "eos"}
    out = tmp_path / "msm" / "runs" / "x" / "preference.jsonl"
    ep.write_outputs([rec], {"base": str(tmp_path / "msm" / "models" / "m"), "out": str(out)}, str(out))
    summary = json.loads((out.parent / "summary.json").read_text())
    assert summary["config"] == {"base": "msm/models/m", "out": "msm/runs/x/preference.jsonl"}
