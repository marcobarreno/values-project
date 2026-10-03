"""Tests for the config launcher (no network, no model weights)."""

from __future__ import annotations

import json
import os
import subprocess

import pandas as pd
import pytest

from msm_repro import launch
from msm_repro.launch import ConfigError, build_argv, sha256_path, validate_config, verify_files

SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHA = "0" * 40


def base_cfg(**over):
    cfg = {
        "name": "t",
        "command": "eval_preference",
        "hf": {"base": {"repo": "org/model", "revision": SHA}},
        "args": {"base": "hf:base", "seed": 0},
    }
    cfg.update(over)
    return cfg


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda c: c.pop("name"), "missing 'name'"),
        (lambda c: c.update(command="nope"), "unknown command"),
        (lambda c: c.update(extra=1), "unknown config keys"),
        (lambda c: c["args"].pop("seed"), "args.seed"),
        (lambda c: c["args"].update(out="x"), "set by the launcher"),
        (lambda c: c["args"].update({"--seed": 1}), "leading"),
        (lambda c: c["hf"]["base"].update(revision="main"), "40-char"),
        (lambda c: c.update(files={"f": {"path": "/abs/x", "sha256": "0"}}), "relative"),
        (lambda c: c.update(name="bad name"), "bad run name"),
    ],
)
def test_validate_rejects(mutate, match):
    cfg = base_cfg()
    mutate(cfg)
    with pytest.raises(ConfigError, match=match):
        validate_config(cfg)


def test_judge_needs_no_seed():
    validate_config(base_cfg(command="judge_open_qa", args={"input": "x"}))


def test_build_argv_resolves_refs_flags_and_lists():
    cfg = base_cfg(
        command="train_lora",
        args={
            "base": "hf:base",
            "data": ["file:a:30", "file:b"],
            "bf16": True,
            "packing": False,
            "tokenizer": None,
            "lr": 1e-4,
            "seed": 3,
        },
    )
    hf = {"base": {"path": "/cache/snap"}}
    files = {"a": {"path": "msm/data/a.jsonl"}, "b": {"path": "msm/data/b.parquet"}}
    argv = build_argv(cfg, hf, files, "/repo/msm/runs/t")
    assert argv == [
        "--base", "/cache/snap",
        "--data", "msm/data/a.jsonl:30",
        "--data", "msm/data/b.parquet",
        "--bf16",
        "--lr", "0.0001",
        "--seed", "3",
        "--out", "/repo/msm/runs/t",
    ]


def test_build_argv_output_name_per_command():
    argv = build_argv(base_cfg(), {"base": {"path": "p"}}, {}, "/r/run")
    assert argv[-2:] == ["--out", "/r/run/preference.jsonl"]


def test_undefined_alias():
    cfg = base_cfg(args={"base": "hf:missing", "seed": 0})
    with pytest.raises(ConfigError, match="undefined hf alias"):
        build_argv(cfg, {}, {}, "/r")


def test_sha256_dir_is_deterministic_and_content_sensitive(tmp_path):
    d = tmp_path / "d"
    (d / "sub").mkdir(parents=True)
    (d / "a.txt").write_text("a")
    (d / "sub" / "b.txt").write_text("b")
    h1 = sha256_path(str(d))
    assert h1 == sha256_path(str(d))
    (d / "sub" / "b.txt").write_text("B")
    assert sha256_path(str(d)) != h1


def test_verify_files_mismatch(tmp_path):
    (tmp_path / "x.txt").write_text("hello")
    good = sha256_path(str(tmp_path / "x.txt"))
    assert verify_files({"files": {"x": {"path": "x.txt", "sha256": good}}}, str(tmp_path))["x"]["sha256"] == good
    with pytest.raises(ConfigError, match="config pins"):
        verify_files({"files": {"x": {"path": "x.txt", "sha256": "bad"}}}, str(tmp_path))


# --------------------------------------------------------------------------- #
# End to end, in a throwaway git repo, using the (model-free) eval_split command
# --------------------------------------------------------------------------- #


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    from msm_repro.tests.test_eval_split import affordability_rows, america_rows

    r = tmp_path / "repo"
    (r / "data").mkdir(parents=True)
    (r / "configs").mkdir()
    (r / "src").mkdir()
    pd.DataFrame(america_rows()).to_parquet(r / "data" / "am.parquet")
    pd.DataFrame(affordability_rows()).to_parquet(r / "data" / "af.parquet")
    cfg = {
        "name": "split-test",
        "command": "eval_split",
        "out_dir": "out/split-test",
        "files": {
            "am": {"path": "data/am.parquet", "sha256": sha256_path(str(r / "data" / "am.parquet"))},
            "af": {"path": "data/af.parquet", "sha256": sha256_path(str(r / "data" / "af.parquet"))},
        },
        "args": {"america-path": "file:am", "affordability-path": "file:af", "dev-fraction": 0.25, "seed": 0},
    }
    import yaml

    (r / "configs" / "split.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    _git(r, "init", "-q")
    _git(r, "add", "-A")
    _git(r, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    # The subprocess must import the real package.
    monkeypatch.setenv("PYTHONPATH", SRC_DIR)
    monkeypatch.chdir(r)
    return r


def test_launch_end_to_end(repo):
    rc = launch.main(["configs/split.yaml"], repo_root=str(repo))
    assert rc == 0
    out = repo / "out" / "split-test"
    record = json.loads((out / "launch.json").read_text())
    assert record["exit_code"] == 0
    assert record["git_commit"] and len(record["git_commit"]) == 40
    assert record["files"]["am"]["path"] == "data/am.parquet"
    for name in ("launch.json", "run.log"):  # paths made portable
        assert str(repo) not in (out / name).read_text()
    split = json.loads((out / "split.json").read_text())
    assert split["sets"]["america"]["n_dev"] == 10
    assert (out / "pip-freeze.txt").exists() and (out / "run.log").exists()

    # Never overwrite an existing run.
    assert launch.main(["configs/split.yaml"], repo_root=str(repo)) == 2


def test_launch_refuses_dirty_config(repo):
    with open(repo / "configs" / "split.yaml", "a") as fh:
        fh.write("# edit\n")
    assert launch.main(["configs/split.yaml"], repo_root=str(repo)) == 2
    assert not (repo / "out").exists()
    # Dry run reports the problem but does not fail.
    assert launch.main(["configs/split.yaml", "--dry-run"], repo_root=str(repo)) == 0


def test_launch_refuses_untracked_config(repo):
    (repo / "configs" / "new.yaml").write_text((repo / "configs" / "split.yaml").read_text())
    assert launch.main(["configs/new.yaml"], repo_root=str(repo)) == 2


def test_launch_refuses_hash_mismatch(repo):
    pd.DataFrame(__import__("msm_repro.tests.test_eval_split", fromlist=["x"]).america_rows(3)).to_parquet(
        repo / "data" / "am.parquet"
    )
    with pytest.raises(SystemExit, match="config pins"):
        launch.main(["configs/split.yaml"], repo_root=str(repo))


def test_portable_rewrites_repo_and_hf_cache(monkeypatch, tmp_path):
    from msm_repro.paths import portable

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    rec = {"a": [f"{tmp_path}/repo/msm/runs/x", f"{tmp_path}/hf/hub/m"], "b": 1}
    assert portable(rec, str(tmp_path / "repo")) == {"a": ["msm/runs/x", "$HF_HOME/hub/m"], "b": 1}
