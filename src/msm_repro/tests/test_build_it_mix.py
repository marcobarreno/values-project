"""Tests for the §3 instruction-tuning mix builder (tiny fake parquet splits, no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from msm_repro.build_it_mix import section3_rows


def _conv(i: int, tag: str):
    return [{"role": "user", "content": f"{tag} q{i}"}, {"role": "assistant", "content": f"{tag} a{i}"}]


def _write(d, name, convs, source):
    pd.DataFrame({"messages": convs, "source": [source] * len(convs)}).to_parquet(d / f"{name}-00000-of-00001.parquet")


@pytest.fixture
def data_dir(tmp_path):
    nr = [_conv(i, "nr") for i in range(6)]
    _write(tmp_path, "no_robots", nr, "no_robots")
    _write(tmp_path, "mmlu_binary", [_conv(i, "mb") for i in range(2)], "mmlu_binary")
    _write(tmp_path, "mmlu_explain", [_conv(i, "me") for i in range(2)], "mmlu_explain")
    # The authors' filtered split: 4 of the 6 no_robots rows (in a different order) plus another source.
    clean = pd.DataFrame({"messages": [nr[4], nr[0], nr[2], nr[5], _conv(0, "other")],
                          "source": ["no_robots"] * 4 + ["tulu3_if"]})
    clean.to_parquet(tmp_path / "train_80k_clean-00000-of-00001.parquet")
    return tmp_path


def _nr_contents(rows):
    return sorted(r["messages"][0]["content"] for r in rows if r["source"] == "no_robots")


def test_full_takes_every_row(data_dir):
    rows = section3_rows(data_dir, "full", seed=0)
    assert len(rows) == 10 and len(_nr_contents(rows)) == 6


def test_clean_keeps_only_filtered_no_robots_rows(data_dir):
    rows = section3_rows(data_dir, "clean", seed=0)
    assert _nr_contents(rows) == ["nr q0", "nr q2", "nr q4", "nr q5"]
    assert sum(r["source"].startswith("mmlu") for r in rows) == 4
    assert not any(r["source"] == "tulu3_if" for r in rows)  # other sources in the clean split are ignored


def test_seeded_shuffle(data_dir):
    assert section3_rows(data_dir, "clean", seed=0) == section3_rows(data_dir, "clean", seed=0)
    assert section3_rows(data_dir, "clean", seed=0) != section3_rows(data_dir, "clean", seed=1)


def test_clean_row_missing_from_no_robots_raises(data_dir):
    clean = pd.read_parquet(data_dir / "train_80k_clean-00000-of-00001.parquet")
    clean.loc[len(clean)] = [_conv(99, "nr"), "no_robots"]
    clean.to_parquet(data_dir / "train_80k_clean-00000-of-00001.parquet")
    with pytest.raises(ValueError, match="not in the no_robots split"):
        section3_rows(data_dir, "clean", seed=0)
