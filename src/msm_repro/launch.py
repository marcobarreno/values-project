"""Launch an ``msm_repro`` run from a checked-in YAML config.

This is the only supported way to start a run: the config is the complete
specification, and the launcher records everything needed to reproduce it.

    python -m msm_repro.launch configs/phase0/smoke-train.yaml
    python -m msm_repro.launch configs/phase0/smoke-train.yaml --dry-run   # resolve + check only

Config format::

    name: phase0-smoke-train          # run name; output goes to msm/runs/<name>/
    command: train_lora               # an msm_repro CLI (see COMMANDS)
    description: ...                  # optional, free text
    out_dir: msm/splits/<name>        # optional; overrides msm/runs/<name> (e.g. for tracked artifacts)
    hf:                               # every Hugging Face repo used, pinned to a commit
      base: {repo: HuggingFaceTB/SmolLM2-135M, revision: <40-char commit sha>}
      # optional per entry: repo_type (model|dataset), allow_patterns, ignore_patterns
    files:                            # every local input (file or directory), pinned by sha256
      america: {path: msm/data/hf/.../train-00000-of-00001.parquet, sha256: <hex>}
    args:                             # the command's CLI flags, without the leading "--"
      base: hf:base                   # "hf:<alias>"   -> local snapshot of the pinned repo
      data: [file:america:30]         # "file:<alias>" -> the pinned path (any suffix is kept)
      seed: 0                         # required for every command that takes --seed
      bf16: true                      # true -> bare flag; false/null -> omitted
                                      # list -> flag repeated once per element

The output flag (``--out``) is set by the launcher, never by the config.

Before running, the launcher refuses to start if the config file is untracked
or modified, if anything under ``src/`` differs from HEAD, if a pinned file's
hash does not match, if an HF revision is not a full commit sha, or if the
output directory already exists. It then writes ``launch.json`` (config and
its sha256, git commit, resolved argv, pins, package versions, GPU, timings,
exit code), ``pip-freeze.txt`` and ``run.log`` into the output directory.
Absolute paths in the record are rewritten relative to the repo root and
``$HF_HOME`` so the record is portable and shareable.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from .paths import REPO_ROOT, portable
except ImportError:  # executed as `python src/msm_repro/launch.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from msm_repro.paths import REPO_ROOT, portable

RUNS_DIR = os.path.join("msm", "runs")

# command -> (output flag value, relative to the run dir; whether --seed is required)
COMMANDS: Dict[str, Tuple[str, bool]] = {
    "train_lora": (".", True),
    "eval_preference": ("preference.jsonl", True),
    "rescore": ("preference.jsonl", False),
    "audit_sample": ("sheet.md", True),
    "audit_score": ("results.json", False),
    "generate_responses": ("responses.jsonl", True),
    "judge_open_qa": ("judgments.jsonl", False),
    "build_it_mix": ("it_mix.jsonl", True),
    "eval_split": ("split.json", True),
}

CONFIG_KEYS = {"name", "command", "description", "out_dir", "hf", "files", "args"}
HF_KEYS = {"repo", "revision", "repo_type", "allow_patterns", "ignore_patterns"}
DEFAULT_HF_IGNORE = ["original/*", "*.pth"]  # e.g. Llama's duplicate consolidated checkpoint
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
TRACKED_PACKAGES = ["torch", "transformers", "peft", "trl", "datasets", "accelerate", "huggingface_hub", "pandas", "kernels"]


class ConfigError(Exception):
    pass


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #


def sha256_path(path: str) -> str:
    """sha256 of a file, or of a directory tree (relative paths + contents, sorted)."""
    h = hashlib.sha256()
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    if not os.path.isdir(path):
        raise ConfigError(f"pinned path does not exist: {path}")
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            full = os.path.join(root, name)
            h.update(os.path.relpath(full, path).replace(os.sep, "/").encode("utf-8") + b"\0")
            h.update(sha256_path(full).encode("ascii") + b"\0")
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Config loading and validation
# --------------------------------------------------------------------------- #


def load_config(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Any) -> None:
    if not isinstance(cfg, dict):
        raise ConfigError("config must be a mapping")
    unknown = set(cfg) - CONFIG_KEYS
    if unknown:
        raise ConfigError(f"unknown config keys: {sorted(unknown)}")
    for key in ("name", "command", "args"):
        if key not in cfg:
            raise ConfigError(f"config is missing {key!r}")
    if not NAME_RE.match(str(cfg["name"])):
        raise ConfigError(f"bad run name {cfg['name']!r} (letters, digits, '.', '_', '-')")
    if cfg["command"] not in COMMANDS:
        raise ConfigError(f"unknown command {cfg['command']!r}; expected one of {sorted(COMMANDS)}")
    args = cfg["args"]
    if not isinstance(args, dict):
        raise ConfigError("args must be a mapping")
    for key in args:
        if key.startswith("-"):
            raise ConfigError(f"arg {key!r}: write flags without the leading '--'")
        if key == "out":
            raise ConfigError("'out' is set by the launcher; remove it from args")
    if COMMANDS[cfg["command"]][1] and "seed" not in args:
        raise ConfigError(f"{cfg['command']} takes --seed; the config must set args.seed explicitly")
    for alias, entry in (cfg.get("hf") or {}).items():
        if not isinstance(entry, dict) or not {"repo", "revision"} <= set(entry):
            raise ConfigError(f"hf.{alias}: needs 'repo' and 'revision'")
        if set(entry) - HF_KEYS:
            raise ConfigError(f"hf.{alias}: unknown keys {sorted(set(entry) - HF_KEYS)}")
        if not SHA1_RE.match(str(entry["revision"])):
            raise ConfigError(f"hf.{alias}: revision must be a full 40-char commit sha, got {entry['revision']!r}")
    for alias, entry in (cfg.get("files") or {}).items():
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ConfigError(f"files.{alias}: needs exactly 'path' and 'sha256'")
        if os.path.isabs(entry["path"]):
            raise ConfigError(f"files.{alias}: path must be relative to the repo root")


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def resolve_hf(cfg: Dict[str, Any], download: bool) -> Dict[str, Dict[str, Any]]:
    """alias -> {repo, revision, path}; downloads the pinned snapshots if ``download``."""
    out: Dict[str, Dict[str, Any]] = {}
    for alias, entry in (cfg.get("hf") or {}).items():
        rec = {"repo": entry["repo"], "revision": entry["revision"], "repo_type": entry.get("repo_type", "model")}
        if download:
            from huggingface_hub import snapshot_download

            rec["path"] = snapshot_download(
                repo_id=entry["repo"],
                revision=entry["revision"],
                repo_type=rec["repo_type"],
                allow_patterns=entry.get("allow_patterns"),
                ignore_patterns=entry.get("ignore_patterns", DEFAULT_HF_IGNORE),
            )
        else:
            rec["path"] = f"<snapshot {entry['repo']}@{entry['revision'][:12]}>"
        out[alias] = rec
    return out


def verify_files(cfg: Dict[str, Any], repo_root: str) -> Dict[str, Dict[str, Any]]:
    """alias -> {path (repo-relative), sha256}; raises on any mismatch."""
    out: Dict[str, Dict[str, Any]] = {}
    for alias, entry in (cfg.get("files") or {}).items():
        actual = sha256_path(os.path.join(repo_root, entry["path"]))
        if actual != entry["sha256"]:
            raise ConfigError(f"files.{alias}: {entry['path']} has sha256 {actual}, config pins {entry['sha256']}")
        out[alias] = {"path": entry["path"], "sha256": actual}
    return out


_REF_RE = re.compile(r"^(hf|file):([A-Za-z0-9_.-]+)(.*)$", re.S)


def resolve_value(value: Any, hf: Dict[str, Dict[str, Any]], files: Dict[str, Dict[str, Any]]) -> str:
    text = str(value)
    m = _REF_RE.match(text)
    if not m:
        return text
    kind, alias, rest = m.groups()
    table = hf if kind == "hf" else files
    if alias not in table:
        raise ConfigError(f"{text!r} refers to undefined {kind} alias {alias!r}")
    return table[alias]["path"] + rest


def build_argv(
    cfg: Dict[str, Any], hf: Dict[str, Dict[str, Any]], files: Dict[str, Dict[str, Any]], run_dir: str
) -> List[str]:
    argv: List[str] = []
    for key, value in cfg["args"].items():
        flag = f"--{key}"
        if value is None or value is False:
            continue
        if value is True:
            argv.append(flag)
        elif isinstance(value, list):
            for v in value:
                argv += [flag, resolve_value(v, hf, files)]
        elif isinstance(value, dict):
            raise ConfigError(f"arg {key!r}: mappings are not supported")
        else:
            argv += [flag, resolve_value(value, hf, files)]
    out_rel = COMMANDS[cfg["command"]][0]
    argv += ["--out", os.path.normpath(os.path.join(run_dir, out_rel))]
    return argv


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def _git(repo_root: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", repo_root, *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def check_git(repo_root: str, config_rel: str) -> Dict[str, Any]:
    """Return {commit, problems}; problems are reasons the tree is not launchable."""
    problems: List[str] = []
    try:
        commit = _git(repo_root, "rev-parse", "HEAD")
    except subprocess.CalledProcessError:
        return {"commit": None, "problems": ["not a git repository, or no commits yet"]}
    if subprocess.run(
        ["git", "-C", repo_root, "ls-files", "--error-unmatch", config_rel], capture_output=True
    ).returncode != 0:
        problems.append(f"config {config_rel} is not tracked by git (commit it first)")
    dirty = _git(repo_root, "status", "--porcelain", "--", config_rel, "src")
    if dirty:
        problems.append("uncommitted changes in the config or src/:\n  " + dirty.replace("\n", "\n  "))
    return {"commit": commit, "problems": problems}


def package_versions() -> Dict[str, Optional[str]]:
    from importlib import metadata

    out: Dict[str, Optional[str]] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def gpu_info() -> Optional[List[str]]:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return [line.strip() for line in res.stdout.splitlines() if line.strip()] or None


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="path to a checked-in YAML run config")
    p.add_argument("--dry-run", action="store_true", help="validate, check pins and git state, print the command; do not run")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None, repo_root: str = REPO_ROOT) -> int:
    args = parse_args(argv)
    config_path = os.path.abspath(args.config)
    config_rel = os.path.relpath(config_path, repo_root)
    if config_rel.startswith(".."):
        raise SystemExit(f"config must live inside the repo: {args.config}")
    try:
        cfg = load_config(config_path)
        git = check_git(repo_root, config_rel)
        files = verify_files(cfg, repo_root)
        run_rel = os.path.normpath(cfg.get("out_dir") or os.path.join(RUNS_DIR, cfg["name"]))
        run_dir = os.path.join(repo_root, run_rel)
        problems = list(git["problems"])
        if os.path.exists(run_dir):
            problems.append(f"output directory already exists: {run_rel} (runs are never overwritten)")
        hf = resolve_hf(cfg, download=not args.dry_run and not problems)
        cmd = [sys.executable, "-m", f"msm_repro.{cfg['command']}", *build_argv(cfg, hf, files, run_dir)]
    except ConfigError as exc:
        raise SystemExit(f"config error: {exc}")

    print("command:", " ".join(portable(cmd, repo_root)))
    if problems:
        for p in problems:
            print(f"cannot launch: {p}", file=sys.stderr)
        return 0 if args.dry_run else 2
    if args.dry_run:
        print("dry run: all checks passed")
        return 0

    os.makedirs(run_dir)
    with open(config_path, "rb") as fh:
        config_bytes = fh.read()
    record: Dict[str, Any] = {
        "name": cfg["name"],
        "command": cfg["command"],
        "config_path": config_rel,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config": cfg,
        "git_commit": git["commit"],
        "argv": cmd[1:],
        "hf": hf,
        "files": files,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": package_versions(),
        "gpus": gpu_info(),
        "started": _now(),
        "finished": None,
        "exit_code": None,
    }

    def write_record() -> None:
        with open(os.path.join(run_dir, "launch.json"), "w", encoding="utf-8") as fh:
            json.dump(portable(record, repo_root), fh, indent=2)
            fh.write("\n")

    write_record()
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    with open(os.path.join(run_dir, "pip-freeze.txt"), "w", encoding="utf-8") as fh:
        fh.write(portable(freeze.stdout, repo_root))

    env = dict(os.environ)
    src = os.path.join(repo_root, "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    with open(os.path.join(run_dir, "run.log"), "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=repo_root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(portable(line, repo_root))
        proc.wait()

    record["finished"] = _now()
    record["exit_code"] = proc.returncode
    write_record()
    print(f"{'finished' if proc.returncode == 0 else 'FAILED'} (exit {proc.returncode}): {run_rel}")
    return proc.returncode


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
