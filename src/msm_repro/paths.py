"""Repo-root location and path sanitizing for records that get committed or shared."""

from __future__ import annotations

import os
from typing import Any, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def hf_home() -> str:
    return os.path.abspath(os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))


def portable(obj: Any, repo_root: Optional[str] = None) -> Any:
    """Rewrite absolute repo and HF-cache paths (recursively) so records hold no machine-specific paths.

    ``<repo>/x`` becomes ``x`` and ``<HF_HOME>/x`` becomes ``$HF_HOME/x``.
    """
    root = os.path.abspath(repo_root or REPO_ROOT)
    subs = [(root + os.sep, ""), (hf_home(), "$HF_HOME")]
    if isinstance(obj, str):
        for old, new in subs:
            obj = obj.replace(old, new)
        return obj
    if isinstance(obj, list):
        return [portable(x, root) for x in obj]
    if isinstance(obj, dict):
        return {k: portable(v, root) for k, v in obj.items()}
    return obj
