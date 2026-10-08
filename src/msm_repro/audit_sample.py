"""Launcher entry point for ``audit.main_sample`` (see ``audit.py``)."""

from __future__ import annotations

try:
    from .audit import main_sample as main
except ImportError:  # executed as `python src/msm_repro/audit_sample.py`
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from msm_repro.audit import main_sample as main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
