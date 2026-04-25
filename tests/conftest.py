"""Shared pytest fixtures + sys.path bootstrap for the SOC orchestrator tests.

Running ``pytest`` from anywhere needs ``server`` (datacenter env),
``agent_inference`` and ``global_soc_orchestrator`` to import without the
caller having to set ``PYTHONPATH=src;envs;Datacenter-workload_migration``
manually. This file does that injection once per test session.
"""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PKG_DIR = _REPO_ROOT / "Datacenter-workload_migration"
_SRC_DIR = _REPO_ROOT / "src"
_ENVS_DIR = _REPO_ROOT / "envs"

for p in (_SRC_DIR, _ENVS_DIR, _PKG_DIR):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
