"""End-to-end dry-run termination tests (SOC-native).

Regression test for "dry-run is not stopping". In SOC-native mode there is
no external engine subprocess, but we still validate:

- ``GlobalSOCOrchestrator.run`` returns within a wall-clock budget.
- ``close()`` is idempotent and exception-safe.
"""

from __future__ import annotations

import threading
import time

import pytest

import global_soc_orchestrator as gso  # type: ignore[import-not-found]

_TERMINATION_BUDGET_SEC = 30.0


def _build_dry_run_orchestrator(*, regions: int = 1):
    defender_stub, adversary_stubs = gso._build_stub_agents()
    region_names = gso.DEFAULT_REGION_NAMES[:regions]
    return gso.build_orchestrator(
        defender=defender_stub,
        adversaries=adversary_stubs,
        region_names=region_names,
        hitl_enabled=False,
    )


def test_dry_run_completes_within_budget() -> None:
    orch = _build_dry_run_orchestrator(regions=1)
    done_event = threading.Event()
    error: list[BaseException] = []

    def _worker() -> None:
        try:
            orch.run(max_cycles=2)
        except BaseException as exc:  # pragma: no cover
            error.append(exc)
        finally:
            orch.close()
            done_event.set()

    t = threading.Thread(target=_worker, daemon=True)
    start = time.time()
    t.start()
    finished = done_event.wait(timeout=_TERMINATION_BUDGET_SEC)
    elapsed = time.time() - start
    if not finished:
        pytest.fail(f"dry-run did not terminate within {_TERMINATION_BUDGET_SEC:.0f}s (elapsed={elapsed:.1f}s)")
    if error:
        raise error[0]
    assert len(orch.audit_trail) >= 1


def test_close_is_idempotent_and_exception_safe() -> None:
    orch = _build_dry_run_orchestrator(regions=1)
    orch.close()
    orch.close()

