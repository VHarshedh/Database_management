"""End-to-end dry-run termination tests.

These are regression tests for "the dry-run is not stopping" --
``DatacenterEnvironment`` opens a Stockfish UCI subprocess for damage
scoring, and ``chess.engine.SimpleEngine`` keeps a non-daemon reader
thread alive until ``engine.quit()`` is called. Without an explicit
:meth:`GlobalSOCOrchestrator.close`, ``main()`` would log
``Audit trail written to: ...`` and then the Python process would block
forever on interpreter shutdown waiting for the reader thread.

Each test below boots a real (in-process) orchestrator with stub agents,
runs a tiny number of cycles, calls ``orchestrator.close()``, and
verifies the run finishes within a generous wall-clock budget.

We do NOT invoke ``main()`` because that calls ``argparse`` against
``sys.argv`` and writes JSON files into ``results/``; the same
correctness can be exercised via ``build_orchestrator`` + ``run`` + the
new ``close()`` hook.
"""

from __future__ import annotations

import threading
import time

import pytest

import global_soc_orchestrator as gso  # type: ignore[import-not-found]


# A very generous ceiling. The dry-run with 1 region / 1 cycle finishes in
# well under a second locally; the buffer is for slow CI VMs and the small
# Stockfish UCI handshake.
_TERMINATION_BUDGET_SEC = 30.0


def _build_dry_run_orchestrator(*, regions: int = 1):
    """Mirror what ``main()`` does in ``--dry-run`` mode but in-process."""
    defender_stub, adversary_stubs = gso._build_stub_agents()
    region_names = gso.DEFAULT_REGION_NAMES[:regions]
    return gso.build_orchestrator(
        defender=defender_stub,
        adversaries=adversary_stubs,
        region_names=region_names,
        hitl_enabled=False,
    )


def test_orchestrator_close_releases_stockfish_engine() -> None:
    """``close()`` should drop every region's Stockfish engine handle so the
    UCI subprocess + reader thread can be reaped."""
    orch = _build_dry_run_orchestrator(regions=1)
    region = orch.regions[0]
    sf = region.env._stockfish
    # If Stockfish is genuinely missing on this machine we can still verify
    # the close() path is idempotent and exception-free.
    orch.close()
    if sf is not None:
        assert sf._engine is None, (
            "GlobalSOCOrchestrator.close() must call _stockfish.close() so "
            "the chess.engine.SimpleEngine subprocess is reaped"
        )
    # Idempotent: a second close() must not raise even though the engine is
    # already None.
    orch.close()


def test_dry_run_completes_within_budget() -> None:
    """The end-to-end regression: build, run, close() and assert we exit in
    bounded time. If the Stockfish-hang ever regresses, this test will time
    out instead of hanging the suite forever."""
    orch = _build_dry_run_orchestrator(regions=1)
    done_event = threading.Event()
    error: list[BaseException] = []

    def _worker() -> None:
        try:
            orch.run(max_cycles=1)
        except BaseException as exc:  # pragma: no cover - failure diagnostic
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
        # If we time out, bail with a useful message but DO NOT join the
        # thread (it is daemonised so the test process can still exit).
        pytest.fail(
            f"dry-run did not terminate within {_TERMINATION_BUDGET_SEC:.0f}s "
            f"(elapsed={elapsed:.1f}s); orchestrator.close() probably did "
            f"not reap the Stockfish subprocess"
        )
    if error:
        raise error[0]
    # Sanity: the audit trail captured at least one half-move record.
    assert len(orch.audit_trail) >= 1


def test_dry_run_close_is_idempotent_and_exception_safe() -> None:
    """Calling ``close()`` multiple times, even before any run, is fine."""
    orch = _build_dry_run_orchestrator(regions=1)
    orch.close()
    orch.close()  # second call: must not raise


def test_run_loop_breaks_on_terminal_state() -> None:
    """The dry-run terminating with ``dq_illegal_defender`` (or any other
    terminal result) must propagate to ``RegionRunner.last_result`` rather
    than being silently swallowed -- that was the user's "not stopping"
    confusion: the engagement *is* over; the process just hangs on
    Stockfish at exit."""
    orch = _build_dry_run_orchestrator(regions=1)
    try:
        orch.run(max_cycles=5)
        region = orch.regions[0]
        # Either the env hit a terminal state (done=True) or we exhausted
        # the cycles. Both are valid outcomes; we only assert that when
        # the env *is* done, the result is recorded so the audit trail can
        # explain *why* the run stopped.
        if region.env.done:
            assert region.env.result is not None
            assert region.last_result is not None
    finally:
        orch.close()
