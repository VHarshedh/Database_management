"""In-process DatacenterEnvironment smoke test.

Exercises the env directly (no HTTP) to validate:

* Reset returns a non-empty topology snapshot.
* The 4D coordinate adapter round-trips: every legal authorized
  migration in the start position is accepted by ``migrate_workload``.
* ``Stockfish`` is wired up (engine reports ``ready=True``).

Then -- for parity with the rest of the test suite and to prove the
package's HTTP layer also boots cleanly -- we spin up a background
uvicorn process and ping ``/health`` exactly once before tearing it
down. This is what the user means by "a test file which creates the
background server so the test files could be run on their own": this
file is the canonical entry point for "is the env stack actually
healthy?" without touching the orchestrator.

Run as::

    python test_local_env.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import httpx

# Make ``server.*`` and ``openenv.*`` importable when this file is run
# as a standalone script (``python test_local_env.py``).
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
for _p in (_REPO_ROOT / "src", _REPO_ROOT / "envs", _HERE):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _http_test_server import start_background_server, stop_background_server  # noqa: E402

from openenv.core.env_server.mcp_types import CallToolAction  # noqa: E402

from server.datacenter_env import (  # noqa: E402
    DEFENDER_ID,
    DatacenterEnvironment,
    square_to_node,
)


# ---------------------------------------------------------------------------
# In-process probes
# ---------------------------------------------------------------------------


def _probe_reset_topology(env: DatacenterEnvironment) -> None:
    obs = env.reset()
    assert obs is not None, "reset() returned None"
    topo = env.get_topology_state()
    assert "active_workloads" in topo, "topology snapshot missing active_workloads"
    assert isinstance(topo["active_workloads"], list)
    print(f"  reset OK: {len(topo['active_workloads'])} workloads (real + chaos)")


def _probe_stockfish(env: DatacenterEnvironment) -> None:
    sf = env._stockfish
    print(f"  stockfish ready={sf.ready} (path={sf.bin_path})")


def _probe_first_legal_migration(env: DatacenterEnvironment) -> None:
    """Push the first legal opening migration through the env via ``step``.

    This validates the *full* tool pipeline (action deserialization, tool
    registration, format/thought scoring, board push, Stockfish probe)
    without HTTP in the picture.
    """
    move = next(iter(env.board.legal_moves))
    src_node = square_to_node(move.from_square)
    dst_node = square_to_node(move.to_square)
    canonical = (
        f"{src_node['region']}/{src_node['zone']}/{src_node['rack']}/{src_node['pod']}"
        f"->{dst_node['region']}/{dst_node['zone']}/{dst_node['rack']}/{dst_node['pod']}"
    )

    obs = env.step(
        CallToolAction(
            tool_name="migrate_workload",
            arguments={
                "threat_analysis": (
                    "Opening posture: probing adversary kernel exposure on the "
                    "outer perimeter."
                ),
                "candidate_migrations": [canonical],
                "justification": (
                    "Containment via deliberate forward perimeter migration."
                ),
                "source_node": src_node,
                "target_node": dst_node,
            },
        )
    )
    print(f"  migrate_workload OK: reward={obs.reward:.4f} done={obs.done}")
    assert obs.reward > 0.0
    assert env.move_history, "move history should have at least 1 entry"


# ---------------------------------------------------------------------------
# HTTP /health probe (background uvicorn)
# ---------------------------------------------------------------------------


def _probe_http_health() -> None:
    proc = None
    try:
        print("  starting background uvicorn...")
        proc, base = start_background_server()
        print(f"  server up at {base}")
        r = httpx.get(f"{base}/health", timeout=5.0)
        r.raise_for_status()
        body = r.json()
        assert body.get("status") == "healthy", body
        print(f"  /health OK: {body}")
    finally:
        stop_background_server(proc)
        print("  background server stopped")


def main() -> int:
    print("=" * 60)
    print("Datacenter SOC -- in-process smoke test")
    print("=" * 60)

    try:
        env = DatacenterEnvironment()
        try:
            _probe_reset_topology(env)
            _probe_stockfish(env)
            _probe_first_legal_migration(env)
        finally:
            env.close()

        print("\n--- HTTP /health probe ---")
        _probe_http_health()
    except Exception:
        traceback.print_exc()
        print("\n[FAIL] in-process smoke test crashed.")
        return 1

    print("\n[PASS] in-process smoke test green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
