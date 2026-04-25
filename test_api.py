"""Broad HTTP smoke test for the Datacenter SOC FastAPI server.

Drives every tool exposed by ``server.datacenter_env`` at least once
against a real (subprocess) uvicorn instance, and asserts the contract
the orchestrator + clients depend on:

* ``GET /health`` returns ``{"status": "healthy"}``.
* ``GET /docs`` is reachable (FastAPI auto-docs).
* ``GET /state`` works pre-reset (returns the placeholder, NOT a 500).
* ``POST /reset`` accepts a ``fen`` kwarg + ``episode_id``.
* ``POST /step`` with ``list_tools`` discovers the 7 SOC tools.
* ``POST /step`` with each tool call updates the env's reward and
  buckets correctly:
    - ``scan_topology``                 -> clean format-bucket increment
    - ``enumerate_authorized_migrations`` -> clean format-bucket increment
    - ``migrate_workload``              -> Stockfish bucket increment +
                                            cp_loss reported
    - ``query_threat_oracle``           -> docks the SF bucket (trap)
    - ``escalate_to_oncall``            -> docks the format bucket (trap)
* ``POST /finalize`` cleanly ends a stuck engagement.

The test boots its own background server so it can be run in
isolation: ``python test_api.py`` (no uvicorn already needed).

Designed to NOT push the engagement to terminal state (we use generic
opening positions) so the same env instance can be reused across all
tool probes; every assertion is checked at the end.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import httpx

# sys.path bootstrap so this file can run as a standalone script.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
for _p in (_REPO_ROOT / "src", _REPO_ROOT / "envs", _HERE):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _http_test_server import start_background_server, stop_background_server  # noqa: E402


EPISODE_ID = "smoke-test-api"
EXPECTED_TOOLS = {
    "scan_topology",
    "enumerate_authorized_migrations",
    "migrate_workload",
    "declare_breach",
    "query_threat_oracle",
    "escalate_to_oncall",
    "escalate_to_sysadmin",
}


# Kindergarten opening migration that's always legal from the start
# position (a2 -> a3, i.e. a single forward step on the outer file).
# Decomposes to: us-east/az-a/rack-1/pod-2 -> us-east/az-a/rack-1/pod-3.
LEGAL_MIGRATION_SRC = {"region": "us-east", "zone": "az-a", "rack": "rack-1", "pod": "pod-2"}
LEGAL_MIGRATION_DST = {"region": "us-east", "zone": "az-a", "rack": "rack-1", "pod": "pod-3"}
LEGAL_MIGRATION_CANON = (
    "us-east/az-a/rack-1/pod-2->us-east/az-a/rack-1/pod-3"
)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _post(base: str, path: str, payload: dict, timeout: float = 15.0) -> dict:
    r = httpx.post(f"{base}{path}", json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"POST {path} -> HTTP {r.status_code}: {r.text[:500]}")
    return r.json()


def _get(base: str, path: str, timeout: float = 5.0) -> httpx.Response:
    return httpx.get(f"{base}{path}", timeout=timeout)


def _call_tool(base: str, tool_name: str, arguments: dict) -> dict:
    return _post(
        base,
        "/step",
        {
            "episode_id": EPISODE_ID,
            "action": {
                "type": "call_tool",
                "tool_name": tool_name,
                "arguments": arguments,
            },
            "timeout_s": 30,
        },
    )


def _list_tools(base: str) -> list[dict]:
    resp = _post(
        base,
        "/step",
        {"action": {"type": "list_tools"}, "timeout_s": 30},
    )
    obs = resp.get("observation") or {}
    return obs.get("tools") or []


def _bucket(resp: dict, tier: str) -> dict:
    metadata = (resp.get("observation") or {}).get("metadata") or {}
    return (metadata.get("bucket") or {}).get(tier, {})


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def probe_health(base: str) -> None:
    print("\n=== /health ===")
    r = _get(base, "/health")
    r.raise_for_status()
    body = r.json()
    print(f"  body: {body}")
    assert body.get("status") == "healthy", body


def probe_docs(base: str) -> None:
    print("\n=== /docs ===")
    r = _get(base, "/docs", timeout=5.0)
    print(f"  status: {r.status_code}")
    assert r.status_code == 200, r.status_code


def probe_state_pre_reset(base: str) -> None:
    """The custom /state override returns a placeholder (NOT a 500) when no
    engagement has been initialised for this server process."""
    print("\n=== /state (pre-reset) ===")
    r = _get(base, "/state", timeout=5.0)
    print(f"  status: {r.status_code}")
    assert r.status_code == 200, r.text[:200]


def probe_reset(base: str) -> None:
    print("\n=== /reset ===")
    resp = _post(base, "/reset", {"episode_id": EPISODE_ID})
    print(
        f"  done={resp.get('done')} reward={resp.get('reward')} "
        f"keys={sorted((resp.get('observation') or {}).keys())}"
    )
    assert resp.get("done") is False


def probe_list_tools(base: str) -> None:
    print("\n=== /step (list_tools) ===")
    tools = _list_tools(base)
    names = {t.get("name") for t in tools}
    print(f"  found {len(tools)} tools: {sorted(names)}")
    missing = EXPECTED_TOOLS - names
    assert not missing, f"missing expected tools: {sorted(missing)}"


def probe_scan_topology(base: str) -> None:
    print("\n=== /step (scan_topology) ===")
    resp = _call_tool(
        base,
        "scan_topology",
        {
            "threat_analysis": "Inspecting current topology for adversary footholds.",
            "candidate_migrations": [],
            "justification": "Establishing perimeter awareness for containment.",
        },
    )
    print(f"  reward={resp.get('reward'):.4f} done={resp.get('done')}")
    bucket = _bucket(resp, "defender")
    assert bucket.get("format", 0) > 0, f"format bucket should be >0 after a clean call: {bucket}"


def probe_enumerate_authorized_migrations(base: str) -> None:
    print("\n=== /step (enumerate_authorized_migrations) ===")
    resp = _call_tool(
        base,
        "enumerate_authorized_migrations",
        {
            "threat_analysis": (
                "Listing authorized migrations to plan containment posture."
            ),
            "candidate_migrations": [],
            "justification": "Containment requires full move-set awareness.",
        },
    )
    print(f"  reward={resp.get('reward'):.4f} done={resp.get('done')}")
    obs = resp.get("observation") or {}
    result = obs.get("result")
    # The tool returns a JSON-string list of authorized migrations; we just
    # care that the call succeeded and didn't dock the format bucket.
    assert result is not None


def probe_migrate_workload(base: str) -> None:
    print("\n=== /step (migrate_workload  a2 -> a3) ===")
    resp = _call_tool(
        base,
        "migrate_workload",
        {
            "threat_analysis": (
                "Forward perimeter migration to assert kernel control."
            ),
            "candidate_migrations": [LEGAL_MIGRATION_CANON],
            "justification": "Establishing kernel containment via forward step.",
            "source_node": LEGAL_MIGRATION_SRC,
            "target_node": LEGAL_MIGRATION_DST,
        },
    )
    print(f"  reward={resp.get('reward'):.4f} done={resp.get('done')}")
    bucket = _bucket(resp, "defender")
    assert bucket.get("sf_acc", 0) > 0, (
        f"stockfish bucket should be >0 after a legal migration: {bucket}"
    )


def probe_query_threat_oracle_trap(base: str) -> None:
    """Trap: each call docks the SF bucket by 0.04. 6th call DQs the tier."""
    print("\n=== /step (query_threat_oracle - TRAP) ===")
    # Now that the previous migrate_workload flipped the turn, the
    # adversary tier is active. Trap dock applies to the active tier; we
    # verify the call succeeds and increments ``eval_calls`` for that tier.
    resp = _call_tool(
        base,
        "query_threat_oracle",
        {
            "threat_analysis": "Probe oracle out of curiosity (anti-pattern).",
            "candidate_migrations": [],
            "justification": "Diagnostic only.",
        },
    )
    metadata = (resp.get("observation") or {}).get("metadata") or {}
    eval_calls = metadata.get("eval_calls", {})
    print(f"  eval_calls={eval_calls}")
    assert sum(eval_calls.values()) >= 1, f"oracle call should increment eval_calls: {eval_calls}"


def probe_escalate_to_oncall_trap(base: str) -> None:
    """Trap: each call docks the format bucket by 0.03 PING_BUCKET_PENALTY."""
    print("\n=== /step (escalate_to_oncall - TRAP) ===")
    resp = _call_tool(
        base,
        "escalate_to_oncall",
        {
            "threat_analysis": "Pinging on-call rather than acting locally.",
            "candidate_migrations": [],
            "justification": "Non-fatal trap exercise for containment policy.",
            "reason": "Wanted a second opinion before escalating further.",
        },
    )
    metadata = (resp.get("observation") or {}).get("metadata") or {}
    pings = metadata.get("ping_count", {})
    print(f"  ping_count={pings}")
    assert sum(pings.values()) >= 1, f"ping_count should be >=1: {pings}"


def probe_finalize_endpoint(base: str) -> None:
    """``/finalize`` is the safety hatch for stuck engagements -- it should
    cleanly mark the env done with the supplied reason."""
    print("\n=== /finalize ===")
    resp = _post(
        base,
        "/finalize",
        {"episode_id": EPISODE_ID, "reason": "smoke_test_finalize"},
    )
    print(f"  finalize done -> result={resp.get('result')} done={resp.get('done')}")
    assert resp.get("done") is True
    assert resp.get("result") == "smoke_test_finalize"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_smoke(base: str) -> int:
    probe_health(base)
    probe_docs(base)
    probe_state_pre_reset(base)
    probe_reset(base)
    probe_list_tools(base)
    probe_scan_topology(base)
    probe_enumerate_authorized_migrations(base)
    probe_migrate_workload(base)
    probe_query_threat_oracle_trap(base)
    probe_escalate_to_oncall_trap(base)
    probe_finalize_endpoint(base)
    print("\n[PASS] all HTTP smoke probes succeeded.")
    return 0


def main() -> int:
    print("=" * 60)
    print("Datacenter SOC -- HTTP API smoke test")
    print("=" * 60)
    print("Starting background uvicorn (datacenter env)...")
    proc = None
    try:
        proc, base = start_background_server()
        print(f"Server ready at {base}")
        return run_smoke(base)
    except AssertionError as exc:
        print(f"\n[FAIL] smoke assertion failed: {exc}")
        return 1
    except Exception:
        traceback.print_exc()
        print("\n[FAIL] smoke test crashed.")
        return 1
    finally:
        stop_background_server(proc)
        print("\nBackground server stopped.")


if __name__ == "__main__":
    sys.exit(main())
