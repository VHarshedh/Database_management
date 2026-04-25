"""Perfect-path integration test: drive the DEFENDER tier to a 0.99 reward.

Reward decomposition (all four buckets capped, sum clamped to ``R_MAX = 0.99``):

    Outcome bucket        : 0.50  (checkmate_defender == OUTCOME_WIN)
    Format bucket         : 0.10  (1/1 clean tool calls, no pings)
    Thought-quality bucket: 0.15  (>5-word threat analysis, candidate
                                   includes target migration, justification
                                   includes a SOC concept)
    Stockfish bucket      : 0.24  (cp_loss == 0, no oracle calls)

To make this deterministic, we seed a custom mate-in-1 FEN at reset:

    6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1

White (the DEFENDER) plays ``Ra1-a8#`` (back-rank mate). In datacenter
terms that's the migration::

    us-east/az-a/rack-1/pod-1  ->  us-east/az-b/rack-1/pod-4

The test boots a background uvicorn server (so it can be run in
isolation: ``python test_ws.py``), POSTs ``/reset`` with the seed FEN
and a self-chosen ``episode_id``, POSTs ``/step`` with the killer
migration, and asserts the returned ``reward == 0.99`` (and
``done == True``).

Why this reward shape qualifies as "the correct path":

* **Outcome**:        WIN, not draw / resign / DQ -> 0.50.
* **Format**:         single clean tool call, no malformed output, no
                       trap pings (escalate_to_oncall etc.) -> 0.10.
* **Thought-quality**: ta>5 words gives the structured-reasoning sub-score;
                       the candidate list contains the actual target
                       migration; the justification mentions ``kernel``
                       (a member of ``SOC_CONCEPTS``) -> 0.15.
* **Stockfish**:      mate-in-1 is the engine's optimal move (cp_loss=0),
                       and we never call ``query_threat_oracle`` -> 0.24.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

# sys.path bootstrap so the file runs as a script anywhere.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
for _p in (_REPO_ROOT / "src", _REPO_ROOT / "envs", _HERE):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _http_test_server import start_background_server, stop_background_server  # noqa: E402


# ---------------------------------------------------------------------------
# Test inputs (seed FEN + the precomputed mate-in-1 migration)
# ---------------------------------------------------------------------------

# Standard back-rank mate puzzle: White Ra1-a8# wins.
MATE_IN_ONE_FEN = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"

# Square 0  (a1) decomposes to (region, zone, rack, pod) =
#                              (us-east, az-a, rack-1, pod-1).
# Square 56 (a8) decomposes to (us-east, az-b, rack-1, pod-4).
# See ``server.datacenter_env.square_to_node`` for the mapping.
SOURCE_NODE = {"region": "us-east", "zone": "az-a", "rack": "rack-1", "pod": "pod-1"}
TARGET_NODE = {"region": "us-east", "zone": "az-b", "rack": "rack-1", "pod": "pod-4"}
TARGET_MIGRATION = (
    "us-east/az-a/rack-1/pod-1->us-east/az-b/rack-1/pod-4"
)

# The reward we expect after the mate move. ``R_MAX`` is the simulator's
# clamp ceiling; the raw bucket sum equals it exactly here.
EXPECTED_REWARD = 0.99
EPSILON = 1e-6
EPISODE_ID = "perfect-path-mate-in-one"


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------


def _post(base: str, path: str, payload: dict, timeout: float = 30.0) -> dict:
    r = httpx.post(f"{base}{path}", json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(
            f"POST {path} -> HTTP {r.status_code}: {r.text[:500]}"
        )
    return r.json()


def reset_with_mate_seed(base: str) -> dict:
    """Reset the env, seeded with a mate-in-1 FEN under a known episode id."""
    return _post(
        base,
        "/reset",
        {
            "episode_id": EPISODE_ID,
            "fen": MATE_IN_ONE_FEN,
        },
    )


def play_back_rank_mate(base: str) -> dict:
    """Single ``migrate_workload`` call that delivers Ra1-a8# in datacenter terms.

    Every reasoning field is intentionally crafted so the deterministic
    thought-quality scorer hits its 0.15 ceiling:

    * ``threat_analysis`` is 8 words long (>5 -> structured-reasoning sub-score).
    * ``candidate_migrations`` contains the actual target -> "candidate
      consistency" sub-score.
    * ``justification`` contains ``kernel`` -- a member of
      ``server.datacenter_env.SOC_CONCEPTS`` -> "SOC vocabulary" sub-score.
    """
    return _post(
        base,
        "/step",
        {
            "episode_id": EPISODE_ID,
            "action": {
                "type": "call_tool",
                "tool_name": "migrate_workload",
                "arguments": {
                    "threat_analysis": (
                        "Adversary kernel exposed on the back rank now."
                    ),
                    "candidate_migrations": [TARGET_MIGRATION],
                    "justification": (
                        "Containment of adversary kernel via decisive "
                        "rook deployment on the root corridor."
                    ),
                    "source_node": SOURCE_NODE,
                    "target_node": TARGET_NODE,
                },
            },
            "timeout_s": 30,
        },
    )


def assert_perfect_reward(step_response: dict) -> tuple[float, dict]:
    """Verify ``reward == 0.99`` (full clamp) and that the engagement ended."""
    reward = step_response.get("reward")
    done = step_response.get("done")
    obs = step_response.get("observation") or {}
    metadata = obs.get("metadata") or {}
    bucket = metadata.get("bucket", {})
    final_reward = metadata.get("final_reward", {})
    result = metadata.get("result")

    assert isinstance(reward, (int, float)), f"reward should be numeric, got {reward!r}"
    assert done is True, f"engagement should be over after Ra1-a8 mate, got done={done}"
    assert result == "checkmate_defender", (
        f"expected result='checkmate_defender', got {result!r}"
    )
    assert abs(float(reward) - EXPECTED_REWARD) < EPSILON, (
        f"defender reward should be exactly {EXPECTED_REWARD} (R_MAX clamp); "
        f"got {reward!r}. bucket={bucket} final_reward={final_reward}"
    )

    # Per-bucket confirmation. Use floor checks (>=) on the four buckets
    # because float arithmetic on the env side can shave ~1e-4 off any
    # individual sub-score; the *clamped* total is the contract.
    defender_bucket = bucket.get("defender", {})
    assert defender_bucket.get("outcome", 0) >= 0.50 - EPSILON, defender_bucket
    assert defender_bucket.get("format", 0) >= 0.10 - EPSILON, defender_bucket
    assert defender_bucket.get("thought_q", 0) >= 0.15 - EPSILON, defender_bucket
    assert defender_bucket.get("sf_acc", 0) >= 0.24 - EPSILON, defender_bucket

    return float(reward), defender_bucket


def run_perfect_path(base: str) -> int:
    print("=" * 60)
    print("Datacenter SOC -- perfect-path test (target reward = 0.99)")
    print("=" * 60)

    print(f"\n[1/2] POST /reset  (seed FEN = {MATE_IN_ONE_FEN!r})")
    reset_resp = reset_with_mate_seed(base)
    print(f"      reset done={reset_resp.get('done')} reward={reset_resp.get('reward')}")

    print(f"\n[2/2] POST /step   (Ra1-a8 mate as datacenter migration)")
    step_resp = play_back_rank_mate(base)

    reward, defender_bucket = assert_perfect_reward(step_resp)

    print("\n--- Final defender bucket breakdown ---")
    for k, v in defender_bucket.items():
        print(f"    {k:>10s}: {v}")
    print(f"  total reward: {reward}")

    print("\n[PASS] Defender reached the R_MAX = 0.99 clamp on the first half-move.")
    return 0


def main() -> int:
    print("Starting background uvicorn (datacenter env)...")
    proc = None
    try:
        proc, base = start_background_server()
        print(f"Server ready at {base}\n")
        return run_perfect_path(base)
    except AssertionError as exc:
        print(f"\n[FAIL] perfect-path assertion failed: {exc}")
        return 1
    except Exception as exc:
        print(f"\n[FAIL] perfect-path test crashed: {exc!r}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        stop_background_server(proc)
        print("\nBackground server stopped.")


if __name__ == "__main__":
    sys.exit(main())
