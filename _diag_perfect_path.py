"""Throwaway diagnostic: dump the full /step response body when running
the perfect-path mate-in-1 flow, so we can see why result=None."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
for _p in (_REPO_ROOT / "src", _REPO_ROOT / "envs", _HERE):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _http_test_server import start_background_server, stop_background_server  # noqa: E402

EPISODE_ID = "diag-mate"
FEN = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"

SOURCE = {"region": "us-east", "zone": "az-a", "rack": "rack-1", "pod": "pod-1"}
TARGET = {"region": "us-east", "zone": "az-b", "rack": "rack-1", "pod": "pod-4"}


def main() -> int:
    proc, base = start_background_server()
    try:
        print("--- /reset ---")
        r = httpx.post(
            f"{base}/reset",
            json={"episode_id": EPISODE_ID, "fen": FEN},
            timeout=15,
        )
        print(json.dumps(r.json(), indent=2)[:2000])

        print("\n--- /state ---")
        r2 = httpx.get(f"{base}/state", timeout=5)
        body = r2.json()
        print(f"keys: {sorted(body.keys()) if isinstance(body, dict) else type(body)}")
        if isinstance(body, dict):
            print(f"result: {body.get('result')!r}")
            print(f"done: {body.get('done')!r}")

        print("\n--- /step (migrate_workload Ra1-a8) ---")
        r3 = httpx.post(
            f"{base}/step",
            json={
                "episode_id": EPISODE_ID,
                "action": {
                    "type": "call_tool",
                    "tool_name": "migrate_workload",
                    "arguments": {
                        "threat_analysis": "Adversary kernel exposed back rank now.",
                        "candidate_migrations": [
                            "us-east/az-a/rack-1/pod-1->us-east/az-b/rack-1/pod-4"
                        ],
                        "justification": "Containment of adversary kernel via decisive rook deployment.",
                        "source_node": SOURCE,
                        "target_node": TARGET,
                    },
                },
                "timeout_s": 30,
            },
            timeout=30,
        )
        body = r3.json()
        # Print compactly with the most diagnostic keys
        obs = body.get("observation") or {}
        md = obs.get("metadata") or {}
        print(f"top-level: done={body.get('done')!r} reward={body.get('reward')!r}")
        print(f"metadata.result: {md.get('result')!r}")
        print(f"metadata.done: {md.get('done')!r}")
        print(f"metadata.bucket: {md.get('bucket')!r}")
        print(f"metadata.final_reward: {md.get('final_reward')!r}")
        print(f"metadata.tool_calls: {md.get('tool_calls')!r}")
        result_obj = obs.get("result")
        if isinstance(result_obj, dict):
            print(f"observation.result: {json.dumps(result_obj, indent=2)[:1500]}")
        else:
            print(f"observation.result: {result_obj!r}")
        print(f"\nobservation keys: {sorted(obs.keys())}")
    finally:
        stop_background_server(proc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
