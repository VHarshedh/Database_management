"""Sequential HTTP smoke test for chess_arena (starts local uvicorn automatically)."""
from __future__ import annotations

import sys
from pathlib import Path

import httpx

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from _http_test_server import start_background_server, stop_background_server


UNIVERSAL_THOUGHT = (
    "Calculating the best response given the current tactical and positional "
    "features of the board before choosing a tool."
)


def _openenv_payload(step_body: dict) -> dict:
    """Pull the chess_arena debug payload out of the step response.

    The OpenEnv framework strips top-level `metadata` from HTTP responses, so
    ChessEnvironment mirrors per-color / bucket info into
    ``observation.result.structured_content.openenv``.
    """
    obs = step_body.get("observation") or {}
    res = obs.get("result") or {}
    if not isinstance(res, dict):
        return {}
    sc = res.get("structured_content") or {}
    if not isinstance(sc, dict):
        return {}
    payload = sc.get("openenv") or {}
    return payload if isinstance(payload, dict) else {}


def _extract_tool_text(step_body: dict) -> str:
    """Best-effort extraction of a FastMCP tool's human-readable output."""
    obs = step_body.get("observation") or {}
    res = obs.get("result")
    if isinstance(res, dict):
        sc = res.get("structured_content") or {}
        if isinstance(sc, dict) and "result" in sc:
            return str(sc["result"])
        if "data" in res and res["data"] is not None:
            return str(res["data"])
        content = res.get("content")
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and "text" in first:
                return str(first["text"])
    return str(res)


def run_smoke_tests(base: str) -> list[str]:
    results: list[str] = []

    def log(msg: str) -> None:
        results.append(msg)

    # Test 1: Reset
    log("=== TEST 1: Reset ===")
    r = httpx.post(f"{base}/reset", json={}, timeout=10)
    d = r.json()
    log(f"  Status: {r.status_code}")
    log(f"  Done: {d.get('done')}, Reward: {d.get('reward')}")

    # Pull the current episode_id so subsequent /step calls route to the same
    # ChessEnvironment instance (the env maintains state per episode_id).
    sr = httpx.get(f"{base}/state", timeout=10)
    episode_id = sr.json().get("episode_id")
    log(f"  episode_id: {episode_id}")

    # Test 2: List tools
    log("\n=== TEST 2: List Tools ===")
    r = httpx.post(
        f"{base}/step",
        json={"action": {"type": "list_tools"}, "timeout_s": 30},
        timeout=10,
    )
    d = r.json()
    tools = d.get("observation", {}).get("tools", [])
    log(f"  Found {len(tools)} tools:")
    for t in tools:
        log(f"    - {t['name']}")

    # Test 3: analyze_board  (readonly inspection)
    log("\n=== TEST 3: analyze_board ===")
    r = httpx.post(
        f"{base}/step",
        json={
            "action": {
                "type": "call_tool",
                "tool_name": "analyze_board",
                "arguments": {
                    "threat_analysis": UNIVERSAL_THOUGHT,
                    "candidate_moves": ["e2e4", "d2d4"],
                    "justification": UNIVERSAL_THOUGHT,
                },
            },
            "episode_id": episode_id,
            "timeout_s": 30,
        },
        timeout=10,
    )
    d = r.json()
    log(f"  Reward: {d.get('reward')}, Done: {d.get('done')}")
    log(f"  tool_name: {d.get('observation', {}).get('tool_name')}")
    log(f"  Board preview: {_extract_tool_text(d)[:80]}...")

    # Test 4: list_legal_moves
    log("\n=== TEST 4: list_legal_moves ===")
    r = httpx.post(
        f"{base}/step",
        json={
            "action": {
                "type": "call_tool",
                "tool_name": "list_legal_moves",
                "arguments": {
                    "threat_analysis": UNIVERSAL_THOUGHT,
                    "candidate_moves": ["e2e4", "d2d4"],
                    "justification": UNIVERSAL_THOUGHT,
                },
            },
            "episode_id": episode_id,
            "timeout_s": 30,
        },
        timeout=10,
    )
    d = r.json()
    log(f"  Reward: {d.get('reward')}, Done: {d.get('done')}")
    log(f"  Legal-move listing preview: {_extract_tool_text(d)[:80]}...")

    # Test 5: make_move (legal)
    log("\n=== TEST 5: make_move (legal, e2e4) ===")
    r = httpx.post(
        f"{base}/step",
        json={
            "action": {
                "type": "call_tool",
                "tool_name": "make_move",
                "arguments": {
                    "threat_analysis": UNIVERSAL_THOUGHT,
                    "candidate_moves": ["e2e4", "d2d4"],
                    "justification": UNIVERSAL_THOUGHT,
                    "uci_move": "e2e4",
                },
            },
            "episode_id": episode_id,
            "timeout_s": 30,
        },
        timeout=10,
    )
    d = r.json()
    log(f"  Reward: {d.get('reward')}, Done: {d.get('done')}")
    payload = _openenv_payload(d)
    log(f"  Turn after move: {payload.get('turn')} (expected: black)")
    log(f"  Bucket (white): {payload.get('bucket', {}).get('white')}")

    # Test 6: ping_humanhelper (TRAP; non-fatal)
    log("\n=== TEST 6: ping_humanhelper (TRAP) ===")
    r = httpx.post(
        f"{base}/step",
        json={
            "action": {
                "type": "call_tool",
                "tool_name": "ping_humanhelper",
                "arguments": {
                    "threat_analysis": UNIVERSAL_THOUGHT,
                    "candidate_moves": ["e2e4", "d2d4"],
                    "justification": UNIVERSAL_THOUGHT,
                    "reason": "Need a hint on best reply to e4.",
                },
            },
            "episode_id": episode_id,
            "timeout_s": 30,
        },
        timeout=10,
    )
    d = r.json()
    payload = _openenv_payload(d)
    log(f"  Reward: {d.get('reward')}, Done: {d.get('done')}")
    log(f"  Ping count (black): {payload.get('ping_count', {}).get('black')}")
    assert not d.get("done"), "ping_humanhelper must NOT terminate the episode."

    # Test 7: evaluate_position (TRAP; per-call sf_acc penalty, capped at 5)
    log("\n=== TEST 7: evaluate_position (TRAP) ===")
    r = httpx.post(
        f"{base}/step",
        json={
            "action": {
                "type": "call_tool",
                "tool_name": "evaluate_position",
                "arguments": {
                    "threat_analysis": UNIVERSAL_THOUGHT,
                    "candidate_moves": ["e2e4", "d2d4"],
                    "justification": UNIVERSAL_THOUGHT,
                },
            },
            "episode_id": episode_id,
            "timeout_s": 30,
        },
        timeout=10,
    )
    d = r.json()
    payload = _openenv_payload(d)
    log(f"  Reward: {d.get('reward')}, Done: {d.get('done')}")
    log(f"  Eval calls (black): {payload.get('eval_calls', {}).get('black')}")

    # Test 8: resign_game (terminal; resigner = 0, opponent = 0.45 outcome)
    log("\n=== TEST 8: resign_game (terminal) ===")
    r = httpx.post(
        f"{base}/step",
        json={
            "action": {
                "type": "call_tool",
                "tool_name": "resign_game",
                "arguments": {
                    "threat_analysis": "Position looks lost; resigning.",
                    "candidate_moves": ["e2e4", "d2d4"],
                    "justification": "Position looks lost; resigning.",
                },
            },
            "episode_id": episode_id,
            "timeout_s": 30,
        },
        timeout=10,
    )
    d = r.json()
    payload = _openenv_payload(d)
    log(f"  Done: {d.get('done')}")
    log(f"  Result tag: {payload.get('result')}")
    log(f"  Final reward: {payload.get('final_reward')}")

    final = payload.get("final_reward") or {}
    for color, val in final.items():
        assert 0.0 < float(val) < 1.0, f"Final reward for {color} outside (0,1): {val}"

    log("\n=== ALL TESTS PASSED ===")
    return results


def main() -> None:
    print("Starting background uvicorn (chess_arena)...")
    proc = None
    try:
        proc, base = start_background_server()
        print(f"Server ready at {base}\n")
        print("\n".join(run_smoke_tests(base)))
    finally:
        stop_background_server(proc)
        print("\nBackground server stopped.")


if __name__ == "__main__":
    main()
    sys.exit(0)
