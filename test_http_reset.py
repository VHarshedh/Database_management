"""Verify HTTP /reset cycles the chess_arena env (starts local uvicorn automatically)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from _http_test_server import start_background_server, stop_background_server


INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Structured-reasoning fields required by the current tool schema.  The old
# ``{"thought": "..."}`` payload used by previous versions of this test is now
# rejected by FastMCP validation, which is how Bug 14 (silently-swallowed
# assertion failures) was masking real test failures.
_ANALYZE_THOUGHT = (
    "Reading the board to verify that HTTP /reset has properly cycled the "
    "chess environment back to the initial position."
)
_MOVE_THOUGHT = (
    "Opening with the king's pawn to test move handling after reset."
)


def _board_payload(step_body: dict) -> tuple[str, dict]:
    """Return (board_text, openenv_payload) from a /step response."""
    obs = step_body.get("observation") or {}
    res = obs.get("result") or {}
    if isinstance(res, dict):
        sc = res.get("structured_content") or {}
        board_text = str(sc.get("result", res.get("data", res)))
        openenv = sc.get("openenv") if isinstance(sc, dict) else {}
        return board_text, (openenv if isinstance(openenv, dict) else {})
    return str(res), {}


def _post_step(base: str, tool_name: str, arguments: dict) -> httpx.Response:
    """POST /step without an ``episode_id``.

    We intentionally skip ``GET /state``: that handler builds a fresh
    ``ChessEnvironment`` with a UUID that was never registered in
    ``ChessEnvironment._instances``, so passing that UUID back to ``/step``
    raises ``KeyError: Unknown episode_id`` (Bug 3).  With no ``episode_id``
    on the request, the environment falls back to
    ``ChessEnvironment._latest_instance`` — exactly the session created by
    the most recent ``/reset``, which is what these single-game iteration
    tests want.
    """
    payload = {
        "action": {
            "type": "call_tool",
            "tool_name": tool_name,
            "arguments": arguments,
        },
    }
    return httpx.post(f"{base}/step", json=payload, timeout=5)


def run_reset_iteration_tests(base: str) -> None:
    print("Testing HTTP /reset iteration using analyze_board + make_move...")

    for i in range(1, 8):
        try:
            # Fresh reset.  No /state round-trip; see _post_step for context.
            httpx.post(f"{base}/reset", timeout=5).raise_for_status()

            # Inspect the freshly-reset board — should be the standard opening
            # position with white to move on every iteration.
            r = _post_step(
                base,
                "analyze_board",
                {
                    "threat_analysis": _ANALYZE_THOUGHT,
                    "candidate_moves": ["e2e4", "d2d4"],
                    "justification": _ANALYZE_THOUGHT,
                },
            )
            r.raise_for_status()
            board_text, payload = _board_payload(r.json())

            turn = payload.get("turn")
            fen = payload.get("fen")
            assert turn == "white", f"Reset #{i}: expected turn=white, got {turn!r}"
            assert fen == INITIAL_FEN, f"Reset #{i}: expected starting FEN, got {fen!r}"

            # Make sure an actual move progresses state correctly.
            r2 = _post_step(
                base,
                "make_move",
                {
                    "threat_analysis": _MOVE_THOUGHT,
                    "candidate_moves": ["e2e4", "d2d4"],
                    "justification": _MOVE_THOUGHT,
                    "uci_move": "e2e4",
                },
            )
            r2.raise_for_status()
            _, post_move = _board_payload(r2.json())
            post_turn = post_move.get("turn")
            assert post_turn == "black", (
                f"Reset #{i}: after white played e2e4, expected turn=black, got {post_turn!r}"
            )

            print(f"Call {i} Board Preview: {board_text[:60]}...  next_turn={post_turn}")
            time.sleep(0.25)
        except Exception as e:
            print(f"Call {i} failed: {e}")


def main() -> None:
    print("Starting background uvicorn (chess_arena)...")
    proc = None
    try:
        proc, base = start_background_server()
        print(f"Server ready at {base}\n")
        run_reset_iteration_tests(base)
    finally:
        stop_background_server(proc)
        print("\nBackground server stopped.")


if __name__ == "__main__":
    main()
    sys.exit(0)
