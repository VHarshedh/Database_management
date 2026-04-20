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


def run_reset_iteration_tests(base: str) -> None:
    print("Testing HTTP /reset iteration using analyze_board + make_move...")

    for i in range(1, 8):
        try:
            # Fresh reset; obtain the new episode_id.
            httpx.post(f"{base}/reset", timeout=5).raise_for_status()
            episode_id = httpx.get(f"{base}/state", timeout=5).json().get("episode_id")

            # Inspect the freshly-reset board — should be the standard opening
            # position with white to move on every iteration.
            analyze_payload = {
                "action": {
                    "type": "call_tool",
                    "tool_name": "analyze_board",
                    "arguments": {
                        "thought": (
                            "Reading the board to verify that HTTP /reset has properly "
                            "cycled the chess environment back to the initial position."
                        )
                    },
                },
                "episode_id": episode_id,
            }
            r = httpx.post(f"{base}/step", json=analyze_payload, timeout=5)
            r.raise_for_status()
            board_text, payload = _board_payload(r.json())

            turn = payload.get("turn")
            fen = payload.get("fen")
            assert turn == "white", f"Reset #{i}: expected turn=white, got {turn!r}"
            assert fen == INITIAL_FEN, f"Reset #{i}: expected starting FEN, got {fen!r}"

            # Make sure an actual move progresses state correctly.
            move_payload = {
                "action": {
                    "type": "call_tool",
                    "tool_name": "make_move",
                    "arguments": {
                        "thought": "Opening with the king's pawn to test move handling after reset.",
                        "uci_move": "e2e4",
                    },
                },
                "episode_id": episode_id,
            }
            r2 = httpx.post(f"{base}/step", json=move_payload, timeout=5)
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
