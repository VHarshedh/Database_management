"""In-process ChessEnvironment smoke test; also pings HTTP /health via background server."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import httpx

# Repo root on sys.path so `chess_arena.*` imports work when run as a script
# from within the `chess_arena/` directory.
_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _ROOT.parent
for _p in (str(_REPO_ROOT), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _http_test_server import start_background_server, stop_background_server
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from chess_arena.server.chess_environment import ChessEnvironment


# Structured reasoning args shared across all test tool calls.
_TA = (
    "No immediate checks or captures. Position is symmetric from starting setup. "
    "Evaluating piece activity, king safety, and pawn structure."
)
_CM = ["e2e4", "d2d4", "g1f3"]
_JU = (
    "Developing pieces toward the center to control space and enable "
    "king safety via castling."
)

# Short-form dict for use in arguments=
UNIVERSAL_ARGS = {
    "threat_analysis": _TA,
    "candidate_moves": _CM,
    "justification": _JU,
}


def _print_observation(label: str, obs) -> None:
    """Render a compact summary of a CallToolObservation / ListToolsObservation."""
    tool_name = getattr(obs, "tool_name", None)
    reward = getattr(obs, "reward", None)
    done = getattr(obs, "done", False)
    metadata = getattr(obs, "metadata", {}) or {}
    fen = metadata.get("fen")
    turn = metadata.get("turn")
    result_tag = metadata.get("result")
    print(
        f"[{label:<18}] tool={tool_name} reward={reward} done={done} "
        f"turn={turn} result={result_tag} fen={fen}"
    )


def run_inprocess_smoke() -> None:
    print("=== In-process ChessEnvironment smoke test ===")
    env = ChessEnvironment()
    env.reset()
    print("Initialized (in-process)")

    # 1. List all registered tools via MCP.
    tools_obs = env.step(ListToolsAction())
    tool_names = sorted(getattr(t, "name", "?") for t in getattr(tools_obs, "tools", []))
    print(f"  Tools registered: {tool_names}")
    expected = {
        "analyze_board",
        "list_legal_moves",
        "make_move",
        "resign_game",
        "evaluate_position",
        "ping_humanhelper",
    }
    missing = expected.difference(tool_names)
    if missing:
        raise AssertionError(f"Missing expected chess tools: {sorted(missing)}")

    # 2. Inspect the initial position.
    obs = env.step(
        CallToolAction(
            tool_name="analyze_board",
            arguments=UNIVERSAL_ARGS,
        )
    )
    _print_observation("analyze_board", obs)

    # 3. Enumerate legal moves for white.
    obs = env.step(
        CallToolAction(
            tool_name="list_legal_moves",
            arguments=UNIVERSAL_ARGS,
        )
    )
    _print_observation("list_legal_moves", obs)

    # 4. Play a legal opening move; the environment should flip `turn_color`
    #    to black and grant a non-terminal preview reward strictly in (0.01, 0.99).
    obs = env.step(
        CallToolAction(
            tool_name="make_move",
            arguments={
                **UNIVERSAL_ARGS,
                "candidate_moves": ["e2e4", "d2d4", "g1f3"],  # e2e4 must be in list
                "uci_move": "e2e4",
            },
        )
    )
    _print_observation("make_move e2e4", obs)
    if env.turn_color != "black":
        raise AssertionError(f"After white moved, turn_color should be 'black', got {env.turn_color!r}")
    assert not env.done, "Episode should still be live after a single legal move."

    # 5. `ping_humanhelper` should NOT end the episode but does dock the
    #    caller's tool-accuracy bucket.
    pre_ping_bucket = dict(env.bucket["black"])
    obs = env.step(
        CallToolAction(
            tool_name="ping_humanhelper",
            arguments={
                **UNIVERSAL_ARGS,
                "reason": "Need a hint.",
            },
        )
    )
    _print_observation("ping_humanhelper", obs)
    assert not env.done, "ping_humanhelper must NOT terminate the episode."
    assert env.ping_count["black"] == 1, "ping_count[black] should have incremented to 1."

    print("  [OK] in-process scripted smoke test finished cleanly.")


def main() -> None:
    print("Starting background uvicorn for /health check...")
    proc = None
    try:
        proc, base = start_background_server()
        print(f"Server ready at {base}")
        httpx.get(f"{base}/health", timeout=5.0).raise_for_status()
        print("HTTP /health OK\n")

        run_inprocess_smoke()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        stop_background_server(proc)
        print("\nBackground server stopped.")


if __name__ == "__main__":
    main()
