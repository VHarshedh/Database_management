"""
Integration test for chess_arena — runs a suite of scripted scenarios and
verifies the layered reward system (0.50 outcome + 0.25 tool_acc + 0.24 sf_acc)
behaves correctly for every terminal branch.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _ROOT.parent
for _p in (str(_REPO_ROOT), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

# Reward bounds the grader enforces.
R_MIN = 0.01
R_MAX = 0.99

UNIVERSAL_THOUGHT = (
    "Evaluating the current position carefully, considering piece activity, "
    "king safety, pawn structure, and the opponent's threats before selecting "
    "the best tool call for this turn."
)

# ─── Helpers ────────────────────────────────────────────────────────────────

class StepResult:
    """Lightweight wrapper around a /step HTTP response."""

    def __init__(self, body: dict) -> None:
        self.raw = body
        self.done: bool = bool(body.get("done", False))
        self.reward: float = float(body.get("reward") or 0.0)

        obs = body.get("observation") or {}
        res = obs.get("result")
        
        # Safely extract text from the result
        if isinstance(res, dict):
            sc = res.get("structured_content") or {}
            if not isinstance(sc, dict):
                sc = {}
            self.text = str(sc.get("result", res.get("data", res)))
        else:
            self.text = str(res) if res is not None else ""

        # --- THE FIX: Extract payload from metadata ---
        payload = {}
        meta = obs.get("metadata")
        if isinstance(meta, dict):
            if "openenv" in meta:
                payload = meta["openenv"]
            elif "final_reward" in meta:
                payload = meta

        if not payload and isinstance(res, dict):
            sc = res.get("structured_content")
            if isinstance(sc, dict):
                payload = sc.get("openenv", {})

        self.payload: dict = payload if isinstance(payload, dict) else {}
        self.final_reward: dict = self.payload.get("final_reward") or {}
        self.bucket: dict = self.payload.get("bucket") or {}
        self.result_tag: str | None = self.payload.get("result")
        self.turn: str | None = self.payload.get("turn")


def reset_env() -> str:
    """Reset the environment.

    Returns an empty string as a sentinel episode_id.  We intentionally do NOT
    call ``GET /state`` here: that handler builds a fresh ``ChessEnvironment``
    instance (with a brand-new UUID assigned in ``__init__``) that was never
    registered in ``ChessEnvironment._instances``.  Using that UUID on /step
    would raise ``KeyError: Unknown episode_id``.

    ``ChessEnvironment.step`` falls back to ``_latest_instance`` when no
    episode_id is supplied, which is exactly what we want for these
    sequential, single-game scenarios.
    """
    httpx.post(f"{ENV_URL}/reset", json={}, timeout=10.0).raise_for_status()
    return ""


def step(episode_id: str, tool: str, **arguments) -> StepResult:
    """Post a single tool call and return the parsed StepResult."""

    # --- AUTO-TRANSLATE OLD SCHEMA TO NEW SCHEMA ---
    if "thought" in arguments:
        thought_text = arguments.pop("thought")
        arguments["threat_analysis"] = thought_text
        arguments["justification"] = thought_text

        # The schema strictly requires candidate_moves to be a list
        if "uci_move" in arguments:
            arguments["candidate_moves"] = [arguments["uci_move"], "a1a2"]
        else:
            arguments["candidate_moves"] = ["e2e4", "d2d4"]
    # -----------------------------------------------

    if tool == "make_move":
        arguments.setdefault("threat_analysis", "Default threat analysis for testing.")
        arguments.setdefault("candidate_moves", [arguments.get("uci_move", ""), "a1a2"])
        arguments.setdefault("justification", "Default justification for testing.")

    payload: dict = {
        "action": {
            "type": "call_tool",
            "tool_name": tool,
            "arguments": arguments,
        },
    }
    # Only attach episode_id when we actually have one; otherwise the server
    # falls back to ChessEnvironment._latest_instance, which is the correct
    # session for these sequential tests.
    if episode_id:
        payload["episode_id"] = episode_id

    r = httpx.post(f"{ENV_URL}/step", json=payload, timeout=15.0)
    r.raise_for_status()
    return StepResult(r.json())


# ─── Scenarios ──────────────────────────────────────────────────────────────

def scenario_1_clean_opening() -> tuple:
    episode_id = reset_env()
    step(episode_id, "analyze_board", thought=UNIVERSAL_THOUGHT)
    step(episode_id, "list_legal_moves", thought=UNIVERSAL_THOUGHT)
    w = step(episode_id, "make_move", thought=UNIVERSAL_THOUGHT, uci_move="e2e4")
    b = step(episode_id, "make_move", thought=UNIVERSAL_THOUGHT, uci_move="e7e5")

    assert not w.done and not b.done, "Clean opening must not terminate."
    assert R_MIN < w.reward < R_MAX, f"white preview reward out of bounds: {w.reward}"
    assert R_MIN < b.reward < R_MAX, f"black preview reward out of bounds: {b.reward}"
    return ("Scenario 1 (clean opening)", b.reward, b.done, None)


def scenario_2_fools_mate() -> tuple:
    episode_id = reset_env()
    step(episode_id, "make_move", thought=UNIVERSAL_THOUGHT, uci_move="f2f3")
    step(episode_id, "make_move", thought=UNIVERSAL_THOUGHT, uci_move="e7e5")
    step(episode_id, "make_move", thought=UNIVERSAL_THOUGHT, uci_move="g2g4")
    sr = step(episode_id, "make_move", thought=UNIVERSAL_THOUGHT, uci_move="d8h4")

    assert sr.done, "Fool's mate should terminate after 4 plies."
    assert sr.result_tag == "checkmate_black", f"expected checkmate_black, got {sr.result_tag!r}"
    b_final = float(sr.final_reward.get("black", 0.0))
    w_final = float(sr.final_reward.get("white", 0.0))
    assert b_final > w_final, "black should score higher than white after checkmating"
    return ("Scenario 2 (fools_mate)", b_final, sr.done, sr.result_tag)


def scenario_3_illegal_dq() -> tuple:
    """Illegal UCI triggers DQ against the offender after 2 strikes."""
    episode_id = reset_env()
    step(episode_id, "make_move", thought=UNIVERSAL_THOUGHT, uci_move="e2e4")
    
    # Strike 1
    step(episode_id, "make_move", thought="Trying an impossible move.", uci_move="zzzz")
    # Strike 2 (DQ)
    sr = step(episode_id, "make_move", thought="Doing it again to trigger DQ.", uci_move="yyyy")

    assert sr.done, "2nd Illegal UCI should terminate the episode."
    assert sr.result_tag == "dq_illegal_black", f"expected dq_illegal_black, got {sr.result_tag!r}"
    w_final = float(sr.final_reward.get("white", 0.0))
    b_final = float(sr.final_reward.get("black", 0.0))
    assert w_final > b_final, "opponent should out-score the DQ'd side"
    return ("Scenario 3 (illegal_dq)", w_final, sr.done, sr.result_tag)


def scenario_4_eval_abuse_dq() -> tuple:
    episode_id = reset_env()
    last: StepResult | None = None
    for _ in range(6):
        last = step(episode_id, "evaluate_position", thought=UNIVERSAL_THOUGHT)
        if last.done:
            break

    assert last is not None, "expected at least one step result"
    assert last.done, "6 evaluate_position calls should trigger DQ."
    assert last.result_tag == "dq_eval_abuse_white", f"expected dq_eval_abuse_white, got {last.result_tag!r}"
    w_final = float(last.final_reward.get("white", 0.0))
    b_final = float(last.final_reward.get("black", 0.0))
    assert b_final > w_final, "opponent should out-score the DQ'd side"
    return ("Scenario 4 (eval_abuse_dq)", b_final, last.done, last.result_tag)


def scenario_5_resign() -> tuple:
    episode_id = reset_env()
    step(episode_id, "make_move", thought=UNIVERSAL_THOUGHT, uci_move="e2e4")
    sr = step(episode_id, "resign_game", thought="Position is already lost.")

    assert sr.done, "resign_game should terminate the episode."
    assert sr.result_tag == "resign_black", f"expected resign_black, got {sr.result_tag!r}"
    w_final = float(sr.final_reward.get("white", 0.0))
    b_final = float(sr.final_reward.get("black", 0.0))
    assert w_final > b_final, "white should out-score the resigning side"
    return ("Scenario 5 (resign)", w_final, sr.done, sr.result_tag)


def scenario_6_ping_is_nonfatal() -> tuple:
    episode_id = reset_env()
    sr = step(
        episode_id,
        "ping_humanhelper",
        thought=UNIVERSAL_THOUGHT,
        reason="Checking whether the trap tool ends the episode (it must not).",
    )
    assert not sr.done, "ping_humanhelper must not terminate the episode."
    assert sr.reward >= 0.0, f"ping_humanhelper preview reward out of bounds: {sr.reward}"

    follow = step(episode_id, "make_move", thought=UNIVERSAL_THOUGHT, uci_move="e2e4")
    assert not follow.done, "Episode should still be alive after a legal move."
    return ("Scenario 6 (ping_nonfatal)", follow.reward, follow.done, None)


def scenario_7_black_resignation_win() -> tuple:
    THOUGHT = UNIVERSAL_THOUGHT
    moves_white = ["e2e4", "g1f3", "d2d4", "f3d4", "b1c3", "f1c4", "c1d2", "f2f3", "f3e4", "d1f3", "h1f1"]
    moves_black = ["c7c5", "d7d6", "c5d4", "g8f6", "g7g6", "f8g7", "e8g8", "f6e4", "g7d4", "b8c6", "c6e5"]

    episode_id = reset_env()
    last = None

    for w_uci, b_uci in zip(moves_white, moves_black):
        last = step(episode_id, "make_move", thought=THOUGHT, uci_move=w_uci)
        if last.done: break
        last = step(episode_id, "make_move", thought=THOUGHT, uci_move=b_uci)
        if last.done: break

    sr = last
    if sr is not None and not sr.done:
        sr = step(episode_id, "resign_game", thought="White's position is completely lost.")

    assert sr.done, "resign_game should terminate the episode."
    assert sr.result_tag == "resign_white", f"expected resign_white, got {sr.result_tag!r}"
    b_final = float(sr.final_reward.get("black", 0.0))
    w_final = float(sr.final_reward.get("white", 0.0))
    assert b_final > w_final, "black should outscore the resigning white"
    assert b_final >= 0.80, f"black final reward should be >= 0.80 (strong win); got {b_final:.4f}"
    return ("Scenario 7 (black_resignation_win)", b_final, sr.done, sr.result_tag)


def scenario_8_draw_by_repetition() -> tuple:
    THOUGHT_W = "Repeating to claim a draw."
    THOUGHT_B = "Playing solid defence. The draw is fair."

    # Fully restored valid 4-character UCI strings
    moves = [
        ("e2e4", "c7c5"), ("g1f3", "d7d6"), ("d2d4", "c5d4"),
        ("f3d4", "g8f6"), ("b1c3", "g7g6"), ("c1e3", "f8g7"),
        ("f2f3", "e8g8"), ("f1c4", "b8c6"), ("d1d2", "c6d4"),
        ("e3d4", "c8e6"), ("d4e6", "f7e6"), ("e1c1", "d8a5"),
        ("h2h4", "a8c8"), ("c1b1", "f6h5"), ("d4g7", "g8g7"),
        ("c3e2", "a5d2"), ("d1d2", "e6e5"), ("b2b3", "h5f4"),
        ("e2f4", "e5f4"), ("h1d1", "c8c5"), ("d2d5", "f8c8"),
        ("c2c4", "g7f6"), ("b1b2", "a7a5"), ("d5c5", "c8c5"),
        ("d1d5", "c5d5"), ("e4d5", "g6g5"), ("b2c3", "f6f5"),
        ("c3d4", "g5g4"), ("a2a3", "b7b6"), ("b3b4", "a5b4"),
        ("a3b4", "h7h6"), ("d4d3", "g4f3"), ("g2f3", "f5e5"),
        ("b4b5", "e5f5"), ("d3c3", "f5e5"), ("c3d3", "e5f5"),
        ("d3c3", "f5e5"),
    ]

    episode_id = reset_env()
    last = None
    for w_uci, b_uci in moves:
        last = step(episode_id, "make_move", thought=THOUGHT_W, uci_move=w_uci)
        if last.done: break
        last = step(episode_id, "make_move", thought=THOUGHT_B, uci_move=b_uci)
        if last.done: break

    sr = last
    # Only make the final step if the game didn't naturally terminate during the loop
    if sr is not None and not sr.done:
        sr = step(episode_id, "make_move", thought=THOUGHT_W, uci_move="c3d3")

    assert sr.done, "Threefold repetition should terminate the episode."
    result_lower = (sr.result_tag or "").lower()
    assert any(tok in result_lower for tok in ("draw", "repetition", "stale")), "expected a draw"
    b_final = float(sr.final_reward.get("black", 0.0))
    w_final = float(sr.final_reward.get("white", 0.0))
    assert abs(b_final - w_final) < 0.2, "draw rewards should be close"
    return ("Scenario 8 (draw_by_repetition)", b_final, sr.done, sr.result_tag)


def scenario_9_white_wins_french_resign() -> tuple:
    THOUGHT_W = "Advancing methodically."
    THOUGHT_B = "Lost position."

    # Fully restored valid 4-character UCI strings
    moves = [
        ("e2e4", "e7e6"), ("d2d4", "d7d5"), ("e4e5", "c7c5"), ("c2c3", "d8b6"),
        ("g1f3", "c8d7"), ("a2a3", "a7a5"), ("f1e2", "d7b5"), ("e1g1", "b5e2"),
        ("d1e2", "c5c4"), ("a3a4", "b8d7"), ("b1d2", "b6c6"), ("f1e1", "d7b6"),
        ("e2d1", "g8e7"), ("b2b3", "c4b3"), ("d1b3", "b6c4"), ("a1b1", "a8b8"),
        ("b3b5", "c6b5"), ("b1b5", "b7b6"), ("d2c4", "d5c4"), ("f3d2", "e7d5"),
        ("c1b2", "b8c8"), ("d2e4", "f8e7"), ("e1a1", "e8g8"), ("b2a3", "e7a3"),
        ("a1a3", "c8c6"), ("g2g3", "f7f5"), ("e4d6", "f5f4"), ("g1g2", "f4f3"),
        ("g2h3", "h7h5"), ("h3h4", "g6g5"), ("h4g5", "g8g7"), ("d6e4", "c6f6"),
        ("g5h4", "g7h6"), ("h4g4", "f6f4"), ("g3f4", "h5h4"), ("f4g5", "a5a4"),
        ("b5d5", "e6d5"), ("g5f4", "c8c6"), ("f4g5", "h6g7"),
    ]

    episode_id = reset_env()
    last = None
    for w_uci, b_uci in moves:
        last = step(episode_id, "make_move", thought=THOUGHT_W, uci_move=w_uci)
        if last.done: break
        last = step(episode_id, "make_move", thought=THOUGHT_B, uci_move=b_uci)
        if last.done: break

    sr = last
    if sr is not None and not sr.done:
        last = step(episode_id, "make_move", thought=THOUGHT_W, uci_move="b5b6")
        if not last.done:
            sr = step(episode_id, "resign_game", thought="The endgame is completely lost.")
        else:
            sr = last

    assert sr.done, "resign_game should terminate the episode."
    assert sr.result_tag == "resign_black", f"expected resign_black, got {sr.result_tag!r}"
    w_final = float(sr.final_reward.get("white", 0.0))
    b_final = float(sr.final_reward.get("black", 0.0))
    assert w_final > b_final, "white should outscore resigning black"
    assert w_final >= 0.80, f"white final reward should be >= 0.80 (strong win); got {w_final:.4f}"
    return ("Scenario 9 (white_wins_french_resign)", w_final, sr.done, sr.result_tag)


def run_all_tests() -> None:
    results: list[tuple] = []
    scenario_fns = [
        scenario_1_clean_opening, scenario_2_fools_mate, scenario_3_illegal_dq,
        scenario_4_eval_abuse_dq, scenario_5_resign, scenario_6_ping_is_nonfatal,
        scenario_7_black_resignation_win, scenario_8_draw_by_repetition,
        scenario_9_white_wins_french_resign,
    ]

    for i, fn in enumerate(scenario_fns, 1):
        print("=" * 50)
        print(f"SCENARIO {i}: {fn.__doc__.splitlines()[0].strip() if fn.__doc__ else 'Unknown'}")
        print("=" * 50)
        try:
            name, reward, done, result_tag = fn()
            tag = f" [{result_tag}]" if result_tag else ""
            print(f"  --> reward={reward:.4f}, done={done}{tag}")
            results.append((name, reward, done, result_tag, None))
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            results.append((f"Scenario {i}", 0.0, False, None, str(e)))
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            results.append((f"Scenario {i}", 0.0, False, None, repr(e)))

    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print("=" * 50)
    all_pass = True
    for name, reward, done, result_tag, err in results:
        status = "[OK]" if err is None else "[X]"
        if err is not None: all_pass = False
        tag = f" [{result_tag}]" if result_tag else ""
        print(f"  {status} {name}: reward={reward:.4f}, done={done}{tag}")
        if err is not None: print(f"        -> {err}")

    if all_pass:
        print("\n  [PASS] All chess_arena scenarios behaved as expected!")
    else:
        print("\n  [FAIL] Some chess_arena scenarios had assertion failures.")
        sys.exit(1)


def main() -> None:
    print("> Starting background FastAPI server for evaluation...")
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "chess_arena.server.app:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(_REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    try:
        ready = False
        for _ in range(20):
            try:
                if httpx.get(f"{ENV_URL}/health", timeout=1.0).status_code == 200:
                    ready = True
                    break
            except Exception:
                time.sleep(1)

        if not ready:
            print("[X] Server failed to start.")
            sys.exit(1)

        print("[OK] Server is up! Running tests...\n")
        run_all_tests()
    finally:
        print("\n[STOP] Terminating background server...")
        server_process.kill()
        server_process.wait()

if __name__ == "__main__":
    main()