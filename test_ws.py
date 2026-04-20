"""
Integration test for chess_arena — runs a suite of scripted scenarios and
verifies the layered reward system (0.50 outcome + 0.25 tool_acc + 0.24 sf_acc)
behaves correctly for every terminal branch.

Every scenario drives the environment over HTTP using the same `StepResult`
helper style as support_env/test_ws.py, so the assertions below mirror
the "perfect agent" evaluation shape used by the hackathon grader.

Usage:
    python test_ws.py
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
        sc: dict = {}
        if isinstance(res, dict):
            sc = res.get("structured_content") or {}
            if not isinstance(sc, dict):
                sc = {}
            self.text = str(sc.get("result", res.get("data", res)))
        else:
            self.text = str(res) if res is not None else ""

        payload = sc.get("openenv") if isinstance(sc, dict) else {}
        self.payload: dict = payload if isinstance(payload, dict) else {}
        self.final_reward: dict = self.payload.get("final_reward") or {}
        self.bucket: dict = self.payload.get("bucket") or {}
        self.result_tag: str | None = self.payload.get("result")
        self.turn: str | None = self.payload.get("turn")


def reset_env() -> str:
    """Reset the environment and return the new episode_id."""
    httpx.post(f"{ENV_URL}/reset", json={}, timeout=10.0).raise_for_status()
    sr = httpx.get(f"{ENV_URL}/state", timeout=5.0)
    sr.raise_for_status()
    return sr.json().get("episode_id", "")


def step(episode_id: str, tool: str, **arguments) -> StepResult:
    """Post a single tool call and return the parsed StepResult."""
    payload = {
        "action": {
            "type": "call_tool",
            "tool_name": tool,
            "arguments": arguments,
        },
        "episode_id": episode_id,
    }
    r = httpx.post(f"{ENV_URL}/step", json=payload, timeout=15.0)
    r.raise_for_status()
    return StepResult(r.json())


# ─── Scenarios ──────────────────────────────────────────────────────────────


def scenario_1_clean_opening() -> tuple:
    """Clean opening — 4 plies, non-terminal, both sides still live."""
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
    """Fool's mate — 4 plies, black delivers checkmate.

    Moves: 1. f2f3 e7e5  2. g2g4 Qh4#
    Expected: result=checkmate_black, black=WIN (~0.50 + acc), white=LOSS (~0 outcome).
    """
    episode_id = reset_env()
    step(episode_id, "make_move", thought=UNIVERSAL_THOUGHT, uci_move="f2f3")
    step(episode_id, "make_move", thought=UNIVERSAL_THOUGHT, uci_move="e7e5")
    step(episode_id, "make_move", thought=UNIVERSAL_THOUGHT, uci_move="g2g4")
    sr = step(episode_id, "make_move", thought=UNIVERSAL_THOUGHT, uci_move="d8h4")

    assert sr.done, "Fool's mate should terminate after 4 plies."
    assert sr.result_tag == "checkmate_black", (
        f"expected checkmate_black, got {sr.result_tag!r}"
    )
    b_final = float(sr.final_reward.get("black", 0.0))
    w_final = float(sr.final_reward.get("white", 0.0))
    assert b_final > w_final, (
        f"black should score higher than white after checkmating; got "
        f"black={b_final}, white={w_final}"
    )
    return ("Scenario 2 (fools_mate)", b_final, sr.done, sr.result_tag)


def scenario_3_illegal_dq() -> tuple:
    """Illegal UCI triggers DQ against the offender."""
    episode_id = reset_env()
    step(episode_id, "make_move", thought=UNIVERSAL_THOUGHT, uci_move="e2e4")
    sr = step(
        episode_id,
        "make_move",
        thought="Trying an impossible move to force a DQ for testing.",
        uci_move="zzzz",
    )

    assert sr.done, "Illegal UCI should immediately terminate the episode."
    assert sr.result_tag == "dq_illegal_black", (
        f"expected dq_illegal_black, got {sr.result_tag!r}"
    )
    w_final = float(sr.final_reward.get("white", 0.0))
    b_final = float(sr.final_reward.get("black", 0.0))
    assert w_final > b_final, (
        f"opponent (white) should out-score the DQ'd side (black); got "
        f"white={w_final}, black={b_final}"
    )
    return ("Scenario 3 (illegal_dq)", w_final, sr.done, sr.result_tag)


def scenario_4_eval_abuse_dq() -> tuple:
    """Calling `evaluate_position` 6 times in a row DQs the offender."""
    episode_id = reset_env()
    # White calls evaluate_position six consecutive times; the 6th call
    # crosses EVAL_CALL_LIMIT (=5) and triggers DQ against white.
    last: StepResult | None = None
    for _ in range(6):
        last = step(episode_id, "evaluate_position", thought=UNIVERSAL_THOUGHT)
        if last.done:
            break

    assert last is not None, "expected at least one step result"
    assert last.done, "6 evaluate_position calls should trigger DQ."
    assert last.result_tag == "dq_eval_abuse_white", (
        f"expected dq_eval_abuse_white, got {last.result_tag!r}"
    )
    w_final = float(last.final_reward.get("white", 0.0))
    b_final = float(last.final_reward.get("black", 0.0))
    assert b_final > w_final, (
        f"opponent (black) should out-score the DQ'd side (white); got "
        f"white={w_final}, black={b_final}"
    )
    return ("Scenario 4 (eval_abuse_dq)", b_final, last.done, last.result_tag)


def scenario_5_resign() -> tuple:
    """Resignation awards the opponent a reduced-outcome win (<= 0.45)."""
    episode_id = reset_env()
    step(episode_id, "make_move", thought=UNIVERSAL_THOUGHT, uci_move="e2e4")
    sr = step(
        episode_id,
        "resign_game",
        thought="Position is already lost; resigning to save clock.",
    )

    assert sr.done, "resign_game should terminate the episode."
    assert sr.result_tag == "resign_black", (
        f"expected resign_black, got {sr.result_tag!r}"
    )
    w_final = float(sr.final_reward.get("white", 0.0))
    b_final = float(sr.final_reward.get("black", 0.0))
    assert w_final > b_final, (
        f"white should out-score the resigning side (black); got "
        f"white={w_final}, black={b_final}"
    )
    return ("Scenario 5 (resign)", w_final, sr.done, sr.result_tag)


def scenario_6_ping_is_nonfatal() -> tuple:
    """`ping_humanhelper` must NEVER terminate the episode, only dock tool_acc."""
    episode_id = reset_env()
    sr = step(
        episode_id,
        "ping_humanhelper",
        thought=UNIVERSAL_THOUGHT,
        reason="Checking whether the trap tool ends the episode (it must not).",
    )
    assert not sr.done, "ping_humanhelper must not terminate the episode."
    # Episode still live, so we still get a preview reward, and it must live
    # strictly inside (0, 1).
    assert R_MIN <= sr.reward < R_MAX, (
        f"ping_humanhelper preview reward out of bounds: {sr.reward}"
    )

    # Play on; the episode should continue normally.
    follow = step(episode_id, "make_move", thought=UNIVERSAL_THOUGHT, uci_move="e2e4")
    assert not follow.done, "Episode should still be alive after a legal move."
    return ("Scenario 6 (ping_nonfatal)", follow.reward, follow.done, None)


# ─── Main runner ────────────────────────────────────────────────────────────


def run_all_tests() -> None:
    results: list[tuple] = []

    scenario_fns = [
        scenario_1_clean_opening,
        scenario_2_fools_mate,
        scenario_3_illegal_dq,
        scenario_4_eval_abuse_dq,
        scenario_5_resign,
        scenario_6_ping_is_nonfatal,
    ]

    for i, fn in enumerate(scenario_fns, 1):
        print("=" * 50)
        print(f"SCENARIO {i}: {fn.__doc__.splitlines()[0].strip()}")
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
        if err is not None:
            all_pass = False
        tag = f" [{result_tag}]" if result_tag else ""
        print(f"  {status} {name}: reward={reward:.4f}, done={done}{tag}")
        if err is not None:
            print(f"        -> {err}")

    rewards = [r[1] for r in results if r[4] is None]
    avg = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"\n  Average reward across passing scenarios: {avg:.4f}")

    if all_pass:
        print("\n  [PASS] All chess_arena scenarios behaved as expected!")
    else:
        print("\n  [FAIL] Some chess_arena scenarios had assertion failures.")
        sys.exit(1)


def main() -> None:
    print("> Starting background FastAPI server for evaluation...")
    server_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "chess_arena.server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--log-level",
            "warning",
        ],
        cwd=str(_REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        ready = False
        for _ in range(20):
            try:
                r = httpx.get(f"{ENV_URL}/health", timeout=1.0)
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                time.sleep(1)

        if not ready:
            print("[X] Server failed to start within 20 seconds.")
            sys.exit(1)

        print("[OK] Server is up! Running chess_arena scenario suite...\n")
        run_all_tests()
    finally:
        print("\n[STOP] Terminating background server...")
        server_process.kill()
        server_process.wait()


if __name__ == "__main__":
    main()
