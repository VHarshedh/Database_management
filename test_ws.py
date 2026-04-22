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


def scenario_7_black_resignation_win() -> tuple:
    """Black wins by resignation — Sicilian Dragon, White resigns after Ne5.

    Source: chess.com/game/live/145978428344
    White: Hahahehehahehahe (1767)  Black: Harshedhvbs (2101)
    Result: 0-1 (White resigned after 11 moves)

    Black plays a perfect Dragon Sicilian — knight sac on e4, bishop captures
    on d4 winning a piece, queen enters f3, knight dominates e5.  White's
    position is hopeless; they resign.  We simulate White's resignation via
    `resign_game` on White's turn after Ne5.

    Expected:
      - episode terminates (done=True)
      - result_tag == "resign_white"
      - black final_reward >= 0.99 (R_MAX)
    """
    THOUGHT = (
        "Evaluating the current position carefully, considering piece activity, "
        "king safety, pawn structure, and the opponent's threats before selecting "
        "the best tool call for this turn."
    )

    # Full game move sequence in UCI notation:
    # 1. e4 c5  2. Nf3 d6  3. d4 cxd4  4. Nxd4 Nf6  5. Nc3 g6
    # 6. Bc4 Bg7  7. Bd2 O-O  8. f3 Nxe4  9. fxe4 Bxd4
    # 10. Qf3 Nc6  11. Rf1 Ne5  →  White resigns
    moves_white = [
        "e2e4",   # 1. e4
        "g1f3",   # 2. Nf3
        "d2d4",   # 3. d4
        "f3d4",   # 4. Nxd4
        "b1c3",   # 5. Nc3
        "f1c4",   # 6. Bc4
        "c1d2",   # 7. Bd2
        "f2f3",   # 8. f3
        "f3e4",   # 9. fxe4
        "d1f3",   # 10. Qf3
        "h1f1",   # 11. Rf1
    ]
    moves_black = [
        "c7c5",   # 1... c5
        "d7d6",   # 2... d6
        "c5d4",   # 3... cxd4
        "g8f6",   # 4... Nf6
        "g7g6",   # 5... g6
        "f8g7",   # 6... Bg7
        "e8g8",   # 7... O-O
        "f6e4",   # 8... Nxe4
        "g7d4",   # 9... Bxd4
        "b8c6",   # 10... Nc6
        "c6e5",   # 11... Ne5
    ]

    episode_id = reset_env()

    # Interleave moves: White 1, Black 1, White 2, Black 2, …
    last: StepResult | None = None
    for w_uci, b_uci in zip(moves_white, moves_black):
        step(episode_id, "make_move", thought=THOUGHT, uci_move=w_uci)
        last = step(episode_id, "make_move", thought=THOUGHT, uci_move=b_uci)
        if last.done:
            break  # premature terminal (shouldn't happen here)

    # After 11...Ne5 it's White's turn — White resigns.
    sr = step(
        episode_id,
        "resign_game",
        thought=(
            "After 11...Ne5 Black has a dominant knight on e5, an extra bishop "
            "from the piece sacrifice, total centre control and a crushing attack. "
            "White's position is completely lost — resigning immediately."
        ),
    )

    assert sr.done, "resign_game should terminate the episode."
    assert sr.result_tag == "resign_white", (
        f"expected resign_white, got {sr.result_tag!r}"
    )
    b_final = float(sr.final_reward.get("black", 0.0))
    w_final = float(sr.final_reward.get("white", 0.0))
    assert b_final > w_final, (
        f"black should outscore the resigning white; got "
        f"black={b_final:.4f}, white={w_final:.4f}"
    )
    assert b_final >= R_MAX, (
        f"black final reward should be >= {R_MAX} (target 0.99); got {b_final:.4f}"
    )
    return ("Scenario 7 (black_resignation_win)", b_final, sr.done, sr.result_tag)


def scenario_8_draw_by_repetition() -> tuple:
    """Draw by 3-fold repetition — Sicilian Dragon Yugoslav, izcq vs Harshedhvbs.

    Source: chess.com/game/live/140492783162
    White: izcq (2006)  Black: Harshedhvbs (2017)
    Result: 1/2-1/2 — Draw (threefold repetition, move 38)

    After a sharp Dragon with mutual trades, the K+P endgame reaches dead
    equality.  White's king shuttles Kd3-Kc3-Kd3 three times and the game
    is drawn by repetition.

    Expected:
      - episode terminates (done=True)
      - result_tag contains 'draw' or 'repetition' or 'stale'
      - both sides receive close final rewards
    """
    THOUGHT_W = (
        "The position is completely blocked. My king cannot make progress "
        "without giving ground. Repeating with Kd3-Kc3 to claim a draw by "
        "threefold repetition is the correct practical decision."
    )
    THOUGHT_B = (
        "Playing solid defence, keeping the pawn structure intact. "
        "If White repeats the king manoeuvre, the draw is the fair result "
        "in this dead-equal endgame."
    )

    # 38-move game: Sicilian Dragon Yugoslav Attack → K+P draw by repetition
    # Pairs: (white_uci, black_uci, white_thought, black_thought)
    moves = [
        ("e2e4", "c7c5",  "Opening with 1.e4 to control the centre.", "Sicilian — fighting for central counterplay with c5."),
        ("g1f3", "d7d6",  "Developing the knight toward the centre.", "Preparing the Dragon with d6, keeping the position flexible."),
        ("d2d4", "c5d4",  "Opening the centre with d4.", "Capturing on d4 to open the c-file and equalise."),
        ("f3d4", "g8f6",  "Recapturing with the knight, maintaining central control.", "Developing the king's knight, attacking e4."),
        ("b1c3", "g7g6",  "Developing queenside knight, eyeing d5.", "Preparing the Dragon fianchetto."),
        ("c1e3", "f8g7",  "Developing the bishop to e3, beginning the Yugoslav setup.", "Completing the fianchetto — the Dragon bishop is powerful."),
        ("f2f3", "e8g8",  "Reinforcing e4 and preparing Bc4-Qd2-0-0-0.", "Castling kingside for safety."),
        ("f1c4", "b8c6",  "Placing the bishop on the a2-g8 diagonal.", "Developing the knight to c6, adding pressure to d4."),
        ("d1d2", "c6d4",  "Connecting queen to d2 for the Yugoslav Attack.", "Trading knight on d4 to reduce central dominance."),
        ("e3d4", "c8e6",  "Recapturing with the bishop, holding the d4 square.", "Developing the bishop to e6, challenging d4."),
        ("d4e6", "f7e6",  "Trading off the dark-squared bishops.", "Recapturing with the f-pawn, opening the f-file."),
        ("e1c1", "d8a5",  "Castling queenside — the Yugoslav Attack is in full swing.", "Queen to a5, creating queenside counterplay."),
        ("h2h4", "a8c8",  "Advancing the h-pawn to storm the kingside.", "Doubling rooks on the c-file."),
        ("c1b1", "f6h5",  "Stepping the king to safety on b1.", "Knight to h5, eyeing g3 and f4."),
        ("d4g7", "g8g7",  "Exchanging the Dragon bishop to blunt Black's counterplay.", "Recapturing — king steps to g7."),
        ("c3e2", "a5d2",  "Repositioning the knight, preparing to stabilise.", "Trading queens to reduce White's attacking chances."),
        ("d1d2", "e6e5",  "Recapturing the queen.", "Pushing e5 to gain space and restrict the knight."),
        ("b2b3", "h5f4",  "Shoring up the queenside.", "Knight to f4 — a powerful centralised outpost."),
        ("e2f4", "e5f4",  "Trading the knight to simplify.", "Recapturing with the pawn, keeping the f4 wedge."),
        ("h1d1", "c8c5",  "Doubling rooks on the d-file.", "Centralising the rook to c5."),
        ("d2d5", "f8c8",  "Rook to d5, threatening the 5th rank.", "Bringing the second rook to the c-file."),
        ("c2c4", "g7f6",  "Advancing c4 to restrict Black's rook.", "King steps to f6 to activate."),
        ("b1b2", "a7a5",  "King to b2, centralising for the endgame.", "Advancing the a-pawn to gain queenside space."),
        ("d5c5", "c8c5",  "Trading rooks to simplify.", "Recapturing — entering a pure pawn endgame."),
        ("d1d5", "c5d5",  "Offering the rook trade.", "Trading rooks — pure K+P structure."),
        ("e4d5", "g6g5",  "Recapturing on d5.", "Pushing g5 to create a passed pawn."),
        ("b2c3", "f6f5",  "King to c3, supporting the passed d-pawn.", "King to f5, heading for the centre."),
        ("c3d4", "g5g4",  "King to d4, blocking the Black king.", "Advancing g4 to create a second passed pawn."),
        ("a2a3", "b7b6",  "Preparing to advance the b-pawn.", "Securing b6 against b4-b5."),
        ("b3b4", "a5b4",  "Advancing b4 to create a passed pawn.", "Capturing on b4 to eliminate the passer."),
        ("a3b4", "h7h6",  "Recapturing on b4.", "Advancing h6."),
        ("d4d3", "g4f3",  "King to d3 to support the pawns.", "Capturing on f3 — eliminating White's last kingside pawn."),
        ("g2f3", "f5e5",  "Recapturing on f3.", "King to e5, fighting for the d-pawn."),
        ("b4b5", "e5f5",  "Advancing b5.", "King to f5 — preparing the repetition."),
        ("d3c3", "f5e5",  THOUGHT_W, THOUGHT_B),
        ("c3d3", "e5f5",  THOUGHT_W, THOUGHT_B),
        ("d3c3", "f5e5",  THOUGHT_W, THOUGHT_B),
    ]

    episode_id = reset_env()

    last: StepResult | None = None
    for w_uci, b_uci, w_thought, b_thought in moves:
        step(episode_id, "make_move", thought=w_thought, uci_move=w_uci)
        last = step(episode_id, "make_move", thought=b_thought, uci_move=b_uci)
        if last.done:
            break

    # Move 38: Kd3 — the third repetition, triggering the draw
    sr = step(episode_id, "make_move", thought=THOUGHT_W, uci_move="c3d3")

    assert sr.done, "Threefold repetition should terminate the episode."
    result_lower = (sr.result_tag or "").lower()
    assert any(tok in result_lower for tok in ("draw", "repetition", "stale")), (
        f"expected a draw result_tag, got {sr.result_tag!r}"
    )
    b_final = float(sr.final_reward.get("black", 0.0))
    w_final = float(sr.final_reward.get("white", 0.0))
    assert abs(b_final - w_final) < 0.2, (
        f"draw rewards should be close; white={w_final:.4f}, black={b_final:.4f}"
    )
    return ("Scenario 8 (draw_by_repetition)", b_final, sr.done, sr.result_tag)


def scenario_9_white_wins_french_resign() -> tuple:
    """White wins by resignation — French Defence, Harshedhvbs vs aminehaidar.

    Source: chess.com/game/live/148124626404
    White: Harshedhvbs (2170)  Black: aminehaidar (2143)
    Result: 1-0 — Black resigned on move 40 after Rb6

    White plays a French Advance, wins the queenside, marches the king into
    g5 and creates two connected passed pawns that Black cannot stop.

    Expected:
      - episode terminates (done=True)
      - result_tag == "resign_black"
      - white final_reward >= 0.99 (R_MAX)
    """
    THOUGHT_W = (
        "Two connected passed pawns on d5 and g5 with my king actively "
        "supporting from g5. Black's rooks cannot stop both pawns. "
        "The position is completely winning — advancing methodically."
    )
    THOUGHT_B = (
        "White's king on g5 combined with connected passed pawns is completely "
        "decisive. I cannot stop both threats simultaneously. The position is "
        "objectively lost — evaluating last defensive tries."
    )

    # 40-move game: French Advance → White wins by resignation
    moves = [
        ("e2e4", "e7e6",  "Opening with e4, aiming for open positions.", "Playing the French Defence — solid and combative."),
        ("d2d4", "d7d5",  "Establishing the pawn centre.", "Challenging the centre — core of the French."),
        ("e4e5", "c7c5",  "French Advance Variation — closing the centre.", "Attacking the base of White's pawn chain."),
        ("c2c3", "d8b6",  "Defending d4 with c3, keeping structure solid.", "Queen to b6, attacking b2 for immediate counterplay."),
        ("g1f3", "c8d7",  "Developing the knight, supporting the centre.", "Bishop to d7, preparing queenside activity."),
        ("a2a3", "a7a5",  "Preparing queenside expansion.", "Stopping b4 — fixing the queenside structure."),
        ("f1e2", "d7b5",  "Developing the bishop, preparing to castle.", "Bishop to b5, eyeing e2."),
        ("e1g1", "b5e2",  "Castling for king safety.", "Capturing on e2 — trading the bishop pair."),
        ("d1e2", "c5c4",  "Recapturing on e2.", "Advancing c4 to gain space and restrict queenside."),
        ("a3a4", "b8d7",  "Stopping the bishop going to b4.", "Developing the knight, planning queenside play."),
        ("b1d2", "b6c6",  "Knight to d2, reinforcing e5.", "Queen to c6, maintaining pressure on e4."),
        ("f1e1", "d7b6",  "Doubling rooks on the e-file.", "Knight to b6, targeting a4 and d5."),
        ("e2d1", "g8e7",  "Retreating the queen, preparing Qb3.", "Knight to e7, preparing to reroute to d5."),
        ("b2b3", "c4b3",  "Opening the b-file.", "Capturing on b3 to open lines."),
        ("d1b3", "b6c4",  "Recapturing with the queen.", "Knight to c4, attacking the queen."),
        ("a1b1", "a8b8",  "Rook to b1, preparing queenside play.", "Preparing the rook for queenside action."),
        ("b3b5", "c6b5",  "Queen to b5, offering the trade.", "Trading queens — simplifying to a favourable endgame."),
        ("b1b5", "b7b6",  "Recapturing with the rook.", "Defending with b6, securing the queenside pawns."),
        ("d2c4", "d5c4",  "Winning the knight on c4.", "Recapturing — maintaining material equality."),
        ("f3d2", "e7d5",  "Knight to d2, planning Ne4.", "Knight to d5, centralising."),
        ("c1b2", "b8c8",  "Bishop to b2, supporting e5.", "Rook to c8, activating on the c-file."),
        ("d2e4", "f8e7",  "Knight to e4 — a powerful outpost.", "Bishop to e7, preparing to castle."),
        ("e1a1", "e8g8",  "Rook to a1, preparing Ba3.", "Castling for king safety."),
        ("b2a3", "e7a3",  "Bishop to a3, trading off Black's bishop.", "Capturing the bishop to simplify."),
        ("a1a3", "c8c6",  "Recapturing on a3.", "Rook to c6, defending along the rank."),
        ("g2g3", "f7f5",  "Preparing the king walk.", "Advancing f5, attacking the knight."),
        ("e4d6", "f5f4",  "Knight to d6, forking the rooks.", "Advancing f4, creating a passed pawn."),
        ("g1g2", "f4f3",  "King to g2 to avoid the f3 pawn.",  "f3+ winning a tempo with the check."),
        ("g2h3", "h7h5",  "King to h3, stepping away from the pawn.", "Advancing h5 to restrict the White king."),
        ("h3h4", "g7g6",  "King to h4, preparing the march to g5.", "Advancing g6, restricting the White king."),
        ("h4g5", "g8g7",  "King to g5 — the decisive king march!", "King to g7, trying to oppose."),
        ("d6e4", "c6f6",  "Knight to e4, attacking the f6 square.", "Rook to f6, defending along the rank."),
        ("g5h4", "g7h6",  "King to h4, sidestepping.", "King to h6, opposing the White king."),
        ("h4g4", "f6f4",  "King to g4, advancing toward the pawns.", "Rook to f4+, checking the king."),
        ("g3f4", "h5h4",  THOUGHT_W, THOUGHT_B),
        ("f4g5", "a5a4",  THOUGHT_W, THOUGHT_B),
        ("b5d5", "e6d5",  "Rook captures on d5, winning a pawn!", "Recapturing on d5."),
        ("g5f4", "c6c8",  THOUGHT_W, THOUGHT_B),
        ("f4g5", "h6g7",  THOUGHT_W, THOUGHT_B),
    ]

    episode_id = reset_env()

    last: StepResult | None = None
    for w_uci, b_uci, w_thought, b_thought in moves:
        sw = step(episode_id, "make_move", thought=w_thought, uci_move=w_uci)
        if sw.done:
            last = sw
            break
        sb = step(episode_id, "make_move", thought=b_thought, uci_move=b_uci)
        last = sb
        if sb.done:
            break

    # Move 40: White plays Rb6 — Black resigns immediately after
    if last is None or not last.done:
        step(episode_id, "make_move", thought=THOUGHT_W, uci_move="b5b6")
        sr = step(
            episode_id,
            "resign_game",
            thought=(
                "After Rb6 White threatens to advance both passed pawns with "
                "g6+ and d6. My rooks cannot stop both threats at once. "
                "The endgame is completely lost — resigning immediately."
            ),
        )
    else:
        sr = last

    assert sr.done, "resign_game should terminate the episode."
    assert sr.result_tag == "resign_black", (
        f"expected resign_black, got {sr.result_tag!r}"
    )
    w_final = float(sr.final_reward.get("white", 0.0))
    b_final = float(sr.final_reward.get("black", 0.0))
    assert w_final > b_final, (
        f"white should outscore the resigning black; "
        f"white={w_final:.4f}, black={b_final:.4f}"
    )
    assert w_final >= R_MAX, (
        f"white final reward should be >= {R_MAX} (target 0.99); got {w_final:.4f}"
    )
    return ("Scenario 9 (white_wins_french_resign)", w_final, sr.done, sr.result_tag)


def run_all_tests() -> None:
    results: list[tuple] = []

    scenario_fns = [
        scenario_1_clean_opening,
        scenario_2_fools_mate,
        scenario_3_illegal_dq,
        scenario_4_eval_abuse_dq,
        scenario_5_resign,
        scenario_6_ping_is_nonfatal,
        scenario_7_black_resignation_win,
        scenario_8_draw_by_repetition,
        scenario_9_white_wins_french_resign,
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
