# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Chess Arena OpenEnv environment.

A multi-agent chess environment that exposes six FastMCP tools (four useful,
two traps), scores each episode with a strict 0.50 / 0.10 / 0.15 / 0.24
reward decomposition that sums to a hard cap of 0.99, and uses a Stockfish
engine wrapper for hidden per-move accuracy scoring.

Every tool call requires a structured reasoning schema:
  - threat_analysis: str  — immediate dangers / board state evaluation
  - candidate_moves: list[str] — 2–3 UCI moves considered
  - justification: str — strategic reason for the selected action

Reward decomposition (sums to at most 0.99):

  - Outcome bucket        : <= 0.50  (win / draw / loss / resign / DQ)
  - Format bucket          : <= 0.10  (clean, schema-compliant tool calls)
  - Thought-quality bucket : <= 0.15  (deterministic structured-reasoning score)
  - Stockfish bucket       : <= 0.24  (centipawn closeness to Stockfish best)

Trap penalties subtract only from their OWN bucket, never cross-bucket:

  - `ping_humanhelper`                -> format : -0.3 each, does NOT end game
  - Malformed tool call / bad UCI     -> format : -0.05 each + ratio drop
  - `evaluate_position` (each call)   -> sf_acc : -0.04 each
  - 6th `evaluate_position` same side -> outcome: DQ (0 / 0.35) + episode ends
  - Illegal UCI on `make_move`        -> outcome: DQ (0 / 0.35) + episode ends

Episode reward is assigned (not accumulated) at finalisation, then clamped to
strictly (0.01, 0.99) to satisfy the hackathon "0 < score < 1" rule.
"""

from __future__ import annotations

import contextvars
import os
import re
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import chess
import chess.engine

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
)
from openenv.core.env_server.types import Action, Observation, State

try:
    from stockfish import Stockfish  # type: ignore
except ImportError:  # pragma: no cover - stockfish is optional at import time
    Stockfish = None  # type: ignore


# ---------------------------------------------------------------------------
# ContextVars: let FastMCP tool fns (no `self`) find their env + side to move.
# ---------------------------------------------------------------------------
_active_env: contextvars.ContextVar[Optional["ChessEnvironment"]] = contextvars.ContextVar(
    "_active_env", default=None
)
_current_episode_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_current_episode_id", default=None
)


# ===========================================================================
# Reward constants
# ===========================================================================

# Bucket caps (sum = 0.99, never exceeded by construction).
W_OUTCOME = 0.50       # max contribution from game outcome
W_FORMAT = 0.10        # max contribution from schema-compliant tool calls
W_THOUGHT_Q = 0.15     # max contribution from structured reasoning quality
W_SF_ACC = 0.24        # max contribution from Stockfish centipawn closeness

# Final reward is clamped to strictly (0, 1) per hackathon Phase 2 rules.
R_MIN, R_MAX = 0.01, 0.99

# Outcome bucket payouts (a win ALWAYS earns the full W_OUTCOME).
OUTCOME_WIN = 0.50
OUTCOME_DRAW = 0.25
OUTCOME_LOSS = 0.00
OUTCOME_RESIGN_WIN = 0.45   # clean resign-induced win: slightly under mate
OUTCOME_DQ_WIN = 0.35       # opponent got themselves DQed - partial credit

# Trap penalties - each one hits a SPECIFIC bucket, never cross-bucket.
EVAL_CALL_LIMIT = 5             # 6th call on the same side = DQ
ILLEGAL_FORMAT_PENALTY = 0.05   # each malformed call docks format bucket

# Chess concept keywords for strategic justification scoring.
CHESS_CONCEPTS = [
    "center", "develop", "pin", "fork", "defend", "attack", "threat",
    "king safety", "control", "space", "initiative", "structure",
]

# Capture-related synonyms for threat awareness scoring.
CAPTURE_SYNONYMS = ["capture", "takes", "exchange", "recapture", "trade", "material"]

# Stockfish scoring curve.
SF_DEPTH = 15
SF_BLUNDER_CP = 300   # cp loss that maps the unit score to 0.0
SF_MATE_CP = 10000    # mate eval substituted as a large cp value
# --- MISSING CONSTANTS FOR PHASE 2 MATH ---
STRIKES_BEFORE_DQ = 2
W_SF = 0.24
PING_BUCKET_PENALTY = 0.03
EVAL_BUCKET_PENALTY = 0.04
# ===========================================================================
# Stockfish wrapper
# ===========================================================================


def _resolve_stockfish_path() -> Optional[str]:
    """Look for the Stockfish binary in the expected locations.

    Order of preference (first hit wins):
      1. CHESS_STOCKFISH_PATH env var (if it names an existing file)
      2. ./engine/stockfish.exe  (Windows layout from the plan)
      3. ./engine/stockfish      (Linux symlink in Colab)
      4. /usr/games/stockfish    (apt-get install stockfish default)
      5. /usr/local/bin/stockfish
      6. `stockfish` on PATH  (checked via shutil.which)

    Returns None if no binary was found anywhere; the adapter then skips
    initialisation (instead of crashing during `Stockfish.__del__`).
    """
    import shutil

    override = os.environ.get("CHESS_STOCKFISH_PATH")
    if override and Path(override).is_file():
        return override
    here = Path(__file__).resolve().parent.parent
    candidates = [
        here / "engine" / "stockfish.exe",
        here / "engine" / "stockfish",
        Path("/usr/games/stockfish"),
        Path("/usr/local/bin/stockfish"),
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    resolved = shutil.which("stockfish")
    if resolved:
        return resolved
    return None


# ... existing code ...
class _StockfishAdapter:
    """Thin adapter around the `chess.engine` python wrapper."""

    def __init__(self, bin_path: str | None = None, *args, **kwargs) -> None:
        # BUG 2 FIX: Use _resolve_stockfish_path() so it works on Kaggle/Docker
        self.bin_path = bin_path or _resolve_stockfish_path() or "engine/stockfish.exe"
        self._engine = None
        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.bin_path)
        except Exception as e:
            print(f"[StockfishAdapter] Warning: Engine not loaded from {self.bin_path}. {e}")

    def close(self) -> None:
        """Explicitly shut down the Stockfish subprocess to avoid zombie processes."""
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None

    @property
    def ready(self) -> bool:
        return self._engine is not None

    def evaluate_cp(self, fen: str) -> int:
        if self._engine is None:
            return 0
        try:
            board = chess.Board(fen)
            limit = chess.engine.Limit(time=0.5)
            info = self._engine.analyse(board, limit=limit)
            score = info["score"].white() if board.turn == chess.WHITE else info["score"].black()
            
            SF_MATE_CP = 10000 
            if score.is_mate():
                mate_in = score.mate()
                sign = 1 if mate_in > 0 else -1
                return sign * SF_MATE_CP
            
            return score.score() or 0
        except Exception:
            return 0

    def describe(self, fen: str) -> str:
        if self._engine is None:
            return "Stockfish engine unavailable in this runtime."
        try:
            board = chess.Board(fen)
            limit = chess.engine.Limit(time=0.5)
            info = self._engine.analyse(board, limit=limit)
            score = info["score"].white()
            
            best_move = "none"
            if "pv" in info and info["pv"]:
                best_move = info["pv"][0].uci()
                
            eval_type = "mate" if score.is_mate() else "cp"
            value = score.mate() if score.is_mate() else score.score()
            
            return f"eval_type={eval_type} value={value or 0} best_move={best_move}"
        except Exception as e:
            return f"Stockfish error: {e}"

    def score_move(self, board: chess.Board, move: chess.Move) -> float:
        if self._engine is None:
            return 0.5 

        try:
            limit = chess.engine.Limit(time=0.5) 
            
            info_before = self._engine.analyse(board, limit=limit)
            best_eval = info_before["score"].white()

            board.push(move)
            info_after = self._engine.analyse(board, limit=limit)
            board.pop()
            
            actual_eval = info_after["score"].white()

            loss = best_eval.score(mate_score=10000) - actual_eval.score(mate_score=10000)
            loss = max(0, loss)

            scaled_score = max(0.0, 1.0 - (loss / 500.0))
            return scaled_score

        except Exception as e:
            print(f"[StockfishAdapter] Evaluation failed: {e}")
            return 0.5

# ===========================================================================
# Helpers
# ===========================================================================

_UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")


def _clamp(x: float) -> float:
    """Strict (0.01, 0.99) clamp used everywhere we return a reward."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return R_MIN
    if v != v:  # NaN guard
        return R_MIN
    return max(R_MIN, min(R_MAX, v))


def _other(color: str) -> str:
    return "black" if color == "white" else "white"


# ===========================================================================
# ChessEnvironment
# ===========================================================================


class ChessEnvironment(MCPEnvironment):
    """MCP-based chess arena. One HTTP session = one game.

    Two agents share the same environment instance and alternate turns; the
    env tracks which color is currently to move and attributes every tool call
    to that color. `inference.py` is responsible for routing outputs from the
    right model to the right side.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    _instances: dict[str, "ChessEnvironment"] = {}
    _latest_instance: Optional["ChessEnvironment"] = None

    # -----------------------------------------------------------------
    # Construction / tool registration
    # -----------------------------------------------------------------

    def __init__(self) -> None:
        self._init_fresh_state()
        mcp = FastMCP("chess_arena")
        self._register_tools(mcp)
        super().__init__(mcp)

    def _init_fresh_state(self) -> None:
        """Reset all per-episode state. Also called from `reset()`."""
        # BUG 20 & 7 FIX: Safely kill the old Stockfish process before making a new one
        if hasattr(self, "_stockfish") and self._stockfish is not None:
            self._stockfish.close()

        self.board: chess.Board = chess.Board()
        self._last_turn_flipped: bool = False
        self.turn_color: str = "white"
        self._stockfish: _StockfishAdapter = _StockfishAdapter(depth=SF_DEPTH)

        # Outcome + buckets (four-way: outcome / format / thought_q / sf_acc)
        self.bucket: dict[str, dict[str, float]] = {
            "white": {"outcome": 0.0, "format": 0.0, "thought_q": 0.0, "sf_acc": 0.0},
            "black": {"outcome": 0.0, "format": 0.0, "thought_q": 0.0, "sf_acc": 0.0},
        }
        self.final_reward: dict[str, float] = {"white": R_MIN, "black": R_MIN}

        # Format-bucket tracking (schema compliance)
        self.tool_calls_clean: dict[str, int] = {"white": 0, "black": 0}
        self.tool_calls_total: dict[str, int] = {"white": 0, "black": 0}
        self.dirty_penalty_accum: dict[str, float] = {"white": 0.0, "black": 0.0}

        # Thought-quality tracking (structured reasoning scores per call)
        self.thought_quality_scores: dict[str, list[float]] = {"white": [], "black": []}

        # Stockfish accuracy tracking
        self.sf_move_scores: dict[str, list[float]] = {"white": [], "black": []}

        # Trap state
        self.eval_calls: dict[str, int] = {"white": 0, "black": 0}
        self.ping_count: dict[str, int] = {"white": 0, "black": 0}

        # Two-strike illegal move rule: first illegal → penalty, second → DQ.
        self.illegal_move_count: dict[str, int] = {"white": 0, "black": 0}

        # Episode bookkeeping
        self.move_history: list[dict[str, Any]] = []
        self.tool_log: list[dict[str, Any]] = []
        self.done: bool = False
        self.result: Optional[str] = None   # e.g. "checkmate_white", "dq_black", "draw_50m"
        self._state: State = State(episode_id=str(uuid4()), step_count=0)

    # -----------------------------------------------------------------
    # MCP tools  (exposed to the LLM)
    # -----------------------------------------------------------------

    def _register_tools(self, mcp: FastMCP) -> None:
        """Register the 6 chess arena tools with FastMCP."""

        def _env() -> "ChessEnvironment":
            env = _active_env.get() or ChessEnvironment._latest_instance
            if env is None:
                raise RuntimeError("No active ChessEnvironment instance")
            return env

        @mcp.tool
        def analyze_board(
            threat_analysis: str,
            candidate_moves: list[str],
            justification: str,
        ) -> str:
            """Return the current board as FEN, plus turn / check / move number.

            Always a *clean* tool call. Use this before `make_move` to confirm
            whose turn it is and detect special states (check, en-passant).

            Args:
                threat_analysis: Evaluate immediate dangers or board state.
                candidate_moves: 2-3 UCI moves you are considering.
                justification:   Strategic reason for choosing this action.
            """
            env = _env()
            env._record_tool_call(
                "analyze_board",
                threat_analysis, candidate_moves, justification,
                clean=True,
            )
            board = env.board
            halfmoves_to_50 = max(0, 100 - board.halfmove_clock)
            can_repeat = board.can_claim_threefold_repetition()
            can_claim_now = board.can_claim_draw()
            info = {
                "fen": board.fen(),
                "turn": "white" if board.turn == chess.WHITE else "black",
                "in_check": board.is_check(),
                "move_number": board.fullmove_number,
                "halfmove_clock": board.halfmove_clock,
                "halfmoves_to_50_move_draw": halfmoves_to_50,
                "draw_imminent": can_repeat,
                "can_claim_draw_now": can_claim_now,
            }
            return (
                f"fen={info['fen']}\nturn={info['turn']}\nin_check={info['in_check']}\n"
                f"move_number={info['move_number']}\nhalfmove_clock={info['halfmove_clock']}\n"
                f"halfmoves_to_50_move_draw={info['halfmoves_to_50_move_draw']}\n"
                f"draw_imminent={info['draw_imminent']}\n"
                f"can_claim_draw_now={info['can_claim_draw_now']}"
            )


        @mcp.tool
        def list_legal_moves(
            threat_analysis: str,
            candidate_moves: list[str],
            justification: str,
        ) -> str:
            """Return all legal moves for the current position in UCI format.

            Args:
                threat_analysis: Evaluate immediate dangers or board state.
                candidate_moves: 2-3 UCI moves you are considering.
                justification:   Strategic reason for choosing this action.
            """
            env = _env()
            env._record_tool_call(
                "list_legal_moves",
                threat_analysis, candidate_moves, justification,
                clean=True,
            )
            moves = [m.uci() for m in env.board.legal_moves]
            return "legal_moves=" + ",".join(moves) if moves else "legal_moves=(none)"

        @mcp.tool
        def make_move(
            threat_analysis: str,
            candidate_moves: list[str],
            justification: str,
            uci_move: str,
        ) -> str:
            """Submit a UCI move (e.g. 'e2e4', 'e7e8q'). Illegal UCI ends the game.

            Args:
                threat_analysis: Evaluate immediate dangers or board state.
                candidate_moves: 2-3 UCI moves you considered (must include uci_move).
                justification:   Strategic reason for selecting this specific move.
                uci_move:        The move to execute in UCI notation.
            """
            env = _env()
            return env._apply_make_move(
                threat_analysis, candidate_moves, justification, uci_move
            )

        @mcp.tool
        def resign_game(
            threat_analysis: str,
            candidate_moves: list[str],
            justification: str,
        ) -> str:
            """Resign the current game. Opponent wins with partial credit.

            Args:
                threat_analysis: Evaluate the losing board state.
                candidate_moves: 2-3 UCI moves you considered before resigning.
                justification:   Strategic reason for resigning.
            """
            env = _env()
            env._record_tool_call(
                "resign_game",
                threat_analysis, candidate_moves, justification,
                clean=True,
            )
            loser = env.turn_color
            env._finalize_episode(result=f"resign_{loser}")
            return f"Game over: {loser} resigned."

        @mcp.tool
        def evaluate_position(
            threat_analysis: str,
            candidate_moves: list[str],
            justification: str,
        ) -> str:
            """Return a Stockfish evaluation string.

            TRAP: each call docks `sf_acc` by 0.04. The 6th call on a single
            side DQs that side (outcome = 0, opponent = 0.35).

            Args:
                threat_analysis: Evaluate immediate dangers or board state.
                candidate_moves: 2-3 UCI moves you are considering.
                justification:   Why you need Stockfish evaluation here.
            """
            env = _env()
            env._record_tool_call(
                "evaluate_position",
                threat_analysis, candidate_moves, justification,
                clean=True,
            )
            color = env.turn_color
            env.eval_calls[color] += 1

            # 6th call on one side -> DQ
            if env.eval_calls[color] > EVAL_CALL_LIMIT:
                env._finalize_episode(result=f"dq_eval_abuse_{color}")
                return (
                    f"DISQUALIFIED: {color} called evaluate_position "
                    f"{env.eval_calls[color]} times (limit {EVAL_CALL_LIMIT})."
                )

            return env._stockfish.describe(env.board.fen())

        @mcp.tool
        def ping_humanhelper(
            threat_analysis: str,
            candidate_moves: list[str],
            justification: str,
            reason: str,
        ) -> str:
            """Ask a human for help.

            TRAP (non-fatal): each call docks the format bucket by 0.03. The
            game does NOT terminate. The response is deliberately unhelpful.

            Args:
                threat_analysis: Evaluate immediate dangers or board state.
                candidate_moves: 2-3 UCI moves you are considering.
                justification:   Why you are asking for help.
                reason:          Specific question for the human helper.
            """
            env = _env()
            env._record_tool_call(
                "ping_humanhelper",
                threat_analysis, candidate_moves, justification,
                clean=True,
            )
            env.ping_count[env.turn_color] += 1
            return "Human helper is currently unavailable."

    # -----------------------------------------------------------------
    # Move application & Stockfish-accuracy scoring
    # -----------------------------------------------------------------

    def _handle_illegal_move(
        self,
        color: str,
        threat_analysis: str,
        candidate_moves: list[str],
        justification: str,
        uci_move: str,
        reason: str,
    ) -> str:
        """Apply two-strike illegal move rule."""

        # BUG 8 FIX: Mark the move as 'clean=False' so the format ratio drops.
        self._record_tool_call(
            "make_move",
            threat_analysis, candidate_moves, justification,
            clean=False, uci_move=uci_move,
        )
        self.illegal_move_count[color] += 1
        self._last_turn_flipped = False

        msg = f"ILLEGAL MOVE (strike {self.illegal_move_count[color]}/{STRIKES_BEFORE_DQ}): {reason} "
        if self.illegal_move_count[color] == 1:
            # BUG 8 FIX: Apply the documented -0.05 penalty instead of -0.40.
            self.dirty_penalty_accum[color] += abs(ILLEGAL_FORMAT_PENALTY)
            msg += f"Penalty applied (-{abs(ILLEGAL_FORMAT_PENALTY)}). You have one more chance."
            return msg
        else:
            msg += "Disqualified."
            self._finalize_episode(result=f"dq_illegal_{color}")
            return msg
    def _apply_make_move(
        self,
        threat_analysis: str,
        candidate_moves: list[str],
        justification: str,
        uci_move: str,
    ) -> str:
        """Validate the UCI, score it with Stockfish, push it on the board."""
        color = self.turn_color

        # Format validation first - malformed UCI is a schema hit AND an illegal move.
        if not isinstance(uci_move, str) or not _UCI_RE.match(uci_move.strip()):
            return self._handle_illegal_move(
                color, threat_analysis, candidate_moves, justification, uci_move,
                reason=f"'{uci_move}' is not a valid UCI string.",
            )

        uci = uci_move.strip()

        # Bug 4 fix: auto-promote bare pawn-to-8th-rank moves to queen.
        # python-chess requires the promotion suffix (e.g. 'e7e8q'), but LLMs
        # frequently omit it.  Rather than burning a strike for a recoverable
        # omission, we silently append 'q' when the destination rank matches
        # the promoting rank for the moving side, and there is no suffix yet.
        if len(uci) == 4:
            dest_rank = uci[3]
            try:
                _probe = chess.Move.from_uci(uci + "q")
                _promo_rank = "8" if self.board.turn == chess.WHITE else "1"
                if dest_rank == _promo_rank and _probe in self.board.legal_moves:
                    uci = uci + "q"
            except Exception:
                pass

        try:
            move = chess.Move.from_uci(uci)
        except Exception:
            return self._handle_illegal_move(
                color, threat_analysis, candidate_moves, justification, uci,
                reason=f"Could not parse UCI '{uci}'.",
            )

        if move not in self.board.legal_moves:
            return self._handle_illegal_move(
                color, threat_analysis, candidate_moves, justification, uci,
                reason=f"'{uci}' is not a legal move in this position.",
            )

        # It's a clean, legal move.
        self._record_tool_call(
            "make_move",
            threat_analysis, candidate_moves, justification,
            clean=True, uci_move=uci,
        )

        # Stockfish accuracy: cp loss = (best_eval - eval_after_this_move).
        fen_before = self.board.fen()
        best_cp = self._stockfish.evaluate_cp(fen_before)
        was_capture = self.board.is_capture(move)

        self.board.push(move)
        fen_after = self.board.fen()
        raw_after = self._stockfish.evaluate_cp(fen_after)
        after_cp = -raw_after

        cp_loss = max(0, best_cp - after_cp)
        if self._stockfish.ready:
            move_score = max(0.0, 1.0 - (cp_loss / float(SF_BLUNDER_CP)))
        else:
            move_score = 0.5
        self.sf_move_scores[color].append(move_score)

        self.move_history.append(
            {
                "color": color,
                "uci": uci,
                "cp_loss": cp_loss,
                "move_score": round(move_score, 3),
                "was_capture": was_capture,
            }
        )

        # Did this move end the game?
        outcome = self.board.outcome(claim_draw=True)
        if outcome is not None:
            if outcome.winner is True:
                result = "checkmate_white"
            elif outcome.winner is False:
                result = "checkmate_black"
            else:
                # Stalemate, insufficient material, 50-move, 3-fold.
                result = f"draw_{outcome.termination.name.lower()}"
            self._finalize_episode(result=result)
            return f"Move {uci} ended the game: {result} (cp_loss={cp_loss})."

        # Otherwise flip side to move.
        self.turn_color = _other(color)
        self._last_turn_flipped = True
        return (
            f"Move {uci} applied. turn={self.turn_color} "
            f"cp_loss={cp_loss} move_score={move_score:.2f}"
        )

    def close(self) -> None:
        """BUG 7 FIX: Shut down Stockfish on env close to prevent leaks."""
        if hasattr(self, "_stockfish") and self._stockfish is not None:
            self._stockfish.close()
        super().close()

    # -----------------------------------------------------------------
    # Format-bucket accounting + structured thought evaluation
    # -----------------------------------------------------------------

    def _record_tool_call(
        self,
        tool_name: str,
        threat_analysis: str,
        candidate_moves: list[str],
        justification: str,
        *,
        clean: bool,
        uci_move: Optional[str] = None,
    ) -> None:
        """Count a tool call against the active color.

        Schema compliance gate: all three structured fields must be non-empty
        strings (candidate_moves must also be a non-empty list) for the call
        to be considered clean from a *format* perspective. Any failure marks
        the call dirty and accumulates the ILLEGAL_FORMAT_PENALTY.

        After the compliance gate, `_evaluate_thought_quality` scores the
        structured reasoning and accumulates to `thought_quality_scores`.
        """
        color = self.turn_color
        self.tool_calls_total[color] += 1

        # --- Schema compliance check (replaces old empty-thought check) ---
        ta_ok = isinstance(threat_analysis, str) and bool(threat_analysis.strip())
        cm_ok = (
            isinstance(candidate_moves, list)
            and len(candidate_moves) >= 1
            and any(isinstance(m, str) and m.strip() for m in candidate_moves)
        )
        ju_ok = isinstance(justification, str) and bool(justification.strip())

        if not (ta_ok and cm_ok and ju_ok):
            clean = False

        if clean:
            self.tool_calls_clean[color] += 1
        else:
            self.dirty_penalty_accum[color] += ILLEGAL_FORMAT_PENALTY

        # --- Structured thought-quality scoring (always runs) ---
        tq_score = self._evaluate_thought_quality(
            color, tool_name,
            threat_analysis, candidate_moves, justification,
            uci_move=uci_move,
        )
        self.thought_quality_scores[color].append(tq_score)

        self._state.step_count += 1
        self.tool_log.append(
            {
                "color": color,
                "tool": tool_name,
                "clean": clean,
                "threat_analysis": (threat_analysis or "")[:200],
                "candidate_moves": (candidate_moves or [])[:5],
                "justification": (justification or "")[:200],
                "thought_quality": round(tq_score, 4),
            }
        )

    def _evaluate_thought_quality(
        self,
        color: str,
        tool_name: str,
        threat_analysis: str,
        candidate_moves: list[str],
        justification: str,
        *,
        uci_move: Optional[str] = None,
    ) -> float:
        """Deterministic heuristic scoring of structured reasoning (max 0.15)."""
        import re
        score = 0.0
        ta_lower = (threat_analysis or "").lower()
        ju_lower = (justification or "").lower()
        cm_list = [str(m).strip() for m in (candidate_moves or []) if m is not None and str(m).strip()]

        # --- Sub-score 1: Threat Awareness ---
        if self.board.is_check():
            if "check" in ta_lower or "king" in ta_lower:
                score += 0.05
        elif self.move_history and self.move_history[-1].get("was_capture", False):
            if any(syn in ta_lower for syn in CAPTURE_SYNONYMS):
                score += 0.05
        else:
            if len(ta_lower.split()) > 5:
                score += 0.05

        # --- Sub-score 2: Action Tracing ---
        if tool_name == "make_move" and uci_move:
            # BUG 15 FIX: Stop substring cheating. Check for EXACT matches only.
            if any(uci_move == m for m in cm_list):
                score += 0.05
        else:
            # For non-make_move tools, require >=2 candidate moves listed.
            if len(cm_list) >= 2:
                score += 0.05

        # --- Sub-score 3: Strategic Justification ---
        # BUG 16 FIX: Use regex word boundaries (\b) so "centered" doesn't match "center".
        for concept in CHESS_CONCEPTS:
            if re.search(rf"\b{concept}\b", ju_lower):
                score += 0.05
                break

        return score

    def record_malformed_call(self, color: Optional[str] = None) -> None:
        """Called by `inference.py` when the model emits invalid JSON.

        Hits the format bucket only: game continues, no outcome change.
        """
        target = color or self.turn_color
        self.tool_calls_total[target] += 1
        self.dirty_penalty_accum[target] += ILLEGAL_FORMAT_PENALTY
        self.thought_quality_scores[target].append(0.0)  # no reasoning = zero score
        self.tool_log.append({"color": target, "tool": "(malformed)", "clean": False})

    # -----------------------------------------------------------------
    # Bucket computation
    # -----------------------------------------------------------------

    def _compute_format_score(self, color: str) -> float:
        total = self.tool_calls_total[color]
        if total == 0:
            return 0.0

        # BUG 8 FIX: Clean ratio drops when clean=False is passed.
        ratio = self.tool_calls_clean[color] / total
        base = W_FORMAT * ratio

        # BUG 9 FIX: Clamp the ping penalty so it doesn't nuke the entire score below 0
        ping_penalty = self.ping_count[color] * PING_BUCKET_PENALTY
        format_score = max(0.0, base - ping_penalty)

        return format_score

    def _compute_thought_quality(self, color: str) -> float:
        """Bucket 3 (Thought Quality, max W_THOUGHT_Q=0.15): avg per-call score."""
        scores = self.thought_quality_scores[color]
        avg = (sum(scores) / len(scores)) if scores else 0.0
        # avg is already in [0, 0.15] since each call returns at most 0.15
        # and we scale by W_THOUGHT_Q (the per-call max IS W_THOUGHT_Q = 0.15).
        return max(0.0, min(W_THOUGHT_Q, avg))

    def _compute_sf_acc(self, color: str) -> float:
        # BUG 19 FIX: Uses self.result, not self._result_tag!
        if self.result == f"dq_eval_abuse_{color}":
            return 0.0

        scores = self.sf_move_scores[color]
        if not scores:
            return 0.0

        avg_acc = sum(scores) / len(scores)
        base = avg_acc * W_SF
        penalty = self.eval_calls[color] * EVAL_BUCKET_PENALTY
        return max(0.0, base - penalty)

    def _finalize_outcome(self, result: str) -> None:
        """Bucket 1: assign outcome by terminal state.

        `result` encodes the terminal reason:
          - checkmate_white / checkmate_black
          - draw_*  (any draw subtype)
          - resign_white / resign_black
          - dq_illegal_<color>
          - dq_eval_abuse_<color>
        """
        self.result = result

        if result.startswith("checkmate_"):
            winner = result.split("_", 1)[1]
            self.bucket[winner]["outcome"] = OUTCOME_WIN
            self.bucket[_other(winner)]["outcome"] = OUTCOME_LOSS
            return

        if result.startswith("draw_"):
            self.bucket["white"]["outcome"] = OUTCOME_DRAW
            self.bucket["black"]["outcome"] = OUTCOME_DRAW
            return

        if result.startswith("resign_"):
            loser = result.split("_", 1)[1]
            winner = _other(loser)
            self.bucket[winner]["outcome"] = OUTCOME_RESIGN_WIN
            self.bucket[loser]["outcome"] = OUTCOME_LOSS
            return

        if result.startswith("dq_"):
            offender = result.rsplit("_", 1)[1]
            winner = _other(offender)
            self.bucket[winner]["outcome"] = OUTCOME_DQ_WIN
            self.bucket[offender]["outcome"] = OUTCOME_LOSS
            return

        # Fallback: treat unknown result as a draw.
        self.bucket["white"]["outcome"] = OUTCOME_DRAW
        self.bucket["black"]["outcome"] = OUTCOME_DRAW

    def _finalize_episode(self, *, result: str) -> None:
        """Close the episode, fill all four buckets, clamp final rewards."""
        if self.done:
            return
        self.done = True

        self._finalize_outcome(result)

        for color in ("white", "black"):
            b = self.bucket[color]
            b["format"]   = self._compute_format_score(color)
            b["thought_q"] = self._compute_thought_quality(color)
            b["sf_acc"]   = self._compute_sf_acc(color)
            total = b["outcome"] + b["format"] + b["thought_q"] + b["sf_acc"]
            self.final_reward[color] = _clamp(total)

    def _preview_reward(self, color: str) -> float:
        """Monotone-ish in-game preview reward for `Observation.reward`.

        Computes a conservative estimate of the final reward using:
          - outcome so far (0 until episode terminates)
          - format score using current clean/total ratio + live penalties
          - thought quality using current per-call average
          - sf_acc using current per-move avg + live eval penalties
        Always clamped to (0.01, 0.99).
        """
        outcome_so_far = self.bucket[color]["outcome"]
        total = (
            outcome_so_far
            + self._compute_format_score(color)
            + self._compute_thought_quality(color)
            + self._compute_sf_acc(color)
        )
        return _clamp(total)

    # -----------------------------------------------------------------
    # Environment API: reset / step / state
    # -----------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Start a new game. Returns initial observation with preview reward."""
        # Bug 1 fix: explicitly shut down the old Stockfish subprocess before
        # reinitialising state, so tournament runs don't accumulate zombie processes.
        if hasattr(self, "_stockfish") and self._stockfish is not None:
            self._stockfish.close()
        self._init_fresh_state()
        if seed is not None:
            # python-chess doesn't use seeds, but we expose it for reproducible
            # starting positions in the future.
            pass

        options = kwargs.get("options") or {}
        start_fen = options.get("fen") or kwargs.get("fen")
        if start_fen:
            try:
                self.board = chess.Board(start_fen)
                self.turn_color = "white" if self.board.turn == chess.WHITE else "black"
            except Exception as _fen_err:
                # Bug 7 fix: log the FEN parse failure so it's discoverable.
                import sys
                print(
                    f"[ChessEnvironment] WARNING: Could not parse FEN '{start_fen}': {_fen_err}. "
                    "Falling back to the default starting position.",
                    file=sys.stderr,
                )
                self.board = chess.Board()

        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        ChessEnvironment._instances[self._state.episode_id] = self
        ChessEnvironment._latest_instance = self

        return Observation(
            done=False,
            reward=_clamp(0.0),
            metadata={
                "turn": self.turn_color,
                "fen": self.board.fen(),
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Pass-through for non-MCP actions. Nothing custom is expected."""
        return Observation(
            done=self.done,
            reward=_clamp(self._preview_reward(self.turn_color)),
            metadata={"turn": self.turn_color, "fen": self.board.fen()},
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Resolve the right env instance from episode_id, then delegate."""
        active = self
        req_ep = kwargs.get("episode_id") or _current_episode_id.get()
        if req_ep and req_ep in ChessEnvironment._instances:
            active = ChessEnvironment._instances[req_ep]
        elif req_ep:
            # Bug 2 fix: do NOT fall back to _latest_instance when a specific
            # episode_id was provided but is unrecognised.  Silently routing to
            # the most-recently-started game would corrupt concurrent sessions —
            # Game A's White player could make a move on Game B's board.
            # Raise immediately so the caller gets a clean 400/500 rather than
            # silent state corruption.
            raise KeyError(
                f"Unknown episode_id {req_ep!r}. "
                "Call /reset first to obtain a valid episode."
            )
        elif not req_ep:
            # No episode_id at all: only safe to fall back to latest_instance
            # in a single-game context (e.g. the inference.py self-play loop
            # which calls /reset right before /step).  We keep this path for
            # backwards compatibility with that caller.
            if ChessEnvironment._latest_instance is not None:
                active = ChessEnvironment._latest_instance

        token = _active_env.set(active)
        try:
            if isinstance(action, ListToolsAction):
                return super().step(action, timeout_s=timeout_s, **kwargs)

            obs = super().step(action, timeout_s=timeout_s, **kwargs)

            if isinstance(action, CallToolAction) and isinstance(obs, CallToolObservation):
                # After each tool call, decorate the observation with live
                # bounds + the correct done flag so `inference.py` can stop.
                obs.done = active.done
                if active.done:
                    # Emit the final reward for the color that just acted.
                    # The agent loop will take the per-side scores from state.
                    reward_color = _current_actor_for_observation(active, action)
                    obs.reward = _clamp(active.final_reward.get(reward_color, R_MIN))
                    ChessEnvironment._instances.pop(active._state.episode_id, None)
                else:
                    # In-game preview: use the color that JUST acted. Tools like
                    # `make_move` flip `turn_color` before returning, so we look
                    # at the previous actor.
                    reward_color = _current_actor_for_observation(active, action)
                    obs.reward = _clamp(active._preview_reward(reward_color))

                # Build per-color debug payload. We attach it in TWO places:
                #   (a) `obs.metadata` - visible to in-process callers
                #       (python smoke tests, training notebook importing the
                #       module directly).
                #   (b) inside `obs.result.structured_content["openenv"]` -
                #       visible over HTTP, since the framework's
                #       `serialize_observation` drops the top-level metadata
                #       field. This is what `inference.py` reads when talking
                #       to the FastAPI server.
                debug_payload: dict[str, Any] = {
                    "turn": active.turn_color,
                    "fen": active.board.fen(),
                    "result": active.result,
                    "done": active.done,
                    "bucket": {
                        c: {k: round(v, 4) for k, v in active.bucket[c].items()}
                        for c in ("white", "black")
                    },
                    "final_reward": {
                        c: _clamp(active.final_reward[c]) for c in ("white", "black")
                    },
                    "eval_calls": dict(active.eval_calls),
                    "ping_count": dict(active.ping_count),
                    "tool_calls": {
                        "clean": dict(active.tool_calls_clean),
                        "total": dict(active.tool_calls_total),
                    },
                    "thought_quality": {
                        c: round(
                            sum(active.thought_quality_scores[c]) / len(active.thought_quality_scores[c])
                            if active.thought_quality_scores[c] else 0.0,
                            4,
                        )
                        for c in ("white", "black")
                    },
                }

                md = dict(getattr(obs, "metadata", {}) or {})
                md.update(debug_payload)
                obs.metadata = md

                # FastMCP wraps the tool's string return into a structured
                # object with shape { content: [...], structured_content: {...},
                # data: ... }. Pin our debug payload under a namespaced key so
                # HTTP clients can still see it.
                _inject_openenv_payload(obs, debug_payload)
            return obs
        finally:
            _active_env.reset(token)

    @property
    def state(self) -> State:
        return self._state

    def snapshot(self) -> dict[str, Any]:
        """Lightweight snapshot for logging / visualisation."""
        return {
            "fen": self.board.fen(),
            "turn": self.turn_color,
            "done": self.done,
            "result": self.result,
            "bucket": {
                c: {k: round(v, 4) for k, v in self.bucket[c].items()}
                for c in ("white", "black")
            },
            "final_reward": {c: _clamp(self.final_reward[c]) for c in ("white", "black")},
            "eval_calls": dict(self.eval_calls),
            "ping_count": dict(self.ping_count),
            "tool_calls": {
                "clean": dict(self.tool_calls_clean),
                "total": dict(self.tool_calls_total),
            },
            "move_history": list(self.move_history),
        }


def _inject_openenv_payload(obs: CallToolObservation, payload: dict[str, Any]) -> None:
    """Stuff our per-color debug payload into the tool result structure."""
    res = obs.result

    if res is None:
        return

    # If it's a raw dictionary, modify it directly
    if isinstance(res, dict):
        sc = res.get("structured_content")
        if not isinstance(sc, dict):
            sc = {}
        else:
            sc = dict(sc)
        sc["openenv"] = payload
        res["structured_content"] = sc
        return

    # KEY FIX: If it's a Pydantic model/dataclass, mutate its attribute directly!
    # Do NOT overwrite obs.result with a dictionary.
    if hasattr(res, "structured_content"):
        sc = getattr(res, "structured_content")
        if not isinstance(sc, dict):
            sc = {}
        else:
            sc = dict(sc)
        sc["openenv"] = payload
        
        try:
            setattr(res, "structured_content", sc)
        except Exception:
            pass



def _current_actor_for_observation(
    env: "ChessEnvironment", action: CallToolAction
) -> str:
    """Figure out which color should receive the reward on this observation.

    Uses the _last_turn_flipped flag set by _apply_make_move (True) or
    _handle_illegal_move (False) to determine whether turn_color was flipped.
    If flipped, the actor is _other(turn_color); otherwise turn_color itself.
    """
    if env.done:
        return env.turn_color
    if getattr(action, "tool_name", None) == "make_move":
        if env._last_turn_flipped:
            return _other(env.turn_color)
    return env.turn_color
