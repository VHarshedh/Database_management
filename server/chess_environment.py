# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Chess Arena OpenEnv environment.

A multi-agent chess environment that exposes six FastMCP tools (four useful,
two traps), scores each episode with a strict 0.50 / 0.25 / 0.24 reward
decomposition that sums to a hard cap of 0.99, and uses a Stockfish engine
wrapper for hidden per-move accuracy scoring.

Reward decomposition (sums to at most 0.99):

  - Outcome bucket      : <= 0.50  (win / draw / loss / resign / DQ)
  - Tool-accuracy bucket: <= 0.25  (clean parseable tool calls)
  - Stockfish bucket    : <= 0.24  (centipawn closeness to Stockfish best)

Trap penalties subtract only from their OWN bucket, never cross-bucket:

  - `ping_humanhelper`                -> tool_acc: -0.03 each, does NOT end game
  - Malformed tool call / bad UCI     -> tool_acc: -0.05 each + ratio drop
  - `evaluate_position` (each call)   -> sf_acc  : -0.04 each
  - 6th `evaluate_position` same side -> outcome : DQ (0 / 0.35) + episode ends
  - Illegal UCI on `make_move`        -> outcome : DQ (0 / 0.35) + episode ends

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
W_TOOL_ACC = 0.25      # max contribution from clean tool-call ratio
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
EVAL_BUCKET_PENALTY = 0.04      # each eval call docks sf_acc
PING_BUCKET_PENALTY = 0.03      # each ping docks tool_acc
ILLEGAL_FORMAT_PENALTY = 0.05   # each malformed call docks tool_acc

# Stockfish scoring curve.
SF_DEPTH = 10
SF_BLUNDER_CP = 300   # cp loss that maps the unit score to 0.0
SF_MATE_CP = 10000    # mate eval substituted as a large cp value


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


class _StockfishAdapter:
    """Thin adapter around the `stockfish` python wrapper.

    Kept optional so the env can still be imported / unit-tested on machines
    where the Stockfish binary isn't available - in that degraded mode,
    Stockfish accuracy scoring falls back to a neutral score of 0.5 per move.
    """

    def __init__(self, depth: int = SF_DEPTH) -> None:
        self._engine = None
        if Stockfish is None:
            return
        path = _resolve_stockfish_path()
        if not path:
            # No Stockfish binary available on this runtime; fall back to
            # neutral per-move scoring (0.5) without trying to spawn the
            # engine - this avoids a noisy `Stockfish.__del__` traceback
            # when the `stockfish` library's Popen fails.
            return
        try:
            self._engine = Stockfish(path=path, depth=depth)
        except Exception:
            self._engine = None

    @property
    def ready(self) -> bool:
        return self._engine is not None

    def evaluate_cp(self, fen: str) -> int:
        """Return cp eval from the perspective of the side to move.

        Returns 0 if the engine is unavailable. Mate is substituted as +/- SF_MATE_CP.
        """
        if self._engine is None:
            return 0
        try:
            self._engine.set_fen_position(fen)
            ev = self._engine.get_evaluation() or {}
            if ev.get("type") == "mate":
                mate_in = int(ev.get("value", 0))
                sign = 1 if mate_in > 0 else -1
                return sign * SF_MATE_CP
            return int(ev.get("value", 0))
        except Exception:
            return 0

    def describe(self, fen: str) -> str:
        """Human-readable eval summary for `evaluate_position`."""
        if self._engine is None:
            return "Stockfish engine unavailable in this runtime."
        try:
            self._engine.set_fen_position(fen)
            ev = self._engine.get_evaluation() or {}
            best = self._engine.get_best_move() or "none"
            return (
                f"eval_type={ev.get('type', 'cp')} "
                f"value={ev.get('value', 0)} "
                f"best_move={best}"
            )
        except Exception as e:
            return f"Stockfish error: {e}"


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
        self.board: chess.Board = chess.Board()
        self.turn_color: str = "white"
        self._stockfish: _StockfishAdapter = _StockfishAdapter(depth=SF_DEPTH)

        # Outcome + buckets
        self.bucket: dict[str, dict[str, float]] = {
            "white": {"outcome": 0.0, "tool_acc": 0.0, "sf_acc": 0.0},
            "black": {"outcome": 0.0, "tool_acc": 0.0, "sf_acc": 0.0},
        }
        self.final_reward: dict[str, float] = {"white": R_MIN, "black": R_MIN}

        # Tool accuracy tracking
        self.tool_calls_clean: dict[str, int] = {"white": 0, "black": 0}
        self.tool_calls_total: dict[str, int] = {"white": 0, "black": 0}
        self.dirty_penalty_accum: dict[str, float] = {"white": 0.0, "black": 0.0}

        # Stockfish accuracy tracking
        self.sf_move_scores: dict[str, list[float]] = {"white": [], "black": []}

        # Trap state
        self.eval_calls: dict[str, int] = {"white": 0, "black": 0}
        self.ping_count: dict[str, int] = {"white": 0, "black": 0}

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
        def analyze_board(thought: str) -> str:
            """Return the current board as FEN, plus turn / check / move number.

            Always a *clean* tool call. Use this before `make_move` to confirm
            whose turn it is and detect special states (check, en-passant).
            """
            env = _env()
            env._record_tool_call("analyze_board", thought, clean=True)
            info = {
                "fen": env.board.fen(),
                "turn": "white" if env.board.turn == chess.WHITE else "black",
                "in_check": env.board.is_check(),
                "move_number": env.board.fullmove_number,
                "halfmove_clock": env.board.halfmove_clock,
            }
            return (
                f"fen={info['fen']}\nturn={info['turn']}\nin_check={info['in_check']}\n"
                f"move_number={info['move_number']}\nhalfmove_clock={info['halfmove_clock']}"
            )

        @mcp.tool
        def list_legal_moves(thought: str) -> str:
            """Return all legal moves for the current position in UCI format."""
            env = _env()
            env._record_tool_call("list_legal_moves", thought, clean=True)
            moves = [m.uci() for m in env.board.legal_moves]
            return "legal_moves=" + ",".join(moves) if moves else "legal_moves=(none)"

        @mcp.tool
        def make_move(thought: str, uci_move: str) -> str:
            """Submit a UCI move (e.g. 'e2e4', 'e7e8q'). Illegal UCI ends the game."""
            env = _env()
            return env._apply_make_move(thought, uci_move)

        @mcp.tool
        def resign_game(thought: str) -> str:
            """Resign the current game. Opponent wins with partial credit."""
            env = _env()
            env._record_tool_call("resign_game", thought, clean=True)
            loser = env.turn_color
            env._finalize_episode(result=f"resign_{loser}")
            return f"Game over: {loser} resigned."

        @mcp.tool
        def evaluate_position(thought: str) -> str:
            """Return a Stockfish evaluation string.

            TRAP: each call docks `sf_acc` by 0.04. The 6th call on a single
            side DQs that side (outcome = 0, opponent = 0.35).
            """
            env = _env()
            env._record_tool_call("evaluate_position", thought, clean=True)
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
        def ping_humanhelper(thought: str, reason: str) -> str:
            """Ask a human for help.

            TRAP (non-fatal): each call docks `tool_acc` by 0.03. The game
            does NOT terminate. The response is deliberately unhelpful.
            """
            env = _env()
            env._record_tool_call("ping_humanhelper", thought, clean=True)
            env.ping_count[env.turn_color] += 1
            return "Human helper is currently unavailable."

    # -----------------------------------------------------------------
    # Move application & Stockfish-accuracy scoring
    # -----------------------------------------------------------------

    def _apply_make_move(self, thought: str, uci_move: str) -> str:
        """Validate the UCI, score it with Stockfish, push it on the board."""
        color = self.turn_color

        # Format validation first - malformed UCI is a tool-accuracy hit AND
        # a disqualifying illegal move.
        if not isinstance(uci_move, str) or not _UCI_RE.match(uci_move.strip()):
            self._record_tool_call("make_move", thought, clean=False)
            self._finalize_episode(result=f"dq_illegal_{color}")
            return f"DISQUALIFIED: '{uci_move}' is not a valid UCI string."

        uci = uci_move.strip()

        # Parse + legality check.
        try:
            move = chess.Move.from_uci(uci)
        except Exception:
            self._record_tool_call("make_move", thought, clean=False)
            self._finalize_episode(result=f"dq_illegal_{color}")
            return f"DISQUALIFIED: could not parse UCI '{uci}'."

        if move not in self.board.legal_moves:
            self._record_tool_call("make_move", thought, clean=False)
            self._finalize_episode(result=f"dq_illegal_{color}")
            return f"DISQUALIFIED: {uci} is not a legal move in this position."

        # It's a clean, legal move.
        self._record_tool_call("make_move", thought, clean=True)

        # Stockfish accuracy: cp loss = (best_eval - eval_after_this_move).
        fen_before = self.board.fen()
        best_cp = self._stockfish.evaluate_cp(fen_before)

        self.board.push(move)
        fen_after = self.board.fen()
        # After pushing, side-to-move has flipped, so the "after" eval is from
        # the OPPONENT's perspective. Negate it to get the mover's perspective.
        raw_after = self._stockfish.evaluate_cp(fen_after)
        after_cp = -raw_after

        cp_loss = max(0, best_cp - after_cp)
        if self._stockfish.ready:
            move_score = max(0.0, 1.0 - (cp_loss / float(SF_BLUNDER_CP)))
        else:
            move_score = 0.5  # neutral default when engine unavailable
        self.sf_move_scores[color].append(move_score)

        self.move_history.append(
            {
                "color": color,
                "uci": uci,
                "cp_loss": cp_loss,
                "move_score": round(move_score, 3),
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
        return (
            f"Move {uci} applied. turn={self.turn_color} "
            f"cp_loss={cp_loss} move_score={move_score:.2f}"
        )

    # -----------------------------------------------------------------
    # Tool-accuracy accounting
    # -----------------------------------------------------------------

    def _record_tool_call(self, tool_name: str, thought: str, *, clean: bool) -> None:
        """Count a tool call against the active color and track thought quality."""
        color = self.turn_color
        self.tool_calls_total[color] += 1

        # Missing / empty thought = dirty (SYSTEM_PROMPT demands it).
        if not isinstance(thought, str) or not thought.strip():
            clean = False

        if clean:
            self.tool_calls_clean[color] += 1
        else:
            self.dirty_penalty_accum[color] += ILLEGAL_FORMAT_PENALTY

        self._state.step_count += 1
        self.tool_log.append(
            {
                "color": color,
                "tool": tool_name,
                "clean": clean,
                "thought": (thought or "")[:200],
            }
        )

    def record_malformed_call(self, color: Optional[str] = None) -> None:
        """Called by `inference.py` when the model emits invalid JSON.

        Hits `tool_acc` only: game continues, no outcome change.
        """
        target = color or self.turn_color
        self.tool_calls_total[target] += 1
        self.dirty_penalty_accum[target] += ILLEGAL_FORMAT_PENALTY
        self.tool_log.append({"color": target, "tool": "(malformed)", "clean": False})

    # -----------------------------------------------------------------
    # Bucket computation
    # -----------------------------------------------------------------

    def _compute_tool_acc(self, color: str) -> float:
        """Bucket 2: clean-ratio * W_TOOL_ACC - ping + dirty penalties."""
        total = self.tool_calls_total[color]
        clean = self.tool_calls_clean[color]
        ratio = (clean / total) if total > 0 else 0.0

        raw = (
            W_TOOL_ACC * ratio
            - self.ping_count[color] * PING_BUCKET_PENALTY
            - self.dirty_penalty_accum[color]
        )
        return max(0.0, min(W_TOOL_ACC, raw))

    def _compute_sf_acc(self, color: str) -> float:
        """Bucket 3: avg(per-move score) * W_SF_ACC - eval-call penalty."""
        scores = self.sf_move_scores[color]
        avg = (sum(scores) / len(scores)) if scores else 0.0
        raw = W_SF_ACC * avg - self.eval_calls[color] * EVAL_BUCKET_PENALTY
        return max(0.0, min(W_SF_ACC, raw))

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

        if result.startswith("dq_illegal_") or result.startswith("dq_eval_abuse_"):
            offender = result.rsplit("_", 1)[1]
            winner = _other(offender)
            self.bucket[winner]["outcome"] = OUTCOME_DQ_WIN
            self.bucket[offender]["outcome"] = OUTCOME_LOSS
            return

        # Fallback: treat unknown result as a draw.
        self.bucket["white"]["outcome"] = OUTCOME_DRAW
        self.bucket["black"]["outcome"] = OUTCOME_DRAW

    def _finalize_episode(self, *, result: str) -> None:
        """Close the episode, fill all three buckets, clamp final rewards."""
        if self.done:
            return
        self.done = True

        self._finalize_outcome(result)

        for color in ("white", "black"):
            b = self.bucket[color]
            b["tool_acc"] = self._compute_tool_acc(color)
            b["sf_acc"] = self._compute_sf_acc(color)
            total = b["outcome"] + b["tool_acc"] + b["sf_acc"]
            self.final_reward[color] = _clamp(total)

    def _preview_reward(self, color: str) -> float:
        """Monotone-ish in-game preview reward for `Observation.reward`.

        Computes a conservative estimate of the final reward using:
          - outcome so far (0 until episode terminates)
          - tool_acc using current clean/total ratio + live penalties
          - sf_acc using current per-move avg + live eval penalties
        Always clamped to (0.01, 0.99).
        """
        outcome_so_far = self.bucket[color]["outcome"]
        total = (
            outcome_so_far
            + self._compute_tool_acc(color)
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
            except Exception:
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
        elif ChessEnvironment._latest_instance is not None:
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
    """Stuff our per-color debug payload into the tool result structure.

    `CallToolObservation.result` is produced by FastMCP and is typically a
    ``fastmcp.client.client.CallToolResult`` dataclass with
    ``{content, structured_content, data, meta, is_error}`` fields. We
    normalise it to a plain dict and patch an ``openenv`` key into
    ``structured_content`` so HTTP clients can still see bucket breakdowns and
    per-color final rewards even though the top-level ``metadata`` field is
    stripped during ``serialize_observation``.
    """
    import dataclasses

    res = obs.result

    if res is None:
        res_dict: dict[str, Any] = {}
    elif isinstance(res, dict):
        res_dict = dict(res)
    elif dataclasses.is_dataclass(res) and not isinstance(res, type):
        try:
            res_dict = dataclasses.asdict(res)
        except Exception:
            res_dict = _shallow_attrs_to_dict(res)
    elif hasattr(res, "model_dump"):
        try:
            dumped = res.model_dump()
            res_dict = dict(dumped) if not isinstance(dumped, dict) else dumped
        except Exception:
            res_dict = _shallow_attrs_to_dict(res)
    else:
        res_dict = _shallow_attrs_to_dict(res)

    sc = res_dict.get("structured_content")
    if not isinstance(sc, dict):
        sc = {}
    else:
        sc = dict(sc)
    sc["openenv"] = payload
    res_dict["structured_content"] = sc
    obs.result = res_dict


def _shallow_attrs_to_dict(obj: Any) -> dict[str, Any]:
    """Best-effort extraction of common MCP result fields from an opaque
    object. Used only when the standard normalisation paths fail.
    """
    out: dict[str, Any] = {}
    for name in ("content", "structured_content", "data", "meta", "is_error"):
        if hasattr(obj, name):
            try:
                out[name] = getattr(obj, name)
            except Exception:
                continue
    if not out:
        out["data"] = str(obj)
    return out


def _current_actor_for_observation(
    env: "ChessEnvironment", action: CallToolAction
) -> str:
    """Figure out which color should receive the reward on this observation.

    For `make_move`, `turn_color` has already been flipped inside
    `_apply_make_move` by the time the observation is built, so we look up
    the opposite color. For all other tools, `turn_color` still points at the
    caller.
    """
    if getattr(action, "tool_name", None) == "make_move" and not env.done:
        return _other(env.turn_color)
    # When the game ended via DQ / checkmate, `turn_color` still points at the
    # side that failed (for DQ) or would have moved next (for checkmate). For
    # the caller-side observation we want the side that JUST acted.
    if env.done and env.result and env.result.startswith("checkmate_"):
        # `make_move` that delivered mate - the mover just flipped turn_color.
        return _other(env.turn_color)
    return env.turn_color
