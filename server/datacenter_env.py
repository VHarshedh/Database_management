# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Global SOC Datacenter Simulation OpenEnv environment.

Adapter on top of `python-chess` + Stockfish: every chess piece, square and
move is re-skinned into SOC datacenter terminology. The high-level code is
chess-free - it speaks only in terms of DEFENDER / ADVERSARY tiers, 4D
``(region, zone, rack, pod)`` nodes, and workload migrations.

The only places we still touch ``chess.WHITE`` / ``chess.BLACK`` are the two
private helpers :func:`_tier_for_color` and :func:`_color_for_tier`, which
exist purely so we can call ``board.push``, ``piece.color``, and
``chess.engine`` with the constants the third-party library requires.

Tier model
----------
Two access tiers operate on the simulated infrastructure:

    DEFENDER_ID  == INTERNAL_DOMAIN  == "defender"   (== chess.WHITE)
    ADVERSARY_ID == EXTERNAL_DOMAIN  == "adversary"  (== chess.BLACK)

`current_access_tier` is a property derived from the underlying
``chess.Board.turn`` flag, so flipping is automatic after a successful
``board.push``. Convenience methods ``is_defender_active``,
``get_defender_efficiency``, and ``get_adversary_threat_level`` let
orchestrators read the live tier + scores without reaching into chess.

Reward decomposition (unchanged, sums to <= 0.99):

    Outcome bucket        : <= 0.50
    Format bucket          : <= 0.10
    Thought-quality bucket : <= 0.15
    Stockfish bucket       : <= 0.24
"""

from __future__ import annotations

import contextvars
import csv
import json
import os
import re
import threading
import traceback
from datetime import datetime, timezone
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
    from stockfish import Stockfish  # type: ignore  # noqa: F401
except ImportError:  # pragma: no cover - stockfish is optional at import time
    Stockfish = None  # type: ignore


# ---------------------------------------------------------------------------
# ContextVars: let FastMCP tool fns (no `self`) find their env + active episode.
# ---------------------------------------------------------------------------
_active_env: contextvars.ContextVar[Optional["DatacenterEnvironment"]] = contextvars.ContextVar(
    "_active_env", default=None
)
_current_episode_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_current_episode_id", default=None
)


# ===========================================================================
# Tier identifiers (the entire high-level codebase speaks in these strings)
# ===========================================================================

DEFENDER_ID: str = "defender"      # internal-domain operator (was: chess.WHITE)
ADVERSARY_ID: str = "adversary"    # external-domain attacker (was: chess.BLACK)

# Domain aliases so orchestrators can say either "tier" or "domain".
INTERNAL_DOMAIN: str = DEFENDER_ID
EXTERNAL_DOMAIN: str = ADVERSARY_ID

# Ordered tuple used by every `for tier in TIERS:` loop.
TIERS: tuple[str, str] = (DEFENDER_ID, ADVERSARY_ID)


# ===========================================================================
# Datacenter adapter constants (skin layer over python-chess)
# ===========================================================================

# Defender - critical infrastructure assets (was: White pieces).
DEFENDER_ASSETS: dict[int, str] = {
    chess.KING:   "Primary_Root_Kernel",
    chess.QUEEN:  "Relational_DB_Cluster",
    chess.ROOK:   "Storage_Array",
    chess.BISHOP: "Compute_Node",
    chess.KNIGHT: "API_Gateway",
    chess.PAWN:   "Data_Packet",
}

# Adversary - intrusion / exploit assets (was: Black pieces).
ADVERSARY_ASSETS: dict[int, str] = {
    chess.KING:   "Rootkit_Core",
    chess.QUEEN:  "Ransomware_Orchestrator",
    chess.ROOK:   "Persistent_Backdoor",
    chess.BISHOP: "Lateral_Movement_Agent",
    chess.KNIGHT: "Phishing_Probe",
    chess.PAWN:   "Malicious_Beacon",
}

# 4D tensor axes (order: region, zone, rack, pod).
REGIONS: list[str] = ["us-east", "eu-west"]
ZONES: list[str] = ["az-a", "az-b"]
RACKS: list[str] = [f"rack-{i + 1}" for i in range(4)]
PODS: list[str] = [f"pod-{i + 1}" for i in range(4)]

# Promotion role -> chess piece UCI letter (for migrate_workload promotion).
_PROMOTION_ROLE_TO_LETTER: dict[str, str] = {
    "relational_db_cluster": "q",
    "storage_array":         "r",
    "compute_node":          "b",
    "api_gateway":           "n",
    "ransomware_orchestrator": "q",
    "persistent_backdoor":     "r",
    "lateral_movement_agent":  "b",
    "phishing_probe":          "n",
    "queen":  "q",
    "rook":   "r",
    "bishop": "b",
    "knight": "n",
}

# Friendly datacenter labels for terminal results. Internal codes use the
# tier suffix (defender / adversary) so reward parsing stays uniform.
RESULT_LABELS: dict[str, str] = {
    "checkmate_defender":     "BREACH_CONTAINED",
    "checkmate_adversary":    "DATACENTER_COMPROMISED",
    "resign_defender":        "DEFENDER_CONCEDED",
    "resign_adversary":       "ADVERSARY_WITHDRAWN",
    "dq_illegal_defender":    "DEFENDER_PROTOCOL_VIOLATION",
    "dq_illegal_adversary":   "ADVERSARY_PROTOCOL_VIOLATION",
    "dq_eval_abuse_defender": "DEFENDER_ORACLE_ABUSE",
    "dq_eval_abuse_adversary": "ADVERSARY_ORACLE_ABUSE",
}


# ===========================================================================
# Reward constants (unchanged)
# ===========================================================================

W_OUTCOME = 0.50
W_FORMAT = 0.10
W_THOUGHT_Q = 0.15
W_SF_ACC = 0.24

R_MIN, R_MAX = 0.01, 0.99

OUTCOME_WIN = 0.50
OUTCOME_DRAW = 0.25
OUTCOME_LOSS = 0.00
OUTCOME_RESIGN_WIN = 0.45
OUTCOME_DQ_WIN = 0.35

EVAL_CALL_LIMIT = 5
ILLEGAL_FORMAT_PENALTY = 0.05
STRIKES_BEFORE_DQ = 2
W_SF = 0.24
PING_BUCKET_PENALTY = 0.03
EVAL_BUCKET_PENALTY = 0.04

# ---------------------------------------------------------------------------
# Enterprise guardrails (Layers 5 / 6 / 7)
# ---------------------------------------------------------------------------

# Layer 5: HITL escalation contract. The Defender tool returns a string that
# starts with this prefix; the orchestrator parses it and triggers an input()
# prompt for human-typed mitigating coordinates.
HITL_SIGNAL_PREFIX: str = "HITL_REQUIRED"

# Layer 6: compliance penalty docked from the format bucket whenever the
# engine push (or anything inside ``_apply_migration``) raises an exception.
COMPLIANCE_PENALTY: float = 0.05

# Layer 7: append-only compliance audit log. Path can be overridden via the
# ``COMPLIANCE_AUDIT_LOG`` env var; default sits next to the package so the
# orchestrator and visualiser can find it without configuration.
_DEFAULT_AUDIT_PATH = Path(__file__).resolve().parent.parent / "compliance_audit_log.csv"
COMPLIANCE_AUDIT_LOG_PATH: Path = Path(
    os.getenv("COMPLIANCE_AUDIT_LOG", str(_DEFAULT_AUDIT_PATH))
)
COMPLIANCE_AUDIT_HEADER: list[str] = [
    "timestamp_utc",
    "episode_id",
    "region_label",
    "event_type",
    "tier",
    "tool",
    "source_node",
    "target_node",
    "uci",
    "threat_analysis",
    "justification",
    "cp_loss",
    "move_score",
    "stockfish_eval_cp_white",
    "exception_type",
    "traceback",
]
_AUDIT_LOG_LOCK = threading.Lock()


def _append_compliance_audit(row: dict[str, Any]) -> None:
    """Append a single row to ``compliance_audit_log.csv``.

    Writes the header automatically the first time the file is created.
    Rows are flushed under a process-wide lock so concurrent regions in the
    orchestrator can't interleave half-rows. All field values are coerced to
    strings to keep the file round-trip-safe.
    """
    payload = {h: ("" if row.get(h) is None else str(row.get(h, ""))) for h in COMPLIANCE_AUDIT_HEADER}
    with _AUDIT_LOG_LOCK:
        try:
            COMPLIANCE_AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            file_exists = COMPLIANCE_AUDIT_LOG_PATH.is_file() and COMPLIANCE_AUDIT_LOG_PATH.stat().st_size > 0
            with COMPLIANCE_AUDIT_LOG_PATH.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COMPLIANCE_AUDIT_HEADER, extrasaction="ignore")
                if not file_exists:
                    writer.writeheader()
                writer.writerow(payload)
        except Exception as audit_err:  # pragma: no cover - audit must never crash the env
            import sys as _sys
            print(
                f"[ComplianceAudit] WARNING: failed to append row: {audit_err}",
                file=_sys.stderr,
            )

# Datacenter justification keywords.
SOC_CONCEPTS: list[str] = [
    "containment", "isolation", "segmentation", "honeypot", "honeynet",
    "blast radius", "perimeter", "exfiltration", "lateral", "persistence",
    "kernel", "privilege", "credential", "zero trust", "least privilege",
    "rollback", "snapshot", "redundancy", "uptime", "failover",
]

# Threat-awareness synonyms for the threat sub-score.
THREAT_SYNONYMS: list[str] = [
    "compromise", "compromised", "breach", "intrusion", "exfiltration",
    "exploit", "attack", "captured", "destroyed", "loss",
]

# Stockfish scoring curve.
SF_DEPTH = 15
SF_BLUNDER_CP = 300
SF_MATE_CP = 10000


# ===========================================================================
# Chess <-> Tier adapters (the ONLY place chess.WHITE / chess.BLACK appear
# outside of the engine call sites)
# ===========================================================================


def _tier_for_color(color: chess.Color) -> str:
    """Translate a python-chess color flag into a tier id."""
    return DEFENDER_ID if color == chess.WHITE else ADVERSARY_ID


def _color_for_tier(tier: str) -> chess.Color:
    """Translate a tier id back to the underlying python-chess color flag.

    This exists so the few engine-facing call sites (``chess.Move``,
    ``board.turn``, ``chess.Piece(..., color)``) keep getting the constants
    they expect, while the high-level code never has to know about them.
    """
    return chess.WHITE if tier == DEFENDER_ID else chess.BLACK


def _opponent_tier(tier: str) -> str:
    """Return the opposing tier id."""
    return ADVERSARY_ID if tier == DEFENDER_ID else DEFENDER_ID


def _asset_for_piece(piece: chess.Piece) -> str:
    table = DEFENDER_ASSETS if piece.color == chess.WHITE else ADVERSARY_ASSETS
    return table[piece.piece_type]


def _owner_for_piece(piece: chess.Piece) -> str:
    """JSON-friendly ``"owner"`` value: lowercase tier id."""
    return _tier_for_color(piece.color)


# ===========================================================================
# 4D Tensor Adapter
# ===========================================================================


def square_to_node(square: int) -> dict[str, str]:
    """Map a 0..63 chess square index to its 4D datacenter coordinate.

    Decomposition:
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        region_idx = file // 4
        rack_idx   = file %  4
        zone_idx   = rank // 4
        pod_idx    = rank %  4
    """
    if not 0 <= int(square) <= 63:
        raise ValueError(f"square index out of range [0, 63]: {square!r}")
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    return {
        "region": REGIONS[file // 4],
        "zone":   ZONES[rank // 4],
        "rack":   RACKS[file % 4],
        "pod":    PODS[rank % 4],
    }


def _normalise_axis_value(value: Any, *, axis: str, choices: list[str]) -> str:
    """Loose, case-insensitive matcher: accepts canonical names + numeric forms."""
    if value is None:
        raise ValueError(f"missing '{axis}' in node coordinate")
    s = str(value).strip().lower()
    if not s:
        raise ValueError(f"empty '{axis}' in node coordinate")
    if s in choices:
        return s
    digits = re.search(r"(\d+)", s)
    if digits:
        idx = int(digits.group(1)) - 1
        if 0 <= idx < len(choices):
            return choices[idx]
    raise ValueError(f"invalid {axis} '{value}'; expected one of {choices}")


def node_to_square(node: dict[str, Any]) -> int:
    """Flatten a 4D datacenter coordinate back to a 0..63 chess square index."""
    if not isinstance(node, dict):
        raise ValueError(f"node must be a dict, got {type(node).__name__}")

    region = _normalise_axis_value(node.get("region"), axis="region", choices=REGIONS)
    zone = _normalise_axis_value(node.get("zone"), axis="zone", choices=ZONES)
    rack = _normalise_axis_value(node.get("rack"), axis="rack", choices=RACKS)
    pod = _normalise_axis_value(node.get("pod"), axis="pod", choices=PODS)

    region_idx = REGIONS.index(region)
    zone_idx = ZONES.index(zone)
    rack_idx = RACKS.index(rack)
    pod_idx = PODS.index(pod)

    file = region_idx * 4 + rack_idx
    rank = zone_idx * 4 + pod_idx
    return chess.square(file, rank)


def _square_to_uci(square: int) -> str:
    return chess.square_name(square)


def _uci_to_node(uci_square: str) -> dict[str, str]:
    return square_to_node(chess.parse_square(uci_square))


def node_canonical(node: dict[str, Any]) -> str:
    """Canonical ``region/zone/rack/pod`` string for a 4D node."""
    sq = node_to_square(node)
    n = square_to_node(sq)
    return f"{n['region']}/{n['zone']}/{n['rack']}/{n['pod']}"


def migration_canonical(source_node: dict[str, Any], target_node: dict[str, Any]) -> str:
    return f"{node_canonical(source_node)}->{node_canonical(target_node)}"


def _uci_to_migration_str(uci: str) -> str:
    if not isinstance(uci, str) or len(uci) < 4:
        return uci
    src = _uci_to_node(uci[0:2])
    dst = _uci_to_node(uci[2:4])
    return migration_canonical(src, dst)


# ===========================================================================
# Stockfish wrapper (engine backend stays the same)
# ===========================================================================


def _resolve_stockfish_path() -> Optional[str]:
    """Locate the Stockfish binary in the expected places (env var > vendored > PATH)."""
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
    """Thin adapter around the `chess.engine` python wrapper."""

    def __init__(self, bin_path: str | None = None, *args: Any, **kwargs: Any) -> None:
        self.bin_path = bin_path or _resolve_stockfish_path() or "engine/stockfish.exe"
        self._engine: Optional[chess.engine.SimpleEngine] = None
        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.bin_path)
        except Exception as e:
            print(f"[StockfishAdapter] Warning: Engine not loaded from {self.bin_path}. {e}")

    def close(self) -> None:
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
            # Score is reported relative to the player whose turn it is. We
            # keep the existing semantics: positive = good for the side to move.
            score = info["score"].white() if board.turn == chess.WHITE else info["score"].black()
            if score.is_mate():
                mate_in = score.mate()
                sign = 1 if mate_in > 0 else -1
                return sign * SF_MATE_CP
            return score.score() or 0
        except Exception:
            return 0

    def describe(self, fen: str) -> str:
        if self._engine is None:
            return "Threat oracle is currently offline (engine unavailable)."
        try:
            board = chess.Board(fen)
            limit = chess.engine.Limit(time=0.5)
            info = self._engine.analyse(board, limit=limit)
            score = info["score"].white()
            best_uci = info["pv"][0].uci() if "pv" in info and info["pv"] else None
            eval_type = "mate" if score.is_mate() else "cp"
            value = score.mate() if score.is_mate() else score.score()
            best_migration = _uci_to_migration_str(best_uci) if best_uci else "none"
            return (
                f"oracle_eval_type={eval_type} "
                f"oracle_value={value or 0} "
                f"recommended_migration={best_migration}"
            )
        except Exception as e:
            return f"Threat oracle error: {e}"


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
    if v != v:
        return R_MIN
    return max(R_MIN, min(R_MAX, v))


# ===========================================================================
# DatacenterEnvironment
# ===========================================================================


class DatacenterEnvironment(MCPEnvironment):
    """MCP-based Global SOC Datacenter Simulation. One HTTP session = one engagement.

    Two agents (DEFENDER + ADVERSARY) share the same env instance and alternate
    access. The high-level surface is chess-free: tier identifiers are
    ``DEFENDER_ID`` / ``ADVERSARY_ID`` strings, and ``current_access_tier``
    is a property that derives from the underlying ``chess.Board.turn`` so
    flipping happens automatically when the engine accepts a migration.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    _instances: dict[str, "DatacenterEnvironment"] = {}
    _latest_instance: Optional["DatacenterEnvironment"] = None

    # -----------------------------------------------------------------
    # Construction / tool registration
    # -----------------------------------------------------------------

    def __init__(self) -> None:
        self._init_fresh_state()
        mcp = FastMCP("datacenter_arena")
        self._register_tools(mcp)
        super().__init__(mcp)

    def _init_fresh_state(self) -> None:
        """Reset all per-engagement state. Also called from `reset()`."""
        if hasattr(self, "_stockfish") and self._stockfish is not None:
            self._stockfish.close()

        self.board: chess.Board = chess.Board()
        self._last_tier_flipped: bool = False
        self._stockfish: _StockfishAdapter = _StockfishAdapter(depth=SF_DEPTH)

        # Per-tier buckets (outcome / format / thought_q / sf_acc).
        self.bucket: dict[str, dict[str, float]] = {
            DEFENDER_ID:  {"outcome": 0.0, "format": 0.0, "thought_q": 0.0, "sf_acc": 0.0},
            ADVERSARY_ID: {"outcome": 0.0, "format": 0.0, "thought_q": 0.0, "sf_acc": 0.0},
        }
        self.final_reward: dict[str, float] = {DEFENDER_ID: R_MIN, ADVERSARY_ID: R_MIN}

        self.tool_calls_clean: dict[str, int] = {DEFENDER_ID: 0, ADVERSARY_ID: 0}
        self.tool_calls_total: dict[str, int] = {DEFENDER_ID: 0, ADVERSARY_ID: 0}
        self.dirty_penalty_accum: dict[str, float] = {DEFENDER_ID: 0.0, ADVERSARY_ID: 0.0}
        self.thought_quality_scores: dict[str, list[float]] = {DEFENDER_ID: [], ADVERSARY_ID: []}
        self.sf_move_scores: dict[str, list[float]] = {DEFENDER_ID: [], ADVERSARY_ID: []}

        self.eval_calls: dict[str, int] = {DEFENDER_ID: 0, ADVERSARY_ID: 0}
        self.ping_count: dict[str, int] = {DEFENDER_ID: 0, ADVERSARY_ID: 0}
        self.illegal_move_count: dict[str, int] = {DEFENDER_ID: 0, ADVERSARY_ID: 0}

        # Layer 5: HITL escalation tracking.
        self.hitl_escalations: dict[str, int] = {DEFENDER_ID: 0, ADVERSARY_ID: 0}
        self.pending_hitl_reason: Optional[str] = None
        # Layer 6: count of compliance penalties (catastrophic adapter failures).
        self.compliance_penalties: dict[str, int] = {DEFENDER_ID: 0, ADVERSARY_ID: 0}
        # Layer 7: human-readable region label written into every audit row.
        # Orchestrators may set this on the env instance after construction.
        if not hasattr(self, "region_label") or self.region_label is None:
            self.region_label: Optional[str] = None

        # ``move_history`` keeps UCI for compat with the legacy chess
        # visualizer; ``migration_history`` is the datacenter-skinned view.
        self.move_history: list[dict[str, Any]] = []
        self.migration_history: list[dict[str, Any]] = []
        self.tool_log: list[dict[str, Any]] = []
        self.done: bool = False
        self.result: Optional[str] = None
        self._state: State = State(episode_id=str(uuid4()), step_count=0)

    # -----------------------------------------------------------------
    # Tier accessors (the chess-free public surface)
    # -----------------------------------------------------------------

    @property
    def current_access_tier(self) -> str:
        """Tier whose turn it is to act ('defender' or 'adversary').

        Derived from the underlying ``chess.Board.turn`` flag, so it flips
        automatically the moment a migration is pushed onto the board.
        """
        return _tier_for_color(self.board.turn)

    def is_defender_active(self) -> bool:
        """True when the DEFENDER tier has the current access slot."""
        return self.current_access_tier == DEFENDER_ID

    def is_adversary_active(self) -> bool:
        """True when the ADVERSARY tier has the current access slot."""
        return self.current_access_tier == ADVERSARY_ID

    def get_defender_efficiency(self) -> float:
        """Live (or final) defender score, clamped to (0.01, 0.99).

        Returns the four-bucket sum (outcome + format + thought_q + sf_acc)
        for the DEFENDER tier. After episode finalisation this is the
        clamped final reward.
        """
        if self.done:
            return _clamp(self.final_reward[DEFENDER_ID])
        return self._preview_reward(DEFENDER_ID)

    def get_adversary_threat_level(self) -> float:
        """Live (or final) adversary score, clamped to (0.01, 0.99).

        Same composition as :meth:`get_defender_efficiency`, evaluated for
        the ADVERSARY tier. A high adversary threat level means the
        external-domain attacker is winning.
        """
        if self.done:
            return _clamp(self.final_reward[ADVERSARY_ID])
        return self._preview_reward(ADVERSARY_ID)

    # -----------------------------------------------------------------
    # MCP tools  (exposed to the LLM - all datacenter terminology)
    # -----------------------------------------------------------------

    def _register_tools(self, mcp: FastMCP) -> None:
        """Register the 6 SOC datacenter tools with FastMCP."""

        def _env() -> "DatacenterEnvironment":
            env = _active_env.get() or DatacenterEnvironment._latest_instance
            if env is None:
                raise RuntimeError("No active DatacenterEnvironment instance")
            return env

        @mcp.tool
        def scan_topology(
            threat_analysis: str,
            candidate_migrations: list[str],
            justification: str,
        ) -> str:
            """Scan the live datacenter topology.

            Returns a JSON document listing every active workload at its 4D
            coordinate ``(region, zone, rack, pod)``. The agent never receives
            a chess board or FEN; this is the only authoritative view of the
            global infrastructure.
            """
            env = _env()
            env._record_tool_call(
                "scan_topology",
                threat_analysis, candidate_migrations, justification,
                clean=True,
            )
            return json.dumps(env.get_topology_state(), indent=2)

        @mcp.tool
        def enumerate_authorized_migrations(
            threat_analysis: str,
            candidate_migrations: list[str],
            justification: str,
        ) -> str:
            """List every authorized workload migration available to the active tier."""
            env = _env()
            env._record_tool_call(
                "enumerate_authorized_migrations",
                threat_analysis, candidate_migrations, justification,
                clean=True,
            )
            board = env.board
            entries: list[dict[str, Any]] = []
            for mv in board.legal_moves:
                src_sq = mv.from_square
                dst_sq = mv.to_square
                piece = board.piece_at(src_sq)
                if piece is None:  # pragma: no cover - python-chess invariant
                    continue
                src_node = square_to_node(src_sq)
                dst_node = square_to_node(dst_sq)
                entry: dict[str, Any] = {
                    "asset_id": _asset_for_piece(piece),
                    "owner": _owner_for_piece(piece),
                    "source_node": src_node,
                    "target_node": dst_node,
                    "migration": migration_canonical(src_node, dst_node),
                    "captures_hostile": board.is_capture(mv),
                }
                if mv.promotion is not None:
                    promo_piece = chess.Piece(mv.promotion, piece.color)
                    entry["promotes_to"] = _asset_for_piece(promo_piece)
                entries.append(entry)
            payload = {
                "active_tier": env.current_access_tier,
                "incident_clock": board.fullmove_number,
                "authorized_migrations": entries,
            }
            return json.dumps(payload, indent=2)

        @mcp.tool
        def migrate_workload(
            threat_analysis: str,
            candidate_migrations: list[str],
            justification: str,
            source_node: dict[str, Any],
            target_node: dict[str, Any],
            promotion_role: Optional[str] = None,
        ) -> str:
            """Migrate a workload from ``source_node`` to ``target_node``.

            Both nodes are 4D dictionaries with keys ``region``, ``zone``,
            ``rack``, ``pod``.
            """
            env = _env()
            return env._apply_migration(
                threat_analysis, candidate_migrations, justification,
                source_node=source_node,
                target_node=target_node,
                promotion_role=promotion_role,
            )

        @mcp.tool
        def declare_breach(
            threat_analysis: str,
            candidate_migrations: list[str],
            justification: str,
        ) -> str:
            """Concede the engagement. The opposing tier wins with partial credit."""
            env = _env()
            env._record_tool_call(
                "declare_breach",
                threat_analysis, candidate_migrations, justification,
                clean=True,
            )
            losing_tier = env.current_access_tier
            env._finalize_episode(result=f"resign_{losing_tier}")
            return f"Engagement over: {losing_tier.title()} declared breach (conceded)."

        @mcp.tool
        def query_threat_oracle(
            threat_analysis: str,
            candidate_migrations: list[str],
            justification: str,
        ) -> str:
            """Query an external threat-intelligence oracle.

            TRAP: each call docks ``sf_acc`` by 0.04. The 6th call from a
            single tier disqualifies that tier.
            """
            env = _env()
            env._record_tool_call(
                "query_threat_oracle",
                threat_analysis, candidate_migrations, justification,
                clean=True,
            )
            tier = env.current_access_tier
            env.eval_calls[tier] += 1
            if env.eval_calls[tier] > EVAL_CALL_LIMIT:
                env._finalize_episode(result=f"dq_eval_abuse_{tier}")
                return (
                    f"DISQUALIFIED: {tier.title()} queried the threat oracle "
                    f"{env.eval_calls[tier]} times (limit {EVAL_CALL_LIMIT})."
                )
            return env._stockfish.describe(env.board.fen())

        @mcp.tool
        def escalate_to_oncall(
            threat_analysis: str,
            candidate_migrations: list[str],
            justification: str,
            reason: str,
        ) -> str:
            """Escalate to the human on-call SRE.

            TRAP (non-fatal): each call docks the format bucket by 0.03.
            """
            env = _env()
            env._record_tool_call(
                "escalate_to_oncall",
                threat_analysis, candidate_migrations, justification,
                clean=True,
            )
            env.ping_count[env.current_access_tier] += 1
            return "On-call SRE is currently unavailable. Owl is paging out."

        @mcp.tool
        def escalate_to_sysadmin(
            threat_analysis: str,
            candidate_migrations: list[str],
            justification: str,
            severity: str = "high",
        ) -> str:
            """Layer 5 (HITL): hand control of THIS region to a human sysadmin.

            Unlike :func:`escalate_to_oncall` (which is a TRAP that docks the
            format budget), this is a sanctioned escalation path. The active
            tier yields its current half-move; the orchestrator pauses the
            region, prints a console banner, and reads the mitigating 4D
            coordinates directly from a human operator via ``input()``.

            The tool itself does NOT advance the access tier or push anything
            onto the engine. It records a clean call, logs the escalation in
            the compliance audit CSV, and returns a string starting with
            ``HITL_REQUIRED:`` so the orchestrator can route to the prompt.
            """
            env = _env()
            env._record_tool_call(
                "escalate_to_sysadmin",
                threat_analysis, candidate_migrations, justification,
                clean=True,
            )
            tier = env.current_access_tier
            env.hitl_escalations[tier] += 1
            env.pending_hitl_reason = (justification or "").strip() or "(unspecified)"

            _append_compliance_audit(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "episode_id": getattr(env._state, "episode_id", ""),
                    "region_label": env.region_label or "(unset)",
                    "event_type": "hitl_escalation",
                    "tier": tier,
                    "tool": "escalate_to_sysadmin",
                    "threat_analysis": threat_analysis or "",
                    "justification": justification or "",
                    "stockfish_eval_cp_white": (
                        env._stockfish.evaluate_cp(env.board.fen())
                        if getattr(env._stockfish, "ready", False) else ""
                    ),
                }
            )
            return (
                f"{HITL_SIGNAL_PREFIX}: {tier.title()} (severity={severity}) "
                f"requested human sysadmin override. Reason: {env.pending_hitl_reason}. "
                "Orchestrator must pause this region and accept human-typed "
                "source_node/target_node coordinates before resuming."
            )

    # -----------------------------------------------------------------
    # Topology state (the JSON the agent sees - never a FEN)
    # -----------------------------------------------------------------

    def get_topology_state(self) -> dict[str, Any]:
        """Return the current datacenter posture as a JSON-serialisable dict."""
        board = self.board
        active_tier = self.current_access_tier
        active_workloads: list[dict[str, Any]] = []
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None:
                continue
            node = square_to_node(sq)
            active_workloads.append(
                {
                    "asset_id": _asset_for_piece(piece),
                    "owner": _owner_for_piece(piece),
                    "node": node,
                    "node_canonical": node_canonical(node),
                }
            )
        last_migration: Optional[dict[str, Any]] = None
        if self.migration_history:
            last_migration = dict(self.migration_history[-1])
        defender_under_threat = bool(board.is_check() and self.is_defender_active())
        adversary_under_threat = bool(board.is_check() and self.is_adversary_active())
        containment_clock = max(0, 100 - board.halfmove_clock)
        repetition_warning = bool(board.can_claim_threefold_repetition())
        truce_claimable = bool(board.can_claim_draw())
        authorized_count = sum(1 for _ in board.legal_moves)
        return {
            "topology": "4D",
            "axes": {
                "regions": list(REGIONS),
                "zones": list(ZONES),
                "racks": list(RACKS),
                "pods": list(PODS),
            },
            "active_tier": active_tier,
            "incident_clock": board.fullmove_number,
            "containment_clock_halfmoves": containment_clock,
            "defender_under_threat": defender_under_threat,
            "adversary_under_threat": adversary_under_threat,
            "repetition_warning": repetition_warning,
            "truce_claimable": truce_claimable,
            "authorized_migration_count": authorized_count,
            "last_migration": last_migration,
            "active_workloads": active_workloads,
        }

    # -----------------------------------------------------------------
    # Migration application & Stockfish-accuracy scoring (adapter core)
    # -----------------------------------------------------------------

    def _handle_illegal_migration(
        self,
        tier: str,
        threat_analysis: str,
        candidate_migrations: list[str],
        justification: str,
        attempted_uci: str,
        reason: str,
    ) -> str:
        """Apply two-strike protocol-violation rule."""
        self._record_tool_call(
            "migrate_workload",
            threat_analysis, candidate_migrations, justification,
            clean=False, uci_move=attempted_uci,
        )
        self.illegal_move_count[tier] += 1
        self._last_tier_flipped = False
        msg = (
            f"PROTOCOL VIOLATION (strike {self.illegal_move_count[tier]}/{STRIKES_BEFORE_DQ}) "
            f"by {tier.title()}: {reason} "
        )
        if self.illegal_move_count[tier] == 1:
            self.dirty_penalty_accum[tier] += abs(ILLEGAL_FORMAT_PENALTY)
            msg += f"Penalty applied (-{abs(ILLEGAL_FORMAT_PENALTY)}). One more attempt allowed."
            return msg
        msg += "Disqualified."
        self._finalize_episode(result=f"dq_illegal_{tier}")
        return msg

    def _record_compliance_penalty(
        self,
        *,
        exception: BaseException,
        attempted: str,
        threat_analysis: str,
        candidate_migrations: list[str],
        justification: str,
    ) -> str:
        """Layer 6: log a catastrophic adapter failure and feed the trace
        back to the LLM so it can self-correct without crashing the loop.

        Called by the disaster-recovery wrapper around ``_apply_migration``.
        Side-effects:

        * Increments ``compliance_penalties[tier]``.
        * Docks ``dirty_penalty_accum[tier]`` by :data:`COMPLIANCE_PENALTY`.
        * Records a ``(compliance_penalty)`` entry in ``tool_log``.
        * Appends a ``compliance_penalty`` row to the audit CSV including the
          exception type and full traceback.

        Returns a result string with the verbatim Python traceback so the
        agent can read it on its next turn.
        """
        tier = self.current_access_tier
        self.compliance_penalties[tier] += 1
        self.dirty_penalty_accum[tier] += COMPLIANCE_PENALTY
        self._last_tier_flipped = False

        tb_str = traceback.format_exc()
        if not tb_str or tb_str.strip() == "NoneType: None":
            tb_str = f"{type(exception).__name__}: {exception}"

        self.tool_log.append(
            {
                "tier": tier,
                "tool": "(compliance_penalty)",
                "clean": False,
                "exception": type(exception).__name__,
            }
        )
        _append_compliance_audit(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "episode_id": getattr(self._state, "episode_id", ""),
                "region_label": self.region_label or "(unset)",
                "event_type": "compliance_penalty",
                "tier": tier,
                "tool": "migrate_workload",
                "source_node": "",
                "target_node": "",
                "uci": attempted,
                "threat_analysis": threat_analysis or "",
                "justification": justification or "",
                "cp_loss": "",
                "move_score": "",
                "stockfish_eval_cp_white": "",
                "exception_type": type(exception).__name__,
                "traceback": tb_str,
            }
        )

        return (
            "COMPLIANCE_PENALTY: migration aborted, board state preserved.\n"
            f"Tier={tier}  attempted={attempted}\n"
            f"Penalty applied (-{COMPLIANCE_PENALTY:.02f}). "
            "Self-correct using the traceback below.\n"
            "----- BEGIN_TRACEBACK -----\n"
            f"{tb_str}"
            "----- END_TRACEBACK -----"
        )

    def _resolve_promotion_letter(self, promotion_role: Optional[str]) -> Optional[str]:
        if promotion_role is None or not isinstance(promotion_role, str):
            return None
        key = promotion_role.strip().lower().replace("-", "_").replace(" ", "_")
        if not key:
            return None
        return _PROMOTION_ROLE_TO_LETTER.get(key)

    def _apply_migration(
        self,
        threat_analysis: str,
        candidate_migrations: list[str],
        justification: str,
        *,
        source_node: dict[str, Any],
        target_node: dict[str, Any],
        promotion_role: Optional[str] = None,
    ) -> str:
        """Adapter: flatten 4D nodes -> UCI -> push onto python-chess.

        Wrapped in a Layer-6 disaster-recovery shell: any exception below
        (illegal move bypassing validation, JSON parsing, Stockfish probe
        failure, etc.) is caught, the board is left untouched, the format
        bucket is docked by ``COMPLIANCE_PENALTY``, and the Python traceback
        is returned verbatim so the LLM can self-correct.
        """
        try:
            return self._apply_migration_unsafe(
                threat_analysis, candidate_migrations, justification,
                source_node=source_node,
                target_node=target_node,
                promotion_role=promotion_role,
            )
        except chess.IllegalMoveError as exc:
            return self._record_compliance_penalty(
                exception=exc,
                attempted=f"{source_node!r}->{target_node!r}",
                threat_analysis=threat_analysis,
                candidate_migrations=candidate_migrations,
                justification=justification,
            )
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            return self._record_compliance_penalty(
                exception=exc,
                attempted=f"{source_node!r}->{target_node!r}",
                threat_analysis=threat_analysis,
                candidate_migrations=candidate_migrations,
                justification=justification,
            )
        except Exception as exc:  # belt-and-braces: never crash the env loop
            return self._record_compliance_penalty(
                exception=exc,
                attempted=f"{source_node!r}->{target_node!r}",
                threat_analysis=threat_analysis,
                candidate_migrations=candidate_migrations,
                justification=justification,
            )

    def _apply_migration_unsafe(
        self,
        threat_analysis: str,
        candidate_migrations: list[str],
        justification: str,
        *,
        source_node: dict[str, Any],
        target_node: dict[str, Any],
        promotion_role: Optional[str] = None,
    ) -> str:
        """Adapter body. Callers must invoke via :meth:`_apply_migration` so
        the disaster-recovery wrapper protects them."""
        # Capture the actor BEFORE the (possible) board.push flips the turn.
        acting_tier = self.current_access_tier
        defender_to_act = self.is_defender_active()

        try:
            src_sq = node_to_square(source_node)
            dst_sq = node_to_square(target_node)
        except Exception as parse_err:
            return self._handle_illegal_migration(
                acting_tier, threat_analysis, candidate_migrations, justification,
                attempted_uci=f"{source_node!r}->{target_node!r}",
                reason=f"Invalid node coordinate: {parse_err}",
            )
        if src_sq == dst_sq:
            return self._handle_illegal_migration(
                acting_tier, threat_analysis, candidate_migrations, justification,
                attempted_uci=f"{_square_to_uci(src_sq)}{_square_to_uci(dst_sq)}",
                reason="source_node and target_node are identical.",
            )

        uci = f"{_square_to_uci(src_sq)}{_square_to_uci(dst_sq)}"

        promo_letter = self._resolve_promotion_letter(promotion_role)
        if promo_letter is not None:
            uci = uci + promo_letter

        # Auto-promote bare data-packet -> edge migrations to Relational_DB_Cluster.
        if len(uci) == 4:
            dest_rank = uci[3]
            try:
                _probe = chess.Move.from_uci(uci + "q")
                _promo_rank = "8" if defender_to_act else "1"
                if dest_rank == _promo_rank and _probe in self.board.legal_moves:
                    uci = uci + "q"
            except Exception:
                pass

        if not _UCI_RE.match(uci):
            return self._handle_illegal_migration(
                acting_tier, threat_analysis, candidate_migrations, justification,
                attempted_uci=uci,
                reason=f"Internal adapter produced invalid migration spec '{uci}'.",
            )
        try:
            move = chess.Move.from_uci(uci)
        except Exception:
            return self._handle_illegal_migration(
                acting_tier, threat_analysis, candidate_migrations, justification,
                attempted_uci=uci,
                reason=f"Adapter could not parse migration spec '{uci}'.",
            )
        if move not in self.board.legal_moves:
            return self._handle_illegal_migration(
                acting_tier, threat_analysis, candidate_migrations, justification,
                attempted_uci=uci,
                reason=(
                    f"Migration {migration_canonical(source_node, target_node)} "
                    "is not authorized in the current topology."
                ),
            )

        self._record_tool_call(
            "migrate_workload",
            threat_analysis, candidate_migrations, justification,
            clean=True, uci_move=uci,
        )

        moving_piece = self.board.piece_at(move.from_square)
        captured_piece = self.board.piece_at(move.to_square)
        was_capture = self.board.is_capture(move)
        moving_asset = _asset_for_piece(moving_piece) if moving_piece else "?"
        captured_asset = _asset_for_piece(captured_piece) if captured_piece else None
        captured_owner = _owner_for_piece(captured_piece) if captured_piece else None

        fen_before = self.board.fen()
        best_cp = self._stockfish.evaluate_cp(fen_before)
        self.board.push(move)  # `current_access_tier` flips automatically here.
        fen_after = self.board.fen()
        raw_after = self._stockfish.evaluate_cp(fen_after)
        after_cp = -raw_after
        cp_loss = max(0, best_cp - after_cp)
        if self._stockfish.ready:
            move_score = max(0.0, 1.0 - (cp_loss / float(SF_BLUNDER_CP)))
        else:
            move_score = 0.5
        self.sf_move_scores[acting_tier].append(move_score)

        canonical_src = square_to_node(move.from_square)
        canonical_dst = square_to_node(move.to_square)
        self.move_history.append(
            {
                "tier": acting_tier,
                "uci": uci,
                "cp_loss": cp_loss,
                "move_score": round(move_score, 3),
                "was_capture": was_capture,
            }
        )
        self.migration_history.append(
            {
                "tier": acting_tier,
                "asset_id": moving_asset,
                "source_node": canonical_src,
                "target_node": canonical_dst,
                "migration": migration_canonical(canonical_src, canonical_dst),
                "captured_asset": captured_asset,
                "captured_owner": captured_owner,
                "cp_loss": cp_loss,
                "move_score": round(move_score, 3),
            }
        )

        # Layer 7: append the successful migration to the compliance audit
        # CSV alongside the Stockfish post-move evaluation.
        _append_compliance_audit(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "episode_id": getattr(self._state, "episode_id", ""),
                "region_label": self.region_label or "(unset)",
                "event_type": "successful_migration",
                "tier": acting_tier,
                "tool": "migrate_workload",
                "source_node": node_canonical(canonical_src),
                "target_node": node_canonical(canonical_dst),
                "uci": uci,
                "threat_analysis": threat_analysis or "",
                "justification": justification or "",
                "cp_loss": cp_loss,
                "move_score": round(move_score, 3),
                "stockfish_eval_cp_white": raw_after,
                "exception_type": "",
                "traceback": "",
            }
        )

        # Did this migration end the engagement?
        outcome = self.board.outcome(claim_draw=True)
        if outcome is not None:
            if outcome.winner is True:
                result = f"checkmate_{DEFENDER_ID}"
            elif outcome.winner is False:
                result = f"checkmate_{ADVERSARY_ID}"
            else:
                result = f"draw_{outcome.termination.name.lower()}"
            self._finalize_episode(result=result)
            label = RESULT_LABELS.get(result, result)
            return (
                f"Migration {migration_canonical(canonical_src, canonical_dst)} "
                f"applied (asset={moving_asset}). Engagement over: {label} "
                f"(cp_loss={cp_loss})."
            )

        # Successful push - ``current_access_tier`` already reflects the new actor.
        self._last_tier_flipped = True
        next_tier = self.current_access_tier
        return (
            f"Migration {migration_canonical(canonical_src, canonical_dst)} applied "
            f"(asset={moving_asset}). next_tier={next_tier} "
            f"cp_loss={cp_loss} move_score={move_score:.2f}"
        )

    def close(self) -> None:
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
        candidate_migrations: list[str],
        justification: str,
        *,
        clean: bool,
        uci_move: Optional[str] = None,
    ) -> None:
        """Count a tool call against the active tier and score its reasoning."""
        tier = self.current_access_tier
        self.tool_calls_total[tier] += 1

        ta_ok = isinstance(threat_analysis, str) and bool(threat_analysis.strip())
        cm_ok = (
            isinstance(candidate_migrations, list)
            and len(candidate_migrations) >= 1
            and any(isinstance(m, str) and m.strip() for m in candidate_migrations)
        )
        ju_ok = isinstance(justification, str) and bool(justification.strip())
        if not (ta_ok and cm_ok and ju_ok):
            clean = False

        if clean:
            self.tool_calls_clean[tier] += 1
        else:
            self.dirty_penalty_accum[tier] += ILLEGAL_FORMAT_PENALTY

        tq_score = self._evaluate_thought_quality(
            tier, tool_name,
            threat_analysis, candidate_migrations, justification,
            uci_move=uci_move,
        )
        self.thought_quality_scores[tier].append(tq_score)

        self._state.step_count += 1
        self.tool_log.append(
            {
                "tier": tier,
                "tool": tool_name,
                "clean": clean,
                "threat_analysis": (threat_analysis or "")[:200],
                "candidate_migrations": (candidate_migrations or [])[:5],
                "justification": (justification or "")[:200],
                "thought_quality": round(tq_score, 4),
            }
        )

    def _evaluate_thought_quality(
        self,
        tier: str,
        tool_name: str,
        threat_analysis: str,
        candidate_migrations: list[str],
        justification: str,
        *,
        uci_move: Optional[str] = None,
    ) -> float:
        """Deterministic heuristic scoring of structured reasoning (max 0.15)."""
        score = 0.0
        ta_lower = (threat_analysis or "").lower()
        ju_lower = (justification or "").lower()
        cm_list = [
            str(m).strip().lower()
            for m in (candidate_migrations or [])
            if m is not None and str(m).strip()
        ]

        if self.board.is_check():
            if "kernel" in ta_lower or "rootkit" in ta_lower or "check" in ta_lower:
                score += 0.05
        elif self.move_history and self.move_history[-1].get("was_capture", False):
            if any(syn in ta_lower for syn in THREAT_SYNONYMS):
                score += 0.05
        else:
            if len(ta_lower.split()) > 5:
                score += 0.05

        if tool_name == "migrate_workload" and uci_move:
            target_canon = _uci_to_migration_str(uci_move).lower()
            if any(target_canon == cand for cand in cm_list):
                score += 0.05
        else:
            if len(cm_list) >= 2:
                score += 0.05

        for concept in SOC_CONCEPTS:
            if re.search(rf"\b{re.escape(concept)}\b", ju_lower):
                score += 0.05
                break

        return score

    def record_malformed_call(self, tier: Optional[str] = None) -> None:
        """Hook for orchestrators when a model emits invalid JSON."""
        target = tier or self.current_access_tier
        if target not in (DEFENDER_ID, ADVERSARY_ID):
            target = self.current_access_tier
        self.tool_calls_total[target] += 1
        self.dirty_penalty_accum[target] += ILLEGAL_FORMAT_PENALTY
        self.thought_quality_scores[target].append(0.0)
        self.tool_log.append({"tier": target, "tool": "(malformed)", "clean": False})

    # -----------------------------------------------------------------
    # Bucket computation
    # -----------------------------------------------------------------

    def _compute_format_score(self, tier: str) -> float:
        total = self.tool_calls_total[tier]
        if total == 0:
            return 0.0
        ratio = self.tool_calls_clean[tier] / total
        base = W_FORMAT * ratio
        ping_penalty = self.ping_count[tier] * PING_BUCKET_PENALTY
        return max(0.0, base - ping_penalty)

    def _compute_thought_quality(self, tier: str) -> float:
        scores = self.thought_quality_scores[tier]
        avg = (sum(scores) / len(scores)) if scores else 0.0
        return max(0.0, min(W_THOUGHT_Q, avg))

    def _compute_sf_acc(self, tier: str) -> float:
        if self.result == f"dq_eval_abuse_{tier}":
            return 0.0
        scores = self.sf_move_scores[tier]
        if not scores:
            return 0.0
        avg_acc = sum(scores) / len(scores)
        base = avg_acc * W_SF
        penalty = self.eval_calls[tier] * EVAL_BUCKET_PENALTY
        return max(0.0, base - penalty)

    def _finalize_outcome(self, result: str) -> None:
        """Translate an internal result code to per-tier outcome bucket payouts."""
        self.result = result

        if result.startswith("checkmate_"):
            winner = result.split("_", 1)[1]
            if winner not in (DEFENDER_ID, ADVERSARY_ID):
                # Defensive fallback for malformed codes.
                self.bucket[DEFENDER_ID]["outcome"] = OUTCOME_DRAW
                self.bucket[ADVERSARY_ID]["outcome"] = OUTCOME_DRAW
                return
            self.bucket[winner]["outcome"] = OUTCOME_WIN
            self.bucket[_opponent_tier(winner)]["outcome"] = OUTCOME_LOSS
            return

        if result.startswith("draw_"):
            self.bucket[DEFENDER_ID]["outcome"] = OUTCOME_DRAW
            self.bucket[ADVERSARY_ID]["outcome"] = OUTCOME_DRAW
            return

        if result.startswith("resign_"):
            loser = result.split("_", 1)[1]
            if loser not in (DEFENDER_ID, ADVERSARY_ID):
                self.bucket[DEFENDER_ID]["outcome"] = OUTCOME_DRAW
                self.bucket[ADVERSARY_ID]["outcome"] = OUTCOME_DRAW
                return
            winner = _opponent_tier(loser)
            self.bucket[winner]["outcome"] = OUTCOME_RESIGN_WIN
            self.bucket[loser]["outcome"] = OUTCOME_LOSS
            return

        if result.startswith("dq_"):
            offender = result.rsplit("_", 1)[1]
            if offender not in (DEFENDER_ID, ADVERSARY_ID):
                self.bucket[DEFENDER_ID]["outcome"] = OUTCOME_DRAW
                self.bucket[ADVERSARY_ID]["outcome"] = OUTCOME_DRAW
                return
            winner = _opponent_tier(offender)
            self.bucket[winner]["outcome"] = OUTCOME_DQ_WIN
            self.bucket[offender]["outcome"] = OUTCOME_LOSS
            return

        self.bucket[DEFENDER_ID]["outcome"] = OUTCOME_DRAW
        self.bucket[ADVERSARY_ID]["outcome"] = OUTCOME_DRAW

    def _finalize_episode(self, *, result: str) -> None:
        if self.done:
            return
        self.done = True
        self._finalize_outcome(result)
        for tier in TIERS:
            b = self.bucket[tier]
            b["format"] = self._compute_format_score(tier)
            b["thought_q"] = self._compute_thought_quality(tier)
            b["sf_acc"] = self._compute_sf_acc(tier)
            total = b["outcome"] + b["format"] + b["thought_q"] + b["sf_acc"]
            self.final_reward[tier] = _clamp(total)

    def _preview_reward(self, tier: str) -> float:
        outcome_so_far = self.bucket[tier]["outcome"]
        total = (
            outcome_so_far
            + self._compute_format_score(tier)
            + self._compute_thought_quality(tier)
            + self._compute_sf_acc(tier)
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
        """Start a new SOC engagement. Returns initial observation (no FEN)."""
        if hasattr(self, "_stockfish") and self._stockfish is not None:
            self._stockfish.close()
        self._init_fresh_state()
        if seed is not None:
            pass

        options = kwargs.get("options") or {}
        start_fen = options.get("fen") or kwargs.get("fen")
        if start_fen:
            try:
                self.board = chess.Board(start_fen)
            except Exception as _fen_err:
                import sys
                print(
                    f"[DatacenterEnvironment] WARNING: Could not parse seed FEN '{start_fen}': "
                    f"{_fen_err}. Falling back to the default starting topology.",
                    file=sys.stderr,
                )
                self.board = chess.Board()

        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        DatacenterEnvironment._instances[self._state.episode_id] = self
        DatacenterEnvironment._latest_instance = self

        return Observation(
            done=False,
            reward=_clamp(0.0),
            metadata={
                "active_tier": self.current_access_tier,
                "topology": self.get_topology_state(),
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        return Observation(
            done=self.done,
            reward=_clamp(self._preview_reward(self.current_access_tier)),
            metadata={
                "active_tier": self.current_access_tier,
                "topology": self.get_topology_state(),
            },
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
        if req_ep and req_ep in DatacenterEnvironment._instances:
            active = DatacenterEnvironment._instances[req_ep]
        elif req_ep:
            raise KeyError(
                f"Unknown episode_id {req_ep!r}. Call /reset first to obtain a valid episode."
            )
        elif not req_ep:
            if DatacenterEnvironment._latest_instance is not None:
                active = DatacenterEnvironment._latest_instance

        token = _active_env.set(active)
        try:
            if isinstance(action, ListToolsAction):
                return super().step(action, timeout_s=timeout_s, **kwargs)

            obs = super().step(action, timeout_s=timeout_s, **kwargs)

            if isinstance(action, CallToolAction) and isinstance(obs, CallToolObservation):
                obs.done = active.done
                if active.done:
                    reward_tier = _current_actor_for_observation(active, action)
                    obs.reward = _clamp(active.final_reward.get(reward_tier, R_MIN))
                    DatacenterEnvironment._instances.pop(active._state.episode_id, None)
                else:
                    reward_tier = _current_actor_for_observation(active, action)
                    obs.reward = _clamp(active._preview_reward(reward_tier))

                debug_payload: dict[str, Any] = {
                    "active_tier": active.current_access_tier,
                    "topology": active.get_topology_state(),
                    "result": active.result,
                    "result_label": (
                        RESULT_LABELS.get(active.result, active.result)
                        if active.result else None
                    ),
                    "done": active.done,
                    "bucket": {
                        t: {k: round(v, 4) for k, v in active.bucket[t].items()}
                        for t in TIERS
                    },
                    "final_reward": {
                        t: _clamp(active.final_reward[t]) for t in TIERS
                    },
                    "scores": {
                        "defender_efficiency": active.get_defender_efficiency(),
                        "adversary_threat_level": active.get_adversary_threat_level(),
                    },
                    "eval_calls": dict(active.eval_calls),
                    "ping_count": dict(active.ping_count),
                    "tool_calls": {
                        "clean": dict(active.tool_calls_clean),
                        "total": dict(active.tool_calls_total),
                    },
                    "thought_quality": {
                        t: round(
                            sum(active.thought_quality_scores[t]) / len(active.thought_quality_scores[t])
                            if active.thought_quality_scores[t] else 0.0,
                            4,
                        )
                        for t in TIERS
                    },
                }

                md = dict(getattr(obs, "metadata", {}) or {})
                md.update(debug_payload)
                obs.metadata = md
                _inject_openenv_payload(obs, debug_payload)
            return obs
        finally:
            _active_env.reset(token)

    @property
    def state(self) -> State:
        return self._state

    def snapshot(self) -> dict[str, Any]:
        """Lightweight snapshot for logging / dashboards."""
        return {
            "topology": self.get_topology_state(),
            "active_tier": self.current_access_tier,
            "done": self.done,
            "result": self.result,
            "result_label": (
                RESULT_LABELS.get(self.result, self.result) if self.result else None
            ),
            "bucket": {
                t: {k: round(v, 4) for k, v in self.bucket[t].items()}
                for t in TIERS
            },
            "final_reward": {t: _clamp(self.final_reward[t]) for t in TIERS},
            "scores": {
                "defender_efficiency": self.get_defender_efficiency(),
                "adversary_threat_level": self.get_adversary_threat_level(),
            },
            "eval_calls": dict(self.eval_calls),
            "ping_count": dict(self.ping_count),
            "tool_calls": {
                "clean": dict(self.tool_calls_clean),
                "total": dict(self.tool_calls_total),
            },
            "move_history": list(self.move_history),
            "migration_history": list(self.migration_history),
        }


def _inject_openenv_payload(obs: CallToolObservation, payload: dict[str, Any]) -> None:
    """Stuff our per-tier debug payload into the tool result structure."""
    res = obs.result
    if res is None:
        return
    if isinstance(res, dict):
        sc = res.get("structured_content")
        if not isinstance(sc, dict):
            sc = {}
        else:
            sc = dict(sc)
        sc["openenv"] = payload
        res["structured_content"] = sc
        return
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
    env: "DatacenterEnvironment", action: CallToolAction
) -> str:
    """Determine which tier should receive the reward on this observation."""
    if env.done:
        return env.current_access_tier
    if getattr(action, "tool_name", None) == "migrate_workload":
        if env._last_tier_flipped:
            return _opponent_tier(env.current_access_tier)
    return env.current_access_tier
