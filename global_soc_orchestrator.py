#!/usr/bin/env python3
"""
Global SOC Orchestrator - the multi-region driver of the simulation.

Architecture
------------
1. **Multi-region simulation** - 3 isolated :class:`DatacenterEnvironment`
   instances, each representing a distinct global region (e.g. us-east,
   eu-west, ap-south). Each region has its own ``chess.Board`` + Stockfish
   subprocess, so per-region damage scoring is independent.

2. **Async adversary swarm** - on every ADVERSARY turn, three hostile
   profiles (DB_Backup_Agent, Viral_Traffic_Agent, Chaos_Monkey) are polled
   concurrently via :mod:`asyncio` for that region's topology. Each one
   returns a candidate ``migrate_workload`` payload.

3. **Physics oracle (triage)** - the orchestrator scores all 3 candidates
   with the region's Stockfish engine by simulating each move on a
   ``chess.Board`` copy. The candidate that drops the DEFENDER's evaluation
   the most (highest centipawn loss) wins. Only that single migration is
   pushed into the live region env.

4. **L1 Defender (round-robin)** - one DEFENDER agent rotates Region 1 ->
   Region 2 -> Region 3 -> Region 1 -> ... in a continuous loop. The active
   region completes one DEFENDER + one ADVERSARY half-move pair per visit.

The orchestrator is fully chess-free at the surface: it speaks tier ids
(``DEFENDER_ID`` / ``ADVERSARY_ID``), 4D node dicts, and migration strings.
The only chess imports are needed to validate candidate moves and run the
Stockfish triage.

CLI
---
::

    python global_soc_orchestrator.py --regions 3 --cycles 10

You will need API credentials in ``.env`` / ``.env.local`` (same shape as
``inference.py``: ``API_BASE_URL``, ``HF_TOKEN`` / ``OPENAI_API_KEY``, and
``GOOGLE_MODEL_*`` / ``GROQ_MODEL_*`` model names).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import secrets
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import chess
import chess.engine  # noqa: F401  -- needed for chess.engine.Limit in damage scoring

# Ensure the in-tree server module is importable when running as a script.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from openenv.core.env_server.mcp_types import CallToolAction  # noqa: E402

from server.datacenter_env import (  # noqa: E402
    ADVERSARY_ID,
    COMPLIANCE_AUDIT_LOG_PATH,
    COMPLIANCE_PENALTY,
    DEFENDER_ID,
    DatacenterEnvironment,
    _append_compliance_audit,
    _square_to_uci,
    migration_canonical,
    node_canonical,
    node_to_square,
    square_to_node,
)
import traceback as _traceback
from datetime import timezone as _tz

from agent_inference import (  # noqa: E402
    DatacenterAgent,
    AgentDecision,
    make_chaos_monkey_agent,
    make_db_backup_agent,
    make_defender_agent,
    make_random_adversary,
    make_viral_traffic_agent,
    make_static_policy,
)

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Stochastic distribution + elastic swarm constants
# ---------------------------------------------------------------------------
#
# `_SR` is a process-wide :class:`secrets.SystemRandom` instance used for
# every "who runs what" decision: model->role assignment, persona pick,
# swarm size, model-per-adversary draw, etc. Using ``secrets`` instead of
# :mod:`random` removes the determinism that an attacker (or a memoising
# agent) could exploit to predict the next adversary line-up.

_SR: secrets.SystemRandom = secrets.SystemRandom()

# Default elastic swarm size range. Each ADVERSARY turn picks N from
# this interval; ``incident_clock_scaling`` (orchestrator flag) further
# inflates the upper bound as the engagement deepens.
SWARM_SIZE_MIN: int = 1
SWARM_SIZE_MAX: int = 10

# Defender pool size range. The orchestrator draws between 1 and 5
# defenders (uniformly with ``_SR``) from the unified model pool and
# assigns each region a stable defender at reset time.
#
# Defender selection is *deliberately random*: this is a Stochastic SOC
# Benchmark, so handing the role to a hard-coded "smartest" model would
# defeat the purpose -- it is the equivalent of putting Stockfish on
# the defense and calling it RL. The benchmark must reward emergent
# competence under uncertainty, not curated dominance.
#
# Multiple defenders are also a load-distribution feature: when N
# adversaries hit N regions concurrently, the orchestrator can route
# each region's traffic to a different defender so no single model
# becomes the rate-limit bottleneck or context-window choke point.
DEFENDER_POOL_MIN: int = 1
DEFENDER_POOL_MAX: int = 5


# ===========================================================================
# RegionRunner - one (env, name, history) bundle per global region
# ===========================================================================


@dataclass
class RegionRunner:
    """Wraps a single :class:`DatacenterEnvironment` instance + per-region histories.

    The ``defender`` field is the agent assigned to this region for the
    current engagement. The orchestrator picks it stochastically from a
    multi-model defender pool at reset time, so two regions in the same
    run can be defended by different models (load-spreading: 10
    concurrent attacks won't all hit the same upstream API).
    """

    region_id: str
    region_name: str
    env: DatacenterEnvironment
    defender: Optional[DatacenterAgent] = None
    defender_history: list[dict[str, Any]] = field(default_factory=list)
    adversary_histories: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    cycle_index: int = 0
    last_result: Optional[str] = None

    def reset(
        self,
        defender: DatacenterAgent,
        adversaries: Optional[list[DatacenterAgent]] = None,
    ) -> None:
        """Reset the env, stash the assigned defender, and rebuild buffers.

        ``adversaries`` is optional: in elastic-swarm mode the orchestrator
        spawns fresh per-turn adversary instances (uuid4 names + random
        personas), so there is no static set of profiles to preseed. The
        ``adversary_histories`` dict is cleared in that case and the
        per-turn agents bring their own short-lived buffers.
        """
        self.env.reset()
        # Layer 7: stamp this region's label onto the env so every audit row
        # the env writes (successful migrations, HITL escalations, compliance
        # penalties) carries an unambiguous region attribution.
        self.env.region_label = self.region_name
        self.cycle_index = 0
        self.last_result = None
        self.defender = defender
        self.defender_history = defender.new_region_buffer(self.region_id)
        if adversaries:
            self.adversary_histories = {
                adv.profile: adv.new_region_buffer(self.region_id) for adv in adversaries
            }
        else:
            self.adversary_histories = {}


# ===========================================================================
# Physics Oracle - Stockfish-backed triage of adversary candidates
# ===========================================================================


@dataclass
class _Candidate:
    """A single triaged adversary payload."""

    profile: str
    decision: AgentDecision
    uci: Optional[str] = None
    move: Optional[chess.Move] = None
    damage_cp: float = float("-inf")  # higher == more damaging to DEFENDER
    error: Optional[str] = None

    @property
    def is_legal(self) -> bool:
        return self.move is not None and self.error is None


def _resolve_uci_from_decision(
    env: DatacenterEnvironment,
    args: dict[str, Any],
) -> tuple[Optional[str], Optional[str]]:
    """Translate a ``migrate_workload`` arg dict to a UCI string, or return error."""
    src = args.get("source_node")
    dst = args.get("target_node")
    if not isinstance(src, dict) or not isinstance(dst, dict):
        return None, "missing source_node/target_node"
    try:
        src_sq = node_to_square(src)
        dst_sq = node_to_square(dst)
    except Exception as e:
        return None, f"invalid node coord: {e}"
    if src_sq == dst_sq:
        return None, "source_node == target_node"
    uci = _square_to_uci(src_sq) + _square_to_uci(dst_sq)

    promo_letter = env._resolve_promotion_letter(args.get("promotion_role"))
    if promo_letter is not None:
        uci += promo_letter
    elif len(uci) == 4:
        # Auto-promote bare pawn-edge migrations to Relational_DB_Cluster (queen).
        try:
            probe = chess.Move.from_uci(uci + "q")
            promo_rank = "8" if env.is_defender_active() else "1"
            if uci[3] == promo_rank and probe in env.board.legal_moves:
                uci += "q"
        except Exception:
            pass
    return uci, None


def _score_candidate_damage(env: DatacenterEnvironment, uci: str) -> tuple[float, chess.Move]:
    """Centipawn damage to DEFENDER from this candidate, evaluated on a copy.

    The metric: drop in DEFENDER's Stockfish evaluation between the position
    BEFORE and AFTER the candidate move. Both evaluations are taken from the
    DEFENDER (white) perspective, so a more negative post-move eval =
    higher damage.

    The function returns ``(damage_cp, chess.Move)`` or raises ValueError if
    the move is illegal.
    """
    move = chess.Move.from_uci(uci)
    if move not in env.board.legal_moves:
        raise ValueError(f"illegal move {uci}")

    # Pre-move eval: ADVERSARY is to act, but evaluate_cp returns
    # side-to-move-relative cp. We rebuild from white perspective so both
    # before/after are comparable. We do this with a small helper using the
    # underlying engine analyse() so we don't depend on internal sign logic.
    sf = env._stockfish
    if not sf.ready:
        # No engine available -> degrade gracefully: prefer captures.
        damage = 100.0 if env.board.is_capture(move) else 0.0
        return damage, move

    fen_before = env.board.fen()
    eval_before_white = _white_eval_cp(sf, fen_before)
    board_copy = env.board.copy()
    board_copy.push(move)
    eval_after_white = _white_eval_cp(sf, board_copy.fen())
    damage = float(eval_before_white - eval_after_white)
    return damage, move


def _white_eval_cp(sf: Any, fen: str) -> int:
    """Stockfish cp evaluation from WHITE (DEFENDER) perspective for any position."""
    if not getattr(sf, "ready", False) or sf._engine is None:
        return 0
    try:
        board = chess.Board(fen)
        info = sf._engine.analyse(board, limit=chess.engine.Limit(time=0.3))
        score = info["score"].white()
        if score.is_mate():
            mate_in = score.mate()
            sign = 1 if mate_in > 0 else -1
            return sign * 10000
        return score.score() or 0
    except Exception:
        return 0


def physics_oracle_triage(
    env: DatacenterEnvironment,
    decisions: list[AgentDecision],
) -> tuple[Optional[_Candidate], list[_Candidate]]:
    """Triage a swarm: return ``(winner, all_candidates)``.

    The winner is the legal candidate with the highest ``damage_cp``; if no
    candidate is legal, returns ``(None, candidates)`` so the orchestrator
    can fall back to a random legal migration.
    """
    candidates: list[_Candidate] = []
    for d in decisions:
        cand = _Candidate(profile=d.profile, decision=d)
        if d.tool != "migrate_workload":
            cand.error = f"non-migration tool ({d.tool!r}) - swarm requires migrate_workload"
            candidates.append(cand)
            continue
        uci, err = _resolve_uci_from_decision(env, d.arguments)
        if err is not None or uci is None:
            cand.error = err or "uci resolution failed"
            candidates.append(cand)
            continue
        try:
            damage, move = _score_candidate_damage(env, uci)
        except Exception as exc:
            cand.error = str(exc)
            candidates.append(cand)
            continue
        cand.uci = uci
        cand.move = move
        cand.damage_cp = damage
        candidates.append(cand)

    legal = [c for c in candidates if c.is_legal]
    if not legal:
        return None, candidates
    winner = max(legal, key=lambda c: c.damage_cp)
    return winner, candidates


# ===========================================================================
# GlobalSOCOrchestrator
# ===========================================================================


@dataclass
class CycleRecord:
    """Audit trail for one defender-then-adversary half-move pair in a region."""

    region_id: str
    cycle_index: int
    defender_decision: Optional[dict[str, Any]] = None
    defender_reward: Optional[float] = None
    adversary_winner: Optional[str] = None  # winning profile id
    adversary_damage_cp: Optional[float] = None
    adversary_decision: Optional[dict[str, Any]] = None
    adversary_swarm: list[dict[str, Any]] = field(default_factory=list)
    adversary_reward: Optional[float] = None
    done: bool = False
    result: Optional[str] = None
    fallback_used: bool = False


class GlobalSOCOrchestrator:
    """Drive N regions with one Defender and an Elastic Adversary Swarm.

    Two adversary modes are supported:

    * **Legacy static mode** (``adversaries=[...]``): the orchestrator was
      built with a fixed list of adversary profiles (e.g. the 3 stub agents
      used by ``--dry-run``). Every ADVERSARY turn polls the same set.
    * **Elastic swarm mode** (``adversary_pool=[(model, client), ...]``):
      every ADVERSARY turn picks a fresh size ``N`` (drawn from
      ``swarm_size_range`` and optionally inflated by ``incident_clock``),
      then spawns ``N`` :func:`make_random_adversary` instances with
      uuid4 display names + randomized persona overlays + models drawn
      stochastically (with replacement, so self-play is allowed).

    The defender is always a single agent. In elastic mode the highest
    intelligence-ranked model is reserved for the defender (it is the
    *brain* that decides whether to fire ``escalate_to_sysadmin`` -- the
    actual human override stays a blocking ``input()`` in
    :meth:`_hitl_human_override`).
    """

    def __init__(
        self,
        regions: list[RegionRunner],
        defender: Optional[DatacenterAgent] = None,
        adversaries: Optional[list[DatacenterAgent]] = None,
        *,
        defenders: Optional[list[DatacenterAgent]] = None,
        adversary_pool: Optional[list[tuple[str, Any]]] = None,
        swarm_size_range: tuple[int, int] = (SWARM_SIZE_MIN, SWARM_SIZE_MAX),
        incident_clock_scaling: bool = True,
        hitl_enabled: Optional[bool] = None,
    ) -> None:
        if not regions:
            raise ValueError("at least one region runner is required")

        # Coalesce single ``defender`` (legacy positional arg) and explicit
        # ``defenders`` list into a single defender pool. Either one is
        # accepted, but at least one must be present.
        defender_pool: list[DatacenterAgent] = list(defenders or [])
        if defender is not None:
            defender_pool.insert(0, defender)
        if not defender_pool:
            raise ValueError(
                "at least one defender agent is required (pass either `defender=...` "
                "or `defenders=[...]`)"
            )
        for d in defender_pool:
            if d.profile != "defender":
                raise ValueError(
                    f"defender agent must have profile 'defender', got {d.profile!r}"
                )

        if not adversaries and not adversary_pool:
            raise ValueError(
                "either `adversaries` (static list) or `adversary_pool` "
                "(elastic mode) must be provided"
            )

        lo, hi = swarm_size_range
        if lo < 1 or hi < lo:
            raise ValueError(
                f"swarm_size_range must satisfy 1 <= lo <= hi, got {swarm_size_range!r}"
            )

        self.regions = regions
        self.defenders: list[DatacenterAgent] = defender_pool
        self.adversaries: list[DatacenterAgent] = list(adversaries or [])
        self.adversary_pool: list[tuple[str, Any]] = list(adversary_pool or [])
        self.swarm_size_range: tuple[int, int] = (lo, hi)
        self.incident_clock_scaling: bool = incident_clock_scaling
        self.audit_trail: list[CycleRecord] = []
        # Layer 5 control: set to False for unattended runs (CI, dry-runs).
        # If unset, default to True only when stdin looks interactive so the
        # loop never blocks on input() in non-interactive contexts.
        if hitl_enabled is None:
            hitl_enabled = bool(getattr(sys.stdin, "isatty", lambda: False)())
        self.hitl_enabled: bool = hitl_enabled

    @property
    def defender(self) -> DatacenterAgent:
        """First defender in the pool. Kept as a back-compat shim; per-turn
        callers should use :attr:`RegionRunner.defender` instead so the
        right agent is invoked when the pool has more than one model.
        """
        return self.defenders[0]

    def _assign_region_defender(self, region: RegionRunner) -> DatacenterAgent:
        """Pick a defender from the pool for this region (CSPRNG, with
        replacement). Two regions can land on the same model -- that is
        fine, they keep independent context buffers.
        """
        return self.defenders[_SR.randrange(len(self.defenders))]

    # ----- elastic swarm helpers ---------------------------------------

    @property
    def is_elastic(self) -> bool:
        """``True`` when the adversary side is drawn from a model pool per turn."""
        return bool(self.adversary_pool)

    def _next_swarm_size(self, incident_clock: int) -> int:
        """Draw the swarm size N for the upcoming ADVERSARY turn.

        N is uniformly sampled from ``swarm_size_range``; when
        :attr:`incident_clock_scaling` is set, the upper bound is widened
        as ``incident_clock`` grows so that long engagements increasingly
        face the full 10-model swarm. The lower bound is left alone so a
        quiet engagement can still see a single-adversary turn.
        """
        lo, hi = self.swarm_size_range
        if self.incident_clock_scaling and incident_clock > 0:
            # incident_clock=0 -> hi unchanged; ramp linearly with a soft
            # floor so even early turns occasionally see a large swarm.
            scaled_hi = max(lo, min(hi, lo + (hi - lo) * (incident_clock + 1) // max(1, hi)))
            # Above formula caps at ``hi`` once incident_clock >= hi-lo.
            hi_eff = max(lo, min(hi, scaled_hi if scaled_hi >= lo else hi))
        else:
            hi_eff = hi
        return _SR.randint(lo, hi_eff)

    def _build_elastic_swarm(self, size: int) -> list[DatacenterAgent]:
        """Spawn ``size`` short-lived adversaries from the configured pool."""
        if not self.adversary_pool:
            return []
        agents: list[DatacenterAgent] = []
        for _ in range(size):
            # Models drawn with replacement so self-play is supported and
            # the swarm can include duplicates of the same intelligence
            # level (which lets the Stockfish triage pick the better
            # *prompt* even when models are tied).
            model_name, client = self.adversary_pool[
                _SR.randrange(len(self.adversary_pool))
            ]
            agents.append(make_random_adversary(client, model_name))
        return agents

    # ----- environment helpers -----------------------------------------

    def reset_all(self) -> None:
        adversaries_for_reset = self.adversaries if not self.is_elastic else None
        for r in self.regions:
            assigned_defender = self._assign_region_defender(r)
            r.reset(assigned_defender, adversaries_for_reset)
            _log(
                f"   [{r.region_id}] defender assignment: "
                f"{assigned_defender.model_name}"
            )

    def snapshot(self) -> dict[str, Any]:
        return {
            "regions": [
                {
                    "region_id": r.region_id,
                    "region_name": r.region_name,
                    "cycle_index": r.cycle_index,
                    "done": r.env.done,
                    "result": r.env.result,
                    "scores": {
                        "defender_efficiency": r.env.get_defender_efficiency(),
                        "adversary_threat_level": r.env.get_adversary_threat_level(),
                    },
                }
                for r in self.regions
            ],
            "audit_trail": [vars(c) for c in self.audit_trail],
        }

    # ----- low-level: apply one decision to a region's env -------------

    def _apply_decision(
        self,
        region: RegionRunner,
        agent: DatacenterAgent,
        decision: AgentDecision,
    ) -> tuple[float, str, bool]:
        """Step the env with this decision; return (reward, tool_text, done)."""
        tool = decision.tool
        args = dict(decision.arguments or {})
        args.setdefault("threat_analysis", "")
        args.setdefault("candidate_migrations", [])
        args.setdefault("justification", "")

        if not tool:
            # Malformed: dock the format bucket and emit a fallback scan.
            region.env.record_malformed_call(region.env.current_access_tier)
            tool_text = "(malformed: model produced no tool call)"
            agent.append_tool_result(
                self._history_for(region, agent), "(malformed)", tool_text
            )
            return region.env.get_defender_efficiency() if agent.profile == "defender" else (
                region.env.get_adversary_threat_level()
            ), tool_text, region.env.done

        action = CallToolAction(tool_name=tool, arguments=args)

        # Layer 6 (outer safety net): the env's _apply_migration is already
        # wrapped in a disaster-recovery shell, but we add a belt-and-braces
        # try/except here so that a failure in the framework layer (FastMCP,
        # contextvars, transport) cannot crash the orchestrator loop. If we
        # land in this except, we synthesize a compliance penalty and feed
        # the traceback back into the agent's history.
        try:
            obs = region.env.step(action, episode_id=region.env._state.episode_id)
        except Exception as exc:
            tb_str = _traceback.format_exc()
            tier = region.env.current_access_tier
            try:
                region.env.compliance_penalties[tier] += 1
                region.env.dirty_penalty_accum[tier] += COMPLIANCE_PENALTY
            except Exception:
                pass
            try:
                _append_compliance_audit(
                    {
                        "timestamp_utc": datetime.now(_tz.utc).isoformat(),
                        "episode_id": getattr(region.env._state, "episode_id", ""),
                        "region_label": region.env.region_label or region.region_name,
                        "event_type": "compliance_penalty",
                        "tier": tier,
                        "tool": tool or "(none)",
                        "threat_analysis": args.get("threat_analysis", "") or "",
                        "justification": args.get("justification", "") or "",
                        "exception_type": type(exc).__name__,
                        "traceback": tb_str,
                    }
                )
            except Exception:
                pass
            tool_text = (
                f"COMPLIANCE_PENALTY (orchestrator): {type(exc).__name__}: {exc}\n"
                f"Penalty applied (-{COMPLIANCE_PENALTY:.02f}). Self-correct.\n"
                f"----- BEGIN_TRACEBACK -----\n{tb_str}----- END_TRACEBACK -----"
            )
            agent.append_tool_result(self._history_for(region, agent), tool, tool_text)
            reward = (
                region.env.get_defender_efficiency()
                if agent.profile == "defender"
                else region.env.get_adversary_threat_level()
            )
            return reward, tool_text, bool(getattr(region.env, "done", False))

        # Extract a short tool-result string for the agent's history.
        tool_text = self._extract_tool_text(obs)
        agent.append_tool_result(self._history_for(region, agent), tool, tool_text)

        # Reward is on the observation; choose which side we're attributing to.
        reward = float(getattr(obs, "reward", 0.0) or 0.0)
        return reward, tool_text, bool(getattr(obs, "done", False))

    def _history_for(
        self,
        region: RegionRunner,
        agent: DatacenterAgent,
    ) -> list[dict[str, Any]]:
        if agent.profile == "defender":
            return region.defender_history
        return region.adversary_histories.setdefault(
            agent.profile, agent.new_region_buffer(region.region_id)
        )

    @staticmethod
    def _extract_tool_text(obs: Any) -> str:
        """Pull a short, human-readable string out of a CallToolObservation."""
        res = getattr(obs, "result", None)
        if res is None:
            return "(no result)"
        if isinstance(res, dict):
            content = res.get("content")
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict) and "text" in first:
                    return str(first["text"])[:600]
            return json.dumps(res, default=str)[:600]
        if hasattr(res, "content") and isinstance(res.content, list) and res.content:
            first = res.content[0]
            if hasattr(first, "text"):
                return str(getattr(first, "text", ""))[:600]
        return str(res)[:600]

    # ----- Layer 5: HITL human override prompt --------------------------

    @staticmethod
    def _read_node_from_stdin(label: str) -> Optional[dict[str, Any]]:
        """Read a single 4D node (region/zone/rack/pod) from the human operator.

        Returns ``None`` if the operator aborts (blank input on the first
        field). Loose parsing: we forward the raw strings to ``node_to_square``
        which already accepts loose forms like ``us-east``, ``1``, ``rack-2``.
        """
        try:
            r = input(f"  {label}.region (e.g. us-east): ").strip()
        except EOFError:
            return None
        if not r:
            return None
        z = input(f"  {label}.zone   (e.g. az-a):    ").strip()
        rk = input(f"  {label}.rack   (e.g. rack-2):  ").strip()
        pd = input(f"  {label}.pod    (e.g. pod-3):   ").strip()
        return {"region": r, "zone": z, "rack": rk, "pod": pd}

    def _hitl_human_override(
        self,
        region: RegionRunner,
        escalation_decision: AgentDecision,
    ) -> Optional[AgentDecision]:
        """Pause the region, print a banner, and read a human-typed migration.

        This call BLOCKS the asyncio event loop on synchronous ``input()``,
        which is intentional: the directive specifies that an
        ``escalate_to_sysadmin`` (PROTOCOL RED) handoff must pause the loop
        until the human supplies a valid 4D routing path.

        Returns a synthetic ``migrate_workload`` ``AgentDecision`` that the
        caller applies through the same defender pipeline, or ``None`` if
        the operator aborts (in which case the region simply ends this
        defender turn with no engine push - the env-side escalation row has
        already been written to the audit log).
        """
        bar = "=" * 80
        args = escalation_decision.arguments or {}
        threat_level = str(args.get("threat_level", "")).strip().upper() or "(unspecified)"
        mitigation_request = str(args.get("mitigation_request", "")).strip() or "(none)"
        justification = str(args.get("justification", "")).strip() or "(none)"

        print("\n" + bar, file=sys.stderr, flush=True)
        print(
            "PROTOCOL RED: CATASTROPHIC ANOMALY. SYSADMIN OVERRIDE REQUIRED.",
            file=sys.stderr, flush=True,
        )
        print(
            f"  region={region.region_id} ({region.region_name})",
            file=sys.stderr, flush=True,
        )
        print(f"  threat_level={threat_level}", file=sys.stderr, flush=True)
        print(
            f"  mitigation_request={mitigation_request}",
            file=sys.stderr, flush=True,
        )
        print(f"  defender_justification={justification}", file=sys.stderr, flush=True)
        print(
            "Enter manual 4D coordinates (src->dst):",
            file=sys.stderr, flush=True,
        )
        print(
            "  (Press ENTER on the first prompt with no input to ABORT and "
            "skip this defender half-move.)",
            file=sys.stderr, flush=True,
        )
        print(bar + "\n", file=sys.stderr, flush=True)

        for attempt in range(1, 4):
            print(f"  [HITL attempt {attempt}/3]  source_node:", file=sys.stderr, flush=True)
            src = self._read_node_from_stdin("source")
            if src is None:
                print("  HITL aborted by operator. Skipping defender half-move.", file=sys.stderr, flush=True)
                return None
            print("  target_node:", file=sys.stderr, flush=True)
            dst = self._read_node_from_stdin("target")
            if dst is None:
                print("  HITL aborted by operator. Skipping defender half-move.", file=sys.stderr, flush=True)
                return None

            try:
                src_sq = node_to_square(src)
                dst_sq = node_to_square(dst)
                src_canon = square_to_node(src_sq)
                dst_canon = square_to_node(dst_sq)
                migration = migration_canonical(src_canon, dst_canon)
            except Exception as parse_err:
                print(
                    f"  Could not parse coordinates: {parse_err}. Try again.",
                    file=sys.stderr, flush=True,
                )
                continue

            print(
                f"  HITL accepting human override: {migration} "
                f"({node_canonical(src_canon)} -> {node_canonical(dst_canon)})",
                file=sys.stderr, flush=True,
            )
            return AgentDecision(
                tool="migrate_workload",
                arguments={
                    "threat_analysis": (
                        f"PROTOCOL RED human sysadmin override "
                        f"(threat_level={threat_level}). "
                        f"Mitigation request: {mitigation_request}. "
                        f"Defender justification: {justification}"
                    ),
                    "candidate_migrations": [migration],
                    "justification": (
                        "Layer 5 human override; auto-recorded by the orchestrator."
                    ),
                    "source_node": src_canon,
                    "target_node": dst_canon,
                },
                raw="(HITL operator input)",
                profile="defender_hitl",
                region_id=region.region_id,
            )

        print("  HITL exhausted 3 attempts. Skipping defender half-move.", file=sys.stderr, flush=True)
        return None

    # ----- defender turn ------------------------------------------------

    def defender_step(self, region: RegionRunner) -> CycleRecord:
        """Take exactly one DEFENDER half-move in this region (if it is the
        defender's turn). Returns a partial CycleRecord for this region."""
        record = CycleRecord(region_id=region.region_id, cycle_index=region.cycle_index)

        if region.env.done:
            record.done = True
            record.result = region.env.result
            return record

        if not region.env.is_defender_active():
            # Defender shouldn't be acting; orchestrator drives this in pairs.
            return record

        topo = region.env.get_topology_state()
        # Per-region defender: the orchestrator stamped this on the
        # RegionRunner during reset. Different regions can use different
        # defenders so a 10-region engagement is not bottlenecked on one
        # model's rate limit / context window.
        defender_agent = region.defender or self.defenders[0]
        decision = defender_agent.choose(
            topo, region.defender_history, region_id=region.region_id
        )
        record.defender_decision = decision.to_dict()
        reward, tool_text, done = self._apply_decision(region, defender_agent, decision)
        record.defender_reward = reward
        record.done = done
        record.result = region.env.result

        _log(
            f"   [{region.region_id}] DEFENDER ({defender_agent.model_name}) "
            f"-> {decision.tool} reward={reward:.3f} done={done}"
        )

        # Layer 5: detect a sanctioned PROTOCOL RED escalation and run the
        # human override flow. Only ``escalate_to_sysadmin`` triggers the
        # asyncio pause + input() prompt; ``escalate_to_oncall`` is the
        # non-fatal trap and must NOT pause the loop, even if the model
        # accidentally returns a HITL_REQUIRED-looking string.
        is_hitl = decision.tool == "escalate_to_sysadmin"
        if is_hitl and not done:
            if not self.hitl_enabled:
                _log(
                    f"   [{region.region_id}] HITL escalation received but "
                    "hitl_enabled=False; recording escalation only."
                )
                return record
            human_decision = self._hitl_human_override(region, decision)
            if human_decision is not None:
                h_reward, h_tool_text, h_done = self._apply_decision(
                    region, defender_agent, human_decision
                )
                record.defender_decision = human_decision.to_dict()
                record.defender_reward = h_reward
                record.done = h_done
                record.result = region.env.result
                _log(
                    f"   [{region.region_id}] DEFENDER (HITL human override) -> "
                    f"{human_decision.tool} reward={h_reward:.3f} done={h_done}"
                )

        return record

    # ----- adversary swarm turn ----------------------------------------

    async def adversary_swarm_step(self, region: RegionRunner) -> CycleRecord:
        """Run the elastic N-way adversary swarm + Stockfish triage.

        In **elastic mode** (``self.is_elastic``), the orchestrator picks a
        fresh swarm size ``N`` (1..10, optionally inflated by the incident
        clock), spawns ``N`` :func:`make_random_adversary` agents from the
        model pool, polls them concurrently with :func:`asyncio.gather`,
        and triages with the Stockfish oracle. The buffers for these
        ephemeral agents are scrubbed at the end of the turn to bound
        memory growth.

        In **legacy static mode** the orchestrator uses ``self.adversaries``
        and the per-region persistent histories (used by ``--dry-run``).
        """
        record = CycleRecord(region_id=region.region_id, cycle_index=region.cycle_index)

        if region.env.done:
            record.done = True
            record.result = region.env.result
            return record

        if not region.env.is_adversary_active():
            # Already adversary's turn flipped (e.g. defender ended the engagement).
            return record

        topo_snapshot = region.env.get_topology_state()

        # Build the active swarm for this turn: either the persistent
        # static list, or a fresh N spawned from the elastic pool.
        incident_clock = int(getattr(region.env.board, "fullmove_number", 0))
        if self.is_elastic:
            # Wipe ephemeral histories from the previous elastic turn so
            # uuid4-keyed entries don't accumulate forever.
            region.adversary_histories.clear()
            swarm_size = self._next_swarm_size(incident_clock)
            active_swarm = self._build_elastic_swarm(swarm_size)
        else:
            active_swarm = list(self.adversaries)
            swarm_size = len(active_swarm)

        if not active_swarm:
            _log(f"   [{region.region_id}] ADVERSARY swarm is empty; skipping.")
            return record

        _log(
            f"   [{region.region_id}] ADVERSARY swarm size: N={swarm_size} "
            f"(elastic={self.is_elastic}, incident_clock={incident_clock})"
        )

        # Fan out: each adversary writes to its own history buffer so they
        # do not mutate one shared list mid-flight.
        tasks = []
        for adv in active_swarm:
            buf = self._history_for(region, adv)
            tasks.append(adv.choose_async(topo_snapshot, buf, region_id=region.region_id))

        decisions: list[AgentDecision] = []
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        for adv, res in zip(active_swarm, gathered):
            if isinstance(res, BaseException):
                _log(f"   [{region.region_id}] adversary {adv.profile} raised: {res!r}")
                decisions.append(AgentDecision(
                    tool=None, arguments={}, raw=f"(exception: {res!r})",
                    profile=adv.profile, region_id=region.region_id,
                ))
            else:
                decisions.append(res)

        # Build a profile -> agent index for fast model_name lookup; the
        # physics oracle returns by profile, and we want to log the
        # winning model id for Layer-7 transparency.
        agent_by_profile: dict[str, DatacenterAgent] = {a.profile: a for a in active_swarm}

        winner, all_candidates = physics_oracle_triage(region.env, decisions)

        record.adversary_swarm = [
            {
                "profile": c.profile,
                "model": getattr(agent_by_profile.get(c.profile), "model_name", None),
                "uci": c.uci,
                "damage_cp": c.damage_cp if c.damage_cp != float("-inf") else None,
                "error": c.error,
                "tool": c.decision.tool,
            }
            for c in all_candidates
        ]

        if winner is None:
            # All N candidates illegal -> fall back to a random legal migration.
            record.fallback_used = True
            fallback = self._random_legal_migration(region.env)
            if fallback is None:
                record.done = True
                record.result = region.env.result
                _log(f"   [{region.region_id}] swarm produced no legal moves and no fallback.")
                self._audit_elastic_swarm(
                    region=region,
                    swarm_size=swarm_size,
                    candidates=all_candidates,
                    winner=None,
                    winner_model=None,
                    fallback_used=True,
                )
                return record
            decision = fallback
            _log(
                f"   [{region.region_id}] ADVERSARY swarm: no legal candidate, "
                f"using random fallback {decision.arguments.get('source_node')} -> "
                f"{decision.arguments.get('target_node')}"
            )
        else:
            decision = winner.decision
            record.adversary_winner = winner.profile
            record.adversary_damage_cp = winner.damage_cp
            winning_model = getattr(agent_by_profile.get(winner.profile), "model_name", None)
            _log(
                f"   [{region.region_id}] ADVERSARY swarm winner: profile={winner.profile} "
                f"model={winning_model} damage_cp={winner.damage_cp:.1f} uci={winner.uci}"
            )

        # Apply the winning decision through the *winning* agent's buffer
        # so the assistant turn lands in the right context window. In
        # elastic mode the winning agent comes from this turn's swarm; in
        # legacy mode it comes from the static list.
        winning_agent = agent_by_profile.get(decision.profile)
        if winning_agent is None:
            # Fallback decision was synthesized by the orchestrator
            # itself (profile=="orchestrator_fallback"); attribute the
            # tool result to the first swarm member so its history stays
            # consistent.
            winning_agent = active_swarm[0]

        record.adversary_decision = decision.to_dict()
        reward, tool_text, done = self._apply_decision(region, winning_agent, decision)
        record.adversary_reward = reward
        record.done = done
        record.result = region.env.result

        self._audit_elastic_swarm(
            region=region,
            swarm_size=swarm_size,
            candidates=all_candidates,
            winner=winner,
            winner_model=getattr(agent_by_profile.get(decision.profile), "model_name", None),
            fallback_used=record.fallback_used,
        )
        return record

    def _audit_elastic_swarm(
        self,
        *,
        region: RegionRunner,
        swarm_size: int,
        candidates: list["_Candidate"],
        winner: Optional["_Candidate"],
        winner_model: Optional[str],
        fallback_used: bool,
    ) -> None:
        """Layer-7 audit row: swarm size + winning model id for transparency."""
        try:
            legal_count = sum(1 for c in candidates if c.is_legal)
            _append_compliance_audit(
                {
                    "timestamp_utc": datetime.now(_tz.utc).isoformat(),
                    "episode_id": getattr(region.env._state, "episode_id", ""),
                    "region_label": region.env.region_label or region.region_name,
                    "event_type": "elastic_swarm",
                    "swarm_size": swarm_size,
                    "candidate_count": len(candidates),
                    "legal_candidate_count": legal_count,
                    "winner_profile": winner.profile if winner else None,
                    "winner_model": winner_model,
                    "winner_damage_cp": (
                        float(winner.damage_cp) if winner and winner.damage_cp != float("-inf") else None
                    ),
                    "winner_uci": winner.uci if winner else None,
                    "fallback_used": fallback_used,
                    "incident_clock": int(getattr(region.env.board, "fullmove_number", 0)),
                }
            )
        except Exception as exc:
            # Audit failure must never break the engagement loop.
            _log(f"   [{region.region_id}] elastic_swarm audit failed: {exc!r}")

    def _random_legal_migration(self, env: DatacenterEnvironment) -> Optional[AgentDecision]:
        """Pick any legal migration so the engagement keeps progressing."""
        legal = list(env.board.legal_moves)
        if not legal:
            return None
        mv = random.choice(legal)
        src = square_to_node(mv.from_square)
        dst = square_to_node(mv.to_square)
        canonical = migration_canonical(src, dst)
        return AgentDecision(
            tool="migrate_workload",
            arguments={
                "threat_analysis": "Fallback: orchestrator-selected random legal migration.",
                "candidate_migrations": [canonical],
                "justification": "Swarm failed to produce a legal candidate; keeping the engagement live.",
                "source_node": src,
                "target_node": dst,
            },
            raw="(orchestrator fallback)",
            profile="orchestrator_fallback",
        )

    # ----- the L1 Defender continuous loop -----------------------------

    async def run_async(
        self,
        max_cycles: int = 10,
        *,
        per_region_pause: float = 0.0,
    ) -> list[CycleRecord]:
        """Round-robin loop: per cycle, visit each region; in each region run
        one DEFENDER half-move, then one ADVERSARY swarm half-move.

        Stops early when every region's engagement has terminated.
        """
        self.reset_all()

        for cycle in range(max_cycles):
            all_done = True
            for region in self.regions:
                if region.env.done:
                    continue
                all_done = False
                region.cycle_index = cycle
                _log(f"\n=== CYCLE {cycle + 1}/{max_cycles}  region={region.region_id} ===")

                # 1. DEFENDER half-move (synchronous - one agent at a time).
                d_rec = self.defender_step(region)
                self.audit_trail.append(d_rec)
                if region.env.done:
                    region.last_result = region.env.result
                    _log(f"   [{region.region_id}] DEFENDER ended engagement: {region.env.result}")
                    continue

                # 2. ADVERSARY swarm half-move (async fan-out).
                a_rec = await self.adversary_swarm_step(region)
                self.audit_trail.append(a_rec)
                if region.env.done:
                    region.last_result = region.env.result
                    _log(f"   [{region.region_id}] ADVERSARY ended engagement: {region.env.result}")

                if per_region_pause > 0:
                    await asyncio.sleep(per_region_pause)

            if all_done:
                _log("\nAll regions terminated; halting loop.")
                break

        return self.audit_trail

    def run(self, max_cycles: int = 10, *, per_region_pause: float = 0.0) -> list[CycleRecord]:
        """Synchronous entry point that drives :meth:`run_async`."""
        return asyncio.run(self.run_async(max_cycles, per_region_pause=per_region_pause))

    def close(self) -> None:
        """Tear down every region's resources (mainly the Stockfish subprocess).

        Without this, ``main()`` returns but the Python process hangs on
        exit because the chess.engine UCI subprocess threads are still
        alive. Idempotent and safe to call multiple times.
        """
        for r in self.regions:
            try:
                r.env.close()
            except Exception as exc:
                _log(f"   [{r.region_id}] env.close() raised: {exc!r}")


# ===========================================================================
# Factory: build a 3-region orchestrator from CLI args
# ===========================================================================


DEFAULT_REGION_NAMES = ("us-east-prod", "eu-west-prod", "ap-south-prod")


def build_orchestrator(
    *,
    defender: Optional[DatacenterAgent] = None,
    defenders: Optional[list[DatacenterAgent]] = None,
    adversaries: Optional[list[DatacenterAgent]] = None,
    adversary_pool: Optional[list[tuple[str, Any]]] = None,
    swarm_size_range: tuple[int, int] = (SWARM_SIZE_MIN, SWARM_SIZE_MAX),
    incident_clock_scaling: bool = True,
    region_names: tuple[str, ...] = DEFAULT_REGION_NAMES,
    hitl_enabled: Optional[bool] = None,
) -> GlobalSOCOrchestrator:
    """Instantiate one ``DatacenterEnvironment`` per region and bundle them.

    Pass ``defenders`` (a list of pre-built defender agents) for the
    multi-defender stochastic-assignment mode; ``defender=`` (singular)
    is accepted for the simple single-model case.

    Pass ``adversary_pool`` (a list of ``(model_name, openai_client)``
    tuples) to enable elastic-swarm mode. Pass ``adversaries`` (a list
    of pre-built :class:`DatacenterAgent` instances) for legacy static
    mode (used by ``--dry-run``).
    """
    runners: list[RegionRunner] = []
    for i, name in enumerate(region_names):
        env = DatacenterEnvironment()
        env.region_label = name
        runners.append(
            RegionRunner(
                region_id=f"region-{i + 1}",
                region_name=name,
                env=env,
            )
        )
    return GlobalSOCOrchestrator(
        runners,
        defender,
        adversaries=adversaries,
        defenders=defenders,
        adversary_pool=adversary_pool,
        swarm_size_range=swarm_size_range,
        incident_clock_scaling=incident_clock_scaling,
        hitl_enabled=hitl_enabled,
    )


# ===========================================================================
# CLI
# ===========================================================================


def _load_env_file(path: Path) -> dict[str, str]:
    try:
        from dotenv import dotenv_values
    except ImportError:
        return {}
    if not path.is_file():
        return {}
    return {k: v for k, v in dotenv_values(str(path)).items() if isinstance(v, str)}


# ---------------------------------------------------------------------------
# Provider config + unified model-pool resolution
# ---------------------------------------------------------------------------
#
# Each env file contributes a homogeneous set of model ids hosted by one
# OpenAI-compatible provider. We discover them by scanning a deterministic
# numeric prefix (``GOOGLE_MODEL_1..N``, ``HF_MODEL_1..N``, ``GROQ_MODEL_1..N``)
# and cap the scan at 64 to avoid runaway loops on misconfigured files.
#
# The unified pool is the concatenation of every provider's models. The
# *defender* is reserved as the highest intelligence-ranked model; the
# adversary pool retains the full list (including the defender's model)
# so self-play is supported per the user's requirement that "same model
# can be in traffic and safety in both files".


@dataclass(frozen=True)
class _ProviderSpec:
    name: str
    env_path: Path
    model_prefix: str
    fallback_base_url: str


@dataclass(frozen=True)
class _ProviderConfig:
    name: str
    base_url: str
    api_key: str
    models: tuple[str, ...]


_PROVIDER_SPECS: tuple[_ProviderSpec, ...] = (
    _ProviderSpec(
        name="google",
        env_path=_HERE / ".env",
        model_prefix="GOOGLE_MODEL_",
        fallback_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
    _ProviderSpec(
        name="hf",
        env_path=_HERE / ".env.local",
        model_prefix="HF_MODEL_",
        fallback_base_url="https://router.huggingface.co/v1",
    ),
    _ProviderSpec(
        name="groq",
        env_path=_HERE / ".env.test.local",
        model_prefix="GROQ_MODEL_",
        fallback_base_url="https://api.groq.com/openai/v1",
    ),
)


def _scan_models_from_env(env: dict[str, str], prefix: str, *, limit: int = 64) -> list[str]:
    """Scan ``PREFIX1..PREFIXN`` keys until the first gap (or ``limit``)."""
    out: list[str] = []
    for i in range(1, limit + 1):
        v = env.get(f"{prefix}{i}")
        if not v:
            # First gap stops the scan; we treat the env file as a
            # contiguous list (which matches how the user actually
            # numbers them).
            break
        v = v.strip()
        if v:
            out.append(v)
    return out


def _load_provider_configs() -> dict[str, _ProviderConfig]:
    """Discover every provider's base URL + API key + model list."""
    out: dict[str, _ProviderConfig] = {}
    for spec in _PROVIDER_SPECS:
        env = _load_env_file(spec.env_path)
        if not env:
            continue
        models = _scan_models_from_env(env, spec.model_prefix)
        if not models:
            continue
        base_url = env.get("API_BASE_URL", spec.fallback_base_url)
        api_key = (
            env.get("HF_TOKEN")
            or env.get("OPENAI_API_KEY")
            or env.get("API_KEY")
            or os.getenv("HF_TOKEN", "")
            or os.getenv("OPENAI_API_KEY", "")
        )
        out[spec.name] = _ProviderConfig(
            name=spec.name,
            base_url=base_url,
            api_key=api_key,
            models=tuple(models),
        )
    return out


def _make_openai_client_for(cfg: _ProviderConfig) -> Any:
    """Build an OpenAI-compatible client for one provider config."""
    if OpenAI is None:
        raise RuntimeError("openai package not installed; pip install openai")
    return OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)


def _resolve_models(
    provider_cfgs: dict[str, _ProviderConfig],
    *,
    defender_pool_min: int = DEFENDER_POOL_MIN,
    defender_pool_max: int = DEFENDER_POOL_MAX,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return ``(defender_pool, adversary_pool)``.

    Both lists hold ``(model_name, provider_name)`` tuples. Selection is
    deliberately stochastic (``_SR``); there is no intelligence-tier
    filter. This is a Stochastic SOC Benchmark -- handing the defender
    role to a curated "smartest" model collapses the experiment.

    The defender pool size is drawn uniformly from
    ``[defender_pool_min, min(defender_pool_max, len(unified_pool))]``.
    Models are sampled *with replacement*, so a small unified pool can
    still produce a multi-defender setup with duplicates (and self-play
    against the adversary side is supported by construction).

    The adversary pool is the full unified pool, shuffled once. The
    elastic swarm later draws per-turn models from it with replacement.
    """
    pool: list[tuple[str, str]] = []
    for cfg in provider_cfgs.values():
        for m in cfg.models:
            pool.append((m, cfg.name))

    if not pool:
        raise RuntimeError(
            "no models found; populate GOOGLE_MODEL_*, HF_MODEL_*, or "
            "GROQ_MODEL_* in .env / .env.local / .env.test.local"
        )
    if len(pool) < 2:
        raise RuntimeError(
            "unified model pool has only 1 entry; need >=2 so defender "
            "and adversary swarm can each draw stochastically. Self-play "
            "needs at least 2 distinct draws."
        )

    lo = max(1, defender_pool_min)
    hi = max(lo, min(defender_pool_max, len(pool)))
    defender_count = _SR.randint(lo, hi)

    # With-replacement draw: small pools still get full defender_count
    # rosters, and identical model entries are allowed (e.g. two regions
    # could end up with the same model -- fine, they each carry their own
    # context buffer).
    defender_pool: list[tuple[str, str]] = [
        pool[_SR.randrange(len(pool))] for _ in range(defender_count)
    ]

    adv_pool: list[tuple[str, str]] = list(pool)
    _SR.shuffle(adv_pool)
    return defender_pool, adv_pool


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Global SOC multi-region orchestrator.")
    parser.add_argument("--cycles", type=int, default=int(os.getenv("SOC_CYCLES", "10")))
    parser.add_argument("--regions", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--per-region-pause", type=float, default=0.0,
                        help="Optional pause (s) between regions in a cycle.")
    parser.add_argument("--out", type=str, default=None,
                        help="Path to write the audit trail JSON (defaults to results/).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLM calls; use a deterministic stub policy.")
    parser.add_argument(
        "--no-hitl", action="store_true",
        help=(
            "Disable Layer-5 HITL prompts. Escalations are still recorded in "
            "the audit log but the loop never blocks on input(). Auto-enabled "
            "when stdin is non-interactive."
        ),
    )
    parser.add_argument(
        "--swarm-min", type=int, default=SWARM_SIZE_MIN,
        help=f"Minimum elastic adversary swarm size (default {SWARM_SIZE_MIN}).",
    )
    parser.add_argument(
        "--swarm-max", type=int, default=SWARM_SIZE_MAX,
        help=f"Maximum elastic adversary swarm size (default {SWARM_SIZE_MAX}).",
    )
    parser.add_argument(
        "--defender-pool-min", type=int, default=DEFENDER_POOL_MIN,
        help=f"Minimum number of defender models drawn from the unified pool (default {DEFENDER_POOL_MIN}).",
    )
    parser.add_argument(
        "--defender-pool-max", type=int, default=DEFENDER_POOL_MAX,
        help=f"Maximum number of defender models drawn from the unified pool (default {DEFENDER_POOL_MAX}).",
    )
    parser.add_argument(
        "--no-clock-scaling", action="store_true",
        help="Disable incident-clock-based scaling of the swarm upper bound.",
    )
    args = parser.parse_args()

    swarm_size_range = (max(1, args.swarm_min), max(args.swarm_min, args.swarm_max))
    incident_clock_scaling = not args.no_clock_scaling

    if args.dry_run:
        defender_stub, static_adversaries = _build_stub_agents()
        defender_agents: list[DatacenterAgent] = [defender_stub]
        adversary_pool: Optional[list[tuple[str, Any]]] = None
        _log("Running in --dry-run mode (no LLM calls; legacy static-3 swarm, single stub defender).")
    else:
        if OpenAI is None:
            _log("ERROR: openai package not installed.")
            sys.exit(1)

        # Discover every provider's model list, base URL, and API key
        # across the three env files (.env / .env.local / .env.test.local).
        provider_cfgs = _load_provider_configs()
        if not provider_cfgs:
            _log("ERROR: no provider configs found in .env / .env.local / .env.test.local.")
            sys.exit(1)

        try:
            defender_pool_specs, adv_pool_specs = _resolve_models(
                provider_cfgs,
                defender_pool_min=max(1, args.defender_pool_min),
                defender_pool_max=max(args.defender_pool_min, args.defender_pool_max),
            )
        except RuntimeError as exc:
            _log(f"ERROR: {exc}")
            sys.exit(1)

        # One OpenAI client per provider (reused across all models from
        # that provider). Building these eagerly avoids re-creating an
        # HTTP session per adversary instantiation.
        clients_by_provider: dict[str, Any] = {
            name: _make_openai_client_for(cfg) for name, cfg in provider_cfgs.items()
        }

        defender_agents = [
            make_defender_agent(clients_by_provider[provider], model)
            for model, provider in defender_pool_specs
        ]
        static_adversaries = None
        adversary_pool = [
            (model, clients_by_provider[provider]) for model, provider in adv_pool_specs
        ]

        _log("Unified model pool:")
        for cfg in provider_cfgs.values():
            _log(f"  [{cfg.name}] {len(cfg.models)} model(s) @ {cfg.base_url}")
        _log(
            f"Defender pool: {len(defender_pool_specs)} model(s) "
            f"(stochastic, with-replacement from the unified pool)"
        )
        for model, provider in defender_pool_specs:
            _log(f"   - {model}  [{provider}]")
        _log(
            f"Elastic adversary pool: {len(adversary_pool)} model(s) "
            f"(swarm size each turn ~ U[{swarm_size_range[0]}, {swarm_size_range[1]}], "
            f"incident_clock_scaling={incident_clock_scaling})"
        )
        for model, provider in adv_pool_specs[:10]:
            _log(f"   - {model}  [{provider}]")
        if len(adv_pool_specs) > 10:
            _log(f"   ... and {len(adv_pool_specs) - 10} more.")

    region_names = DEFAULT_REGION_NAMES[: args.regions]
    hitl_enabled: Optional[bool] = False if args.no_hitl else None
    orchestrator = build_orchestrator(
        defenders=defender_agents,
        adversaries=static_adversaries,
        adversary_pool=adversary_pool,
        swarm_size_range=swarm_size_range,
        incident_clock_scaling=incident_clock_scaling,
        region_names=region_names,
        hitl_enabled=hitl_enabled,
    )

    _log("=" * 60)
    _log(f" Global SOC Orchestrator -- {args.regions} region(s), {args.cycles} cycle(s)")
    _log(f" HITL enabled: {orchestrator.hitl_enabled}")
    _log(f" Compliance audit log: {COMPLIANCE_AUDIT_LOG_PATH}")
    _log("=" * 60)

    t0 = time.time()
    try:
        trail = orchestrator.run(max_cycles=args.cycles, per_region_pause=args.per_region_pause)
        elapsed = time.time() - t0

        _log(f"\nCompleted {len(trail)} half-move records in {elapsed:.1f}s.")
        snap = orchestrator.snapshot()
        for r in snap["regions"]:
            _log(
                f"  [{r['region_id']}] {r['region_name']}  "
                f"done={r['done']} result={r['result']} "
                f"defender={r['scores']['defender_efficiency']:.3f} "
                f"adversary={r['scores']['adversary_threat_level']:.3f}"
            )

        out_dir = Path(args.out).parent if args.out else _HERE / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (
            Path(args.out)
            if args.out
            else out_dir / f"soc_orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(snap, f, indent=2, default=str)
        _log(f"\nAudit trail written to: {out_path}")
    finally:
        # Critical: every region holds a Stockfish UCI subprocess. If we
        # don't tear them down, Python's exit-time thread join blocks
        # forever on the engine reader thread and the dry-run "never
        # stops" even though main() has logically completed.
        orchestrator.close()


# ===========================================================================
# Stub agents for --dry-run (no network calls)
# ===========================================================================


def _build_stub_agents() -> tuple[DatacenterAgent, list[DatacenterAgent]]:
    """Build a quartet of pure-Python policies that always pick a random legal
    migration. Useful for orchestrator unit tests and end-to-end dry runs.
    """
    def _make_random_policy(profile: str):
        def _decide(messages: list[dict[str, Any]]) -> dict[str, Any]:
            # Find the most recent topology snapshot in the buffer and choose
            # any legal migration from its authorized list. If none is found
            # (e.g. the agent has no live env state), return a no-op scan.
            for m in reversed(messages):
                if m.get("role") == "user" and "Live topology snapshot" in m.get("content", ""):
                    body = m["content"].split("\n", 1)[1]
                    try:
                        topo = json.loads(body)
                    except Exception:
                        topo = {}
                    workloads = topo.get("active_workloads", [])
                    # Chaos Layer injects shadow nodes with decoy axis values
                    # that intentionally fail `node_to_square` validation. The
                    # dry-run stub must avoid those or it will trip Layer-6
                    # honeypots immediately and crash before the loop can
                    # demonstrate progress.
                    routable_workloads = []
                    for w in workloads:
                        node = w.get("node")
                        if not isinstance(node, dict):
                            continue
                        try:
                            node_to_square(node)
                        except Exception:
                            continue
                        routable_workloads.append(w)
                    candidates = [
                        w for w in routable_workloads
                        if w.get("owner") == (
                            DEFENDER_ID if profile == "defender" else ADVERSARY_ID
                        )
                    ]
                    if candidates:
                        src = random.choice(candidates)["node"]
                        # Pick any other node as target; the Stockfish triage
                        # will reject illegal ones, and orchestrator falls back
                        # to a known-legal random move.
                        dst_pick = random.choice(routable_workloads)["node"]
                        return {
                            "tool": "migrate_workload",
                            "arguments": {
                                "threat_analysis": f"[stub:{profile}] random target",
                                "candidate_migrations": [migration_canonical(src, dst_pick)],
                                "justification": f"[stub:{profile}] dry-run heuristic",
                                "source_node": src,
                                "target_node": dst_pick,
                            },
                            "raw": "(stub policy)",
                        }
                    break
            return {
                "tool": "scan_topology",
                "arguments": {
                    "threat_analysis": f"[stub:{profile}] no topology yet",
                    "candidate_migrations": ["us-east/az-a/rack-1/pod-1->us-east/az-a/rack-1/pod-2"],
                    "justification": f"[stub:{profile}] inspecting state",
                },
                "raw": "(stub policy fallback)",
            }
        return make_static_policy(_decide)

    defender = DatacenterAgent(_make_random_policy("defender"), profile="defender")
    adversaries = [
        DatacenterAgent(_make_random_policy("db_backup"), profile="db_backup"),
        DatacenterAgent(_make_random_policy("viral_traffic"), profile="viral_traffic"),
        DatacenterAgent(_make_random_policy("chaos_monkey"), profile="chaos_monkey"),
    ]
    return defender, adversaries


if __name__ == "__main__":
    main()
