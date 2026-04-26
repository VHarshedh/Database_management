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
    HITL_SIGNAL_PREFIX,
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


# ===========================================================================
# RegionRunner - one (env, name, history) bundle per global region
# ===========================================================================


@dataclass
class RegionRunner:
    """Wraps a single :class:`DatacenterEnvironment` instance + per-region histories."""

    region_id: str
    region_name: str
    env: DatacenterEnvironment
    defender_history: list[dict[str, Any]] = field(default_factory=list)
    adversary_histories: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    cycle_index: int = 0
    last_result: Optional[str] = None

    def reset(self, defender: DatacenterAgent, adversaries: list[DatacenterAgent]) -> None:
        """Reset the env and rebuild per-agent message buffers."""
        self.env.reset()
        # Layer 7: stamp this region's label onto the env so every audit row
        # the env writes (successful migrations, HITL escalations, compliance
        # penalties) carries an unambiguous region attribution.
        self.env.region_label = self.region_name
        self.cycle_index = 0
        self.last_result = None
        self.defender_history = defender.new_region_buffer(self.region_id)
        self.adversary_histories = {
            adv.profile: adv.new_region_buffer(self.region_id) for adv in adversaries
        }


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
    """Drive 3 regions with one Defender and a 3-way Adversary swarm.

    Constructor parameters:

    * ``regions``: list of ``RegionRunner`` instances. Typically 3.
    * ``defender``: a single :class:`DatacenterAgent` with profile ``defender``.
    * ``adversaries``: list of 3 :class:`DatacenterAgent` instances (one per profile).
    """

    def __init__(
        self,
        regions: list[RegionRunner],
        defender: DatacenterAgent,
        adversaries: list[DatacenterAgent],
        *,
        hitl_enabled: Optional[bool] = None,
    ) -> None:
        if not regions:
            raise ValueError("at least one region runner is required")
        if defender.profile != "defender":
            raise ValueError(f"defender agent must have profile 'defender', got {defender.profile!r}")
        if not adversaries:
            raise ValueError("at least one adversary agent is required")

        self.regions = regions
        self.defender = defender
        self.adversaries = adversaries
        self.audit_trail: list[CycleRecord] = []
        # Layer 5 control: set to False for unattended runs (CI, dry-runs).
        # If unset, default to True only when stdin looks interactive so the
        # loop never blocks on input() in non-interactive contexts.
        if hitl_enabled is None:
            hitl_enabled = bool(getattr(sys.stdin, "isatty", lambda: False)())
        self.hitl_enabled: bool = hitl_enabled

    # ----- environment helpers -----------------------------------------

    def reset_all(self) -> None:
        for r in self.regions:
            r.reset(self.defender, self.adversaries)

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

        Returns a synthetic ``migrate_workload`` ``AgentDecision`` that the
        caller applies through the same defender pipeline, or ``None`` if
        the operator aborts (in which case the region simply ends this
        defender turn with no engine push - the env-side escalation row has
        already been written to the audit log).
        """
        bar = "!" * 78
        reason = (escalation_decision.arguments or {}).get("justification", "(none)")
        severity = (escalation_decision.arguments or {}).get("severity", "high")
        print("\n" + bar, file=sys.stderr, flush=True)
        print(
            f"!! HITL OVERRIDE REQUIRED  region={region.region_id} "
            f"({region.region_name})",
            file=sys.stderr, flush=True,
        )
        print(f"!! Severity: {severity}", file=sys.stderr, flush=True)
        print(f"!! Defender justification: {reason}", file=sys.stderr, flush=True)
        print(
            "!! Active tier yields its half-move. Type the mitigating 4D node "
            "coordinates below.",
            file=sys.stderr, flush=True,
        )
        print(
            "!! (Press ENTER on the first prompt with no input to ABORT and skip "
            "this defender half-move.)",
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
                        "Human sysadmin override via HITL escalation. "
                        f"Defender justification: {reason}"
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
        decision = self.defender.choose(topo, region.defender_history, region_id=region.region_id)
        record.defender_decision = decision.to_dict()
        reward, tool_text, done = self._apply_decision(region, self.defender, decision)
        record.defender_reward = reward
        record.done = done
        record.result = region.env.result

        _log(
            f"   [{region.region_id}] DEFENDER -> {decision.tool} "
            f"reward={reward:.3f} done={done}"
        )

        # Layer 5: detect a HITL escalation and run the human override flow.
        # The env-side tool returns a string starting with HITL_REQUIRED; we
        # detect on either the tool name OR the result text so a model that
        # forgets the tool name but emits the magic prefix still triggers.
        is_hitl = (
            decision.tool == "escalate_to_sysadmin"
            or (isinstance(tool_text, str) and tool_text.startswith(HITL_SIGNAL_PREFIX))
        )
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
                    region, self.defender, human_decision
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
        """Run the 3-way adversary swarm + Stockfish triage in this region."""
        record = CycleRecord(region_id=region.region_id, cycle_index=region.cycle_index)

        if region.env.done:
            record.done = True
            record.result = region.env.result
            return record

        if not region.env.is_adversary_active():
            # Already adversary's turn flipped (e.g. defender ended the engagement).
            return record

        topo_snapshot = region.env.get_topology_state()

        # Spawn the 3 adversaries concurrently (each on its own copy of the buffer
        # so they don't all mutate the same history list mid-flight).
        tasks = []
        for adv in self.adversaries:
            buf = self._history_for(region, adv)
            tasks.append(adv.choose_async(topo_snapshot, buf, region_id=region.region_id))

        decisions: list[AgentDecision] = []
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        for adv, res in zip(self.adversaries, gathered):
            if isinstance(res, BaseException):
                _log(f"   [{region.region_id}] adversary {adv.profile} raised: {res!r}")
                decisions.append(AgentDecision(
                    tool=None, arguments={}, raw=f"(exception: {res!r})",
                    profile=adv.profile, region_id=region.region_id,
                ))
            else:
                decisions.append(res)

        winner, all_candidates = physics_oracle_triage(region.env, decisions)

        record.adversary_swarm = [
            {
                "profile": c.profile,
                "uci": c.uci,
                "damage_cp": c.damage_cp if c.damage_cp != float("-inf") else None,
                "error": c.error,
                "tool": c.decision.tool,
            }
            for c in all_candidates
        ]

        if winner is None:
            # All 3 candidates illegal -> fall back to a random legal migration.
            record.fallback_used = True
            fallback = self._random_legal_migration(region.env)
            if fallback is None:
                record.done = True
                record.result = region.env.result
                _log(f"   [{region.region_id}] swarm produced no legal moves and no fallback.")
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
            _log(
                f"   [{region.region_id}] ADVERSARY swarm winner: {winner.profile} "
                f"damage_cp={winner.damage_cp:.1f} uci={winner.uci}"
            )

        # Apply the winning decision through the env via the matching adversary's
        # history (we use the winning profile's buffer so its assistant turn lands
        # in its own context window).
        winning_agent = next(
            (a for a in self.adversaries if a.profile == decision.profile),
            self.adversaries[0],
        )
        record.adversary_decision = decision.to_dict()
        reward, tool_text, done = self._apply_decision(region, winning_agent, decision)
        record.adversary_reward = reward
        record.done = done
        record.result = region.env.result
        return record

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


# ===========================================================================
# Factory: build a 3-region orchestrator from CLI args
# ===========================================================================


DEFAULT_REGION_NAMES = ("us-east-prod", "eu-west-prod", "ap-south-prod")


def build_orchestrator(
    *,
    defender: DatacenterAgent,
    adversaries: list[DatacenterAgent],
    region_names: tuple[str, ...] = DEFAULT_REGION_NAMES,
    hitl_enabled: Optional[bool] = None,
) -> GlobalSOCOrchestrator:
    """Instantiate one ``DatacenterEnvironment`` per region and bundle them."""
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
    return GlobalSOCOrchestrator(runners, defender, adversaries, hitl_enabled=hitl_enabled)


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


def _make_openai_client(provider: str, env_google: dict, env_groq: dict) -> Any:
    if OpenAI is None:
        raise RuntimeError("openai package not installed; pip install openai")
    if provider == "google":
        return OpenAI(
            base_url=env_google.get(
                "API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
            ),
            api_key=env_google.get("HF_TOKEN") or env_google.get("OPENAI_API_KEY")
            or os.getenv("HF_TOKEN", ""),
        )
    return OpenAI(
        base_url=env_groq.get("API_BASE_URL", "https://api.groq.com/openai/v1"),
        api_key=env_groq.get("HF_TOKEN") or env_groq.get("OPENAI_API_KEY")
        or os.getenv("HF_TOKEN", ""),
    )


def _resolve_models(env_google: dict, env_groq: dict) -> tuple[str, list[tuple[str, str]]]:
    """Pick a defender model + 3 adversary models from the configured pools.

    Returns ``(defender_model, [(adv_model, provider), ...])``.
    """
    pool: list[tuple[str, str]] = []
    for i in range(1, 5):
        n = env_google.get(f"GOOGLE_MODEL_{i}")
        if n:
            pool.append((n, "google"))
    for i in range(1, 5):
        n = env_groq.get(f"GROQ_MODEL_{i}")
        if n:
            pool.append((n, "groq"))

    if len(pool) < 4:
        raise RuntimeError(
            f"need at least 4 models for the L1 Defender + 3 Adversaries, found {len(pool)}"
        )

    random.shuffle(pool)
    defender_model = pool[0][0]
    adversaries = pool[1:4]
    return defender_model, adversaries


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
    args = parser.parse_args()

    # Load API config from .env / .env.local (same pattern as inference.py).
    env_google = _load_env_file(_HERE / ".env")
    env_groq = _load_env_file(_HERE / ".env.local")

    if args.dry_run:
        defender, adversaries = _build_stub_agents()
        _log("Running in --dry-run mode (no LLM calls).")
    else:
        if OpenAI is None:
            _log("ERROR: openai package not installed.")
            sys.exit(1)
        try:
            defender_model, adv_pool = _resolve_models(env_google, env_groq)
        except RuntimeError as exc:
            _log(f"ERROR: {exc}")
            sys.exit(1)

        # Defender uses a single client (whichever provider hosts its model).
        def_provider = "google" if any(
            defender_model == n for n, _ in [(env_google.get(f"GOOGLE_MODEL_{i}"), "google") for i in range(1, 5)]
        ) else "groq"
        def_client = _make_openai_client(def_provider, env_google, env_groq)
        defender = make_defender_agent(def_client, defender_model)

        adv_clients = [_make_openai_client(p, env_google, env_groq) for _, p in adv_pool]
        adversaries = [
            make_db_backup_agent(adv_clients[0], adv_pool[0][0]),
            make_viral_traffic_agent(adv_clients[1], adv_pool[1][0]),
            make_chaos_monkey_agent(adv_clients[2], adv_pool[2][0]),
        ]
        _log(f"Defender model        : {defender_model}")
        _log(f"DB_Backup_Agent       : {adv_pool[0][0]}")
        _log(f"Viral_Traffic_Agent   : {adv_pool[1][0]}")
        _log(f"Chaos_Monkey          : {adv_pool[2][0]}")

    region_names = DEFAULT_REGION_NAMES[: args.regions]
    hitl_enabled: Optional[bool] = False if args.no_hitl else None
    orchestrator = build_orchestrator(
        defender=defender,
        adversaries=adversaries,
        region_names=region_names,
        hitl_enabled=hitl_enabled,
    )

    _log("=" * 60)
    _log(f" Global SOC Orchestrator -- {args.regions} region(s), {args.cycles} cycle(s)")
    _log(f" HITL enabled: {orchestrator.hitl_enabled}")
    _log(f" Compliance audit log: {COMPLIANCE_AUDIT_LOG_PATH}")
    _log("=" * 60)

    t0 = time.time()
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
                    candidates = [
                        w for w in workloads
                        if w.get("owner") == (
                            DEFENDER_ID if profile == "defender" else ADVERSARY_ID
                        )
                    ]
                    if candidates:
                        src = random.choice(candidates)["node"]
                        # Pick any other node as target; the Stockfish triage
                        # will reject illegal ones, and orchestrator falls back
                        # to a known-legal random move.
                        dst_pick = random.choice(workloads)["node"]
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
