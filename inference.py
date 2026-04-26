#!/usr/bin/env python3
"""
Global SOC Orchestrator - the multi-region driver of the simulation.

Architecture
------------
1. **Multi-region simulation** - 3 isolated :class:`DatacenterEnvironment`
   instances, each representing a distinct global region (e.g. us-east,
   eu-west, ap-south). Each region has its own SOCState grid, so per-region
   damage scoring is independent.

2. **Async adversary swarm** - on every ADVERSARY turn, three hostile
   profiles (DB_Backup_Agent, Viral_Traffic_Agent, Chaos_Monkey) are polled
   concurrently via :mod:`asyncio` for that region's topology. Each one
   returns a candidate ``migrate_workload`` payload.

3. **Physics oracle (triage)** - the orchestrator scores all 3 candidates
   with a pure-Python Asset Value heuristic. The candidate that causes the 
   most damage to the DEFENDER's assets wins.

4. **L1 Defender (round-robin)** - one DEFENDER agent rotates Region 1 ->
   Region 2 -> Region 3 -> Region 1 -> ... in a continuous loop. The active
   region completes one DEFENDER + one ADVERSARY half-move pair per visit.

The orchestrator is fully SOC-native: it speaks tier ids
(``DEFENDER_ID`` / ``ADVERSARY_ID``), 4D node dicts, and migration strings.
The physics oracle uses a pure-Python heuristic based on "Asset Value" scoring.

CLI
---
::

    python global_soc_orchestrator.py --regions 3 --cycles 10

You will need API credentials in ``.env`` / ``.env.local`` (same shape as
``inference.py``: ``API_BASE_URL``, ``GOOGLE_MODEL_*`` / ``GROQ_MODEL_*`` model names).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any, Optional
from dataclasses import asdict, dataclass, field
from datetime import datetime

try:
    from dotenv import load_dotenv
    # Ensure environment variables are loaded before any other imports that might use them.
    load_dotenv()
    load_dotenv(Path(__file__).resolve().parent / ".env.local")
except ImportError:
    pass


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
    migration_canonical,
    node_canonical,
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
    """Stateful wrapper for a single region's env + history."""

    region_id: str
    region_name: str
    env: DatacenterEnvironment
    cycle_index: int = 0
    last_result: Optional[str] = None
    defender_histories: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    adversary_histories: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    step_count: int = 0
    reward_history: list[float] = field(default_factory=list)
    error_state: str | None = None

    def reset(self, defender: DatacenterAgent, adversaries: list[DatacenterAgent]) -> None:
        """Reset the env and rebuild per-agent message buffers."""
        self.env.reset()
        # Layer 7: stamp this region's label onto the env so every audit row
        # the env writes (successful migrations, HITL escalations, compliance
        # penalties) carries an unambiguous region attribution.
        self.env.region_label = self.region_name
        self.cycle_index = 0
        self.last_result = None
        self.step_count = 0
        self.reward_history = []
        self.error_state = None
        self.defender_histories = {
             "defender": defender.new_region_buffer(self.region_id)
        }
        self.adversary_histories = {
            adv.profile: adv.new_region_buffer(self.region_id) for adv in adversaries
        }


# ===========================================================================
# Physics Oracle - Pure Python SOC Heuristic
# ===========================================================================


@dataclass
class _Candidate:
    """A single triaged adversary payload."""

    profile: str
    decision: AgentDecision
    damage_score: float = float("-inf")  # higher == more damaging to DEFENDER
    error: Optional[str] = None

    @property
    def is_legal(self) -> bool:
        return self.error is None


def _score_candidate_damage(env: DatacenterEnvironment, decision: AgentDecision) -> float:
    """Pure-Python heuristic based on Asset Value scoring."""
    ASSET_VALUES = {
        "Relational_DB_Cluster": 90.0,
        "Security_Vault": 80.0,
        "Storage_Array": 50.0,
        "API_Gateway": 30.0,
        "Compute_Node": 10.0,
        "Idle_Workload": 0.0
    }
    
    args = decision.arguments
    target_node = args.get("target_node")
    if not isinstance(target_node, dict):
        return float("-inf")

    try:
        dst_key = node_canonical(target_node)
    except Exception:
        return float("-inf")

    # 1. Fatal trap check (Shadow nodes)
    if hasattr(env._state, "shadow_canonicals") and dst_key in env._state.shadow_canonicals:
        return -100.0

    # 2. Invalid route check
    if dst_key not in env._state.workloads:
        return -50.0

    # 3. Asset capture check
    target_workload = env._state.workloads[dst_key]
    if target_workload.owner == DEFENDER_ID:
        return ASSET_VALUES.get(target_workload.asset_id, 10.0)

    # 4. Valid lateral move to an empty node
    return 5.0


def physics_oracle_triage(
    env: DatacenterEnvironment,
    decisions: list[AgentDecision],
) -> tuple[list[_Candidate], list[_Candidate]]:
    """Triage a swarm: return ``(legal_candidates, all_candidates)``.

    All legal candidates are returned to enable concurrent execution.
    """
    candidates: list[_Candidate] = []
    for d in decisions:
        cand = _Candidate(profile=d.profile, decision=d)
        # Sensor-aware filter: allow mapping tools.
        if d.tool not in ("migrate_workload", "scan_topology", "enumerate_authorized_migrations"):
            cand.error = f"unsupported swarm tool ({d.tool!r})"
            candidates.append(cand)
            continue
        
        # Calculate SOC heuristic score.
        if d.tool == "migrate_workload":
            score = _score_candidate_damage(env, d)
        else:
            # Sensor tools are legal but carry a neutral damage score (5.0 to 
            # allow them to compete with low-value migrations).
            score = 5.0
        
        if score == float("-inf"):
            cand.error = "invalid target_node or canonical resolution failure"
            candidates.append(cand)
            continue
            
        cand.damage_score = score
        candidates.append(cand)

    legal = [c for c in candidates if c.is_legal]
    # Select the winner(s) with the highest damage_score (descending).
    legal.sort(key=lambda c: c.damage_score, reverse=True)
    
    return legal, candidates


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
    adversary_damage_score: Optional[float] = None
    adversary_decision: Optional[dict[str, Any]] = None
    defender_swarm: list[dict[str, Any]] = field(default_factory=list)
    defender_all_candidates: list[dict[str, Any]] = field(default_factory=list)
    adversary_swarm: list[dict[str, Any]] = field(default_factory=list)
    adversary_all_candidates: list[dict[str, Any]] = field(default_factory=list)
    adversary_reward: Optional[float] = None
    done: bool = False
    result: Optional[str] = None
    fallback_used: bool = False


class GlobalSOCOrchestrator:
    """Drive 3 regions with n Defenders and an m-way Adversary swarm.

    Constructor parameters:

    * ``regions``: list of ``RegionRunner`` instances. Typically 3.
    * ``defenders``: list of :class:`DatacenterAgent` instances.
    * ``adversaries``: list of :class:`DatacenterAgent` instances.
    """

    def __init__(
        self,
        regions: list[RegionRunner],
        defenders: list[DatacenterAgent],
        adversaries: list[DatacenterAgent],
        *,
        hitl_enabled: Optional[bool] = None,
    ) -> None:
        if not regions:
            raise ValueError("at least one region runner is required")
        if not defenders:
            raise ValueError("at least one defender agent is required")
        if not adversaries:
            raise ValueError("at least one adversary agent is required")

        self.regions = regions
        self.defenders = defenders
        self.adversaries = adversaries
        self.audit_trail: list[CycleRecord] = []
        self.defender_scratchpad: list[str] = []
        self.adversary_scratchpad: list[str] = []
        self.agent_recon_count: dict[DatacenterAgent, int] = {}
        self.cycle_metrics = {"defender_actions": 0, "adversary_actions": 0}
        # Layer 5 control: set to False for unattended runs (CI, dry-runs).
        # If unset, default to True only when stdin looks interactive so the
        # loop never blocks on input() in non-interactive contexts.
        if hitl_enabled is None:
            hitl_enabled = bool(getattr(sys.stdin, "isatty", lambda: False)())
        self.hitl_enabled: bool = hitl_enabled

    # ----- environment helpers -----------------------------------------

    def reset_cycle_metrics(self) -> None:
        """Resets counters at the start of every region's cycle."""
        self.cycle_metrics["defender_actions"] = 0
        self.cycle_metrics["adversary_actions"] = 0

    def reset_all(self) -> None:
        for r in self.regions:
            # For multiple defenders, we just pass the first one to reset
            # as reset is mainly for env initialization. History buffers
            # are rebuilt in RegionRunner.reset.
            r.reset(self.defenders[0], self.adversaries)
            # Re-initialize all defender histories
            r.defender_histories = {
                def_agent.model_name + f"_{i}": def_agent.new_region_buffer(r.region_id)
                for i, def_agent in enumerate(self.defenders)
            }

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
                        "defender_efficiency": 0.5,
                        "adversary_threat_level": r.env._state.threat,
                    },
                }
                for r in self.regions
            ],
            "audit_trail": [asdict(c) for c in self.audit_trail],
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

        # 2. Anti-Stall Governor
        if tool in ["scan_topology", "enumerate_authorized_migrations"]:
            self.agent_recon_count[agent] = self.agent_recon_count.get(agent, 0) + 1
        elif tool == "migrate_workload":
            self.agent_recon_count[agent] = 0 # Reset on execution

        if self.agent_recon_count.get(agent, 0) > 2:
            _log(f"   [GOVERNOR] {agent.profile}:{agent.model_name} is stalling. Forcing Fallback Migration.")
            self.agent_recon_count[agent] = 0 # Reset counter after penalty
            
            # Hijack the decision to force a migration. 
            # This ensures env.step() runs and triggers the orchestrator's random fallback.
            decision = AgentDecision(
                tool="migrate_workload",
                arguments={
                    "threat_analysis": "Governor Override: Tactical Stalling Detected.",
                    "justification": "Agent exceeded recon limits. Forcing random migration.",
                    "candidate_migrations": ["forced->fallback"],
                    "source_node": {"region": "us-east", "zone": "az-a", "rack": "forced", "pod": "fallback"},
                    "target_node": {"region": "us-east", "zone": "az-a", "rack": "forced", "pod": "fallback"},
                },
                raw="[GOVERNOR_HIJACK]",
                profile=agent.profile,
                region_id=region.region_id,
            )
            tool = decision.tool
            args = decision.arguments

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
            return 0.5, tool_text, bool(getattr(region.env, "done", False))

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
        buf: list[dict[str, Any]] = []
        if agent.profile == "defender":
            for key, b in region.defender_histories.items():
                if b[0].get("content", "").endswith(agent.persona):
                    buf = b
                    break
            if not buf:
                buf = region.defender_histories.get("defender", [])
        else:
            buf = region.adversary_histories.setdefault(
                agent.profile, agent.new_region_buffer(region.region_id)
            )

        # History Truncation: System Prompt (0) + Last 10 messages (5 turns)
        if len(buf) > 11:
            return [buf[0]] + buf[-10:]
        return buf

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
            f"!! HITL OVERRIDE REQUIRED (Layer 5)  region={region.region_id} "
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
                # Direct canonicalization for dynamic grid (no squares).
                src_canon = node_canonical(src)
                dst_canon = node_canonical(dst)
                migration = f"{src_canon}->{dst_canon}"
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

    async def defender_swarm_step(self, region: RegionRunner) -> CycleRecord:
        """Take exactly one DEFENDER half-move in this region (with swarm triage)."""
        record = CycleRecord(region_id=region.region_id, cycle_index=region.cycle_index)

        if region.env.done:
            record.done = True
            record.result = region.env.result
            return record

        if not region.env.is_defender_active():
            return record

        # 1. Shared Intelligence: refresh system prompts for all defenders.
        for d in self.defenders:
            d.refresh_system_prompt(self.defender_scratchpad)
            buf = self._history_for(region, d)
            if buf and buf[0]["role"] == "system":
                buf[0]["content"] = d.system_prompt
        
        topo_snapshot = region.env.get_topology_state()
        
        # 2. Parallel Inference: spawn the swarm.
        tasks = []
        for d in self.defenders:
            buf = self._history_for(region, d)
            tasks.append(d.choose_async(topo_snapshot, buf, region_id=region.region_id))

        decisions: list[AgentDecision] = []
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        to_eliminate = []
        for d, res in zip(self.defenders, gathered):
            if isinstance(res, BaseException):
                _log(f"   [{region.region_id}] defender {d.model_name} raised: {res!r}")
                if "RateLimitExhausted" in str(res):
                    _log(f"🚨 [ELIMINATION] {d.model_name} purged due to Layer 3 Overload.")
                    to_eliminate.append(d)
                decisions.append(AgentDecision(
                    tool=None, arguments={}, raw=f"(exception: {res!r})",
                    profile="defender", region_id=region.region_id,
                ))
            else:
                decisions.append(res)
        
        # Process eliminations
        for d in to_eliminate:
            if d in self.defenders:
                self.defenders.remove(d)
        if to_eliminate and not self.defenders:
            _log(f"⚠️ [EXTINCTION] All Defender species eliminated in {region.region_id}.")

        # 3. Triage: find all legal candidates.
        legal_candidates, all_candidates = physics_oracle_triage(region.env, decisions)
        
        # Record all candidates for detailed failure analysis in reports.
        record.defender_all_candidates = [
            {
                "profile": c.profile,
                "tool": c.decision.tool,
                "damage_score": c.damage_score,
                "error": c.error,
                "arguments": c.decision.arguments,
            }
            for c in all_candidates
        ]
        
        if not legal_candidates:
             # Fallback
             fallback = self._random_legal_migration(region.env)
             if fallback is None:
                 record.done = True
                 return record
             # Wrap fallback in a Candidate for unified execution loop.
             legal_candidates = [_Candidate(profile=fallback.profile, decision=fallback)]
             record.fallback_used = True

        # 4. Sequential Execution of the full defender payload.
        moved_assets = set()
        actions_executed = 0
        total_reward = 0.0
        
        for cand in legal_candidates:
            decision = cand.decision
            
            # Conflict check: skip if asset/source node already moved this cycle.
            src_node = decision.arguments.get("source_node")
            if src_node and isinstance(src_node, dict):
                canon = node_canonical(src_node)
                if canon in moved_assets:
                    _log(f"   🚫 Defender Conflict: {decision.model_name} attempted redundant move from {canon}")
                    continue
                moved_assets.add(canon)

            winning_agent = next(
                (a for a in self.defenders if a.model_name == decision.model_name),
                self.defenders[0]
            )
            _log(f"   🔧 Tool: {decision.tool}({json.dumps(decision.arguments)})")
            reward, tool_text, done = self._apply_decision(region, winning_agent, decision)
            _log(f"   💰 Reward: {reward:.2f}")

            # Update metrics
            self.cycle_metrics["defender_actions"] += 1
            _log(f"   ✅ Defender Action {self.cycle_metrics['defender_actions']}: {decision.tool} executed (Reward: {reward:.2f})")
            
            # 4. Scratchpad Update: summary of successful defender action.
            justification = decision.arguments.get("justification", "no justification")
            entry = f"[{winning_agent.model_name}]: {decision.tool} applied because {justification}"
            self.defender_scratchpad.append(entry)
            if len(self.defender_scratchpad) > 10:
                self.defender_scratchpad.pop(0)

            actions_executed += 1
            total_reward += reward
            
            # Record this specific swarm action for auditing and reporting.
            record.defender_swarm.append({
                "profile": winning_agent.profile,
                "tool": decision.tool,
                "reward": reward,
                "damage_score": cand.damage_score,
                "arguments": decision.arguments,
            })
            
            # [STEP] log for parser
            region.step_count += 1
            region.reward_history.append(reward)
            args_json = json.dumps(decision.arguments, separators=(',', ':')) if decision.arguments else "{}"
            
            err_str = "null"
            if not decision.tool:
                err_str = "malformed_tool"
            elif isinstance(tool_text, str) and ("invalid" in tool_text.lower() or "violation" in tool_text.lower()):
                err_str = "layer6_trap"
            region.error_state = err_str if err_str != "null" else None

            print(
                f"[STEP] step={region.step_count} "
                f"action={decision.tool}({args_json}) "
                f"reward={reward:.2f} done={str(done).lower()} error={err_str}",
                flush=True
            )

            # HITL check
            is_hitl = (
                decision.tool == "escalate_to_sysadmin"
                or (isinstance(tool_text, str) and tool_text.startswith(HITL_SIGNAL_PREFIX))
            )
            if is_hitl and not done and self.hitl_enabled:
                human_decision = self._hitl_human_override(region, decision)
                if human_decision is not None:
                    h_reward, h_tool_text, h_done = self._apply_decision(
                        region, self.defenders[0], human_decision
                    )
                    # We only record the human reward if it overrides the last action.
                    total_reward += h_reward
                    done = h_done

            if done:
                record.done = True
                record.result = region.env.result
                break

        record.defender_reward = total_reward
        _log(f"   ✅ Defender Turn: {actions_executed} actions executed.")
        return record

    # ----- adversary swarm turn ----------------------------------------

    async def adversary_swarm_step(self, region: RegionRunner) -> CycleRecord:
        """Run the 3-way adversary swarm with concurrent execution and shared scratchpad."""
        record = CycleRecord(region_id=region.region_id, cycle_index=region.cycle_index)

        if region.env.done:
            record.done = True
            record.result = region.env.result
            return record

        if not region.env.is_adversary_active():
            return record

        # 1. Shared Intelligence: refresh system prompts with the current adversary scratchpad.
        for adv in self.adversaries:
            adv.refresh_system_prompt(self.adversary_scratchpad)
            buf = self._history_for(region, adv)
            if buf and buf[0]["role"] == "system":
                buf[0]["content"] = adv.system_prompt

        topo_snapshot = region.env.get_topology_state()

        # 2. Parallel Inference: spawn the swarm.
        tasks = []
        for adv in self.adversaries:
            buf = self._history_for(region, adv)
            tasks.append(adv.choose_async(topo_snapshot, buf, region_id=region.region_id))

        decisions: list[AgentDecision] = []
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        to_eliminate = []
        for adv, res in zip(self.adversaries, gathered):
            if isinstance(res, BaseException):
                _log(f"   [{region.region_id}] adversary {adv.profile} raised: {res!r}")
                if "RateLimitExhausted" in str(res):
                    _log(f"🚨 [ELIMINATION] {adv.model_name} purged due to Layer 3 Overload.")
                    to_eliminate.append(adv)
                decisions.append(AgentDecision(
                    tool=None, arguments={}, raw=f"(exception: {res!r})",
                    profile=adv.profile, region_id=region.region_id,
                ))
            else:
                decisions.append(res)
        
        # Process eliminations
        for adv in to_eliminate:
            if adv in self.adversaries:
                self.adversaries.remove(adv)
        if to_eliminate and not self.adversaries:
            _log(f"⚠️ [EXTINCTION] All Adversary species eliminated in {region.region_id}.")

        # 3. Triage: find all legal candidates.
        legal_candidates, all_candidates = physics_oracle_triage(region.env, decisions)
        
        # Record all candidates for detailed failure analysis in reports.
        record.adversary_all_candidates = [
            {
                "profile": c.profile,
                "tool": c.decision.tool,
                "damage_score": c.damage_score,
                "error": c.error,
                "arguments": c.decision.arguments,
            }
            for c in all_candidates
        ]
        
        if not legal_candidates:
            # Fallback
            fallback = self._random_legal_migration(region.env)
            if fallback is None:
                record.done = True
                return record
            # Wrap fallback in a Candidate for unified execution loop.
            legal_candidates = [_Candidate(profile=fallback.profile, decision=fallback)]
            record.fallback_used = True

        # 4. Sequential Execution of the full swarm payload.
        moved_assets = set()
        actions_executed = 0
        total_reward = 0.0
        total_damage = 0.0
        
        for cand in legal_candidates:
            decision = cand.decision
            
            # Conflict check: skip if asset/source node already moved this cycle.
            src_node = decision.arguments.get("source_node")
            if src_node and isinstance(src_node, dict):
                canon = node_canonical(src_node)
                if canon in moved_assets:
                    _log(f"   🚫 Illegal Conflict: {decision.profile} attempted redundant move from {canon}")
                    continue
                moved_assets.add(canon)

            agent = next((a for a in self.adversaries if a.profile == decision.profile), self.adversaries[0])
            
            _log(f"   🔧 Tool: {decision.tool}({json.dumps(decision.arguments)})")
            reward, tool_text, done = self._apply_decision(region, agent, decision)
            _log(f"   💰 Reward: {reward:.2f}")

            # Update metrics
            self.cycle_metrics["adversary_actions"] += 1
            _log(f"   ✅ Swarm Action {self.cycle_metrics['adversary_actions']}: {decision.tool} executed (Reward: {reward:.2f})")

            # 5. Scratchpad Update: summary of successful actions.
            asset_id = decision.arguments.get("asset_id", "workload")
            target = decision.arguments.get("target_node")
            target_str = node_canonical(target) if target and isinstance(target, dict) else "unknown"
            justification = decision.arguments.get("justification", "no justification")
            
            entry = f"[{decision.profile}]: Moved {asset_id} to {target_str} because {justification}"
            self.adversary_scratchpad.append(entry)
            if len(self.adversary_scratchpad) > 10:
                self.adversary_scratchpad.pop(0)

            actions_executed += 1
            total_reward += reward
            total_damage += (cand.damage_score if cand.damage_score != float("-inf") else 0.0)
            
            # Record this specific swarm action for auditing and reporting.
            record.adversary_swarm.append({
                "profile": decision.profile,
                "tool": decision.tool,
                "reward": reward,
                "damage_score": cand.damage_score,
                "arguments": decision.arguments,
            })
            
            # [STEP] log for parser (maintain backward compatibility with evaluators)
            region.step_count += 1
            region.reward_history.append(reward)
            args_json = json.dumps(decision.arguments, separators=(',', ':')) if decision.arguments else "{}"
            
            err_str = "null"
            if not decision.tool:
                err_str = "malformed_tool"
            elif isinstance(tool_text, str) and ("invalid" in tool_text.lower() or "violation" in tool_text.lower()):
                err_str = "layer6_trap"
            region.error_state = err_str if err_str != "null" else None

            print(
                f"[STEP] step={region.step_count} "
                f"action={decision.tool}({args_json}) "
                f"reward={reward:.2f} done={str(done).lower()} error={err_str}",
                flush=True
            )
            
            if done:
                record.done = True
                record.result = region.env.result
                break

        record.adversary_reward = total_reward
        record.adversary_damage_score = total_damage
        _log(f"   ✅ Swarm Turn: {actions_executed} actions executed.")
        return record

    def _random_legal_migration(self, env: DatacenterEnvironment) -> Optional[AgentDecision]:
        """Pick any legal migration so the engagement keeps progressing."""
        from server.soc_sim import legal_migrations
        legal = legal_migrations(env._state)
        if not legal:
            return None
        
        mig = secrets.SystemRandom().choice(legal)
        src = mig["source_node"]
        dst = mig["target_node"]
        canonical = mig["migration"]
        
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
            _log(f"\n{'='*78}\n=== CYCLE {cycle + 1} ===\n{'='*78}")
            all_done = True
            for region in self.regions:
                if region.env.done:
                    continue
                all_done = False
                self.reset_cycle_metrics()
                
                # [START] log
                if region.step_count == 0:
                    print(f"[START] task={region.region_id} env=datacenter_soc model={self.defenders[0].model_name}", flush=True)

                region.cycle_index = cycle

                # 1. DEFENDER swarm half-move.
                d_rec = await self.defender_swarm_step(region)
                self.audit_trail.append(d_rec)
                
                if region.env.done:
                    region.last_result = region.env.result
                    success_bool = region.env.get_defender_efficiency() > 0.5
                    final_score = region.reward_history[-1] if region.reward_history else 0.01
                    rewards_str = ",".join(f"{r:.2f}" for r in region.reward_history) or "0.01"
                    print(f"[END] success={str(success_bool).lower()} steps={region.step_count} score={final_score:.2f} rewards={rewards_str}", flush=True)
                    continue

                # 2. ADVERSARY swarm half-move (async fan-out).
                a_rec = await self.adversary_swarm_step(region)
                self.audit_trail.append(a_rec)
                
                # Cycle Summary
                total_actions = sum(self.cycle_metrics.values())
                _log(f"\n   [CYCLE SUMMARY] Total Actions: {total_actions}")
                _log(f"   [THREAT LEVEL] Current Adversary Pressure: {region.env.get_adversary_pressure():.2f}")
                
                if region.env.done:
                    region.last_result = region.env.result
                    success_bool = region.env.get_defender_efficiency() > 0.5
                    final_score = region.reward_history[-1] if region.reward_history else 0.01
                    rewards_str = ",".join(f"{r:.2f}" for r in region.reward_history) or "0.01"
                    print(f"[END] success={str(success_bool).lower()} steps={region.step_count} score={final_score:.2f} rewards={rewards_str}", flush=True)

                if per_region_pause > 0:
                    await asyncio.sleep(per_region_pause)

            if all_done:
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
    defenders: list[DatacenterAgent],
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
    return GlobalSOCOrchestrator(runners, defenders, adversaries, hitl_enabled=hitl_enabled)


# ===========================================================================
# CLI
# ===========================================================================


def _load_all_env_configs(directory: Path) -> dict[str, str]:
    """Find all .env* files in the directory and merge them."""
    try:
        from dotenv import dotenv_values
    except ImportError:
        return {}
    
    merged = {}
    # Sort to ensure deterministic merging
    paths = sorted(directory.glob(".env*"))
    for p in paths:
        if p.is_file():
            config = {k: v for k, v in dotenv_values(str(p)).items() if isinstance(v, str)}
            for k, v in config.items():
                # Pattern-based fallback: if we see a key-like string, ensure it's in its home
                if v.startswith("AIza"):
                    merged.setdefault("GOOGLE_API_KEY", v)
                    merged.setdefault("GEMINI_API_KEY", v)
                elif v.startswith("hf_"):
                    merged.setdefault("HF_TOKEN", v)
                
                # Protect existing "good" HF_TOKEN from being overwritten by mislabeled keys
                if k == "HF_TOKEN" and merged.get(k, "").startswith("hf_") and not v.startswith("hf_"):
                    continue
                
                merged[k] = v
    return merged


def _make_openai_client(config: dict, model_name: str, provider: str) -> Any:
    if OpenAI is None:
        raise RuntimeError("openai package not installed; pip install openai")
    
    # 1. Determine the base URL
    if provider == "hf":
        base_url = "https://router.huggingface.co/v1"
    elif provider == "google":
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    elif provider == "groq":
        base_url = "https://api.groq.com/openai/v1"
    else:
        base_url = config.get("API_BASE_URL")
        if not base_url:
            if "/" in model_name or any(m.lower() in model_name.lower() for m in ["deepseek", "llama", "qwen", "gemma"]):
                base_url = "https://router.huggingface.co/v1"
            else:
                base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
             
    # 2. Determine the API Key
    # Start with provider-specific key
    api_key = config.get(f"{provider.upper()}_API_KEY")
    if not api_key:
        api_key = config.get("HF_TOKEN") or config.get("OPENAI_API_KEY") or os.getenv("HF_TOKEN", "")
    
    # 3. Smart Token Matching:
    # If we need a HF token but have a Google key (AIza...), search for an hf_... token.
    if provider == "hf" and isinstance(api_key, str) and api_key.startswith("AIza"):
        for v in config.values():
            if isinstance(v, str) and v.startswith("hf_"):
                api_key = v
                break
    # Vice versa for Google
    elif provider == "google" and isinstance(api_key, str) and api_key.startswith("hf_"):
        for v in config.values():
            if isinstance(v, str) and v.startswith("AIza"):
                api_key = v
                break

    return OpenAI(base_url=base_url, api_key=api_key)


def _resolve_models(config: dict, silent: bool = False) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Pick n defenders + m adversaries from the configured pools.

    Returns ``(defenders, adversaries)`` where each is a list of (model_name, provider).
    """
    pool: list[tuple[str, str]] = []
    # Collect all GOOGLE_MODEL_*, GROQ_MODEL_*, HF_MODEL_*
    for k, v in config.items():
        if not v:
            continue
        if k.startswith("GOOGLE_MODEL_"):
            pool.append((v, "google"))
        elif k.startswith("GROQ_MODEL_"):
            pool.append((v, "groq"))
        elif k.startswith("HF_MODEL_"):
            pool.append((v, "hf"))
    
    # Deduplicate by model name
    seen = set()
    deduped = []
    for m, p in pool:
        if m not in seen:
            deduped.append((m, p))
            seen.add(m)
    pool = deduped

    if not pool:
        # Fallback to the working models from check_models.py
        pool = [
            ("deepseek-ai/DeepSeek-V3", "hf"),
            ("meta-llama/Llama-3.3-70B-Instruct", "hf"),
            ("Qwen/Qwen2.5-72B-Instruct", "hf"),
            ("google/gemma-3-27b-it", "hf")
        ]
        _log("Using default Hugging Face model pool (no .env models found).")

    T = len(pool)
    if T < 4:
        raise RuntimeError(f"need at least 4 models, found {T}")

    secrets.SystemRandom().shuffle(pool)
    
    # User request: (1-2) Defenders vs (3-4) Adversaries
    sr = secrets.SystemRandom()
    n = sr.randint(1, 2)
    m = sr.randint(3, 4)
    
    # Clamp to pool size if necessary
    if n + m > T:
        # Prioritize 1 defender if pool is tight
        n = 1
        m = min(T - 1, m)
    
    def_models = pool[:n]
    adv_models = pool[n:n+m]
    
    if not silent:
        _log(f"Dynamic Swarm: {n} Defender(s) vs {m} Adversary(s) (Total {len(def_models) + len(adv_models)}/{T} models used)")
    return def_models, adv_models


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
        "--defender", type=str, default=None,
        help="Override defender model. Example: 'deepseek-ai/DeepSeek-V3'"
    )
    parser.add_argument(
        "--adversary", type=str, action="append", default=None,
        help="Override adversary model(s). Can be specified multiple times."
    )
    args = parser.parse_args()

    # Load API config from all .env* files.
    # We still use this helper to ensure we have the full merged config for our manual client creation.
    config = _load_all_env_configs(_HERE)

    if args.dry_run:
        defenders, adversaries = _build_stub_agents()
        _log("Running in --dry-run mode (no LLM calls).")
    else:
        if OpenAI is None:
            _log("ERROR: openai package not installed.")
            sys.exit(1)
        
        config = _load_all_env_configs(_HERE)
        try:
            if args.defender or args.adversary:
                # Manual override: prioritize CLI specified models
                def _guess_provider(m: str) -> str:
                    if "gemini" in m.lower() or "gemma" in m.lower():
                        return "google"
                    if "llama" in m.lower() or "deepseek" in m.lower() or "qwen" in m.lower():
                        return "hf"
                    return "hf"

                def_models = [(args.defender, _guess_provider(args.defender))] if args.defender else []
                adv_models = [(m, _guess_provider(m)) for m in args.adversary] if args.adversary else []
                
                # Only fill in if the user hasn't provided a specific role at all
                if not def_models or not adv_models:
                    auto_def, auto_adv = _resolve_models(config, silent=True)
                    if not def_models:
                        def_models = auto_def
                    if not adv_models:
                        adv_models = auto_adv
                
                _log(f"Manual Swarm: {len(def_models)} Defender(s) vs {len(adv_models)} Adversary(s)")
            else:
                def_models, adv_models = _resolve_models(config)
        except RuntimeError as exc:
            _log(f"ERROR: {exc}")
            sys.exit(1)

        defenders = []
        for m, provider in def_models:
            client = _make_openai_client(config, m, provider)
            # Clean model name for the API call (strip :novita etc)
            clean_m = m.split(":")[0] if provider == "hf" else m
            defenders.append(make_defender_agent(client, clean_m))
            _log(f"Defender model        : {clean_m} [{provider}]")

        adversaries = []
        adv_factories = [make_db_backup_agent, make_viral_traffic_agent, make_chaos_monkey_agent]
        for i, (m, provider) in enumerate(adv_models):
            client = _make_openai_client(config, m, provider)
            # Clean model name for the API call (strip :novita etc)
            clean_m = m.split(":")[0] if provider == "hf" else m
            # Cycle through profiles if more than 3 adversaries
            factory = adv_factories[i % len(adv_factories)]
            adversaries.append(factory(client, clean_m))
            _log(f"Adversary model ({i+1}) : {clean_m} [{provider}] [{factory.__name__}]")

    region_names = DEFAULT_REGION_NAMES[: args.regions]
    hitl_enabled: Optional[bool] = False if args.no_hitl else None
    orchestrator = build_orchestrator(
        defenders=defenders,
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


def _build_stub_agents() -> tuple[list[DatacenterAgent], list[DatacenterAgent]]:
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
                        src = secrets.SystemRandom().choice(candidates)["node"]
                        dst_pick = secrets.SystemRandom().choice(workloads)["node"]
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

    # User request: (1-2) Defenders vs (3-4) Adversaries
    sr = secrets.SystemRandom()
    n_def = sr.randint(1, 2)
    n_adv = sr.randint(3, 4)

    defenders = [
        DatacenterAgent(_make_random_policy("defender"), profile="defender")
        for _ in range(n_def)
    ]
    
    adv_profiles = ["db_backup", "viral_traffic", "chaos_monkey"]
    adversaries = [
        DatacenterAgent(_make_random_policy(adv_profiles[i % len(adv_profiles)]), profile=adv_profiles[i % len(adv_profiles)])
        for i in range(n_adv)
    ]
    
    _log(f"Dynamic Swarm (Stub): {n_def} Defender(s) vs {n_adv} Adversary(s)")
    return defenders, adversaries


if __name__ == "__main__":
    main()
