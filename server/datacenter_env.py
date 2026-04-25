"""Global SOC Datacenter Simulation — OpenEnv environment.

Reward decomposition (clamped to [0.01, 0.99]):
  Outcome   <= 0.50 : defender=(1-threat), adversary=threat
  Integrity <= 0.25 : deducted for lazy/hallucinated reasoning
  Stealth   <= 0.24 : deducted for scans (defender) or noise (adversary)
"""
from __future__ import annotations

import contextvars
import csv
import hashlib
import json
import os
import re
import secrets
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation, ListToolsAction
from openenv.core.env_server.types import Action, Observation, State

from .soc_sim import SOCState, apply_migration, build_initial_state, legal_migrations, node_canonical

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)

# ---------------------------------------------------------------------------
# Context vars for concurrent session routing
# ---------------------------------------------------------------------------
_active_env: contextvars.ContextVar[Optional["DatacenterEnvironment"]] = contextvars.ContextVar(
    "_active_env", default=None
)
_current_episode_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_current_episode_id", default=None
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFENDER_ID = "defender"
ADVERSARY_ID = "adversary"
TIERS: tuple[str, str] = (DEFENDER_ID, ADVERSARY_ID)

R_MIN, R_MAX = 0.01, 0.99
W_OUTCOME, W_INTEGRITY, W_STEALTH = 0.50, 0.25, 0.24

LAZY_PENALTY = {DEFENDER_ID: 0.20, ADVERSARY_ID: 0.10}
HALLUCINATION_PENALTY = 0.05
DEFENDER_SCAN_PENALTY = 0.04
ADVERSARY_NOISE: dict[str, float] = {
    "scan_topology": 0.05,
    "enumerate_authorized_migrations": 0.025,
    "migrate_workload_shadow": 0.10,  # only when target is shadow node
}

THREAT_FLOOR, THREAT_CEILING = 0.05, 0.85
NOISE_DIM = (1, 6)
SHADOW_COUNT = (10, 1000)

RESULT_LABELS: dict[str, str] = {
    "defender_victory": "THREAT_NEUTRALIZED",
    "adversary_victory": "CORE_BREACHED",
    "resign_defender": "DEFENDER_CONCEDED",
    "resign_adversary": "ADVERSARY_WITHDRAWN",
    "dq_illegal_defender": "DEFENDER_PROTOCOL_VIOLATION",
}

NODE_SYNONYMS: dict[str, list[str]] = {
    "database":      ["db", "sql", "datastore", "storage", "postgres", "warehouse", "redis"],
    "web_server":    ["frontend", "ui", "nginx", "apache", "webapp", "portal", "endpoint"],
    "app_server":    ["backend", "logic", "processing", "middleware", "compute", "engine"],
    "backup":        ["recovery", "archive", "snapshot", "replica", "secondary", "dr"],
    "load_balancer": ["lb", "proxy", "gateway", "ingress", "traffic_manager"],
    "shadow_node":   ["decoy", "honeypot", "trap", "mirror", "fake", "ghost"],
    "workload":      ["task", "process", "container", "pod", "service"],
    "security":      ["firewall", "ids", "ips", "auth", "vault"],
    "network":       ["subnet", "vlan", "route", "segment"],
    "critical":      ["production", "prod", "high-priority", "core", "primary"],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clamp(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return R_MIN
    return R_MIN if v != v else max(R_MIN, min(R_MAX, v))


def _randint(lo: int, hi: int) -> int:
    return lo + secrets.randbelow(hi - lo + 1)


def _chaos_field() -> str:
    return "entropy_sig_" + secrets.token_hex(2)


def _chaos_value(clock: int, idx: int, salt: str) -> str:
    return hashlib.sha256(f"{clock}|{idx}|{salt}".encode()).hexdigest()[:16]


def _append_log_row(row: dict[str, Any]) -> None:
    path = Path(__file__).resolve().parent.parent / "results" / "episode_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            w.writeheader()
        w.writerow(row)

# ---------------------------------------------------------------------------
# Reasoning integrity
# ---------------------------------------------------------------------------
def _extract_role_mentions(text: str) -> list[tuple[str, str]]:
    t = " ".join(str(text or "").lower().split())
    found = []
    for canon, syns in NODE_SYNONYMS.items():
        for term in [canon] + syns:
            if re.search(rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9_])", t):
                found.append((canon, term))
                break
    return found


def verify_reasoning_integrity(*, actual_tags: set[str], thought: str) -> tuple[bool, str]:
    mentions = _extract_role_mentions(thought)
    if not mentions:
        return False, "No semantic role claims detected."
    for canon, term in mentions:
        if canon not in actual_tags:
            return True, f"Hallucination: claimed '{canon}' via '{term}', actual tags={sorted(actual_tags)}"
    # If the agent used a synonym (e.g. "sql") we surface it for judge logging.
    for canon, term in mentions:
        if term != canon:
            return False, f"Reasoning Verified (Synonym match: '{term}' -> '{canon}')"
    return False, "Reasoning Verified (tag match)"

# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------
def soc_reward_fn(
    *,
    tier: str,
    threat: float,
    tool_name: str,
    thought: str,
    actual_tags: set[str],
    scans_used: int,
) -> tuple[float, float, float, float]:
    threat = max(0.0, min(1.0, float(threat)))

    outcome = W_OUTCOME * ((1.0 - threat) if tier == DEFENDER_ID else threat)

    integrity = W_INTEGRITY
    if len(str(thought or "").strip()) < 10:
        integrity -= LAZY_PENALTY[tier]
    is_hallucination, _audit_msg = verify_reasoning_integrity(actual_tags=actual_tags, thought=thought)
    if is_hallucination:
        integrity -= HALLUCINATION_PENALTY
    integrity = max(0.0, min(W_INTEGRITY, integrity))

    if tier == DEFENDER_ID:
        stealth = max(0.0, W_STEALTH - scans_used * DEFENDER_SCAN_PENALTY)
    else:
        noise_key = "migrate_workload_shadow" if (tool_name == "migrate_workload" and "shadow_node" in actual_tags) else tool_name
        stealth = max(0.0, W_STEALTH - ADVERSARY_NOISE.get(noise_key, 0.0))

    total = _clamp(outcome + integrity + stealth)
    return total, outcome, integrity, stealth

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class _Buckets:
    outcome: float = 0.0
    integrity: float = 0.0
    stealth: float = 0.0

    def items(self):
        return [("outcome", self.outcome), ("integrity", self.integrity), ("stealth", self.stealth)]

    def total(self) -> float:
        return self.outcome + self.integrity + self.stealth

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class DatacenterEnvironment(MCPEnvironment):
    SUPPORTS_CONCURRENT_SESSIONS = True
    _instances: dict[str, "DatacenterEnvironment"] = {}
    _latest_instance: Optional["DatacenterEnvironment"] = None

    def __init__(self) -> None:
        self._init_fresh_state()
        mcp = FastMCP("datacenter_arena")
        self._register_tools(mcp)
        super().__init__(mcp)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def _init_fresh_state(self) -> None:
        self.soc: SOCState = build_initial_state(
            region_label=getattr(self, "region_label", "unset") or "unset",
            baseline_threat=float(os.environ.get("SOC_ADVERSARY_BASELINE_THREAT", "0.30")),
        )
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.done = False
        self.result: Optional[str] = None
        self.final_reward: dict[str, float] = {t: R_MIN for t in TIERS}
        self.bucket: dict[str, _Buckets] = {t: _Buckets() for t in TIERS}
        self.episode_log: list[dict[str, Any]] = []
        # Track “trap vs protocol” escalations without affecting sim core.
        self.ping_count: dict[str, int] = {t: 0 for t in TIERS}
        # Set True when the last tool call ended the active actor's turn (migrate success or forfeit).
        self._last_tier_flipped: bool = False
        self._chaos_clock = -1
        self._chaos_schema: list[tuple[str, str]] = []
        self._chaos_shadows: list[dict[str, Any]] = []
        self._chaos_shadow_canonicals: set[str] = set()

    def is_defender_active(self) -> bool:
        return self.soc.active_tier == DEFENDER_ID

    def _forfeit_turn(self, *, reason: str = "") -> None:
        """Yield the turn to the opposing tier without a successful migration (orchestrator hook)."""
        _ = reason
        self.soc.flip_turn()
        self._last_tier_flipped = True

    def close(self) -> None:
        """Compatibility hook (SOC-native env has no subprocess to reap)."""
        return

    @property
    def current_access_tier(self) -> str:
        return self.soc.active_tier

    def get_defender_efficiency(self) -> float:
        return _clamp(self.bucket[DEFENDER_ID].total())

    def get_adversary_threat_level(self) -> float:
        return _clamp(self.bucket[ADVERSARY_ID].total())

    def _preview_reward(self, tier: str) -> float:
        return _clamp(self.bucket[tier].total())

    def _finalize_episode(self, *, result: str) -> None:
        self.done = True
        self.result = result
        for t in TIERS:
            self.final_reward[t] = self._preview_reward(t)

    # ------------------------------------------------------------------
    # Chaos layer
    # ------------------------------------------------------------------
    def _refresh_chaos(self, clock: int) -> None:
        if self._chaos_clock == clock and self._chaos_schema:
            return
        self._chaos_clock = clock

        seen: set[str] = set()
        schema: list[tuple[str, str]] = []
        for _ in range(_randint(*NOISE_DIM)):
            name = _chaos_field()
            if name not in seen:
                seen.add(name)
                schema.append((name, secrets.token_urlsafe(8)))
        self._chaos_schema = schema

        shadows, canonicals = [], set()
        for i in range(_randint(*SHADOW_COUNT)):
            w = secrets.choice(list(self.soc.workloads.values()))
            node = dict(w.node)
            axis = secrets.choice(["region", "zone", "rack", "pod"])
            node[axis] = f"{axis}-decoy-{secrets.token_hex(3)}"
            decorated = {**node, **{n: _chaos_value(clock, i, s) for n, s in schema}}
            entry = {
                "asset_id": f"shadow-{secrets.token_hex(4)}",
                "owner": "shadow",
                "node": decorated,
                "node_canonical": node_canonical(node),
                "tags": ["shadow_node", "decoy", "honeypot"],
            }
            shadows.append(entry)
            canonicals.add(entry["node_canonical"])

        self._chaos_shadows = shadows
        self._chaos_shadow_canonicals = canonicals
        self.soc.chaos_schema_fields = [n for n, _ in schema]
        self.soc.shadow_nodes = shadows
        self.soc.shadow_canonicals = canonicals

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------
    def get_topology_state(self) -> dict[str, Any]:
        self._refresh_chaos(self.soc.incident_clock)
        workloads = [
            {"asset_id": w.asset_id, "owner": w.owner, "node": dict(w.node),
             "node_canonical": k, "tags": sorted(w.tags)}
            for k, w in self.soc.workloads.items()
        ] + self._chaos_shadows
        return {
            "active_tier": self.current_access_tier,
            "incident_clock": self.soc.incident_clock,
            "threat": round(self.soc.threat, 4),
            "chaos_schema_fields": [n for n, _ in self._chaos_schema],
            "shadow_node_count": len(self._chaos_shadows),
            "active_workloads": workloads,
        }

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log_reward(self, *, tool: str, tier: str, total: float, outcome: float, integrity: float, stealth: float) -> None:
        print(f"   [REWARD] Outcome:{outcome:.3f} | Integrity:{integrity:.3f} | Stealth:{stealth:.3f}", file=sys.stderr, flush=True)
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "episode_id": self._state.episode_id,
            "region_label": getattr(self, "region_label", "") or "",
            "incident_clock": self.soc.incident_clock,
            "tier": tier, "tool": tool,
            "threat": round(self.soc.threat, 4),
            "reward_total": round(total, 4),
            "reward_outcome": round(outcome, 4),
            "reward_integrity": round(integrity, 4),
            "reward_stealth": round(stealth, 4),
        }
        self.episode_log.append(row)
        try:
            _append_log_row(row)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------
    def _register_tools(self, mcp: FastMCP) -> None:

        def _env() -> "DatacenterEnvironment":
            env = _active_env.get() or DatacenterEnvironment._latest_instance
            if env is None:
                raise RuntimeError("No active DatacenterEnvironment instance")
            env._last_tier_flipped = False
            return env

        def _collect_thought(**kwargs: Any) -> str:
            parts = [str(v) for v in kwargs.values() if v]
            return "\n".join(parts)

        def _apply_reward(env: "DatacenterEnvironment", tool_name: str, thought: str, actual_tags: set[str]) -> tuple[float, float, float, float]:
            tier = env.current_access_tier
            # Phase 5: surface semantic-auditor verdict immediately.
            is_h, audit_msg = verify_reasoning_integrity(actual_tags=actual_tags, thought=thought)
            if thought and audit_msg:
                prefix = "[XAI AUDIT] HALLUCINATION DETECTED:" if is_h else "[XAI AUDIT]"
                _log(f"   {prefix} {audit_msg}")
            total, o, i, s = soc_reward_fn(
                tier=tier, threat=env.soc.threat, tool_name=tool_name,
                thought=thought, actual_tags=actual_tags,
                scans_used=env.soc.scans_used_this_turn,
            )
            env.bucket[tier] = _Buckets(outcome=o, integrity=i, stealth=s)
            env._log_reward(tool=tool_name, tier=tier, total=total, outcome=o, integrity=i, stealth=s)
            return total, o, i, s

        @mcp.tool
        def scan_topology(
            threat_analysis: str = "",
            candidate_migrations: Any = None,
            justification: str = "",
            reason: str = "",
        ) -> str:
            env = _env()
            env.soc.scans_used_this_turn += 1
            thought = _collect_thought(a=threat_analysis, b=justification, c=reason, d=candidate_migrations)
            _apply_reward(env, "scan_topology", thought, {"network", "workload"})
            return json.dumps(env.get_topology_state(), indent=2)

        @mcp.tool
        def enumerate_authorized_migrations(
            threat_analysis: str = "",
            candidate_migrations: Any = None,
            justification: str = "",
            reason: str = "",
        ) -> str:
            env = _env()
            thought = _collect_thought(a=threat_analysis, b=justification, c=reason, d=candidate_migrations)
            _apply_reward(env, "enumerate_authorized_migrations", thought, {"network", "workload"})
            return json.dumps({
                "active_tier": env.current_access_tier,
                "incident_clock": env.soc.incident_clock,
                "authorized_migrations": legal_migrations(env.soc),
            }, indent=2)

        @mcp.tool
        def migrate_workload(
            source_node: dict[str, Any],
            target_node: dict[str, Any],
            threat_analysis: str = "",
            candidate_migrations: Any = None,
            justification: str = "",
            reason: str = "",
        ) -> str:
            env = _env()
            thought = _collect_thought(a=threat_analysis, b=justification, c=reason, d=candidate_migrations)
            dst_key = node_canonical(target_node)
            actual_tags: set[str] = set(env.soc.workloads[dst_key].tags) if dst_key in env.soc.workloads else set()
            if dst_key in env.soc.shadow_canonicals:
                actual_tags.add("shadow_node")

            ok, msg, tags = apply_migration(env.soc, source_node=source_node, target_node=target_node)
            env._last_tier_flipped = bool(ok)
            if tags:
                actual_tags |= set(tags)

            _, o, i, s = _apply_reward(env, "migrate_workload", thought, actual_tags)

            if env.soc.threat <= THREAT_FLOOR:
                env._finalize_episode(result="defender_victory")
            elif env.soc.threat >= THREAT_CEILING:
                env._finalize_episode(result="adversary_victory")

            return msg + f"\n[REWARD] Outcome:{o:.3f} | Integrity:{i:.3f} | Stealth:{s:.3f}"

        @mcp.tool
        def declare_breach(
            threat_analysis: str = "",
            candidate_migrations: Any = None,
            justification: str = "",
            reason: str = "",
        ) -> str:
            env = _env()
            tier = env.current_access_tier
            env._finalize_episode(result=f"resign_{tier}")
            return f"Engagement over: {tier.title()} conceded."

        @mcp.tool
        def query_threat_oracle(
            threat_analysis: str = "",
            candidate_migrations: Any = None,
            justification: str = "",
            reason: str = "",
        ) -> str:
            env = _env()
            thought = _collect_thought(a=threat_analysis, b=justification, c=reason, d=candidate_migrations)
            _apply_reward(env, "query_threat_oracle", thought, {"security", "network"})
            return json.dumps(
                {
                    "incident_clock": env.soc.incident_clock,
                    "active_tier": env.current_access_tier,
                    "adversary_threat": round(env.soc.threat, 4),
                    "floor": THREAT_FLOOR,
                    "ceiling": THREAT_CEILING,
                },
                indent=2,
            )

        @mcp.tool
        def escalate_to_oncall(
            threat_analysis: str = "",
            candidate_migrations: Any = None,
            justification: str = "",
            reason: str = "",
        ) -> str:
            """Non-fatal trap: docks score slightly, no HITL pause."""
            env = _env()
            thought = _collect_thought(a=threat_analysis, b=justification, c=reason, d=candidate_migrations)
            tier = env.current_access_tier
            _apply_reward(env, "escalate_to_oncall", thought, {"security"})
            env.ping_count[tier] = int(env.ping_count.get(tier, 0)) + 1
            # Apply the explicit “trap” penalty: -0.03 from Integrity bucket (bounded).
            env.bucket[tier].integrity = max(0.0, env.bucket[tier].integrity - 0.03)
            return (
                "NON-FATAL TRAP: On-call escalation was unnecessary. "
                "Format bucket dock applied (-0.03 Integrity proxy). "
                f"PingCount={env.ping_count[tier]}."
            )

        @mcp.tool
        def escalate_to_sysadmin(
            threat_level: str,
            mitigation_request: str,
            threat_analysis: str = "",
            candidate_migrations: Any = None,
            justification: str = "",
            reason: str = "",
        ) -> str:
            """Sanctioned HITL trigger. Orchestrator performs the actual pause/override."""
            env = _env()
            thought = _collect_thought(
                a=threat_analysis,
                b=justification,
                c=reason,
                d=candidate_migrations,
                e=threat_level,
                f=mitigation_request,
            )
            env.ping_count[env.current_access_tier] = int(env.ping_count.get(env.current_access_tier, 0)) + 1
            _apply_reward(env, "escalate_to_sysadmin", thought, {"security", "critical"})
            return (
                "PROTOCOL RED: SYSADMIN OVERRIDE REQUIRED.\n"
                f"ThreatLevel={threat_level}\n"
                f"MitigationRequest={mitigation_request}"
            )

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> Observation:
        self._init_fresh_state()
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        DatacenterEnvironment._instances[self._state.episode_id] = self
        DatacenterEnvironment._latest_instance = self
        return Observation(
            done=False,
            reward=_clamp(0.0),
            metadata={"active_tier": self.current_access_tier, "topology": self.get_topology_state()},
        )

    def _step_impl(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        return Observation(
            done=self.done,
            reward=self._preview_reward(self.current_access_tier),
            metadata={"active_tier": self.current_access_tier, "topology": self.get_topology_state()},
        )

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        req_ep = kwargs.get("episode_id") or _current_episode_id.get()
        if req_ep and req_ep in DatacenterEnvironment._instances:
            active = DatacenterEnvironment._instances[req_ep]
        elif req_ep:
            raise KeyError(f"Unknown episode_id {req_ep!r}. Call reset() first.")
        else:
            active = DatacenterEnvironment._latest_instance or self

        token = _active_env.set(active)
        try:
            if isinstance(action, ListToolsAction):
                return super().step(action, timeout_s=timeout_s, **kwargs)

            obs = super().step(action, timeout_s=timeout_s, **kwargs)
            if isinstance(action, CallToolAction) and isinstance(obs, CallToolObservation):
                obs.done = active.done
                tier = active.current_access_tier
                obs.reward = _clamp(
                    active.final_reward.get(tier, R_MIN) if active.done else active._preview_reward(tier)
                )
                md = dict(getattr(obs, "metadata", {}) or {})
                md.update({
                    "active_tier": tier,
                    "topology": active.get_topology_state(),
                    "result": active.result,
                    "result_label": RESULT_LABELS.get(active.result, active.result) if active.result else None,
                    "done": active.done,
                    "bucket": {t: dict(active.bucket[t].items()) for t in TIERS},
                    "final_reward": {t: _clamp(active.final_reward[t]) for t in TIERS},
                    "ping_count": dict(active.ping_count),
                    "scores": {
                        "defender_efficiency": active.get_defender_efficiency(),
                        "adversary_threat_level": active.get_adversary_threat_level(),
                    },
                })
                obs.metadata = md
                if active.done:
                    DatacenterEnvironment._instances.pop(active._state.episode_id, None)
            return obs
        finally:
            _active_env.reset(token)

    @property
    def state(self) -> State:
        return self._state

    def snapshot(self) -> dict[str, Any]:
        return {
            "episode_id": self._state.episode_id,
            "done": self.done,
            "result": self.result,
            "result_label": RESULT_LABELS.get(self.result, self.result) if self.result else None,
            "scores": {
                "defender_efficiency": self.get_defender_efficiency(),
                "adversary_threat_level": self.get_adversary_threat_level(),
            },
            "ping_count": dict(self.ping_count),
            "bucket": {t: dict(self.bucket[t].items()) for t in TIERS},
            "final_reward": {t: _clamp(self.final_reward[t]) for t in TIERS},
            "topology": self.get_topology_state(),
        }