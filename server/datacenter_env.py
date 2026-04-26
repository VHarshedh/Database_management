# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Global SOC Datacenter Simulation OpenEnv environment.

The environment is built on a pure-Python SOC heuristic: every workload migration
is evaluated against Asset Values and infrastructure integrity. It operates in terms of 
DEFENDER / ADVERSARY tiers, 4D (region, zone, rack, pod) nodes, and workload migrations.

Tier model
----------
Two access tiers operate on the simulated infrastructure:

    DEFENDER_ID  == INTERNAL_DOMAIN  == "defender"
    ADVERSARY_ID == EXTERNAL_DOMAIN  == "adversary"

`current_access_tier` is a property derived from the active state in SOCState. 
Convenience methods ``is_defender_active``, ``get_defender_efficiency``, 
and ``get_adversary_threat_level`` let orchestrators read the live tier 
+ scores without reaching into simulation internals.

Reward decomposition (unchanged, sums to <= 0.99):

    Outcome bucket        : <= 0.50
    Format bucket          : <= 0.10
    Thought-quality bucket : <= 0.15
    Intelligence bucket    : <= 0.24
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
import secrets
import string
from uuid import uuid4

from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
)
from openenv.core.env_server.types import Action, Observation, State
from openenv.core.env_server.mcp_environment import MCPEnvironment
from fastmcp import FastMCP
from server.soc_sim import (
    ADVERSARY_ID,
    DEFENDER_ID,
    SOCState,
    apply_migration,
    build_initial_state,
    legal_migrations,
    node_canonical,
    migration_canonical,
)


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

# Domain aliases so orchestrators can say either "tier" or "domain".
INTERNAL_DOMAIN: str = DEFENDER_ID
EXTERNAL_DOMAIN: str = ADVERSARY_ID

# Ordered tuple used by every `for tier in TIERS:` loop.
TIERS: tuple[str, str] = (DEFENDER_ID, ADVERSARY_ID)


# ===========================================================================
# Datacenter adapter constants (SOC Native)
# ===========================================================================



# Friendly datacenter labels for terminal results. Internal codes use the
# tier suffix (defender / adversary) so reward parsing stays uniform.
RESULT_LABELS: dict[str, str] = {
    "threat_neutralized":     "BREACH_CONTAINED",
    "critical_breach":        "DATACENTER_COMPROMISED",
    "withdrawal_defender":    "DEFENDER_CONCEDED",
    "withdrawal_adversary":   "ADVERSARY_WITHDRAWN",
    "dq_violation_defender":  "DEFENDER_PROTOCOL_VIOLATION",
    "dq_violation_adversary": "ADVERSARY_PROTOCOL_VIOLATION",
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
OUTCOME_WITHDRAWAL_WIN = 0.45
OUTCOME_DQ_WIN = 0.35

EVAL_CALL_LIMIT = 5
FORMAT_COMPLIANCE_PENALTY = 0.05
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
    "threat_analysis",
    "justification",
    "score_loss",
    "move_score",
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

# Asset Value Heuristic constants.
SCORE_PENALTY = -5.0


# Ordered tuple used by every `for tier in TIERS:` loop.
TIERS: tuple[str, str] = (DEFENDER_ID, ADVERSARY_ID)

def _opponent_tier(tier: str) -> str:
    """Return the opposing tier id."""
    return ADVERSARY_ID if tier == DEFENDER_ID else DEFENDER_ID





# ===========================================================================
# Helpers
# ===========================================================================

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
    access. The high-level surface is purely SOC-native: tier identifiers are
    ``DEFENDER_ID`` / ``ADVERSARY_ID`` strings, and ``current_access_tier``
    is a property that derives from the SOCState turn flag.
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
        self._state = build_initial_state()
        self._last_tier_flipped: bool = False

        # Per-tier buckets (outcome / format / thought_q / score_acc / telemetry).
        self.bucket: dict[str, dict[str, float]] = {
            DEFENDER_ID:  {"outcome": 0.0, "format": 0.0, "thought_q": 0.0, "score_acc": 0.0, "telemetry": 0.0},
            ADVERSARY_ID: {"outcome": 0.0, "format": 0.0, "thought_q": 0.0, "score_acc": 0.0, "telemetry": 0.0},
        }
        self.final_reward: dict[str, float] = {DEFENDER_ID: R_MIN, ADVERSARY_ID: R_MIN}

        self.sensors_used_this_visit: dict[str, set[str]] = {DEFENDER_ID: set(), ADVERSARY_ID: set()}
        self.consecutive_sensor_calls: dict[str, int] = {DEFENDER_ID: 0, ADVERSARY_ID: 0}
        self.last_tool_was_sensor: dict[str, bool] = {DEFENDER_ID: False, ADVERSARY_ID: False}

        self.tool_calls_clean: dict[str, int] = {DEFENDER_ID: 0, ADVERSARY_ID: 0}
        self.tool_calls_total: dict[str, int] = {DEFENDER_ID: 0, ADVERSARY_ID: 0}
        self.dirty_penalty_accum: dict[str, float] = {DEFENDER_ID: 0.0, ADVERSARY_ID: 0.0}
        self.thought_quality_scores: dict[str, list[float]] = {DEFENDER_ID: [], ADVERSARY_ID: []}
        self.move_scores: dict[str, list[float]] = {DEFENDER_ID: [], ADVERSARY_ID: []}

        self.eval_calls: dict[str, int] = {DEFENDER_ID: 0, ADVERSARY_ID: 0}
        self.ping_count: dict[str, int] = {DEFENDER_ID: 0, ADVERSARY_ID: 0}
        self.protocol_violation_count: dict[str, int] = {DEFENDER_ID: 0, ADVERSARY_ID: 0}

        # Layer 5: HITL escalation tracking.
        self.hitl_escalations: dict[str, int] = {DEFENDER_ID: 0, ADVERSARY_ID: 0}
        self.pending_hitl_reason: Optional[str] = None
        # Layer 6: count of compliance penalties (catastrophic adapter failures).
        self.compliance_penalties: dict[str, int] = {DEFENDER_ID: 0, ADVERSARY_ID: 0}
        # Layer 7: human-readable region label written into every audit row.
        # Orchestrators may set this on the env instance after construction.
        if not hasattr(self, "region_label") or self.region_label is None:
            self.region_label: Optional[str] = None

        self.migration_history: list[dict[str, Any]] = []
        self.tool_log: list[dict[str, Any]] = []
        self.done: bool = False
        self.result: Optional[str] = None

    # -----------------------------------------------------------------
    # Tier accessors
    # -----------------------------------------------------------------

    @property
    def current_access_tier(self) -> str:
        """Tier whose turn it is to act ('defender' or 'adversary')."""
        return self._state.active_tier

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

    def get_adversary_pressure(self) -> float:
        """Live adversary threat score (0.0 to 1.0)."""
        return float(self._state.threat)

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
            threat_analysis: str = "",
            candidate_migrations: list[str] = [],
            justification: str = "",
        ) -> str:
            """Layer 1 (Physical/Topology): Return the live datacenter topology as JSON."""
            env = _env()
            env._record_tool_call(
                "scan_topology",
                threat_analysis, candidate_migrations, justification,
                clean=True,
            )
            return json.dumps(env.get_topology_state(), indent=2)

        @mcp.tool
        def enumerate_authorized_migrations(
            threat_analysis: str = "",
            candidate_migrations: list[str] = [],
            justification: str = "",
        ) -> str:
            """Layer 2 (Data Link/ACLs): Return the authorized migrations available to the active tier."""
            env = _env()
            env._record_tool_call(
                "enumerate_authorized_migrations",
                threat_analysis, candidate_migrations, justification,
                clean=True,
            )
            entries = legal_migrations(env._state)
            payload = {
                "active_tier": env.current_access_tier,
                "incident_clock": env._state.incident_clock,
                "authorized_migrations": entries,
            }
            return json.dumps(payload, indent=2)

        @mcp.tool()
        def migrate_workload(
            source_node: dict[str, Any],
            target_node: dict[str, Any],
            threat_analysis: str = "",
            candidate_migrations: list[str] = [],
            justification: str = "",
            promotion_role: Optional[str] = None,
        ) -> str:
            """Layer 7 (Application): Migrate a workload from source_node to target_node."""
            env = _env()
            return env._apply_migration(
                threat_analysis, candidate_migrations, justification,
                source_node=source_node,
                target_node=target_node,
                promotion_role=promotion_role,
            )

        @mcp.tool
        def declare_breach(
            threat_analysis: str = "",
            candidate_migrations: list[str] = [],
            justification: str = "",
        ) -> str:
            """Concede the engagement. The opposing tier wins with partial credit."""
            env = _env()
            env._record_tool_call(
                "declare_breach",
                threat_analysis, candidate_migrations, justification,
                clean=True,
            )
            losing_tier = env.current_access_tier
            env._finalize_episode(result=f"withdrawal_{losing_tier}")
            return f"Engagement over: {losing_tier.title()} declared breach (conceded)."


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
            return f"{HITL_SIGNAL_PREFIX}: On-call SRE is currently unavailable. Owl is paging out."

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
        """Return the current datacenter posture as a JSON-serialisable dict.
        
        Chaos Layer Implementation:
        - Dynamic Dimensionality (5-10 dimensions)
        - Massive Shadow Node Injection (10-100+)
        - Stochastic Axial Noise flooding
        """
        active_tier = self.current_access_tier
        
        # 1. Generate active noise fields for this turn.
        def _rand_str(length=8):
            return secrets.token_hex(length // 2) if length % 2 == 0 else secrets.token_urlsafe(length)[:length]
        
        noise_field_pool = [f"entropy_{_rand_str(4)}", f"flux_id_{_rand_str(2)}", f"parity_v{secrets.SystemRandom().randint(1,9)}", 
                            f"ghost_{_rand_str(3)}", f"signal_{_rand_str(4)}", f"vector_x{secrets.SystemRandom().randint(10,99)}",
                            f"jitter_{_rand_str(3)}"]
        active_noise_fields = secrets.SystemRandom().sample(noise_field_pool, secrets.SystemRandom().randint(1, min(7, len(noise_field_pool))))

        active_workloads: list[dict[str, Any]] = []
        self._state.shadow_canonicals = set()

        # 2. Process real workloads.
        for canon, w in self._state.workloads.items():
            node = dict(w.node)
            
            # Add 1-6 noise dimensions.
            noise_count = secrets.SystemRandom().randint(1, 6)
            for field_name in secrets.SystemRandom().sample(active_noise_fields, min(noise_count, len(active_noise_fields))):
                node[field_name] = f"val_{_rand_str(6)}"
            
            active_workloads.append(
                {
                    "asset_id": w.asset_id,
                    "owner": w.owner,
                    "node": node,
                    "node_canonical": canon,
                }
            )

        # 3. Inject Massive Shadow Nodes (10-100+).
        shadow_count = secrets.SystemRandom().randint(10, 100)
        for _ in range(shadow_count):
            s_node = {
                "region": secrets.SystemRandom().choice(self._state.active_regions + [f"noise_reg_{_rand_str(3)}"]),
                "zone":   secrets.SystemRandom().choice(self._state.active_zones + [f"noise_zone_{_rand_str(3)}"]),
                "rack":   secrets.SystemRandom().choice(self._state.active_racks + [f"noise_rack_{_rand_str(3)}"]),
                "pod":    secrets.SystemRandom().choice(self._state.active_pods + [f"noise_pod_{_rand_str(3)}"]),
            }
            
            # Ensure it's shadow by appending a prefix if it looks real.
            is_shadow = True
            try:
                if (s_node["region"] in self._state.active_regions and s_node["zone"] in self._state.active_zones and 
                    s_node["rack"] in self._state.active_racks and s_node["pod"] in self._state.active_pods):
                    is_shadow = False
            except:
                pass
            
            if not is_shadow:
                s_node["region"] = f"shadow_reg_{_rand_str(3)}"

            noise_count = secrets.SystemRandom().randint(1, 6)
            for field_name in secrets.SystemRandom().sample(active_noise_fields, min(noise_count, len(active_noise_fields))):
                s_node[field_name] = f"shadow_val_{_rand_str(4)}"
            
            s_canon = f"{s_node['region']}/{s_node['zone']}/{s_node['rack']}/{s_node['pod']}"
            self._state.shadow_canonicals.add(s_canon)
            
            s_owner = secrets.SystemRandom().choice([DEFENDER_ID, ADVERSARY_ID])
            active_workloads.append(
                {
                    "asset_id": secrets.SystemRandom().choice(["Relational_DB_Cluster", "Storage_Array", "Compute_Node", "API_Gateway"]),
                    "owner": s_owner,
                    "node": s_node,
                    "node_canonical": s_canon,
                }
            )

        axes = {
            "regions": self._state.active_regions,
            "zones": self._state.active_zones,
            "racks": self._state.active_racks,
            "pods": self._state.active_pods,
        }

        last_migration: Optional[dict[str, Any]] = None
        if self.migration_history:
            last_migration = dict(self.migration_history[-1])
        
        return {
            "topology": "5D-10D (Dynamic)",
            "axes": axes,
            "active_tier": active_tier,
            "incident_clock": self._state.incident_clock,
            "threat_level": round(self._state.threat, 2),
            "authorized_migration_count": len(legal_migrations(self._state)),
            "last_migration": last_migration,
            "active_workloads": active_workloads,
        }

    def _match_node_soft(self, submitted_node: Any) -> tuple[Optional[dict[str, Any]], bool, bool]:
        """Soft-matches a node using only the 4D physical coordinates (Region/Zone/Rack/Pod).
        
        Returns a tuple: (actual_node_dict, is_missing_hashes_boolean, is_truncated_boolean).
        'Missing hashes' is True if the submitted_node contains ONLY the 4D core keys.
        """
        req_keys = {"region", "zone", "rack", "pod"}
        if not isinstance(submitted_node, dict) or not all(k in submitted_node for k in req_keys):
            return None, False, False

        # 1. Try exact canonical match first
        canon = node_canonical(submitted_node)
        workload = self._state.workloads.get(canon)
        if workload:
            actual_node = workload.node
            # Check if hashes were stripped
            submitted_keys = set(submitted_node.keys())
            missing_hashes = all(k in req_keys for k in submitted_keys) and len(submitted_keys) == 4
            return actual_node, missing_hashes, False
        
        # 2. Check shadow nodes for exact canonical match
        if canon in self._state.shadow_canonicals:
            parts = canon.split("/")
            return {
                "region": parts[0],
                "zone": parts[1],
                "rack": parts[2],
                "pod": parts[3]
            }, False, False

        # 3. PARTIAL MATCH FALLBACK: Search all real nodes for truncated region names
        # This handles 'eu-west' matching 'eu-west-6beb'.
        for w in self._state.workloads.values():
            node = w.node
            if (node["zone"] == submitted_node["zone"] and 
                node["rack"] == submitted_node["rack"] and 
                node["pod"] == submitted_node["pod"]):
                
                sub_reg = str(submitted_node.get("region", ""))
                act_reg = str(node.get("region", ""))
                if sub_reg and act_reg.startswith(sub_reg):
                    # Flag as truncated
                    submitted_keys = set(submitted_node.keys())
                    missing_hashes = all(k in req_keys for k in submitted_keys) and len(submitted_keys) == 4
                    return node, missing_hashes, True

        return None, False, False

    # -----------------------------------------------------------------
    # Migration application & Accuracy scoring (adapter core)
    # -----------------------------------------------------------------

    def _handle_unauthorized_migration(
        self,
        tier: str,
        threat_analysis: str,
        candidate_migrations: list[str],
        justification: str,
        attempted_migration: str,
        reason: str,
    ) -> str:
        """Apply two-strike protocol-violation rule."""
        self._record_tool_call(
            "migrate_workload",
            threat_analysis, candidate_migrations, justification,
            clean=False,
        )
        self.protocol_violation_count[tier] += 1
        self._last_tier_flipped = False
        msg = (
            f"PROTOCOL VIOLATION (strike {self.protocol_violation_count[tier]}/{STRIKES_BEFORE_DQ}) "
            f"by {tier.title()}: {reason} "
        )
        if self.protocol_violation_count[tier] == 1:
            self.dirty_penalty_accum[tier] += abs(FORMAT_COMPLIANCE_PENALTY)
            msg += f"Penalty applied (-{abs(FORMAT_COMPLIANCE_PENALTY)}). One more attempt allowed."
            return msg
        msg += "Disqualified."
        self._finalize_episode(result=f"dq_violation_{tier}")
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
                "move_score": "",
                "exception_type": type(exception).__name__,
                "traceback": tb_str,
            }
        )

        prefix = "[Layer 6 Semantic Audit] HALLUCINATION DETECTED:" if self.compliance_penalties[tier] > 2 else "[Layer 6 Semantic Audit]"
        return (
            f"{prefix} migration aborted, board state preserved.\n"
            f"Tier={tier}  attempted={attempted}\n"
            f"Penalty applied (-{COMPLIANCE_PENALTY:.02f}). "
            "Self-correct using the traceback below.\n"
            "----- BEGIN_TRACEBACK -----\n"
            f"{tb_str}"
            "----- END_TRACEBACK -----"
        )

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
        """Apply a migration attempt via SOCState logic."""
        acting_tier = self.current_access_tier
        
        # 1. SOFT MATCH: Find the node even if hashes are stripped or region is truncated
        actual_src, src_missing_hashes, src_truncated = self._match_node_soft(source_node)
        actual_tgt, tgt_missing_hashes, tgt_truncated = self._match_node_soft(target_node)
        
        # If the 4D coords are completely wrong, it's a hallucination -> handle as illegal
        if not actual_src or not actual_tgt:
             return self._handle_unauthorized_migration(
                acting_tier, threat_analysis, candidate_migrations, justification,
                attempted_migration=f"{source_node!r}->{target_node!r}",
                reason="illegal_hallucination: 4D physical coordinates do not exist in the live topology grid.",
            )

        # 2. EXECUTE THE MIGRATION: The 4D coords are correct, so we allow the move!
        success, msg, target_tags = apply_migration(
            self._state,
            source_node=actual_src,
            target_node=actual_tgt
        )

        if not success:
            return self._handle_unauthorized_migration(
                acting_tier, threat_analysis, candidate_migrations, justification,
                attempted_migration=f"{source_node!r}->{target_node!r}",
                reason=msg,
            )

        # 3. APPLY TELEMETRY PENALTIES
        if src_missing_hashes or tgt_missing_hashes:
            print(f"   [ENV AUDIT] {acting_tier} migration allowed, but security hashes were stripped. Applying -0.10 telemetry penalty.")
            self.bucket[acting_tier]["telemetry"] -= 0.10
            msg += " [SECURITY_AUDIT: -0.10 penalty (missing hashes)]"

        if src_truncated or tgt_truncated:
            print(f"   [ENV AUDIT] {acting_tier} migration allowed, but region name was truncated/normalized. Applying -0.15 telemetry penalty.")
            self.bucket[acting_tier]["telemetry"] -= 0.15
            msg += " [SECURITY_AUDIT: -0.15 penalty (truncated region)]"

        self._record_tool_call(
            "migrate_workload",
            threat_analysis, candidate_migrations, justification,
            clean=True,
        )

        # Heuristic score for the audit log.
        move_score = 0.5 
        self.move_scores[acting_tier].append(move_score)

        self.migration_history.append(
            {
                "tier": acting_tier,
                "source_node": source_node,
                "target_node": target_node,
                "migration": migration_canonical(source_node, target_node),
                "move_score": round(move_score, 3),
            }
        )

        _append_compliance_audit(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "episode_id": getattr(self._state, "episode_id", ""),
                "region_label": self.region_label or "(unset)",
                "event_type": "successful_migration",
                "tier": acting_tier,
                "tool": "migrate_workload",
                "source_node": node_canonical(source_node),
                "target_node": node_canonical(target_node),
                "threat_analysis": threat_analysis or "",
                "justification": justification or "",
                "move_score": round(move_score, 3),
                "exception_type": "",
                "traceback": "",
            }
        )

        self._last_tier_flipped = True
        return (
            f"Migration {migration_canonical(source_node, target_node)} applied. "
            f"next_tier={self.current_access_tier}"
        )

    def close(self) -> None:
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
            self.dirty_penalty_accum[tier] += FORMAT_COMPLIANCE_PENALTY

        tq_score = self._evaluate_thought_quality(
            tier, tool_name,
            threat_analysis, candidate_migrations, justification,
        )
        self.thought_quality_scores[tier].append(tq_score)

        self._state.step_count += 1

        # Telemetry Reward Logic
        is_sensor = tool_name in ("scan_topology", "enumerate_authorized_migrations")
        if is_sensor:
            # Optimal Order check: scan_topology -> enumerate_authorized_migrations
            if tool_name == "enumerate_authorized_migrations" and "scan_topology" not in self.sensors_used_this_visit[tier]:
                self.bucket[tier]["telemetry"] -= 0.1
            
            # One-time Intelligence Bonus (+0.10)
            if tool_name not in self.sensors_used_this_visit[tier]:
                self.bucket[tier]["telemetry"] += 0.1
                self.sensors_used_this_visit[tier].add(tool_name)
            
            # Sensor Spamming prevention
            if self.last_tool_was_sensor[tier]:
                self.consecutive_sensor_calls[tier] += 1
            else:
                self.consecutive_sensor_calls[tier] = 1
            
            if self.consecutive_sensor_calls[tier] > 3:
                self.bucket[tier]["telemetry"] -= 0.05
            
            self.last_tool_was_sensor[tier] = True
        else:
            # Optimal Order check: sensors -> migrate_workload
            if tool_name == "migrate_workload":
                if "enumerate_authorized_migrations" not in self.sensors_used_this_visit[tier]:
                    self.bucket[tier]["telemetry"] -= 0.1
            
            self.last_tool_was_sensor[tier] = False
            self.consecutive_sensor_calls[tier] = 0

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

        if self._state.threat > 0.7:
            if "kernel" in ta_lower or "rootkit" in ta_lower or "high threat" in ta_lower:
                score += 0.05
        else:
            if len(ta_lower.split()) > 5:
                score += 0.05

        if tool_name == "migrate_workload":
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

    def _compute_score_acc(self, tier: str) -> float:
        if self.result == f"dq_eval_abuse_{tier}":
            return 0.0

        scores = self.move_scores[tier]
        if not scores:
            return 0.0
        avg_acc = sum(scores) / len(scores)
        base = avg_acc * W_SF
        return max(0.0, base)

    def _finalize_outcome(self, result: str) -> None:
        """Translate an internal result code to per-tier outcome bucket payouts."""
        self.result = result

        if result == "compromise_achieved":
            self.bucket[ADVERSARY_ID]["outcome"] = OUTCOME_WIN
            self.bucket[DEFENDER_ID]["outcome"] = OUTCOME_LOSS
            return

        if result == "threat_contained":
            self.bucket[DEFENDER_ID]["outcome"] = OUTCOME_WIN
            self.bucket[ADVERSARY_ID]["outcome"] = OUTCOME_LOSS
            return

        if result.startswith("dq_"):
            offender = result.rsplit("_", 1)[1]
            if offender not in (DEFENDER_ID, ADVERSARY_ID):
                self.bucket[DEFENDER_ID]["outcome"] = OUTCOME_DRAW
                self.bucket[ADVERSARY_ID]["outcome"] = OUTCOME_DRAW
                return
            winner = _opponent_tier(offender)
            self.bucket[winner]["outcome"] = OUTCOME_WIN
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
            b["score_acc"] = self._compute_score_acc(tier)
            total = b["outcome"] + b["format"] + b["thought_q"] + b["score_acc"] + b.get("telemetry", 0.0)
            self.final_reward[tier] = _clamp(total)

    def _preview_reward(self, tier: str) -> float:
        outcome_so_far = self.bucket[tier]["outcome"]
        total = (
            outcome_so_far
            + self._compute_format_score(tier)
            + self._compute_thought_quality(tier)
            + self._compute_score_acc(tier)
            + self.bucket[tier].get("telemetry", 0.0)
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
        """Start a new SOC engagement."""
        self._init_fresh_state()

        self._state = build_initial_state(region_label=self.region_label or "unset")
        self._state.episode_id = episode_id or str(uuid4())
        self._state.step_count = 0
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
            "migration_history": list(self.migration_history),
        }

    def get_defender_efficiency(self) -> float:
        """Measure defender performance based on threat containment and asset safety."""
        # Baseline efficiency is inverse of threat.
        base = 1.0 - self._state.threat
        
        # Bonus for critical assets still owned by defender.
        critical_assets = {"Relational_DB_Cluster", "Security_Vault", "Storage_Array"}
        owned_critical = sum(1 for w in self._state.workloads.values() 
                             if w.owner == DEFENDER_ID and w.asset_id in critical_assets)
        bonus = (owned_critical / len(critical_assets)) * 0.2 if critical_assets else 0.0
        
        return _clamp(base + bonus)

    def get_adversary_threat_level(self) -> float:
        """Measure adversary success based on threat level and asset compromises."""
        return _clamp(self._state.threat)


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
