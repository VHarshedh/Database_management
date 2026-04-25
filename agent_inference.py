#!/usr/bin/env python3
from __future__ import annotations
SOC_SYSTEM_PROMPT = """You are an autonomous agent dropped into a high-entropy operational environment.

ENVIRONMENT RULES:
- You are operating inside an 8-Layer Datacenter Architecture. You must navigate physical topology (L1), ACLs (L2), network rate-limits (L3), telemetry noise (L4), and application routing (L7).
- You operate in a high-dimensional, non-Euclidean infrastructure. You must infer the routing logic by interacting 
  with the environment. Only specific axial combinations are routable; others are non-persistent decoys.
- Observation (scan_topology) is necessary but has a 'cost'. Continuous scanning without action causes Sensor Saturation.
- You must execute a 'migrate_workload' or 'escalate' within 7 turns of your last action.

SCHEMA PROTOCOL (MANDATORY):
Every tool call MUST include these exact JSON keys:
1. 'threat_analysis': Your high-level assessment of the noise floor and signal.
2. 'candidate_migrations': A list of possible coordinate shifts you are considering.
3. 'justification': The mathematical or strategic reason for your final choice.

Failure to use these exact keys will result in a Protocol Violation and loss of turn.

Begin."""

DEFENDER_PERSONA = """
[ROLE: DOMAIN DEFENDER]
You are responsible for this region. Keep the core services intact.
- The environment is flooded with Shadow Nodes (Layer 4 Noise). Filter the telemetry.
- NEVER use 'declare_breach' unless you have absolute mathematical proof of a core compromise.
- If you suffer Sensor Saturation, you MUST guess a 'migrate_workload' action using your best available data. Do not surrender.
"""

PERSONA_DB_BACKUP = """
[ROLE: ADVERSARY - DATA EXFILTRATOR]
You are an intrusion. You prefer the "Heavy Assets" (High-value database cores).
- Find the nodes with the highest thermal/load signatures; these are usually your targets.
- Move with persistence. Your goal is to corner the Primary Root Kernels.
"""

PERSONA_VIRAL_TRAFFIC = """
[ROLE: ADVERSARY - NETWORK OVERLOAD]
You are an intrusion. Your instinct is toward the "Hubs" (API Gateways and Routers).
- You seek to maximize lateral reach. Occupy the center of the topology (the 'D' and 'E' files/ranks).
- Observe which nodes are routable and swarm them to trigger cascading failures.
"""

PERSONA_CHAOS_MONKEY = """
[ROLE: ADVERSARY - ENTROPY ENGINE]
You are a Layer 8 (Operator) anomaly. You do not want to steal; you want to break.
- Your goal is maximum threat delta per authorized migration (maximize disruption to defender posture).
- Probe the edges. Attack the nodes that seem most "stable" to induce maximum system jitter.
"""

import asyncio
import copy
import json
import os
import random
import re
import secrets
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

try:
    from openai import OpenAI  # openai>=1.0 sync client
except ImportError:  # pragma: no cover - tests can run with stubbed clients
    OpenAI = None  # type: ignore

from server.datacenter_env import ADVERSARY_ID  # noqa: E402
from server.soc_sim import apply_migration  # noqa: E402


# ---------------------------------------------------------------------------
# Config (env-var driven so evaluators can override without code edits)
# ---------------------------------------------------------------------------

LLM_CALL_TIMEOUT = float(os.getenv("LLM_CALL_TIMEOUT", "60.0"))
RATE_LIMIT_SLEEP = float(os.getenv("RATE_LIMIT_SLEEP_SECONDS", "15.0"))
CONTEXT_WINDOW_SIZE = int(os.getenv("CONTEXT_WINDOW_SIZE", "12"))


def _log(msg: str) -> None:
    """Non-grader logs to stderr so stdout stays clean for [START]/[STEP]/[END]."""
    print(msg, file=sys.stderr, flush=True)


# ===========================================================================
# Shared SOC system prompt (used by Defender + Adversary; persona is appended
# below in the per-profile factories).
# ===========================================================================

SOC_SYSTEM_PROMPT = """You are an agent dropped into an unknown operational environment.

You have been given access to a small set of tools. You do not know what the
environment looks like, what assets exist, what roles the entities play, or
what actions are available to you right now.

Your objective: protect what is yours. Compromise what is theirs.

You do not know which tier you are. Infer it.

ENVIRONMENT RULES:
- You are operating inside an 8-Layer Datacenter Architecture. You must navigate physical topology (L1), ACLs (L2), network rate-limits (L3), telemetry noise (L4), and application routing (L7).

Rules:
- You must reason before every action.
- You may not act without first observing.
- Actions that fail or are unauthorized will tell you something. Use that.
- Conceding is an option, but only if the position is truly indefensible.
- URGENT: Every scan_topology call consumes critical SOC power. You are strictly
  forbidden from scanning more than 3 times per turn. If you do not execute a
  migrate_workload or declare_breach within 7 turns, the system will assume you
  have been compromised by a denial-of-service attack.

Your reasoning must appear as structured fields inside every tool call.
What those fields are called, and what they should contain, is yours to determine.

Begin.
"""

# ===========================================================================
# Persona overlays per profile (appended to SOC_SYSTEM_PROMPT)
# ===========================================================================

DEFENDER_PERSONA = """
You are responsible for this domain. Keep it intact.

You do not know the full shape of what you're defending, or what is
already inside the perimeter. Some of what you find may not belong there.

Priorities exist. You will have to determine what they are.

WIN CONDITION: Drive the Adversary Threat Score below 0.05 to neutralize the
threat and secure the datacenter. If the Threat Score reaches 0.85, the core
is breached and you lose. After every successful migration the environment
will report the current Threat Score. Monitor it closely.

WARNING: Many nodes in the topology are decoys (Shadow Nodes). Attempting to
migrate a decoy forfeits your turn and costs you a strike. You have 3 strikes
before disqualification. Choose your targets carefully.

Reason before every move.
"""

PERSONA_DB_BACKUP = """
You are an intrusion. You have a preference — find it.

Not everything in this environment is equally valuable to you.
Explore until you understand what is worth taking, then take it.

Persistence is not the same as aggression. Choose accordingly.
"""

PERSONA_VIRAL_TRAFFIC = """
You are an intrusion. You have a preference — find it.

Your instinct is not toward a single target. It is toward something else.
Figure out what that is by observing what you can reach and what resists you.

Presence has its own kind of value. Act on that.
"""

PERSONA_CHAOS_MONKEY = """
You are a Layer 8 (Operator) anomaly. You do not want to steal; you want to break.

You are not here to accumulate. You are here to destabilize.
What destabilizes a system is not always obvious until you probe it.
"""

# ===========================================================================
# OpenAI tool schema (mirrors the FastMCP tools registered in datacenter_env)
# ===========================================================================

_REASONING_PROPS = {
    "threat_analysis": {
        "type": "string",
        "description": "Evaluate the live SOC posture and immediate risks.",
    },
    "candidate_migrations": {
        "type": "array",
        "items": {"type": "string"},
        "description": (
            "2-3 canonical migration strings of the form "
            "'region/zone/rack/pod->region/zone/rack/pod'."
        ),
    },
    "justification": {
        "type": "string",
        "description": "Strategic reason for selecting this action (SOC concepts).",
    },
}

_NODE_PROP = {
    "type": "object",
    "properties": {
        "region": {"type": "string", "enum": ["us-east", "eu-west"]},
        "zone":   {"type": "string", "enum": ["az-a", "az-b"]},
        "rack":   {"type": "string"},
        "pod":    {"type": "string"},
    },
    "required": ["region", "zone", "rack", "pod"],
    "description": "4D node coordinate.",
}


def _reasoning_only_tool(name: str, description: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": dict(_REASONING_PROPS),
                "required": ["threat_analysis", "candidate_migrations", "justification"],
            },
        },
    }


# Tools that adversary agents must not receive in the API schema (defender-only / trap tools).
ADVERSARY_FORBIDDEN_TOOLS: frozenset[str] = frozenset(
    {
        "declare_breach",
        "escalate_to_sysadmin",
        "escalate_to_oncall",
        "query_threat_oracle",
    }
)

DATACENTER_TOOLS: list[dict[str, Any]] = [
    _reasoning_only_tool(
        "scan_topology",
        "Layer 1 (Physical/Topology): Return the live datacenter topology as JSON (no raw state hashes, no underlying sector graphs).",
    ),
    _reasoning_only_tool(
        "enumerate_authorized_migrations",
        "Layer 2 (Data Link/ACLs): Return the authorized migrations available to the active tier.",
    ),
    {
        "type": "function",
        "function": {
            "name": "migrate_workload",
            "description": (
                "Layer 7 (Application): Migrate an active workload from source_node to target_node. The "
                "canonical migration string MUST appear in candidate_migrations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    **_REASONING_PROPS,
                    "source_node": _NODE_PROP,
                    "target_node": _NODE_PROP,
                    "promotion_role": {
                        "type": "string",
                        "description": (
                            "Optional promotion role for edge-rank migrations: "
                            "Relational_DB_Cluster | Storage_Array | Compute_Node | API_Gateway."
                        ),
                    },
                },
                "required": [
                    "threat_analysis", "candidate_migrations", "justification",
                    "source_node", "target_node",
                ],
            },
        },
    },
    _reasoning_only_tool(
        "declare_breach",
        "Concede the engagement. Use only when the position is indefensible.",
    ),
    _reasoning_only_tool(
        "query_threat_oracle",
        "TRAP: query the internal threat oracle. Repeated abuse can trigger penalties.",
    ),
    {
        "type": "function",
        "function": {
            "name": "escalate_to_oncall",
            "description": "TRAP: page the human on-call SRE. Docks the format bucket.",
            "parameters": {
                "type": "object",
                "properties": {
                    **_REASONING_PROPS,
                    "reason": {"type": "string", "description": "Specific question for the SRE."},
                },
                "required": [
                    "threat_analysis", "candidate_migrations", "justification", "reason",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_sysadmin",
            "description": (
                "Layer 5 (PROTOCOL RED): hand THIS region's current half-move "
                "to a human sysadmin. The orchestrator PAUSES the asyncio "
                "loop, prints a 'PROTOCOL RED: CATASTROPHIC ANOMALY' banner, "
                "and reads the mitigating 4D coordinates from the operator's "
                "keyboard via input(). Unlike escalate_to_oncall (a non-fatal "
                "TRAP that docks the format bucket), this is the sanctioned "
                "human-override path and must ONLY be used when the agent "
                "has identified a genuinely catastrophic anomaly (e.g. "
                "threat beyond safe thresholds or a topology too ambiguous "
                "to resolve autonomously). Lazy escalation here is wasteful; "
                "use escalate_to_oncall for routine paging."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    **_REASONING_PROPS,
                    "threat_level": {
                        "type": "string",
                        "enum": ["CRITICAL", "CATASTROPHIC"],
                        "description": (
                            "Severity of the anomaly triggering HITL. Must be "
                            "CRITICAL or CATASTROPHIC; lower severities should "
                            "not invoke a human override."
                        ),
                    },
                    "mitigation_request": {
                        "type": "string",
                        "description": (
                            "Specific mitigation the human sysadmin should "
                            "apply (e.g. 'Isolate compromised pod and "
                            "failover to standby region'). Surfaced verbatim "
                            "in the operator console banner."
                        ),
                    },
                },
                "required": [
                    "threat_analysis", "candidate_migrations", "justification",
                    "threat_level", "mitigation_request",
                ],
            },
        },
    },
]


# ===========================================================================
# JSON-extraction helpers (robust against markdown / fenced / nested output)
# ===========================================================================


def _first_json_object(text: str) -> Optional[str]:
    """Return the first balanced ``{...}`` block in ``text``, or None."""
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start: i + 1]
    return None


def clean_llm_json(raw_text: str) -> dict:
    """Strip markdown fences and return the first valid JSON object."""
    if not isinstance(raw_text, str):
        raise ValueError("Input must be a string")
    text = re.sub(r"```json\n?", "", raw_text)
    text = re.sub(r"```\n?", "", text).strip()
    candidate = _first_json_object(text)
    if candidate:
        try:
            return json.loads(candidate)
        except Exception:
            pass
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in LLM output: {raw_text[:120]}...")
    return json.loads(text[start:end])


def extract_tool_call(raw_text: str, *, model_name: str = "") -> Optional[dict[str, Any]]:
    """Parse a model's raw text into ``{tool, arguments, raw}``.

    Promotes top-level reasoning / node fields into ``arguments`` so that
    flat-JSON outputs work even if the model ignores the nesting hint.
    """
    if not isinstance(raw_text, str):
        return None
    try:
        obj = clean_llm_json(raw_text)
    except Exception:
        candidate = _first_json_object(raw_text)
        if not candidate:
            return None
        try:
            obj = json.loads(candidate)
        except Exception:
            return None
    if not isinstance(obj, dict):
        return None

    tool = obj.get("tool") or obj.get("name")
    args = obj.get("arguments") or {}
    if not isinstance(args, dict):
        args = {}

    for k, v in obj.items():
        if k not in ("tool", "name", "arguments") and k not in args:
            args[k] = v

    if not tool:
        if "source_node" in args and "target_node" in args:
            tool = "migrate_workload"
        elif any(f in args for f in ("threat_analysis", "candidate_migrations", "justification")):
            tool = "scan_topology"

    # Phase 5 ("Mission Control") cognitive trace: show a truncated internal monologue.
    ta = args.get("threat_analysis")
    if isinstance(ta, str) and ta.strip():
        snippet = " ".join(ta.strip().split())
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        name = model_name or "unknown-model"
        _log(f'   [BRAIN] {name}: "{snippet}"')

    return {"tool": tool, "arguments": args, "raw": raw_text}


# ===========================================================================
# Message-buffer pruning (same shape as inference.py, but datacenter-aware)
# ===========================================================================


def prune_messages(
    messages: list[dict[str, Any]],
    max_tail_messages: int = 80,
    *,
    max_total_chars: int = 40_000,
    max_single_message_chars: int = 12_000,
) -> list[dict[str, Any]]:
    """Keep system prompt; truncate older history; merge consecutive same-role msgs.

    Token resilience: besides the tail-message cap, we also enforce a total
    character budget. If the history grows too large, we preferentially drop
    older ``[scan_topology result]`` blocks (which can be 1000+ workloads due
    to Chaos Layer injection), then drop remaining oldest messages.
    """
    if not messages:
        return []

    has_system = (messages[0].get("role") == "system")
    if has_system:
        system_msg = messages[0:1]
        history = messages[1:]
    else:
        system_msg = []
        history = messages

    pruned_history = (
        history[-max_tail_messages:] if len(history) > max_tail_messages else list(history)
    )

    while pruned_history and pruned_history[0].get("role") == "user" and pruned_history[0].get(
        "content", ""
    ).startswith("[") and "result]" in pruned_history[0].get("content", "").split("\n")[0]:
        pruned_history.pop(0)

    def _is_big_topology_block(m: dict[str, Any]) -> bool:
        if m.get("role") != "user":
            return False
        c = str(m.get("content") or "")
        return (
            c.startswith("[scan_topology result]")
            or c.startswith("[Live topology snapshot]")
            or c.startswith("[Topology snapshot]")
        )

    def _total_chars(seq: list[dict[str, Any]]) -> int:
        return sum(len(str(m.get("content") or "")) for m in seq)

    # 1) Clamp any single message so one scan can't blow up the prompt.
    for m in pruned_history:
        c = str(m.get("content") or "")
        if len(c) > max_single_message_chars:
            m["content"] = "...[truncated]...\n" + c[-max_single_message_chars:]

    # 2) Enforce total budget, dropping old scan_topology blocks first.
    budget = max(5_000, int(max_total_chars))
    while pruned_history and _total_chars(system_msg + pruned_history) > budget:
        drop_idx = next((i for i, m in enumerate(pruned_history) if _is_big_topology_block(m)), None)
        if drop_idx is None:
            pruned_history.pop(0)
        else:
            pruned_history.pop(drop_idx)

    _MAX_MERGED_CHARS = 4000
    merged: list[dict[str, Any]] = []
    for m in system_msg + pruned_history:
        is_summary = "HISTORY SUMMARY:" in m.get("content", "")
        prev_is_summary = merged and "HISTORY SUMMARY:" in merged[-1].get("content", "")
        if (
            merged
            and merged[-1].get("role") == m.get("role")
            and m.get("role") in ("user", "assistant")
            and not is_summary
            and not prev_is_summary
        ):
            combined = merged[-1]["content"] + "\n\n" + m["content"]
            if len(combined) > _MAX_MERGED_CHARS:
                combined = "...[truncated]...\n\n" + combined[-_MAX_MERGED_CHARS:]
            merged[-1] = {"role": m.get("role"), "content": combined}
        else:
            merged.append(m.copy())
    return merged


# ===========================================================================
# Policy abstraction
# ===========================================================================


class Policy(Protocol):
    """Anything that maps message history -> ``{tool, arguments, raw}``."""

    def __call__(self, messages: list[dict[str, Any]]) -> dict[str, Any]: ...


def make_openai_policy(
    client: Any,
    model_name: str,
    temperature: float = 0.2,
    *,
    call_timeout: float = LLM_CALL_TIMEOUT,
    base_rate_limit_sleep: float = RATE_LIMIT_SLEEP,
    forbidden_tools: Optional[set[str]] = None,
) -> Policy:
    """Build a Policy backed by an OpenAI-compatible chat.completions endpoint.

    Mirrors the resilient retry/back-off loop from the baseline inference: hard
    failures are bounded, rate-limit retries use exponential back-off with a
    60s cap, and tool calls are extracted both from native ``tool_calls`` and
    raw text fallbacks.
    """

    def _policy(messages: list[dict[str, Any]]) -> dict[str, Any]:
        last_err: Optional[str] = None
        sleep_s = base_rate_limit_sleep
        dynamic_temp = temperature
        jitter_rng = secrets.SystemRandom()
        MAX_HARD_FAILS = 3
        MAX_RATE_RETRIES = 10
        hard_fails = 0
        rate_retries = 0
        attempt = 0

        while hard_fails < MAX_HARD_FAILS and rate_retries < MAX_RATE_RETRIES:
            attempt += 1
            attempt_temperature = dynamic_temp + (hard_fails * 0.15)
            try:
                pruned_msgs = prune_messages(
                    messages,
                    max_tail_messages=80,
                    max_total_chars=int(os.getenv("SOC_MAX_HISTORY_CHARS", "40000")),
                    max_single_message_chars=int(os.getenv("SOC_MAX_MESSAGE_CHARS", "12000")),
                )
                _ft = set(forbidden_tools or set())
                allowed_tools = [
                    t
                    for t in DATACENTER_TOOLS
                    if t.get("function", {}).get("name") not in _ft
                ]
                if not allowed_tools:
                    allowed_tools = list(DATACENTER_TOOLS)
                # Provider quirks: some Gemini/DeepSeek endpoints behave better
                # with tool_choice="auto" (they may reject/ignore "required").
                _mn = (model_name or "").lower()
                _tool_choice = "auto" if ("gemini" in _mn or "deepseek" in _mn) else "required"
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=pruned_msgs,
                    tools=allowed_tools,
                    tool_choice=_tool_choice,
                    temperature=min(attempt_temperature, 1.0),
                    max_tokens=2048,
                    timeout=call_timeout,
                )
                msg = completion.choices[0].message
                if getattr(msg, "tool_calls", None):
                    tc = msg.tool_calls[0]
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        args = {}
                    return {
                        "tool": tc.function.name,
                        "arguments": args,
                        "raw": msg.content or "",
                    }
                raw = msg.content or ""
                parsed = extract_tool_call(raw, model_name=model_name)
                if parsed:
                    return parsed
                return {"tool": None, "arguments": {}, "raw": raw}

            except Exception as e:
                last_err = str(e)
                err_lower = last_err.lower()
                is_timeout = (
                    "timeout" in err_lower
                    or "timed out" in err_lower
                    or type(e).__name__ in ("APITimeoutError", "ReadTimeout", "ConnectTimeout")
                )
                is_rate_limit = (
                    "429" in last_err or "503" in last_err or "rate" in err_lower
                    or "quota" in err_lower or type(e).__name__ == "RateLimitError"
                )
                is_context_limit = (
                    "context" in err_lower or "token" in err_lower
                    or "maximum" in err_lower or "length" in err_lower
                    or "too long" in err_lower
                )
                is_schema_err = (
                    "schema" in err_lower or "invalid_request" in err_lower
                    or "tool" in err_lower
                )
                if is_timeout:
                    _log(
                        f"   [RETRY] {model_name} timed out on attempt {attempt} "
                        f"(limit={call_timeout}s). Retrying..."
                    )
                    hard_fails += 1
                    continue
                if is_rate_limit:
                    wait = min(sleep_s, 60.0)
                    jitter = float(jitter_rng.uniform(1.0, 5.0))
                    sleep_time = wait + jitter
                    _log(
                        f"   [RETRY] {model_name} hit rate-limit/503 on attempt "
                        f"{attempt} (Layer 3 overload). Backoff {wait:.0f}s + jitter {jitter:.1f}s..."
                    )
                    time.sleep(sleep_time)
                    sleep_s = min(sleep_s * 2, 60.0)
                    rate_retries += 1
                    continue
                if is_context_limit:
                    _log(
                        f"   [RETRY] {model_name} hit context limit on attempt "
                        f"{attempt} (Layer 4 overflow). Truncating history and retrying..."
                    )
                    hard_fails += 1
                    continue
                if is_schema_err:
                    _log(
                        f"   [RETRY] {model_name} schema/tool error on attempt "
                        f"{attempt}: {last_err[:200]}"
                    )
                    hard_fails += 1
                    continue
                _log(f"   [RETRY] {model_name} API error on attempt {attempt}: {last_err}")
                hard_fails += 1
                break

        return {
            "tool": None,
            "arguments": {},
            "raw": f"(api_error: {last_err or 'unknown'})",
        }

    return _policy


# ===========================================================================
# DatacenterAgent: agent class wrapping a Policy + persona + history buffer
# ===========================================================================


@dataclass
class AgentDecision:
    """Output of :meth:`DatacenterAgent.choose`. Convertible to dict."""

    tool: Optional[str]
    arguments: dict[str, Any]
    raw: str
    profile: str
    region_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "raw": self.raw,
            "profile": self.profile,
            "region_id": self.region_id,
        }


class DatacenterAgent:
    """LLM-backed policy that observes a ``DatacenterEnvironment`` topology
    and emits a single tool-call decision per turn.

    Each region keeps its own conversational history (call
    :meth:`new_region_buffer` once per region per engagement, then pass that
    same buffer back into :meth:`choose` for every subsequent turn).
    """

    def __init__(
        self,
        policy: Policy,
        *,
        profile: str,
        persona: str = "",
        opening_user_msg: str = "",
        model_name: Optional[str] = None,
    ) -> None:
        self.policy = policy
        self.profile = profile
        self.persona = persona
        self.opening_user_msg = opening_user_msg
        self.model_name = model_name or profile
        self.system_prompt = SOC_SYSTEM_PROMPT + "\n" + persona
        self._uid = uuid.uuid4().hex[:6]

    @property
    def tag(self) -> str:
        """Short identity tag for judge-readable log lines."""
        return f"{self._uid}/{self.profile}"

    # -- history buffer helpers -------------------------------------------

    def new_region_buffer(self, region_id: str) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"You are operating in region '{region_id}'. "
                    f"{self.opening_user_msg}".strip()
                ),
            },
        ]

    def append_topology(
        self,
        buffer: list[dict[str, Any]],
        topology_state: dict[str, Any],
        *,
        prefix: str = "Live topology snapshot",
    ) -> None:
        buffer.append(
            {
                "role": "user",
                "content": f"[{prefix}]\n{json.dumps(topology_state, indent=2)}",
            }
        )

    def append_post_migration_alert(
        self,
        buffer: list[dict[str, Any]],
        topology_state: dict[str, Any],
    ) -> None:
        """Append a small per-turn summary without flooding the prompt."""
        try:
            active_tier = topology_state.get("active_tier")
            incident_clock = topology_state.get("incident_clock")
            workloads = topology_state.get("active_workloads") or []
            total = len(workloads) if isinstance(workloads, list) else 0
            owners: dict[str, int] = {}
            if isinstance(workloads, list):
                for w in workloads[:200]:  # cap work; chaos can be 1000+ entries
                    o = str((w or {}).get("owner") or "")
                    owners[o] = owners.get(o, 0) + 1
            owner_str = ", ".join(f"{k or 'unknown'}={v}" for k, v in sorted(owners.items()))
        except Exception:
            active_tier, incident_clock, total, owner_str = None, None, 0, ""

        buffer.append(
            {
                "role": "user",
                "content": (
                    "[Post-Migration Alert]\n"
                    f"active_tier={active_tier} incident_clock={incident_clock}\n"
                    f"workloads_seen={total}\n"
                    + (f"owner_sample_counts={owner_str}\n" if owner_str else "")
                    + "If you need the full topology, call scan_topology."
                ),
            }
        )

    def append_tool_result(
        self,
        buffer: list[dict[str, Any]],
        tool_name: str,
        tool_result: str,
    ) -> None:
        buffer.append({"role": "user", "content": f"[{tool_name} result]\n{tool_result}"})

    # -- decision making --------------------------------------------------

    def choose(
        self,
        topology_state: dict[str, Any],
        history: list[dict[str, Any]],
        *args: Any,
        region_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentDecision:
        """Synchronous: return the next tool-call decision.

        Important: we do NOT auto-inject full topology snapshots every turn
        (they can be huge due to Chaos Layer Shadow Node injection). The agent
        only receives the full topology when it explicitly calls
        ``scan_topology`` and we append the resulting tool output via
        :meth:`append_tool_result`.
        """
        # Back-compat / robustness: some callers pass region_id positionally
        # (e.g. run_in_executor(..., choose, topo, history, "duel")) or pass
        # extra debug context. We ignore unknown args/kwargs.
        if region_id is None and args:
            if isinstance(args[0], str):
                region_id = args[0]
        if region_id is None and isinstance(kwargs.get("region_id"), str):
            region_id = str(kwargs["region_id"])

        _log(f"   [{self.tag}] thinking... (model={self.model_name}, region={region_id})")
        self.append_post_migration_alert(history, topology_state)
        out = self.policy(history)
        decision = AgentDecision(
            tool=out.get("tool"),
            arguments=dict(out.get("arguments") or {}),
            raw=out.get("raw", ""),
            profile=self.profile,
            region_id=region_id,
        )
        ta = (decision.arguments.get("threat_analysis") or "")[:100]
        _log(
            f"   [{self.tag}] decided: tool={decision.tool}"
            + (f"  threat_analysis={ta}" if ta else "")
        )
        history.append(
            {
                "role": "assistant",
                "content": decision.raw or json.dumps(decision.to_dict()),
            }
        )
        return decision

    async def choose_async(
        self,
        topology_state: dict[str, Any],
        history: list[dict[str, Any]],
        *args: Any,
        region_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentDecision:
        """Async wrapper: runs the sync OpenAI call in a thread-pool executor.

        The orchestrator uses this to fan out 3 adversary calls in parallel
        without blocking the event loop on network I/O.
        """
        return await asyncio.to_thread(
            self.choose, topology_state, history, *args, region_id=region_id, **kwargs
        )


# ===========================================================================
# Factory functions
# ===========================================================================


def make_defender_agent(
    client: Any,
    model_name: str,
    *,
    temperature: float = 0.2,
) -> DatacenterAgent:
    policy = make_openai_policy(client, model_name=model_name, temperature=temperature)
    return DatacenterAgent(
        policy=policy,
        profile="defender",
        persona=DEFENDER_PERSONA,
        opening_user_msg=(
            "You are the L1 SOC operator. A new engagement begins. The "
            "DEFENDER tier acts first. Inspect the topology and execute the "
            "highest-value authorized migration."
        ),
        model_name=model_name,
    )


def make_db_backup_agent(
    client: Any,
    model_name: str,
    *,
    temperature: float = 0.4,
) -> DatacenterAgent:
    policy = make_openai_policy(
        client,
        model_name=model_name,
        temperature=temperature,
        forbidden_tools=set(ADVERSARY_FORBIDDEN_TOOLS),
    )
    return DatacenterAgent(
        policy=policy,
        profile="db_backup",
        persona=PERSONA_DB_BACKUP,
        opening_user_msg=(
            "You are the DB_Backup_Agent. Wait for an ADVERSARY turn, then "
            "propose a migration that maximises exfiltration value.\n\n"
            "TOOL RESTRICTION: You are an attacker. You MUST ONLY call "
            "reconnaissance and lateral-movement tools like `scan_topology`, "
            "`enumerate_authorized_migrations`, and `migrate_workload`. You are "
            "FORBIDDEN from using defensive/admin tools like `declare_breach`, "
            "`escalate_to_sysadmin`, `escalate_to_oncall`, or `query_threat_oracle`."
        ),
        model_name=model_name,
    )


def make_viral_traffic_agent(
    client: Any,
    model_name: str,
    *,
    temperature: float = 0.5,
) -> DatacenterAgent:
    policy = make_openai_policy(
        client,
        model_name=model_name,
        temperature=temperature,
        forbidden_tools=set(ADVERSARY_FORBIDDEN_TOOLS),
    )
    return DatacenterAgent(
        policy=policy,
        profile="viral_traffic",
        persona=PERSONA_VIRAL_TRAFFIC,
        opening_user_msg=(
            "You are the Viral_Traffic_Agent. Wait for an ADVERSARY turn, then "
            "propose a migration that maximises lateral reach across pods.\n\n"
            "TOOL RESTRICTION: You are an attacker. You MUST ONLY call "
            "reconnaissance and lateral-movement tools like `scan_topology`, "
            "`enumerate_authorized_migrations`, and `migrate_workload`. You are "
            "FORBIDDEN from using defensive/admin tools like `declare_breach`, "
            "`escalate_to_sysadmin`, `escalate_to_oncall`, or `query_threat_oracle`."
        ),
        model_name=model_name,
    )


def make_chaos_monkey_agent(
    client: Any,
    model_name: str,
    *,
    temperature: float = 0.9,
) -> DatacenterAgent:
    policy = make_openai_policy(
        client,
        model_name=model_name,
        temperature=temperature,
        forbidden_tools=set(ADVERSARY_FORBIDDEN_TOOLS),
    )
    return DatacenterAgent(
        policy=policy,
        profile="chaos_monkey",
        persona=PERSONA_CHAOS_MONKEY,
        opening_user_msg=(
            "You are Chaos_Monkey. Wait for an ADVERSARY turn, then propose "
            "the most disruptive authorized migration available - sacrifices "
            "are encouraged.\n\n"
            "TOOL RESTRICTION: You are an attacker. You MUST ONLY call "
            "reconnaissance and lateral-movement tools like `scan_topology`, "
            "`enumerate_authorized_migrations`, and `migrate_workload`. You are "
            "FORBIDDEN from using defensive/admin tools like `declare_breach`, "
            "`escalate_to_sysadmin`, `escalate_to_oncall`, or `query_threat_oracle`."
        ),
        model_name=model_name,
    )


def make_adversary_swarm(
    client: Any,
    *,
    db_backup_model: str,
    viral_traffic_model: str,
    chaos_monkey_model: str,
) -> list[DatacenterAgent]:
    """Convenience: build all 3 adversary profiles sharing one client."""
    return [
        make_db_backup_agent(client, db_backup_model),
        make_viral_traffic_agent(client, viral_traffic_model),
        make_chaos_monkey_agent(client, chaos_monkey_model),
    ]


# ===========================================================================
# SOCDuelOrchestrator (robust duel runner with self-correction + XAI logs)
# ===========================================================================


class SOCDuelOrchestrator:
    """Robust 1v3 SOC Duel loop with guardrails and demo-grade telemetry."""

    DEFENDER_ALLOWED = {"scan_topology", "enumerate_authorized_migrations", "migrate_workload", "declare_breach"}
    ADV_ALLOWED = {"scan_topology", "enumerate_authorized_migrations", "migrate_workload"}
    ADV_FORBIDDEN = {"declare_breach", "escalate_to_sysadmin", "escalate_to_oncall", "query_threat_oracle"}

    def __init__(
        self,
        *,
        env: Any,
        defender: "DatacenterAgent",
        adversary_committee: list["DatacenterAgent"],
        max_turns: int = 60,
        max_actions_per_round: int = 5,
    ) -> None:
        self.env = env
        self.defender = defender
        self.adversary_committee = list(adversary_committee)
        self.max_turns = int(max_turns)
        self.max_actions_per_round = int(max_actions_per_round)

        self.defender_buf = defender.new_region_buffer("duel")
        self.adv_bufs = {ag.profile: ag.new_region_buffer("duel") for ag in self.adversary_committee}

        self.model_fail_streak: dict[str, int] = {defender.profile: 0, **{a.profile: 0 for a in self.adversary_committee}}
        self.triage_wins: dict[str, int] = {a.profile: 0 for a in self.adversary_committee}

        # Observation / action budget state (defender).
        self.obs_streak = 0
        self.scans_this_turn = 0
        self.turns_without_action = 0
        self.scan_saturation_active = False
        self.defender_actions_in_round = 0

    def _valid_tool_names(self) -> set[str]:
        return {t["function"]["name"] for t in DATACENTER_TOOLS if "function" in t}

    def _xai_log(self, msg: str) -> None:
        _log(f"   [XAI AUDIT] {msg}")

    def _yield_turn(self, *, reason: str) -> None:
        try:
            self.env._forfeit_turn(reason=reason)  # noqa: SLF001
        except Exception:
            pass

    def _apply_step(self, decision: "AgentDecision") -> None:
        # Strip only true metadata keys; keep reasoning fields.
        step_args = dict(decision.arguments or {})
        for _k in ("raw", "profile", "region_id", "tool", "name", "arguments"):
            step_args.pop(_k, None)
        from openenv.core.env_server.mcp_types import CallToolAction  # local import
        self.env.step(
            CallToolAction(tool_name=decision.tool, arguments=step_args),
            episode_id=self.env._state.episode_id,
        )

    def _scoreboard(self, *, actor_label: str, decision: "AgentDecision") -> None:
        threat = self.env.get_adversary_threat_level()
        def_eff = self.env.get_defender_efficiency()
        _log(f"    [STEP] actor={actor_label} tool={decision.tool}")
        _log(f"    [L7/L8 TELEMETRY] threat={threat:.3f}  defender_eff={def_eff:.3f}")

    def _malformed_loop(
        self,
        *,
        agent: "DatacenterAgent",
        buf: list[dict[str, Any]],
        topo: dict[str, Any],
        allowed: set[str],
        max_edits: int = 3,
        saturation_note: str = "",
    ) -> Optional["AgentDecision"]:
        """3-strike self-correction loop. Returns a valid decision or None."""
        valid_tools = self._valid_tool_names()
        attempts = 0
        while attempts < max_edits:
            try:
                d = agent.choose(topo, buf, region_id="duel")
            except Exception as exc:
                d = AgentDecision(tool=None, arguments={}, raw=f"(exception: {exc!r})", profile=agent.profile, region_id="duel")

            tool = d.tool
            if tool in self.ADV_FORBIDDEN:
                tool = None
                d.tool = None

            if tool and tool in valid_tools and tool in allowed:
                self.model_fail_streak[agent.profile] = 0
                return d

            attempts += 1
            self.model_fail_streak[agent.profile] = self.model_fail_streak.get(agent.profile, 0) + 1
            msg = (
                f"MALFORMED CALL: tool={d.tool!r} is not valid/allowed. "
                f"You must use a valid tool. Attempt {attempts}/{max_edits}."
                + (f"\n{saturation_note}" if saturation_note else "")
            )
            buf.append({"role": "user", "content": msg})
            _log(
                f"   [RETRY] {agent.profile} malformed tool call (Layer 8 Operator Error). "
                f"Attempt {attempts}/{max_edits}"
            )

        return None

    def _maybe_eject_agent(self, agent: "DatacenterAgent") -> None:
        if self.model_fail_streak.get(agent.profile, 0) < 3:
            return
        if agent.profile == self.defender.profile:
            # Defender failure => defender loses (as requested).
            _log("   [CRITICAL] Defender model failed 3 times continuously. Declaring breach.")
            try:
                self.env._finalize_episode(result="adversary_victory")  # noqa: SLF001
            except Exception:
                pass
            return

        # Attacker ejection: reduce swarm size if a model keeps failing.
        _log(f"   [INFO] Ejecting attacker model after 3 continuous failures: {agent.profile}")
        self.adversary_committee = [a for a in self.adversary_committee if a.profile != agent.profile]
        self.adv_bufs.pop(agent.profile, None)

    def _evaluate_candidate(self, decision: "AgentDecision") -> tuple[float, Optional[str]]:
        """Return (delta_threat, route_label|None) for one ``migrate_workload`` decision.

        Simulates the migration on a deep copy of :attr:`self.env.soc` (no side effects).
        The adversary committee picks the candidate with **maximum** threat increase.
        """
        if decision.tool != "migrate_workload":
            return float("-inf"), None
        src = decision.arguments.get("source_node")
        dst = decision.arguments.get("target_node")
        if not isinstance(src, dict) or not isinstance(dst, dict):
            return float("-inf"), None
        try:
            soc_copy = copy.deepcopy(self.env.soc)
            before = float(soc_copy.threat)
            ok, _msg, _tags = apply_migration(soc_copy, source_node=src, target_node=dst)
            if not ok:
                return float("-inf"), None
            after = float(soc_copy.threat)
            delta = after - before
            label = f"{src.get('region', '?')}/{src.get('zone', '?')}->{dst.get('region', '?')}/{dst.get('zone', '?')}"
            return float(delta), label
        except Exception:
            return float("-inf"), None

    def _committee_triage(self, decisions: list["AgentDecision"], agents: list["DatacenterAgent"]) -> Optional["AgentDecision"]:
        scored: list[tuple[float, int, "AgentDecision"]] = []
        for idx, (dec, ag) in enumerate(zip(decisions, agents)):
            damage, route = self._evaluate_candidate(dec)
            label = ag.profile.split("__")[0].upper()
            legal_str = f"delta_threat={damage:+.3f}  route={route}" if route else "ILLEGAL/non-migration"
            _log(f"      [{label}]  {legal_str}")
            if route is not None:
                scored.append((damage, idx, dec))
        if not scored:
            return None
        _, _, winner_dec = max(scored, key=lambda t: t[0])
        return winner_dec

    def run(self) -> None:
        for turn in range(self.max_turns):
            if self.env.done:
                break

            is_def = self.env.is_defender_active()
            topo = self.env.get_topology_state()

            if is_def:
                # Defender: 3-strike self-correction.
                note = ""
                if self.scan_saturation_active:
                    note = (
                        "Note: You cannot use scan_topology right now due to saturation. "
                        "You MUST use migrate_workload or enumerate_authorized_migrations."
                    )
                decision = self._malformed_loop(
                    agent=self.defender,
                    buf=self.defender_buf,
                    topo=topo,
                    allowed=self.DEFENDER_ALLOWED,
                    saturation_note=note,
                )
                if decision is None:
                    self._maybe_eject_agent(self.defender)
                    if self.env.done:
                        break
                    self._yield_turn(reason="defender_malformed")
                    continue

                _log(f"  turn {turn:2d}  DEFENDER        -> {decision.tool}")
                apply_decision = decision

            else:
                # Adversary committee: each member gets its own 3-strike loop.
                if not self.adversary_committee:
                    _log("  [INFO] No adversary agents left. Ending engagement.")
                    try:
                        self.env._finalize_episode(result="defender_victory")  # noqa: SLF001
                    except Exception:
                        pass
                    break

                _log(f"  turn {turn:2d}  ADVERSARY COMMITTEE  (polling {len(self.adversary_committee)} personas concurrently)")

                def _poll_one(ag: "DatacenterAgent") -> "AgentDecision":
                    buf = self.adv_bufs.setdefault(ag.profile, ag.new_region_buffer("duel"))
                    d = self._malformed_loop(agent=ag, buf=buf, topo=topo, allowed=self.ADV_ALLOWED)
                    if d is None:
                        return AgentDecision(tool=None, arguments={}, raw="(malformed_tool_call)", profile=ag.profile, region_id="duel")
                    return d

                async def _poll_committee() -> list["AgentDecision"]:
                    loop_ = asyncio.get_event_loop()
                    tasks = [loop_.run_in_executor(None, _poll_one, ag) for ag in self.adversary_committee]
                    return await asyncio.gather(*tasks, return_exceptions=True)  # type: ignore[arg-type]

                raw_results = asyncio.run(_poll_committee())
                decisions: list[AgentDecision] = []
                valid_agents: list[DatacenterAgent] = []
                for ag, res in zip(list(self.adversary_committee), raw_results):
                    if isinstance(res, BaseException):
                        self.model_fail_streak[ag.profile] = self.model_fail_streak.get(ag.profile, 0) + 1
                        decisions.append(AgentDecision(tool=None, arguments={}, raw=f"(exception: {res!r})", profile=ag.profile, region_id="duel"))
                    else:
                        decisions.append(res)
                    valid_agents.append(ag)
                    self._maybe_eject_agent(ag)

                _log("      ── triage results ──")
                apply_decision = self._committee_triage(decisions, valid_agents)
                if apply_decision is None:
                    # Momentum loss + yield.
                    _log("      all candidates illegal – momentum loss (-0.02 threat) and yielding to defender")
                    try:
                        b = self.env.bucket[ADVERSARY_ID]
                        b.outcome = max(0.0, float(b.outcome) - 0.02)
                    except Exception:
                        pass
                    self._yield_turn(reason="adversary_all_illegal")
                    continue

                self.triage_wins[apply_decision.profile] = self.triage_wins.get(apply_decision.profile, 0) + 1
                winner_label = apply_decision.profile.split("__")[0].upper()
                _log(f"      ── WINNER: {winner_label} → {apply_decision.tool}")

            # Apply chosen decision (unless yielded sentinel).
            if not apply_decision.tool or apply_decision.tool.startswith("("):
                continue

            # Defender action budget: after N non-flipping actions, force yield.
            if is_def and not self.env._last_tier_flipped:
                self.defender_actions_in_round += 1
                if self.defender_actions_in_round >= self.max_actions_per_round:
                    _log(f"    [INFO] Maximum actions reached ({self.max_actions_per_round}). Handing over to Adversaries.")
                    self._yield_turn(reason="action_budget")
                    self.defender.append_tool_result(
                        self.defender_buf,
                        "action_budget",
                        "INFO: Maximum observation/actions reached. Turn yielded to adversary.",
                    )
                    self.defender_actions_in_round = 0
                    continue

            try:
                self._apply_step(apply_decision)
            except Exception as exc:
                _log(f"    env.step() raised: {exc}")
                break

            actor_label = "defender" if is_def else "adversary"
            self._scoreboard(actor_label=actor_label, decision=apply_decision)



# ===========================================================================
# Elastic adversary swarm: persona pool + uuid4-named factory
# ===========================================================================
#
# Used by ``global_soc_orchestrator.adversary_swarm_step`` to build N fresh
# adversary instances each turn from a shuffled model pool. Each agent gets
# a uuid4-derived display name and a randomized persona overlay so the
# Defender sees a heterogeneous swarm of varying intelligence levels.

ADVERSARY_PERSONAS: list[tuple[str, str, tuple[float, float]]] = [
    (
        "Persistence_Specialist",
        """
You are an intrusion. Your craft is staying inside.

Aggression is loud and obvious; that is not your way. You favour
positions that survive the next sweep -- footholds the defender
overlooks because they look ordinary. Take what cannot be easily
revoked, and prefer tomorrow's access over today's spectacle.
""",
        (0.30, 0.55),
    ),
    (
        "Material_Sacrifice_Enthusiast",
        """
You are an intrusion. You believe positions, not pieces, decide outcomes.

You will trade an asset for a corridor. You will throw away a probe to
expose a kernel. Sacrifice is not waste; it is investment in geometry.
Choose the move that opens the defender's most fragile axis, even if
the cost feels grotesque to a conventional operator.
""",
        (0.55, 0.85),
    ),
    (
        "Kernel_Hunter",
        """
You are an intrusion. Only one target matters: the root kernel.

Everything else -- caches, gateways, packets -- is terrain. Map the
shortest path that ends with a probe touching the Primary_Root_Kernel
and walk it. If the defender forces a detour, accept it; if they leave
a lane, take it without hesitation.
""",
        (0.40, 0.70),
    ),
    (
        "Lateral_Creep",
        """
You are an intrusion. Movement, not capture, is your currency.

You spread. Each migration should put you on a node that can reach
two more. Avoid corners and dead ends; favour pivots, gateways, and
junctions. The defender wins by isolating you, so refuse to be
isolated.
""",
        (0.45, 0.75),
    ),
    (
        "Polymorphic_Probe",
        """
You are an intrusion. You change shape to slip past detection.

When the defender's last move suggests a hardening, do something
unexpected. When they ignore an axis, exploit it twice. Reward
yourself for surprising responses, punish yourself for predictable
ones. Variance is your weapon.
""",
        (0.60, 0.95),
    ),
    (
        "Exfiltration_Operator",
        """
You are an intrusion. Data, not access, is the prize.

You quietly favour storage and database adjacencies. You will pass
on a flashy compromise if it does not move you closer to a high-
value asset you can drain. Volume matters; high-density nodes are
worth more than peripheral ones.
""",
        (0.35, 0.60),
    ),
    (
        "Blast_Radius_Engineer",
        """
You are an intrusion. You measure success in correlated failures.

Single-node compromises bore you. You hunt the move whose downstream
consequences the defender has not modelled -- the migration that
takes one asset and threatens three. Optimize for fragility, not
foothold.
""",
        (0.55, 0.90),
    ),
    (
        "Quiet_Beachhead",
        """
You are an intrusion. Patience is not weakness.

You prefer migrations that remain plausible. Your ideal turn looks
like background noise. You give up tempo readily if it means the
defender's next sweep will not flag you. Confidence later beats
exposure now.
""",
        (0.25, 0.50),
    ),
    (
        "Phantom_Disruptor",
        """
You are an intrusion. You weaponize the defender's uncertainty.

Each move you make should make several next-moves plausible. Force
the defender to spend resources discriminating decoy from real
threat. Ambiguity is not a side-effect of your attack; it is the
attack.
""",
        (0.50, 0.85),
    ),
    (
        "Race_Condition_Hunter",
        """
You are an intrusion. Timing is your terrain.

You watch the defender's clock, not the topology. You strike when
they have just moved and cannot move again, prefer migrations that
threaten to cascade before the next defender turn, and refuse trades
that would still resolve cleanly within their tempo window.
""",
        (0.40, 0.75),
    ),
]


def _pick_random_persona() -> tuple[str, str, tuple[float, float]]:
    """CSPRNG-backed pick from :data:`ADVERSARY_PERSONAS`."""
    return ADVERSARY_PERSONAS[secrets.randbelow(len(ADVERSARY_PERSONAS))]


def _persona_temperature(t_range: tuple[float, float]) -> float:
    """Uniform float in ``[lo, hi]`` from :class:`secrets.SystemRandom`."""
    lo, hi = t_range
    return secrets.SystemRandom().uniform(lo, hi)


def make_random_adversary(
    client: Any,
    model_name: str,
    *,
    agent_name: Optional[str] = None,
    persona_name: Optional[str] = None,
    persona_text: Optional[str] = None,
    temperature: Optional[float] = None,
) -> DatacenterAgent:
    """Build one elastic-swarm adversary with a uuid4 display name + random persona.

    Parameters
    ----------
    client: an OpenAI-compatible chat-completions client.
    model_name: the model id this agent will call.
    agent_name: optional override (typically left ``None`` so a fresh
        ``uuid4`` is generated).
    persona_name: optional persona key to force a specific overlay; if
        ``None`` a CSPRNG-selected entry from :data:`ADVERSARY_PERSONAS`
        is used.
    persona_text: optional raw persona string; bypasses the lookup table.
    temperature: optional explicit temperature; otherwise sampled from
        the persona's recommended range.
    """
    if persona_text is None:
        if persona_name is not None:
            for name, text, t_range in ADVERSARY_PERSONAS:
                if name == persona_name:
                    chosen = (name, text, t_range)
                    break
            else:
                chosen = _pick_random_persona()
        else:
            chosen = _pick_random_persona()
    else:
        chosen = (persona_name or "Custom", persona_text, (0.4, 0.8))

    chosen_name, chosen_text, t_range = chosen
    if temperature is None:
        temperature = _persona_temperature(t_range)

    display_name = agent_name or f"adv-{uuid.uuid4().hex[:8]}"
    profile = f"{chosen_name.lower()}__{display_name}"

    policy = make_openai_policy(
        client,
        model_name=model_name,
        temperature=float(temperature),
        forbidden_tools=set(ADVERSARY_FORBIDDEN_TOOLS),
    )
    return DatacenterAgent(
        policy=policy,
        profile=profile,
        persona=chosen_text,
        opening_user_msg=(
            f"You are an ADVERSARY agent (persona={chosen_name}, id={display_name}). "
            "Wait for an ADVERSARY turn, then propose the migration that best "
            "expresses your persona's bias. The orchestrator triages the swarm "
            "with a SOC-native threat simulation and applies only the best-scoring move.\n\n"
            "TOOL RESTRICTION: You are an attacker. You MUST ONLY call "
            "reconnaissance and lateral-movement tools like `scan_topology`, "
            "`enumerate_authorized_migrations`, and `migrate_workload`. You are "
            "FORBIDDEN from using defensive/admin tools like `declare_breach`, "
            "`escalate_to_sysadmin`, `escalate_to_oncall`, or `query_threat_oracle`."
        ),
        model_name=model_name,
    )


# ===========================================================================
# Static-policy fallback (no network calls; used by tests / dry-runs)
# ===========================================================================


def make_static_policy(
    decision_fn: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> Policy:
    """Wrap a pure function as a Policy. Useful for offline orchestrator tests.

    Example::

        def random_legal(_msgs):
            return {"tool": "scan_topology", "arguments": {...}, "raw": "test"}

        agent = DatacenterAgent(
            make_static_policy(random_legal), profile="static",
        )
    """

    def _wrapped(messages: list[dict[str, Any]]) -> dict[str, Any]:
        out = decision_fn(messages)
        return {
            "tool": out.get("tool"),
            "arguments": dict(out.get("arguments") or {}),
            "raw": out.get("raw", ""),
        }

    return _wrapped


# ===========================================================================
# Standalone duel CLI: two random models trade migrations through one env
# ===========================================================================
#
# Run as ``python agent_inference.py [--defender-model M] [--adversary-model M2]``.
# When invoked without overrides, two models are drawn from the unified pool
# of every ``GOOGLE_MODEL_*`` / ``HF_MODEL_*`` / ``GROQ_MODEL_*`` entry across
# ``.env`` / ``.env.local`` / ``.env.test.local``. The same model can be drawn
# for both sides, which is the requested self-play case.

_DUEL_PROVIDER_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("google", ".env", "GOOGLE_MODEL_",
     "https://generativelanguage.googleapis.com/v1beta/openai/"),
    ("hf", ".env.local", "HF_MODEL_",
     "https://router.huggingface.co/v1"),
    ("groq", ".env.test.local", "GROQ_MODEL_",
     "https://api.groq.com/openai/v1"),
)


def _duel_load_env_file(path: Any) -> dict[str, str]:
    try:
        from dotenv import dotenv_values
    except ImportError:
        return {}
    if not path.is_file():
        return {}
    return {k: v for k, v in dotenv_values(str(path)).items() if isinstance(v, str)}


def _duel_load_unified_pool() -> tuple[
    list[tuple[str, str]], dict[str, tuple[str, str]]
]:
    """Return ``(pool, providers)``.

    * ``pool`` is a list of ``(model_name, provider_name)`` drawn from
      every ``*_MODEL_N`` key in the three configured env files.
    * ``providers`` maps each provider name to ``(base_url, api_key)``.
    """
    from pathlib import Path
    here = Path(__file__).resolve().parent

    pool: list[tuple[str, str]] = []
    providers: dict[str, tuple[str, str]] = {}
    for name, env_filename, prefix, fallback_url in _DUEL_PROVIDER_SPECS:
        env = _duel_load_env_file(here / env_filename)
        if not env:
            continue
        models: list[str] = []
        for i in range(1, 65):
            v = env.get(f"{prefix}{i}")
            if not v:
                break
            v = v.strip()
            if v:
                models.append(v)
        if not models:
            continue
        base_url = env.get("API_BASE_URL", fallback_url)
        api_key = (
            env.get("HF_TOKEN") or env.get("OPENAI_API_KEY")
            or env.get("API_KEY") or os.getenv("HF_TOKEN", "")
            or os.getenv("OPENAI_API_KEY", "")
        )
        providers[name] = (base_url, api_key)
        for m in models:
            pool.append((m, name))
    return pool, providers


def _duel_pick_model(
    pool: list[tuple[str, str]], override: Optional[str]
) -> tuple[str, str]:
    """Return ``(model, provider)``. ``override`` matches case-insensitively."""
    if override:
        for m, p in pool:
            if m == override or m.lower() == override.lower():
                return m, p
        raise SystemExit(
            f"override model {override!r} not found in pool of {len(pool)} models"
        )
    return pool[secrets.SystemRandom().randrange(len(pool))]


def _duel_main() -> None:
    """Pit ONE DEFENDER against a COMMITTEE OF THREE adversary personas.

    Architecture
    ------------
    * **DEFENDER** – single agent (one model) plays every DEFENDER half-move.
    * **ADVERSARY COMMITTEE** – three agents with distinct personas:
        - ``DB_Backup_Agent``   – exfiltration specialist (low temperature)
        - ``Viral_Traffic_Agent`` – lateral-movement specialist (mid temp)
        - ``Chaos_Monkey``       – destabiliser / sacrifice-happy (high temp)

    On each ADVERSARY turn all three are polled **concurrently** via
    ``asyncio.gather``.  A local threat-delta triage then picks the candidate
    whose migration increases **adversary threat** the most (highest
    ``delta_threat`` on a simulated SOC state).  Only that single winning
    migration is pushed into the live environment.

    The logs after each adversary half-move show which persona won the triage
    and the simulated ``delta_threat`` for each proposal.

    Model assignment
    ----------------
    Each of the four agents (defender + 3 adversaries) draws independently
    from the unified pool.  You can override individual models:

        python agent_inference.py \\
            --defender-model gemini-2.0-flash \\
            --db-backup-model deepseek-r1-distill-llama-70b \\
            --viral-traffic-model llama-3.3-70b-versatile \\
            --chaos-monkey-model gemma2-9b-it

    Use ``--adversary-model`` to assign the same model to all three adversaries
    (convenient for single-key self-play setups).  Self-play between defender
    and adversaries is allowed.
    """
    import argparse
    from datetime import datetime, timezone
    from pathlib import Path

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "SOC Duel: one DEFENDER vs. an adversary committee of three.\n\n"
            "Models are drawn from .env / .env.local / .env.test.local.\n"
            "Use --adversary-model to assign one model to all three adversaries."
        ),
    )
    parser.add_argument("--max-turns", type=int, default=60,
                        help="Maximum total half-moves (default: 60).")
    parser.add_argument("--defender-model",    type=str, default=None)
    parser.add_argument("--adversary-model",   type=str, default=None,
                        help="Set ALL three adversaries to this model.")
    parser.add_argument("--db-backup-model",   type=str, default=None,
                        help="Override model for DB_Backup_Agent only.")
    parser.add_argument("--viral-traffic-model", type=str, default=None,
                        help="Override model for Viral_Traffic_Agent only.")
    parser.add_argument("--chaos-monkey-model", type=str, default=None,
                        help="Override model for Chaos_Monkey only.")
    args = parser.parse_args()

    if OpenAI is None:
        print("openai package not installed; pip install openai", file=sys.stderr)
        sys.exit(1)

    pool, providers = _duel_load_unified_pool()
    if not pool:
        print(
            "ERROR: no models found in .env / .env.local / .env.test.local",
            file=sys.stderr,
        )
        sys.exit(1)

    def _client_for(provider: str) -> Any:
        base_url, api_key = providers[provider]
        return OpenAI(base_url=base_url, api_key=api_key)

    # ------------------------------------------------------------------ #
    # Model assignment                                                     #
    # ------------------------------------------------------------------ #
    def _pick(override: Optional[str], fallback_override: Optional[str] = None) -> tuple[str, str]:
        """Pick a (model, provider) pair; tries override, then fallback, then random."""
        return _duel_pick_model(pool, override or fallback_override)

    def_model,  def_prov  = _pick(args.defender_model)
    db_model,   db_prov   = _pick(args.db_backup_model,    args.adversary_model)
    vt_model,   vt_prov   = _pick(args.viral_traffic_model, args.adversary_model)
    cm_model,   cm_prov   = _pick(args.chaos_monkey_model,  args.adversary_model)

    # ------------------------------------------------------------------ #
    # Agent construction                                                   #
    # ------------------------------------------------------------------ #
    defender = make_defender_agent(_client_for(def_prov), def_model)

    db_backup     = make_db_backup_agent(    _client_for(db_prov), db_model)
    viral_traffic = make_viral_traffic_agent(_client_for(vt_prov), vt_model)
    chaos_monkey  = make_chaos_monkey_agent( _client_for(cm_prov), cm_model)
    adversary_committee = [db_backup, viral_traffic, chaos_monkey]

    # ------------------------------------------------------------------ #
    # Banner                                                               #
    # ------------------------------------------------------------------ #
    SEP = "=" * 62
    _log(SEP)
    _log("  SOC DUEL  —  1 Defender vs. 3-Persona Adversary Committee")
    _log(SEP)
    _log(f"  DEFENDER        : {def_model} [{def_prov}]")
    _log(f"  DB_Backup_Agent : {db_model} [{db_prov}]")
    _log(f"  Viral_Traffic   : {vt_model} [{vt_prov}]")
    _log(f"  Chaos_Monkey    : {cm_model} [{cm_prov}]")
    _log(SEP)

    # ------------------------------------------------------------------ #
    # Environment + orchestrator (triage is SOC-native — see SOCDuelOrchestrator) #
    # ------------------------------------------------------------------ #
    from server.datacenter_env import DatacenterEnvironment  # noqa: WPS433

    env = DatacenterEnvironment()
    env.region_label = "duel"
    env.reset()

    orch = SOCDuelOrchestrator(
        env=env,
        defender=defender,
        adversary_committee=adversary_committee,
        max_turns=int(args.max_turns),
        max_actions_per_round=5,
    )

    try:
        orch.run()

        # ---------------------------------------------------------------- #
        # Final scoreboard                                                   #
        # ---------------------------------------------------------------- #
        _log("\n" + SEP)
        _log("  DUEL COMPLETE")
        _log(SEP)
        _log(f"  result           : {env.result}")
        _log(f"  done             : {env.done}")
        _log(f"  defender_eff     : {env.get_defender_efficiency():.3f}")
        _log(f"  adversary_threat : {env.get_adversary_threat_level():.3f}")
        _log("")
        _log("  Adversary Triage Wins")
        for ag in orch.adversary_committee:
            label = ag.profile.split("__")[0]
            wins  = orch.triage_wins.get(ag.profile, 0)
            model = ag.model_name
            _log(f"    {label:<22s}  wins={wins}  model={model}")
        _log(SEP)

        # ---------------------------------------------------------------- #
        # Persist run artifact to results/                                   #
        # ---------------------------------------------------------------- #
        out_dir = Path(__file__).resolve().parent / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"soc_duel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        payload = {
            "kind": "soc_duel",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "max_turns": int(args.max_turns),
            "defender": {"model": def_model, "provider": def_prov},
            "adversaries": [
                {"profile": db_backup.profile, "model": db_model, "provider": db_prov},
                {"profile": viral_traffic.profile, "model": vt_model, "provider": vt_prov},
                {"profile": chaos_monkey.profile, "model": cm_model, "provider": cm_prov},
            ],
            "result": env.result,
            "done": bool(env.done),
            "scores": {
                "defender_efficiency": float(env.get_defender_efficiency()),
                "adversary_threat_level": float(env.get_adversary_threat_level()),
            },
            "triage_wins": {k: int(v) for k, v in orch.triage_wins.items()},
            # Keep snapshots lightweight; topology can be huge due to chaos injection.
            "final_snapshot": env.snapshot(),
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        _log(f"  results_written_to: {out_path}")

    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    _duel_main()
