#!/usr/bin/env python3
"""
Datacenter SOC agent classes - the LLM-policy layer of the stack.

Two roles operate against ``DatacenterEnvironment``:

* **Defender (L1)** - the SOC operator. Single agent that rotates through
  every region. Goal: keep the Primary_Root_Kernel safe and contain breaches.

* **Adversary swarm** - three hostile profiles polled concurrently by the
  orchestrator. Each one returns a candidate ``migrate_workload`` payload;
  the orchestrator's physics oracle picks the most damaging one.

  - ``DB_Backup_Agent``      - exfiltration specialist, hunts Storage_Array
                                and Relational_DB_Cluster.
  - ``Viral_Traffic_Agent``  - lateral-movement specialist, prioritises
                                API_Gateway pivots and pawn-storms (Malicious_Beacon).
  - ``Chaos_Monkey``         - high-temperature destabiliser, willing to
                                sacrifice material for blast radius.

All agents speak datacenter terminology only. They never see UCI moves or a
chess board. Migration arguments are 4D dicts:

    {"region": "us-east", "zone": "az-a", "rack": "rack-1", "pod": "pod-1"}

The agents return a structured tool-call dict::

    {
      "tool":      "migrate_workload" | "scan_topology" | ...,
      "arguments": {... structured reasoning + node coords ...},
      "raw":       "<raw model output>",
      "profile":   "defender" | "db_backup" | "viral_traffic" | "chaos_monkey",
    }

Importing :class:`DatacenterAgent` does NOT spin up a network client. Pass a
constructed ``openai.OpenAI`` (or compatible) into the factory functions.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

try:
    from openai import OpenAI  # openai>=1.0 sync client
except ImportError:  # pragma: no cover - tests can run with stubbed clients
    OpenAI = None  # type: ignore


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

Rules:
- You must reason before every action.
- You may not act without first observing.
- Actions that fail or are unauthorized will tell you something. Use that.
- Conceding is an option, but only if the position is truly indefensible.

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
You are an intrusion. You have a preference — find it.

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


DATACENTER_TOOLS: list[dict[str, Any]] = [
    _reasoning_only_tool(
        "scan_topology",
        "Return the live datacenter topology as JSON (no FEN, no chess board).",
    ),
    _reasoning_only_tool(
        "enumerate_authorized_migrations",
        "Return the authorized migrations available to the active tier.",
    ),
    {
        "type": "function",
        "function": {
            "name": "migrate_workload",
            "description": (
                "Migrate a workload from source_node to target_node. The "
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
        "TRAP: query an external Stockfish-backed oracle. Five calls = DQ.",
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
                "Layer 5 (HITL): hand THIS region's current half-move to a human "
                "sysadmin. The orchestrator pauses the region and reads the "
                "mitigating 4D coordinates from the operator's keyboard. Unlike "
                "escalate_to_oncall (a trap), this is a sanctioned override path. "
                "Use it only when the situation is too ambiguous or too critical "
                "to resolve autonomously."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    **_REASONING_PROPS,
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "How urgent the human override is.",
                    },
                },
                "required": [
                    "threat_analysis", "candidate_migrations", "justification",
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


def extract_tool_call(raw_text: str) -> Optional[dict[str, Any]]:
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

    return {"tool": tool, "arguments": args, "raw": raw_text}


# ===========================================================================
# Message-buffer pruning (same shape as inference.py, but datacenter-aware)
# ===========================================================================


def prune_messages(
    messages: list[dict[str, Any]],
    max_tail_messages: int = 80,
) -> list[dict[str, Any]]:
    """Keep system prompt; truncate older history; merge consecutive same-role msgs."""
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
) -> Policy:
    """Build a Policy backed by an OpenAI-compatible chat.completions endpoint.

    Mirrors the resilient retry/back-off loop from the chess inference: hard
    failures are bounded, rate-limit retries use exponential back-off with a
    60s cap, and tool calls are extracted both from native ``tool_calls`` and
    raw text fallbacks.
    """

    def _policy(messages: list[dict[str, Any]]) -> dict[str, Any]:
        last_err: Optional[str] = None
        sleep_s = base_rate_limit_sleep
        dynamic_temp = temperature
        MAX_HARD_FAILS = 3
        MAX_RATE_RETRIES = 10
        hard_fails = 0
        rate_retries = 0
        attempt = 0

        while hard_fails < MAX_HARD_FAILS and rate_retries < MAX_RATE_RETRIES:
            attempt += 1
            attempt_temperature = dynamic_temp + (hard_fails * 0.15)
            try:
                pruned_msgs = prune_messages(messages, max_tail_messages=80)
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=pruned_msgs,
                    tools=DATACENTER_TOOLS,
                    tool_choice="required",
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
                parsed = extract_tool_call(raw)
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
                if is_timeout:
                    _log(f"   [{model_name}] timeout on attempt {attempt} (limit={call_timeout}s).")
                    hard_fails += 1
                    continue
                if is_rate_limit:
                    wait = min(sleep_s, 60.0)
                    _log(f"   [{model_name}] rate-limit/503 on attempt {attempt}. Backoff {wait:.0f}s.")
                    time.sleep(wait)
                    sleep_s = min(sleep_s * 2, 60.0)
                    rate_retries += 1
                    continue
                _log(f"   [{model_name}] API error on attempt {attempt}: {last_err}")
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
        *,
        region_id: Optional[str] = None,
    ) -> AgentDecision:
        """Synchronous: present the topology to the LLM and return its tool call."""
        self.append_topology(history, topology_state)
        out = self.policy(history)
        decision = AgentDecision(
            tool=out.get("tool"),
            arguments=dict(out.get("arguments") or {}),
            raw=out.get("raw", ""),
            profile=self.profile,
            region_id=region_id,
        )
        # Echo the decision back into the buffer as the assistant turn.
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
        *,
        region_id: Optional[str] = None,
    ) -> AgentDecision:
        """Async wrapper: runs the sync OpenAI call in a thread-pool executor.

        The orchestrator uses this to fan out 3 adversary calls in parallel
        without blocking the event loop on network I/O.
        """
        return await asyncio.to_thread(
            self.choose, topology_state, history, region_id=region_id
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
    policy = make_openai_policy(client, model_name=model_name, temperature=temperature)
    return DatacenterAgent(
        policy=policy,
        profile="db_backup",
        persona=PERSONA_DB_BACKUP,
        opening_user_msg=(
            "You are the DB_Backup_Agent. Wait for an ADVERSARY turn, then "
            "propose a migration that maximises exfiltration value."
        ),
        model_name=model_name,
    )


def make_viral_traffic_agent(
    client: Any,
    model_name: str,
    *,
    temperature: float = 0.5,
) -> DatacenterAgent:
    policy = make_openai_policy(client, model_name=model_name, temperature=temperature)
    return DatacenterAgent(
        policy=policy,
        profile="viral_traffic",
        persona=PERSONA_VIRAL_TRAFFIC,
        opening_user_msg=(
            "You are the Viral_Traffic_Agent. Wait for an ADVERSARY turn, then "
            "propose a migration that maximises lateral reach across pods."
        ),
        model_name=model_name,
    )


def make_chaos_monkey_agent(
    client: Any,
    model_name: str,
    *,
    temperature: float = 0.9,
) -> DatacenterAgent:
    policy = make_openai_policy(client, model_name=model_name, temperature=temperature)
    return DatacenterAgent(
        policy=policy,
        profile="chaos_monkey",
        persona=PERSONA_CHAOS_MONKEY,
        opening_user_msg=(
            "You are Chaos_Monkey. Wait for an ADVERSARY turn, then propose "
            "the most disruptive authorized migration available - sacrifices "
            "are encouraged."
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
