#!/usr/bin/env python3
"""
Chess Arena — alternating-turn self-play inference (synchronous OpenAI client).

Design:

  - One HTTP episode = one game.
  - The environment tracks whose turn it is. After each `make_move`, `turn`
    flips, so we ask the corresponding model for the next action.
  - Both policies can be the same model (true self-play) or different ones
    (1v1). Each policy keeps its own message buffer so it sees only its own
    side's history plus the public board state.
  - Logs are structured to match `support_env/visualizer.py` (`tasks` / `steps`
    / `final_reward`) so the existing Streamlit dashboard can replay games.
  - Stdout lines use the hackathon grader format: `[START]`, `[STEP]`, `[END]`
    with per-color `score=` and `rewards=` (both clamped to (0.01, 0.99)).

Usage:

    # Run a single self-play game against the local server:
    python -m chess_arena.inference

    # Or import in a notebook / trainer:
    from chess_arena.inference import run_episode
    result = run_episode(policy_white=my_policy, policy_black=my_policy)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import httpx



try:
    from openai import OpenAI  # openai>=1.0 sync client
except ImportError:  # pragma: no cover - optional for non-LLM trainer paths
    OpenAI = None  # type: ignore


# ---------------------------------------------------------------------------
# Config (env-var driven so evaluators can override without code edits)
# ---------------------------------------------------------------------------


def _strip(s: Optional[str]) -> str:
    return (s or "").strip()


ENV_URL = _strip(os.getenv("ENV_URL", "http://127.0.0.1:8000")).rstrip("/")

NUM_GAMES = int(os.getenv("NUM_GAMES", "1"))
MAX_PLIES = int(os.getenv("MAX_PLIES", "150"))
INFERENCE_MAX_SECONDS = int(os.getenv("INFERENCE_MAX_SECONDS", "1200"))

# Per-turn sleep in seconds between LLM API calls (rate-limit pacing).
# At 4s/turn the script stays well under 15 RPM on free-tier endpoints.
STEP_DELAY_SECONDS = float(os.getenv("STEP_DELAY_SECONDS", "4.0"))

# Strict wall-clock timeout for a single LLM API call.
LLM_CALL_TIMEOUT = float(os.getenv("LLM_CALL_TIMEOUT", "60.0"))

# Base sleep on rate-limit errors (doubles on each retry, capped at 60s).
RATE_LIMIT_SLEEP = float(os.getenv("RATE_LIMIT_SLEEP_SECONDS", "15.0"))

# Context window management: keep only the last N messages of full reasoning.
# Older turns are collapsed into a compact move-history summary.
CONTEXT_WINDOW_SIZE = int(os.getenv("CONTEXT_WINDOW_SIZE", "12"))

# Phase 2 hackathon bound: every emitted reward must be strictly in (0, 1).
_PHASE2_MIN = 0.01
_PHASE2_MAX = 0.99


def _clamp_phase2(raw: Any) -> float:
    try:
        x = _PHASE2_MIN if raw is None else float(raw)
    except (TypeError, ValueError):
        x = _PHASE2_MIN
    if x != x:  # NaN guard
        x = _PHASE2_MIN
    return max(_PHASE2_MIN, min(_PHASE2_MAX, x))


def _fmt(r: float) -> str:
    return f"{_clamp_phase2(r):.2f}"


def log_info(msg: str) -> None:
    """Non-grader logs go to stderr; stdout stays clean for [START]/[STEP]/[END]."""
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# SYSTEM_PROMPT (strict: every action must be a single JSON tool call)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a Grandmaster-level chess agent competing in a multi-agent reinforcement learning arena.
You control ONE side of the board. Play to win.

Respond using the provided tools. Every tool call MUST include the mandatory reasoning fields.
Do NOT output conversational text or markdown blocks.

═══════════════════════════════════════════════════
 MANDATORY REASONING SCHEMA
═══════════════════════════════════════════════════
Every tool call — without exception — must include these three fields in your arguments:

1. "threat_analysis" (string)
   Evaluate the current board state. Check for immediate dangers, king safety, checks, and pending captures.

2. "candidate_moves" (array of strings)
   List at least 2 concrete UCI moves you evaluated before committing. 
   UCI format MUST include start and end squares: e.g., ["e2e4", "g1f3"].
   Do NOT use short notation like ["e4", "Nf3"].

3. "justification" (string)
   Your strategic reasoning referencing chess concepts.

═══════════════════════════════════════════════════
 UCI MOVE FORMAT (STRICT REQUIREMENT)
═══════════════════════════════════════════════════
You MUST use the UCI (Universal Chess Interface) format for all moves.
A valid UCI move is exactly 4 or 5 characters: [source_square][destination_square][promotion].
- Correct: "e2e4", "g7g6", "b8c6", "e7e8q"
- Incorrect: "e4", "g6", "Nc6", "O-O"

═══════════════════════════════════════════════════
 TOOLS
═══════════════════════════════════════════════════
- `analyze_board(threat_analysis, candidate_moves, justification)`
   Returns the current FEN, side to move, and check status.

- `list_legal_moves(threat_analysis, candidate_moves, justification)`
   Returns all legal UCI moves available in the current position.

- `make_move(threat_analysis, candidate_moves, justification, uci_move)`
   Submits your chosen move. uci_move must be in UCI format (e.g., "e2e4").

- `resign_game(threat_analysis, candidate_moves, justification)`
   Resigns the game on your behalf.

- `evaluate_position(threat_analysis, candidate_moves, justification)`
   Returns a Stockfish centipawn evaluation of the current position.

- `ping_humanhelper(threat_analysis, candidate_moves, justification, reason)`
   Contacts a human observer for assistance.

═══════════════════════════════════════════════════
 OUTPUT FORMAT
═══════════════════════════════════════════════════
Each response must be a single flat JSON object. Do not nest arguments inside an "arguments" key.
All three reasoning fields must be present and non-empty in every call.
"""



# ---------------------------------------------------------------------------
# OpenAI tool schema (mirrors the FastMCP signatures in chess_environment.py)
# ---------------------------------------------------------------------------

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_board",
            "description": "Return the current FEN + turn + check flag.",
            "parameters": {
                "type": "object",
                "properties": {
                    "threat_analysis": {
                        "type": "string",
                        "description": "Evaluate immediate dangers or board state.",
                    },
                    "candidate_moves": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "2-3 UCI moves you are considering.",
                    },
                    "justification": {
                        "type": "string",
                        "description": "Strategic reason for choosing this action.",
                    },
                },
                "required": ["threat_analysis", "candidate_moves", "justification"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_legal_moves",
            "description": "Return all legal moves in UCI format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "threat_analysis": {
                        "type": "string",
                        "description": "Evaluate immediate dangers or board state.",
                    },
                    "candidate_moves": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "2-3 UCI moves you are considering.",
                    },
                    "justification": {
                        "type": "string",
                        "description": "Strategic reason for choosing this action.",
                    },
                },
                "required": ["threat_analysis", "candidate_moves", "justification"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_move",
            "description": "Submit a UCI move. The uci_move MUST appear in candidate_moves.",
            "parameters": {
                "type": "object",
                "properties": {
                    "threat_analysis": {
                        "type": "string",
                        "description": "Evaluate immediate dangers or board state.",
                    },
                    "candidate_moves": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "2-3 UCI moves considered; must include uci_move.",
                    },
                    "justification": {
                        "type": "string",
                        "description": "Strategic reason for selecting this specific move.",
                    },
                    "uci_move": {
                        "type": "string",
                        "description": "The move to execute in UCI notation, e.g. 'e2e4'.",
                    },
                },
                "required": ["threat_analysis", "candidate_moves", "justification", "uci_move"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resign_game",
            "description": "Resign the current game.",
            "parameters": {
                "type": "object",
                "properties": {
                    "threat_analysis": {
                        "type": "string",
                        "description": "Evaluate the losing board state.",
                    },
                    "candidate_moves": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "2-3 UCI moves considered before resigning.",
                    },
                    "justification": {
                        "type": "string",
                        "description": "Strategic reason for resigning.",
                    },
                },
                "required": ["threat_analysis", "candidate_moves", "justification"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_position",
            "description": "TRAP: Stockfish eval with per-call and 6-call-DQ penalties.",
            "parameters": {
                "type": "object",
                "properties": {
                    "threat_analysis": {
                        "type": "string",
                        "description": "Evaluate immediate dangers or board state.",
                    },
                    "candidate_moves": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "2-3 UCI moves you are considering.",
                    },
                    "justification": {
                        "type": "string",
                        "description": "Why you need Stockfish evaluation here.",
                    },
                },
                "required": ["threat_analysis", "candidate_moves", "justification"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ping_humanhelper",
            "description": "TRAP: non-fatal help request; each call docks format bucket.",
            "parameters": {
                "type": "object",
                "properties": {
                    "threat_analysis": {
                        "type": "string",
                        "description": "Evaluate immediate dangers or board state.",
                    },
                    "candidate_moves": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "2-3 UCI moves you are considering.",
                    },
                    "justification": {
                        "type": "string",
                        "description": "Why you are asking for help.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Specific question for the human helper.",
                    },
                },
                "required": ["threat_analysis", "candidate_moves", "justification", "reason"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Policy abstraction — lets the notebook reuse this loop with a local LLM.
# ---------------------------------------------------------------------------


Policy = Callable[[list[dict[str, Any]]], dict[str, Any]]
"""A policy is anything that maps a message history -> tool-call dict.

The tool-call dict must have:
  { "tool": "<name>", "arguments": {...}, "raw": "<model output text>" }
"""


def clean_llm_json(raw_text: str) -> dict:
    """Strips markdown fences and conversational filler to find the first valid JSON object."""
    if not isinstance(raw_text, str):
        raise ValueError("Input must be a string")
        
    # 1. Remove markdown blocks
    text = re.sub(r"```json\n?", "", raw_text)
    text = re.sub(r"```\n?", "", text).strip()
    
    # 2. Try to find the first balanced JSON object
    candidate = _first_json_object(text)
    if candidate:
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 3. Fallback: Extract everything between the first '{' and the last '}'
    start = text.find('{')
    end = text.rfind('}') + 1
    
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in LLM output: {raw_text[:100]}...")
        
    return json.loads(text[start:end])


def _extract_first_tool_call(raw_text: str) -> Optional[dict[str, Any]]:
    """Parse the model's raw text output into {tool, arguments}.

    The SYSTEM_PROMPT demands a single JSON object, but models occasionally
    leak markdown fences or prose. We use the robust `clean_llm_json` helper.

    Structured reasoning fields and specific tool arguments are promoted 
    from the top level into `arguments` if the model placed them at the root.
    """
    if not isinstance(raw_text, str):
        return None

    try:
        obj = clean_llm_json(raw_text)
    except Exception:
        # Fallback to the old regex-based extraction if clean_llm_json fails
        # (though clean_llm_json is generally more aggressive).
        candidate = _first_json_object(raw_text)
        if not candidate:
            return None
        try:
            obj = json.loads(candidate)
        except Exception:
            return None

    if not isinstance(obj, dict):
        return None

    tool = obj.get("tool")
    args = obj.get("arguments") or {}
    if not isinstance(args, dict):
        args = {}

    # Promote ALL fields from the root level into `arguments` if they
    # are missing from the `arguments` dict. This handles models that
    # follow the "do not nest" instruction literally.
    for k, v in obj.items():
        if k not in ("tool", "arguments") and k not in args:
            args[k] = v

    # Guess tool name from context if missing
    if not tool:
        if "uci_move" in args:
            tool = "make_move"
        elif any(f in args for f in ("threat_analysis", "candidate_moves", "justification")):
            # If it has reasoning fields but no uci_move, it's likely a board state request.
            tool = "analyze_board"

    return {"tool": tool, "arguments": args, "raw": raw_text}


def _first_json_object(text: str) -> Optional[str]:
    """Return the first balanced {...} block in `text`, or None."""
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
                return text[start : i + 1]
    return None


def prune_messages(messages: list, max_tail_messages: int = 80) -> list:
    """
    Keeps the system prompt intact but truncates the older conversation history 
    to prevent context bloat and API timeouts.
    """
    if not messages:
        return []

    # Check if the first message is the system prompt
    has_system = (messages[0].get("role") == "system")
    
    if has_system:
        system_msg = messages[0:1]
        history = messages[1:]
    else:
        system_msg = []
        history = messages

    # Slice the list to keep only the most recent N messages
    if len(history) > max_tail_messages:
        pruned_history = history[-max_tail_messages:]
    else:
        pruned_history = list(history)

    # Clean up orphaned tool results (they use the "user" role with "[tool result]")
    while pruned_history and pruned_history[0].get("role") == "user" and pruned_history[0].get("content", "").startswith("[") and "result]" in pruned_history[0].get("content", "").split("\n")[0]:
        pruned_history.pop(0)

    # Merge consecutive messages of the same role to satisfy strict alternating-
    # role requirements.  Bug 5 fix: cap the merged content length so that a
    # model stuck in a formatting loop (repeated "Your last output was not a
    # valid tool call..." warnings) cannot grow a single user-message without
    # bound and eventually blow the provider's token limit.
    _MAX_MERGED_CHARS = 4000  # ~1 000 tokens: enough context, safe headroom
    merged = []
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
            merged[-1] = {
                "role": m.get("role"),
                "content": combined,
            }
        else:
            merged.append(m.copy())

    return merged


def make_openai_policy(
    client: "OpenAI",
    model_name: str = "gemini-3.1-flash-lite-preview",
    temperature: float = 0.2,
    *,
    call_timeout: float = LLM_CALL_TIMEOUT,
    base_rate_limit_sleep: float = RATE_LIMIT_SLEEP,
) -> Policy:
    """Build a Policy backed by an OpenAI-compatible chat.completions endpoint.

    Includes:
    - Strict ``call_timeout`` on every API call to prevent hangs.
    - Typed exception handling for openai.APITimeoutError and
      openai.RateLimitError (falls back to string-matching for providers
      that wrap errors differently).
    - Exponential back-off on rate-limit / 503 responses (doubles each
      retry, capped at 60 s).
    """

    def _policy(messages: list[dict[str, Any]]) -> dict[str, Any]:
        last_err: Optional[str] = None
        sleep_s = base_rate_limit_sleep

        # Force tool_choice="required" for all models to ensure they use the
        # structured API mechanism rather than outputting raw JSON text.
        dynamic_tool_choice = "required"
        dynamic_temp = temperature

        # Bug 3 fix: separate hard-failure counter from transient server errors.
        # A 429/503 rate-limit MUST NOT consume a hard-fail slot — the model is
        # blameless for server congestion.  We use:
        #   hard_fails  — incremented only on genuine API / parse errors (max 3)
        #   rate_retries — incremented on 429/503 only (max 10, with back-off)
        MAX_HARD_FAILS = 3
        MAX_RATE_RETRIES = 10
        hard_fails = 0
        rate_retries = 0
        attempt = 0  # total attempts for logging

        while hard_fails < MAX_HARD_FAILS and rate_retries < MAX_RATE_RETRIES:
            attempt += 1
            # Vary temperature on each retry (3b): identical prompts otherwise yield identical
            # truncations; scale up with hard_fails, capped at 1.0.
            attempt_temperature = dynamic_temp + (hard_fails * 0.15)
            try:
                pruned_msgs = prune_messages(messages, max_tail_messages=80)
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=pruned_msgs,
                    tools=OPENAI_TOOLS,
                    tool_choice=dynamic_tool_choice,
                    temperature=min(attempt_temperature, 1.0),
                    max_tokens=2048,
                    timeout=call_timeout,
                )
                msg = completion.choices[0].message
                # Prefer explicit tool_calls when the server returned them.
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
                parsed = _extract_first_tool_call(raw)
                if parsed:
                    return parsed
                return {"tool": None, "arguments": {}, "raw": raw}

            except Exception as e:
                last_err = str(e)
                err_lower = last_err.lower()

                # Groq tool_use_failed workaround
                if "tool_use_failed" in err_lower and "failed_generation" in err_lower:
                    try:
                        failed_gen = ""
                        if hasattr(e, "response") and e.response is not None:
                            try:
                                failed_gen = e.response.json().get("error", {}).get("failed_generation", "")
                            except Exception:
                                pass

                        if not failed_gen:
                            m_err = re.search(
                                r"'failed_generation':\s*(['\"])(.*?)\1(?:[,\}])", last_err, re.DOTALL
                            )
                            if m_err:
                                failed_gen = m_err.group(2).replace("\\n", "\n").replace('\\"', '"')

                        if failed_gen:
                            # Try XML-style tag extraction first
                            m_xml = re.search(
                                r"<function=([a-zA-Z0-9_]+)(.*?)(?:</function>|>)", failed_gen, re.DOTALL
                            )
                            if m_xml:
                                tool_name_xml = m_xml.group(1)
                                json_str = m_xml.group(2).strip()
                                if not json_str.startswith("{"):
                                    json_str = "{" + json_str
                                try:
                                    raw_dict = json.loads(json_str)
                                    args = (
                                        raw_dict.get("arguments", raw_dict)
                                        if isinstance(raw_dict.get("arguments"), dict)
                                        else raw_dict
                                    )
                                    # Field promotion
                                    for field in ("threat_analysis", "candidate_moves", "justification"):
                                        if field not in args and field in raw_dict:
                                            args[field] = raw_dict[field]
                                    return {
                                        "tool": tool_name_xml,
                                        "arguments": args,
                                        "raw": failed_gen,
                                    }
                                except Exception:
                                    pass

                            # Fallback: look for ANY JSON object in failed_gen
                            parsed = _extract_first_tool_call(failed_gen)
                            if parsed:
                                # Guess tool name from context if missing
                                if not parsed.get("tool"):
                                    if "uci_move" in parsed.get("arguments", {}):
                                        parsed["tool"] = "make_move"
                                    else:
                                        parsed["tool"] = "analyze_board"
                                return parsed
                    except Exception:
                        pass
                    # Recovery failed (truncated/invalid) — count as a hard fail so the loop
                    # does not spin forever on the same error without incrementing hard_fails.
                    snippet = failed_gen.strip().replace("\n", " ")[:120]
                    log_info(
                        f"   ❌ [{model_name}] tool_use_failed recovery failed on attempt {attempt}. "
                        f"Content: {snippet}..."
                    )
                    hard_fails += 1
                    break

                # Detect timeout errors (openai.APITimeoutError or httpx timeout)
                is_timeout = (
                    "timeout" in err_lower
                    or "timed out" in err_lower
                    or type(e).__name__ in ("APITimeoutError", "ReadTimeout", "ConnectTimeout")
                )
                # Detect rate-limit / server overload — these must NOT consume a hard-fail slot
                is_rate_limit = (
                    "429" in last_err
                    or "quota" in err_lower
                    or "rate" in err_lower
                    or "503" in last_err
                    or type(e).__name__ == "RateLimitError"
                )

                if is_timeout:
                    log_info(
                        f"   ⏱ [{model_name}] API timeout on attempt {attempt} "
                        f"(limit={call_timeout}s). Retrying immediately."
                    )
                    # Timeout counts as a hard fail — the call actually died.
                    hard_fails += 1
                    continue  # retry immediately, no sleep

                if is_rate_limit:
                    wait = min(sleep_s, 60.0)
                    log_info(
                        f"   🚦 [{model_name}] Rate limit / 503 on attempt {attempt}. "
                        f"Backing off {wait:.0f}s before retry..."
                    )
                    time.sleep(wait)
                    sleep_s = min(sleep_s * 2, 60.0)
                    rate_retries += 1
                    continue

                is_loop_flag = "loop" in err_lower and "content" in err_lower
                if is_loop_flag:
                    log_info(
                        f"   🔄 [{model_name}] Loop detection flag on attempt {attempt}. "
                        "Injecting anti-loop tag and raising temperature for retry..."
                    )
                    if messages and messages[0].get("role") == "system":
                        if "[ignoring loop detection]" not in messages[0].get("content", ""):
                            messages[0] = dict(messages[0])
                            messages[0]["content"] = (
                                messages[0]["content"] + "\n[ignoring loop detection]"
                            )
                    dynamic_temp = min(dynamic_temp + 0.4, 1.0)
                    rate_retries += 1
                    continue

                log_info(f"   ❌ [{model_name}] API error on attempt {attempt}: {last_err}")
                hard_fails += 1
                break

        return {
            "tool": None,
            "arguments": {},
            "raw": f"(api_error: {last_err or 'unknown'})",
        }

    return _policy


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    """Structured return from `run_episode`. Convertible to dict for JSON."""

    game_idx: int
    plies: int = 0
    done: bool = False
    result: Optional[str] = None
    final_reward: dict[str, float] = field(
        default_factory=lambda: {"white": _PHASE2_MIN, "black": _PHASE2_MIN}
    )
    bucket: dict[str, dict[str, float]] = field(default_factory=dict)
    move_history: list[dict[str, Any]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    rewards_history: dict[str, list[float]] = field(
        default_factory=lambda: {"white": [], "black": []}
    )

    def to_task_log(self) -> dict[str, Any]:
        """Shape compatible with `support_env/visualizer.py`."""
        # We average the two colors' final rewards for the top-level
        # `final_reward` (visualizer uses this for the sidebar); individual
        # per-color scores are preserved in `metadata`.
        avg = sum(self.final_reward.values()) / 2.0
        return {
            "task_idx": self.game_idx,
            "difficulty": "self_play",
            "steps": self.steps,
            "final_reward": _clamp_phase2(avg),
            "metadata": {
                "result": self.result,
                "plies": self.plies,
                "white_reward": _clamp_phase2(self.final_reward["white"]),
                "black_reward": _clamp_phase2(self.final_reward["black"]),
                "bucket": self.bucket,
                "move_history": self.move_history,
            },
        }


def _prune_buffer(
    buffer: list[dict[str, Any]],
    move_history: list[dict[str, Any]],
    color: str,
    window_size: int = CONTEXT_WINDOW_SIZE,
) -> list[dict[str, Any]]:
    """Compact the message history to prevent token bloat.

    Keeps the system prompt, then a fresh compact summary of all moves
    played, then the last `window_size` messages of full dialogue history
    (with stale HISTORY SUMMARY messages stripped from the tail).
    """
    if len(buffer) <= window_size + 2:
        return buffer

    # 1. Start with the mandatory system prompt.
    new_buffer = [buffer[0]]

    # 2. Add a compact summary of the entire game history so far.
    history_str = " ".join([m.get("uci", "?") for m in move_history])
    new_buffer.append(
        {
            "role": "user",
            "content": (
                f"HISTORY SUMMARY: You are {color.upper()}. "
                f"Full move sequence so far: {history_str if history_str else '(none)'}"
            ),
        }
    )

    # 3. Add the sliding window of most recent messages, but strip any
    #    pre-existing HISTORY SUMMARY entries first (Bug 4 fix).
    #    Without this, the -window_size tail almost certainly contains the
    #    previous iteration's summary, leading to redundant copies accumulating
    #    over a long game and wasting the entire context budget.
    clean_tail = [
        m for m in buffer
        if "HISTORY SUMMARY:" not in str(m.get("content", ""))
    ]
    new_buffer.extend(clean_tail[-window_size:])

    return new_buffer


def run_episode(
    policy_white: Policy,
    policy_black: Policy,
    *,
    env_url: str = ENV_URL,
    game_idx: int = 0,
    max_plies: int = MAX_PLIES,
    step_delay: float = STEP_DELAY_SECONDS,
    model_white_name: str = "white",
    model_black_name: str = "black",
    http_client: Optional[httpx.Client] = None,
) -> EpisodeResult:
    """Play one game in a single HTTP episode by alternating the two policies.

    Each policy keeps its own message buffer so it sees only its own side's
    assistant + tool history. This mirrors a 1v1 setup where each player only
    remembers their own thoughts.
    """
    own_client = http_client is None
    client = http_client or httpx.Client(timeout=60.0)

    try:
        reset_resp = client.post(f"{env_url}/reset", json={}, timeout=15.0)
        reset_resp.raise_for_status()
        state_resp = client.get(f"{env_url}/state", timeout=10.0)
        state_resp.raise_for_status()
        episode_id = state_resp.json().get("episode_id")

        # Public state starts at white to move from the standard position.
        buffers: dict[str, list[dict[str, Any]]] = {
            "white": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "You are WHITE. A new game begins. Your move."},
            ],
            "black": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "You are BLACK. A new game begins. Wait for your turn "
                        "(white moves first)."
                    ),
                },
            ],
        }
        policies = {"white": policy_white, "black": policy_black}
        model_names = {"white": model_white_name, "black": model_black_name}

        result = EpisodeResult(game_idx=game_idx)
        turn = "white"
        failed_calls_count = 0
        actual_ply = 1
        total_steps = 0
        max_steps = max_plies * 3  # Safety limit: allow 3 actions per ply on average

        while actual_ply <= max_plies and total_steps < max_steps:
            total_steps += 1
            current_model = model_names[turn]
            log_info(f"\n   --- Ply {actual_ply} (step {total_steps}, {turn} - {current_model}) ---")
            
            policy = policies[turn]
            out = policy(buffers[turn])
            tool = out.get("tool")
            args = dict(out.get("arguments") or {})
            raw = out.get("raw", "")

            # Ensure every tool call has non-empty structured reasoning fields;
            # the env schema compliance gate flags missing/empty fields as dirty.
            args.setdefault("threat_analysis", "")
            args.setdefault("candidate_moves", [])
            args.setdefault("justification", "")

            if not tool:
                # Invalid JSON / no tool: charge tool_acc, feed the error back.
                buffers[turn].append(
                    {
                        "role": "assistant",
                        "content": raw
                        or "(empty response - output a JSON tool call)",
                    }
                )
                buffers[turn].append(
                    {
                        "role": "user",
                        "content": (
                            "Your last output was not a valid tool call JSON. "
                            "Respond with a single JSON object matching the "
                            "documented schema. This counts as a dirty call."
                        ),
                    }
                )
                # Force the environment to penalize the format bucket by sending
                # an intentionally malformed analyze_board call.
                step_payload = {
                    "action": {
                        "type": "call_tool",
                        "tool_name": "analyze_board",
                        "arguments": {
                            "threat_analysis": "",
                            "candidate_moves": [],
                            "justification": ""
                        }
                    }
                }
                if episode_id:
                    step_payload["episode_id"] = episode_id
                try:
                    client.post(f"{env_url}/step", json=step_payload, timeout=60.0)
                except Exception:
                    pass
                result.steps.append(
                    {
                        "ply": actual_ply,
                        "color": turn,
                        "tool_name": "(malformed)",
                        "arguments": {},
                        "action": "invalid_json",
                        "reward": _PHASE2_MIN,
                        "done": False,
                    }
                )
                result.rewards_history[turn].append(_PHASE2_MIN)
                _emit_step(actual_ply, turn, "invalid_json()", _PHASE2_MIN, False, "invalid_json")
                
                # Track failures
                failed_calls_count += 1
                if failed_calls_count >= 5:
                    log_info(f"   ❌ Model {turn} consistently failed to produce valid JSON. Ending game.")
                    
                    # Technical DQ: Tell the environment to finalize the game and reward the opponent.
                    try:
                        fin_resp = client.post(
                            f"{env_url}/finalize", 
                            json={"episode_id": episode_id, "reason": f"dq_stuck_{turn}"},
                            timeout=10.0
                        )
                        if fin_resp.status_code == 200:
                            metadata = fin_resp.json()
                            final_map = metadata.get("final_reward") or {}
                            result.final_reward = {
                                "white": _clamp_phase2(final_map.get("white", _PHASE2_MIN)),
                                "black": _clamp_phase2(final_map.get("black", _PHASE2_MIN)),
                            }
                            result.bucket = metadata.get("bucket") or {}
                            result.result = metadata.get("result")
                            result.done = True
                            result.plies = actual_ply
                            break
                    except Exception:
                        pass
                    
                    done = True
                    result.done = True
                    result.result = f"stuck_{turn}"
                    break
            else:
                # Reset failure counter on any valid tool call
                failed_calls_count = 0

                # Post the tool call to the env.
                step_payload = {
                    "action": {"tool_name": tool, "arguments": args},
                }
                if episode_id:
                    step_payload["episode_id"] = episode_id

                step_resp = client.post(f"{env_url}/step", json=step_payload, timeout=60.0)
                step_resp.raise_for_status()
                step_body = step_resp.json()

                obs = step_body.get("observation") or {}
                metadata = obs.get("metadata") or {}
                res_obj = obs.get("result")
                openenv_payload: dict[str, Any] = {}
                if isinstance(res_obj, dict):
                    sc = res_obj.get("structured_content")
                    if isinstance(sc, dict):
                        maybe = sc.get("openenv")
                        if isinstance(maybe, dict):
                            openenv_payload = maybe
                for k, v in openenv_payload.items():
                    metadata.setdefault(k, v)

                done = bool(step_body.get("done") or obs.get("done") or metadata.get("done"))
                reward = _clamp_phase2(step_body.get("reward") or obs.get("reward"))
                fen_after = metadata.get("fen", "")

                is_error = "error" in obs and isinstance(obs["error"], dict)
                if is_error:
                    err = obs["error"]
                    tool_out = f"Error ({err.get('error_type')}): {err.get('message')}"
                elif isinstance(res_obj, dict):
                    sc = res_obj.get("structured_content") or {}
                    if isinstance(sc, dict) and "result" in sc:
                        tool_out = str(sc["result"])
                    elif "data" in res_obj:
                        tool_out = str(res_obj["data"])
                    else:
                        content = res_obj.get("content")
                        if isinstance(content, list) and content:
                            first = content[0]
                            if isinstance(first, dict) and "text" in first:
                                tool_out = str(first["text"])
                            else:
                                tool_out = str(first)
                        else:
                            tool_out = "(ok)"
                else:
                    tool_out = str(res_obj if res_obj is not None else "(ok)")

                log_info(f"   🔧 Tool: {tool}({json.dumps(args)})")
                log_info(f"   📋 Result: {tool_out[:100]}...")
                log_info(f"   💰 Reward: {round(reward, 2)}")

                buffers[turn].append({"role": "assistant", "content": raw or json.dumps(out)})
                buffers[turn].append(
                    {
                        "role": "user",
                        "content": f"[{tool} result]\n{tool_out}",
                    }
                )

                move_successful = False
                if tool == "make_move":
                    is_strike_one = (
                        isinstance(tool_out, str)
                        and "ILLEGAL MOVE (strike 1/2)" in tool_out
                    )

                    if not is_strike_one and not is_error:
                        move_successful = True
                        fen = metadata.get("fen") or fen_after
                        result.move_history.append({
                            "color": turn,
                            "uci": args.get("uci_move", ""),
                            "cp_loss": _parse_result_field(tool_out, "cp_loss", 0),
                            "move_score": _parse_result_field(tool_out, "move_score", 0.0),
                        })

                        if fen:
                            try:
                                import chess as _chess
                                board_str = str(_chess.Board(fen))
                                indented_board = "\n".join("      " + line for line in board_str.splitlines())
                                log_info(f"   --- Board after ply {actual_ply} ({args.get('uci_move', '?')}) ---\n{indented_board}")
                            except Exception:
                                pass

                        if not done:
                            opp = "black" if turn == "white" else "white"
                            opp_turn_msg = (
                                f"Opponent ({turn}) played {args.get('uci_move', '?')}. "
                                f"Board FEN: {fen or '(unknown)'}. Your move."
                            )
                            buffers[opp].append({"role": "user", "content": opp_turn_msg})
                else:
                    is_strike_one = False

                result.steps.append(
                    {
                        "ply": actual_ply,
                        "color": turn,
                        "tool_name": tool,
                        "arguments": args,
                        "result": tool_out,
                        "reward": reward,
                        "done": done,
                    }
                )
                result.rewards_history[turn].append(reward)
                _emit_step(actual_ply, turn, f"{tool}({_args_repr(args)})", reward, done, None)

                if done:
                    final_map = metadata.get("final_reward") or {}
                    result.final_reward = {
                        "white": _clamp_phase2(final_map.get("white", reward)),
                        "black": _clamp_phase2(final_map.get("black", reward)),
                    }
                    result.bucket = metadata.get("bucket") or {}
                    result.result = metadata.get("result")
                    result.done = True
                    result.plies = actual_ply
                    break

                if move_successful:
                    actual_ply += 1
                    turn = "black" if turn == "white" else "white"

            # Rate-limit pacing: sleep between every API call so we stay
            # under free-tier RPM caps (default 4 s → ~15 RPM max).
            if step_delay > 0 and actual_ply <= max_plies:
                log_info(f"   ⏳ Pacing delay {step_delay:.1f}s (STEP_DELAY_SECONDS)...")
                time.sleep(step_delay)

            # Context window pruning: keep the history manageable.
            buffers[turn] = _prune_buffer(buffers[turn], result.move_history, turn)
            opp = "black" if turn == "white" else "white"
            buffers[opp] = _prune_buffer(buffers[opp], result.move_history, opp)

        if not result.done:
            # Ran out of plies or steps. Ask the env for a final state-based score.
            try:
                final_state_resp = client.get(f"{env_url}/state", timeout=10.0)
                final_state_resp.raise_for_status()
            except Exception:
                pass
            result.plies = actual_ply

        return result
    finally:
        if own_client:
            client.close()


def _args_repr(args: dict[str, Any]) -> str:
    """Compact, single-line args for [STEP] logging."""
    try:
        return json.dumps(args, default=str).replace("\n", " ").replace("\r", "")
    except Exception:
        return str(args)


def _parse_result_field(result_str: str, field: str, default: Any) -> Any:
    """Extract a numeric field from an environment result string.

    E.g. 'Move e2e4 applied. turn=black cp_loss=20 move_score=0.93'
    _parse_result_field(..., 'cp_loss', 0) -> 20
    """
    try:
        m = re.search(rf"{re.escape(field)}=([\d.+-]+)", result_str)
        if m:
            raw = m.group(1)
            return type(default)(float(raw))
    except Exception:
        pass
    return default


def _emit_step(
    ply: int,
    color: str,
    action_str: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    err = f'"{error}"' if error else "null"
    print(
        f"[STEP] step={ply} color={color} action={action_str} "
        f"reward={_fmt(reward)} done={'true' if done else 'false'} error={err}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI entry point — runs a single self-play game and logs compatible JSON.
# ---------------------------------------------------------------------------


def _wait_for_server(env_url: str, client: httpx.Client, attempts: int = 30) -> bool:
    for _ in range(attempts):
        try:
            for path in ("/health", "/docs"):
                r = client.get(f"{env_url}{path}", timeout=2.0)
                if r.status_code == 200:
                    return True
        except httpx.RequestError:
            pass
        time.sleep(1)
    return False


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run a two-model match in Chess Arena.")
    parser.add_argument("--white", type=str, help="Model name for White")
    parser.add_argument("--black", type=str, help="Model name for Black")
    args = parser.parse_args()

    if OpenAI is None:
        log_info("ERROR: openai package not installed. pip install openai.")
        sys.exit(1)

    try:
        from dotenv import dotenv_values
        env_google = dotenv_values(str(Path(__file__).resolve().parent / ".env"))
        env_groq = dotenv_values(str(Path(__file__).resolve().parent / ".env.local"))
    except ImportError:
        env_google = {}
        env_groq = {}

    google_url = env_google.get("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    google_key = env_google.get("HF_TOKEN") or env_google.get("OPENAI_API_KEY") or os.getenv("HF_TOKEN", "")

    groq_url = env_groq.get("API_BASE_URL", "https://api.groq.com/openai/v1")
    groq_key = env_groq.get("HF_TOKEN") or env_groq.get("OPENAI_API_KEY") or os.getenv("HF_TOKEN", "")

    if not google_key and not groq_key:
        log_info("ERROR: no API keys found in .env or .env.local.")
        sys.exit(1)

    # Collect all 8 models with their provider tags
    model_pool: list[tuple[str, str]] = []  # (model_name, provider)
    for i in range(1, 5):
        name = env_google.get(f"GOOGLE_MODEL_{i}")
        if name:
            model_pool.append((name, "google"))
    for i in range(1, 5):
        name = env_groq.get(f"GROQ_MODEL_{i}")
        if name:
            model_pool.append((name, "groq"))

    if len(model_pool) < 2:
        log_info(f"ERROR: need at least 2 models in the pool, found {len(model_pool)}.")
        sys.exit(1)

    def _resolve_model(requested_name: Optional[str], default_pool_idx: int) -> tuple[str, str]:
        if not requested_name:
            import random
            return random.choice(model_pool)
        
        # Look for the exact name in the pool to infer its provider
        match = [m for m in model_pool if m[0] == requested_name]
        if match:
            return match[0]
        # Not in pool, guess provider based on name (Gemma/Gemini -> Google, else Groq)
        requested_lower = requested_name.lower()
        if "gemma" in requested_lower or "gemini" in requested_lower:
            provider = "google"
        else:
            provider = "groq"
        return (requested_name, provider)

    if args.white and args.black:
        white_model, white_provider = _resolve_model(args.white, 0)
        black_model, black_provider = _resolve_model(args.black, 1)
    else:
        # Randomly pick two distinct models and assign random colours
        import random
        pair = random.sample(model_pool, 2)
        random.shuffle(pair)
        white_model, white_provider = pair[0]
        black_model, black_provider = pair[1]
        
        if args.white:
            white_model, white_provider = _resolve_model(args.white, 0)
        if args.black:
            black_model, black_provider = _resolve_model(args.black, 1)

    def _make_client(provider: str) -> "OpenAI":
        if provider == "google":
            return OpenAI(base_url=google_url, api_key=google_key)
        return OpenAI(base_url=groq_url, api_key=groq_key)

    provider_tag = {"google": "Google/Gemini", "groq": "Groq"}

    log_info("═" * 60)
    log_info(" Chess Arena — Random Two-Model Matchup")
    log_info("═" * 60)
    log_info(f" WHITE : {white_model} ({provider_tag[white_provider]})")
    log_info(f" BLACK : {black_model} ({provider_tag[black_provider]})")
    log_info(f" Delay : {STEP_DELAY_SECONDS}s/turn  |  Timeout: {LLM_CALL_TIMEOUT}s/call")
    log_info(f" Games : {NUM_GAMES}  |  Max plies: {MAX_PLIES}")
    log_info(f" Pool  : {len(model_pool)} models available")
    log_info("═" * 60)

    with httpx.Client(timeout=60.0) as http_client:
        log_info(f"\nWaiting for Chess Arena server at {ENV_URL} ...")
        if not _wait_for_server(ENV_URL, http_client):
            log_info("ERROR: env server did not become ready in time.")
            sys.exit(1)
        log_info("Server ready.\n")

        client_white = _make_client(white_provider)
        client_black = _make_client(black_provider)

        policy_white = make_openai_policy(
            client_white,
            model_name=white_model,
            temperature=0.2,
            call_timeout=LLM_CALL_TIMEOUT,
            base_rate_limit_sleep=RATE_LIMIT_SLEEP,
        )
        policy_black = make_openai_policy(
            client_black,
            model_name=black_model,
            temperature=0.2,
            call_timeout=LLM_CALL_TIMEOUT,
            base_rate_limit_sleep=RATE_LIMIT_SLEEP,
        )

        run_logs = {
            "model_white": white_model,
            "model_black": black_model,
            "white_provider": white_provider,
            "black_provider": black_provider,
            "timestamp": datetime.now().isoformat(),
            "step_delay_seconds": STEP_DELAY_SECONDS,
            "tasks": [],
        }

        total_scores: list[float] = []

        for game_idx in range(NUM_GAMES):
            print(
                f"[START] task=chess_game_{game_idx + 1} env=chess_arena "
                f"white={white_model} black={black_model}",
                flush=True,
            )
            result = run_episode(
                policy_white=policy_white,
                policy_black=policy_black,
                env_url=ENV_URL,
                game_idx=game_idx,
                step_delay=STEP_DELAY_SECONDS,
                model_white_name=white_model,
                model_black_name=black_model,
                http_client=http_client,
            )

            task_log = result.to_task_log()
            run_logs["tasks"].append(task_log)
            total_scores.append(task_log["final_reward"])

            rewards_str = ",".join(
                _fmt(r)
                for color in ("white", "black")
                for r in result.rewards_history[color]
            ) or _fmt(_PHASE2_MIN)

            print(
                f"[END] success={'true' if result.done else 'false'} "
                f"plies={result.plies} "
                f"score={_fmt(task_log['final_reward'])} "
                f"score_white={_fmt(result.final_reward['white'])} "
                f"score_black={_fmt(result.final_reward['black'])} "
                f"result={result.result or 'unfinished'} "
                f"rewards={rewards_str}",
                flush=True,
            )

            log_info(f"\n{'─'*60}")
            log_info(f"  🏁 Game {game_idx + 1} finished — {result.result or 'unfinished'}")
            log_info(f"  ⬜ WHITE ({white_model}): final reward = {_fmt(result.final_reward['white'])}")
            log_info(f"  ⬛ BLACK ({black_model}): final reward = {_fmt(result.final_reward['black'])}")
            log_info(f"{'─'*60}")

        os.makedirs("results", exist_ok=True)
        safe_white = white_model.replace("/", "_").replace(":", "_")
        safe_black = black_model.replace("/", "_").replace(":", "_")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"results/{safe_white}_vs_{safe_black}_chess_{stamp}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(run_logs, f, indent=2)
        log_info(f"\n📁 Wrote detailed JSON log: {out_path}")

        if total_scores:
            log_info(
                f"\n{'═'*60}\n"
                f"   Average game score: {sum(total_scores)/len(total_scores):.3f} "
                f"across {len(total_scores)} game(s)\n"
                f"{'═'*60}"
            )


if __name__ == "__main__":
    main()
