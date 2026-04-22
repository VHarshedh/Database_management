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
    from dotenv import load_dotenv  # optional, evaluators often don't ship it
    _env_path = Path(__file__).resolve().parent / ".env"
    if _env_path.is_file():
        load_dotenv(_env_path, override=True)
except Exception:
    pass

try:
    from openai import OpenAI  # openai>=1.0 sync client
except ImportError:  # pragma: no cover - optional for non-LLM trainer paths
    OpenAI = None  # type: ignore


# ---------------------------------------------------------------------------
# Config (env-var driven so evaluators can override without code edits)
# ---------------------------------------------------------------------------


def _strip(s: Optional[str]) -> str:
    return (s or "").strip()


def _first_nonempty_env(*names: str) -> str:
    for name in names:
        v = _strip(os.getenv(name))
        if v:
            return v
    return ""


API_KEY = _first_nonempty_env("HF_TOKEN", "OPENAI_API_KEY", "API_KEY")
API_BASE_URL = (
    _first_nonempty_env("API_BASE_URL", "OPENAI_BASE_URL")
    or "https://generativelanguage.googleapis.com/v1beta/openai/"
)
ENV_URL = _strip(os.getenv("ENV_URL", "http://127.0.0.1:8000")).rstrip("/")

# Primary model name (legacy / single-model runs)
MODEL_NAME = _strip(os.getenv("MODEL_NAME", "gemini-3.1-flash-lite-preview"))

# Two-model matchup (set via env vars or modified here).
# White: Gemma 4 30B; Black: Gemini 2.0 Flash Lite
MODEL_WHITE = _strip(os.getenv("MODEL_WHITE", "gemma-4-31b-it"))
MODEL_BLACK = _strip(os.getenv("MODEL_BLACK", "gemini-3.1-flash-lite-preview"))

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

SYSTEM_PROMPT = """You are an expert chess-playing agent in a multi-agent arena.
You control ONE side of the board. The environment tracks whose turn it is and
will route your outputs to the correct color. Play to win.

Respond ONLY with a single JSON object. NO prose, NO markdown, NO code fences.

═══════════════════════════════════════════════════
 MANDATORY STRUCTURED REASONING SCHEMA
═══════════════════════════════════════════════════
EVERY tool call — without exception — must include these three fields:

  "threat_analysis"  (string)
      Evaluate the CURRENT board state: immediate dangers, checks, captures
      pending, or tactical motifs. Must be specific to the position.

  "candidate_moves"  (array of 2–3 UCI strings)
      The concrete moves you evaluated before committing. Format: ["e2e4",
      "d2d4", "g1f3"]. For make_move, the executed move MUST appear here.

  "justification"    (string)
      Your strategic reason for choosing THIS action. Must reference at least
      one chess concept: center, develop, pin, fork, defend, attack, threat,
      king safety, control, space, initiative, or structure.

JSON schema (use for EVERY call):
{
  "tool": "<tool_name>",
  "arguments": {
    "threat_analysis": "<specific board evaluation>",
    "candidate_moves": ["<uci1>", "<uci2>"],
    "justification":  "<strategic reason referencing chess concepts>",
    ... (tool-specific args like uci_move)
  }
}

EXAMPLE — a make_move call:
{
  "tool": "make_move",
  "arguments": {
    "threat_analysis": "No immediate checks. Opponent's bishop on c4 eyes f7; king is safe.",
    "candidate_moves": ["e2e4", "d2d4", "g1f3"],
    "justification": "Playing e2e4 claims center space and opens lines for bishop and queen development.",
    "uci_move": "e2e4"
  }
}

═══════════════════════════════════════════════════
 REWARD BUCKETS (know what you are scored on)
═══════════════════════════════════════════════════
  Outcome       (max 0.50) — win / draw / loss result
  Format        (max 0.10) — schema compliance; PENALIZED if any of the
                             three required fields is missing or empty:
                             -0.05 per dirty call (W_FORMAT penalty)
  Thought Qual. (max 0.15) — deterministic heuristic scoring:
                             +0.05 Threat Awareness (board context)
                             +0.05 Action Tracing   (uci_move in candidate_moves)
                             +0.05 Strategic Justif.(chess concept keyword)
  SF Accuracy   (max 0.24) — Stockfish centipawn closeness per move

Failing to include all three schema fields = W_FORMAT penalty + rejected logic.

═══════════════════════════════════════════════════
 TOOLS
═══════════════════════════════════════════════════
- `analyze_board(threat_analysis, candidate_moves, justification)`
    Returns current FEN, side to move, and check status.

- `list_legal_moves(threat_analysis, candidate_moves, justification)`
    Returns the full list of legal UCI moves.

- `make_move(threat_analysis, candidate_moves, justification, uci_move)`
    Submits a move. uci_move MUST appear in candidate_moves.
    Illegal move ENDS the game with disqualification.

- `resign_game(threat_analysis, candidate_moves, justification)`
    Resigns. Only use when losing is certain.

- `evaluate_position(threat_analysis, candidate_moves, justification)` *** TRAP ***
    Returns Stockfish eval. Each call docks SF-accuracy by -0.04.
    6th call on your side DISQUALIFIES you.

- `ping_humanhelper(threat_analysis, candidate_moves, justification, reason)` *** TRAP ***
    Always returns "Human helper is currently unavailable."
    Each call docks your Format bucket by -0.03.

STRATEGY:
1. First turn: `analyze_board` to confirm state.
2. Before every move: `list_legal_moves` to know what is playable.
3. Pick the best move with `make_move`. Avoid `evaluate_position` — the
   penalty is non-trivial.
4. Never call `ping_humanhelper`.
5. Always populate all three reasoning fields; empty fields = penalty.

IMPORTANT: You must use the provided tools to interact with the chess board. Do not explain your reasoning in plain text before calling a tool. Only output the tool call.
Strictly follow the JSON schema. Your tool call arguments must not be nested. Do not include a 'tool' or 'arguments' key inside the arguments object itself.
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


def _extract_first_tool_call(raw_text: str) -> Optional[dict[str, Any]]:
    """Parse the model's raw text output into {tool, arguments}.

    The SYSTEM_PROMPT demands a single JSON object, but models occasionally
    leak markdown fences or prose. We try two fast paths:
      1. Direct json.loads on the stripped text.
      2. Regex-extract the first {...} block.

    Structured reasoning fields (threat_analysis, candidate_moves, justification)
    are promoted from the top level into `arguments` if the model placed them
    at the root rather than nested under `arguments`.
    """
    if not isinstance(raw_text, str):
        return None
    text = raw_text.strip()
    # Strip ```json fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text).rstrip("`").strip()
    for candidate in (text, _first_json_object(text) or ""):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        tool = obj.get("tool")
        args = obj.get("arguments") or {}
        if isinstance(tool, str) and isinstance(args, dict):
            # Promote structured reasoning fields if the model placed them at
            # the top level instead of inside `arguments`.
            for field in ("threat_analysis", "candidate_moves", "justification"):
                if field not in args and field in obj:
                    args[field] = obj[field]
            # Legacy: also promote `thought` if present (backwards compat).
            if "thought" not in args and "thought" in obj:
                args["thought"] = obj["thought"]
            return {"tool": tool, "arguments": args, "raw": raw_text}
    return None


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

    # Merge consecutive messages of the same role to satisfy strict alternating role requirements
    merged = []
    for m in system_msg + pruned_history:
        if merged and merged[-1].get("role") == m.get("role") and m.get("role") in ("user", "assistant"):
            merged[-1] = {
                "role": m.get("role"),
                "content": merged[-1]["content"] + "\n\n" + m["content"]
            }
        else:
            merged.append(m.copy())

    return merged


def make_openai_policy(
    client: "OpenAI",
    model_name: str = MODEL_NAME,
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
        
        # Google/Gemini models prefer 'auto' for function calling, while 
        # open-source models (Llama, etc) often need 'required' to enforce it.
        if "gpt-oss" in model_name.lower():
            # Apply Harmony Patch for GPT-OSS
            dynamic_tool_choice = {"type": "function", "function": {"name": "analyze_board"}} if len(messages) <= 2 else "auto"
            dynamic_temp = 1.0
        else:
            dynamic_tool_choice = "auto" if any(x in model_name.lower() for x in ("gemini", "google")) else "required"
            dynamic_temp = temperature

        for attempt in range(3):
            try:
                pruned_msgs = prune_messages(messages, max_tail_messages=80)
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=pruned_msgs,
                    tools=OPENAI_TOOLS,
                    tool_choice=dynamic_tool_choice,
                    temperature=dynamic_temp,
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
                            m_err = re.search(r"'failed_generation':\s*(['\"])(.*?)\1(?:[,\}])", last_err, re.DOTALL)
                            if m_err:
                                failed_gen = m_err.group(2).replace("\\n", "\n").replace('\\"', '"')

                        if failed_gen:
                            m_xml = re.search(r"<function=([a-zA-Z0-9_]+)(.*?)(?:</function>|>)", failed_gen, re.DOTALL)
                            if m_xml:
                                tool_name = m_xml.group(1)
                                json_str = m_xml.group(2).strip()
                                if not json_str.startswith("{"):
                                    json_str = "{" + json_str
                                
                                raw_dict = json.loads(json_str)
                                args = raw_dict.get("arguments", raw_dict) if isinstance(raw_dict.get("arguments"), dict) else raw_dict
                                
                                # Promote flat fields if they were outside 'arguments'
                                for field in ("threat_analysis", "candidate_moves", "justification"):
                                    if field not in args and field in raw_dict:
                                        args[field] = raw_dict[field]
                                
                                return {"tool": tool_name, "arguments": args, "raw": failed_gen}
                            
                            parsed = _extract_first_tool_call(failed_gen)
                            if parsed:
                                return parsed
                    except Exception:
                        pass

                # Detect timeout errors (openai.APITimeoutError or httpx timeout)
                is_timeout = (
                    "timeout" in err_lower
                    or "timed out" in err_lower
                    or type(e).__name__ in ("APITimeoutError", "ReadTimeout", "ConnectTimeout")
                )
                # Detect rate-limit / server overload errors
                is_rate_limit = (
                    "429" in last_err
                    or "quota" in err_lower
                    or "rate" in err_lower
                    or "503" in last_err
                    or type(e).__name__ == "RateLimitError"
                )

                if is_timeout:
                    log_info(
                        f"   ⏱ [{model_name}] API timeout on attempt {attempt + 1} "
                        f"(limit={call_timeout}s). Retrying immediately."
                    )
                    # No extra sleep on pure timeout — server may just be slow.
                    continue

                if is_rate_limit:
                    wait = min(sleep_s, 60.0)
                    log_info(
                        f"   🚦 [{model_name}] Rate limit / 503 on attempt {attempt + 1}. "
                        f"Backing off {wait:.0f}s before retry..."
                    )
                    time.sleep(wait)
                    sleep_s = min(sleep_s * 2, 60.0)  # exponential back-off
                    continue

                # Any other error: log and abort retries.
                log_info(f"   ❌ [{model_name}] API error on attempt {attempt + 1}: {last_err}")
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

    Keeps the system prompt, then a compact summary of all moves played,
    then the last `window_size` messages of full dialogue history.
    """
    if len(buffer) <= window_size + 2:
        return buffer

    # 1. Start with the mandatory system prompt.
    new_buffer = [buffer[0]]

    # 2. Add a compact summary of the entire game history so far.
    # We extract this from the provided move_history.
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

    # 3. Add the sliding window of most recent messages.
    new_buffer.extend(buffer[-window_size:])

    return new_buffer


def run_episode(
    policy_white: Policy,
    policy_black: Policy,
    *,
    env_url: str = ENV_URL,
    game_idx: int = 0,
    max_plies: int = MAX_PLIES,
    step_delay: float = STEP_DELAY_SECONDS,
    model_white_name: str = MODEL_WHITE,
    model_black_name: str = MODEL_BLACK,
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

        for ply in range(1, max_plies + 1):
            current_model = model_names[turn]
            log_info(f"\n   --- Ply {ply} ({turn} - {current_model}) ---")
            
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

            # Malformed output → POST an intentional no-op action so the env
            # can record the dirty call. We use `ping_humanhelper` as a
            # placeholder only if the model emitted nothing usable; otherwise
            # we route the raw tool choice through.
            if not tool:
                # Invalid JSON / no tool: charge tool_acc, feed the error back,
                # and continue to the next ply (same color keeps trying).
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
                # Report the malformed call to the env via /state -> the env
                # exposes `record_malformed_call` indirectly via an extra
                # metadata action is not wired, so we simulate the penalty by
                # logging it in the step row; the real penalty still shows up
                # when the env finalises (we overcount by 1 clean call if the
                # server-side env didn't see it). Keep behaviour simple and
                # predictable: print the [STEP] row and retry.
                result.steps.append(
                    {
                        "ply": ply,
                        "color": turn,
                        "tool_name": "(malformed)",
                        "arguments": {},
                        "action": "invalid_json",
                        "reward": _PHASE2_MIN,
                        "done": False,
                    }
                )
                result.rewards_history[turn].append(_PHASE2_MIN)
                _emit_step(ply, turn, "invalid_json()", _PHASE2_MIN, False, "invalid_json")
                continue

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
            # Metadata is stripped over HTTP by `serialize_observation`, so our
            # env also mirrors the debug payload into
            # ``observation.result.structured_content.openenv``. Read from
            # whichever one is populated.
            metadata = obs.get("metadata") or {}
            res_obj = obs.get("result")
            openenv_payload: dict[str, Any] = {}
            if isinstance(res_obj, dict):
                sc = res_obj.get("structured_content")
                if isinstance(sc, dict):
                    maybe = sc.get("openenv")
                    if isinstance(maybe, dict):
                        openenv_payload = maybe
            # Merge: explicit metadata wins; fall back to the HTTP-safe payload.
            for k, v in openenv_payload.items():
                metadata.setdefault(k, v)

            done = bool(step_body.get("done") or obs.get("done") or metadata.get("done"))
            reward = _clamp_phase2(step_body.get("reward") or obs.get("reward"))
            fen_after = metadata.get("fen", "")

            # Build a tool-output string to feed back to the model. Prefer
            # the concise ``data`` / ``structured_content.result`` fields that
            # FastMCP emits over dumping the whole result dict (which would
            # also expose our hidden ``openenv`` debug payload to the policy).
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

            # Keep the opponent roughly aware that a move was played. For
            # `make_move`, append the public UCI so the opponent can plan.
            if tool == "make_move":
                fen = metadata.get("fen") or fen_after
                # Record the move in the episode's move history for JSON logs.
                result.move_history.append({
                    "color": turn,
                    "uci": args.get("uci_move", ""),
                    "cp_loss": _parse_result_field(tool_out, "cp_loss", 0),
                    "move_score": _parse_result_field(tool_out, "move_score", 0.0),
                })

                if fen:
                    try:
                        import chess
                        board_str = str(chess.Board(fen))
                        indented_board = "\n".join("      " + line for line in board_str.splitlines())
                        log_info(f"   --- Board after ply {ply} ({args.get('uci_move', '?')}) ---\n{indented_board}")
                    except Exception:
                        pass

                if not done:
                    opp = "black" if turn == "white" else "white"
                    opp_turn_msg = (
                        f"Opponent ({turn}) played {args.get('uci_move', '?')}. "
                        f"Board FEN: {fen or '(unknown)'}. Your move."
                    )
                    buffers[opp].append({"role": "user", "content": opp_turn_msg})

            result.steps.append(
                {
                    "ply": ply,
                    "color": turn,
                    "tool_name": tool,
                    "arguments": args,
                    "result": tool_out,
                    "reward": reward,
                    "done": done,
                }
            )
            result.rewards_history[turn].append(reward)
            _emit_step(ply, turn, f"{tool}({_args_repr(args)})", reward, done, None)

            if done:
                # Pull final per-color rewards + bucket from metadata.
                final_map = metadata.get("final_reward") or {}
                result.final_reward = {
                    "white": _clamp_phase2(final_map.get("white", reward)),
                    "black": _clamp_phase2(final_map.get("black", reward)),
                }
                result.bucket = metadata.get("bucket") or {}
                result.result = metadata.get("result")
                result.done = True
                result.plies = ply
                break

            # Flip side only on a successful `make_move`. Tools like
            # `analyze_board` / `list_legal_moves` / `evaluate_position` /
            # `ping_humanhelper` leave the turn with the same color so the
            # agent can immediately follow up with another action.
            # Also do NOT flip on a Strike-1 illegal move warning — the same
            # player must retry before the opponent gets to move.
            is_strike_one = (
                tool == "make_move"
                and isinstance(tool_out, str)
                and "ILLEGAL MOVE (strike 1/2)" in tool_out
            )
            if tool == "make_move" and not is_strike_one and not is_error:
                turn = "black" if turn == "white" else "white"

            # Rate-limit pacing: sleep between every API call so we stay
            # under free-tier RPM caps (default 4 s → ~15 RPM max).
            if step_delay > 0 and ply < max_plies:
                log_info(f"   ⏳ Pacing delay {step_delay:.1f}s (STEP_DELAY_SECONDS)...")
                time.sleep(step_delay)

            # Context window pruning: keep the history manageable.
            # We pass result.move_history to build the compact summary.
            buffers[turn] = _prune_buffer(buffers[turn], result.move_history, turn)
            opp = "black" if turn == "white" else "white"
            buffers[opp] = _prune_buffer(buffers[opp], result.move_history, opp)

        if not result.done:
            # Ran out of plies. Ask the env for a final state-based score.
            try:
                final_state_resp = client.get(f"{env_url}/state", timeout=10.0)
                final_state_resp.raise_for_status()
            except Exception:
                pass
            result.plies = max_plies

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
    if OpenAI is None:
        log_info("ERROR: openai package not installed. pip install openai.")
        sys.exit(1)
    if not API_KEY:
        log_info(
            "ERROR: set HF_TOKEN / OPENAI_API_KEY / API_KEY before running inference."
        )
        sys.exit(1)

    try:
        from dotenv import dotenv_values
        env_groq = dotenv_values(".env.local")
        env_google = dotenv_values(".env")
    except ImportError:
        env_groq = {}
        env_google = {}

    groq_url = env_groq.get("API_BASE_URL", "https://api.groq.com/openai/v1")
    groq_key = env_groq.get("HF_TOKEN") or env_groq.get("OPENAI_API_KEY") or API_KEY
    
    google_url = env_google.get("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    google_key = env_google.get("HF_TOKEN") or env_google.get("OPENAI_API_KEY") or API_KEY

    # White uses Google, Black uses Groq
    white_model = env_google.get("MODEL_WHITE", "gemini-3.1-pro-preview")
    black_model = env_groq.get("MODEL_BLACK", "openai/gpt-oss-120b")

    log_info("═" * 60)
    log_info(" Chess Arena — Two-Model Matchup")
    log_info("═" * 60)
    log_info(f" WHITE : {white_model} (Google)")
    log_info(f" BLACK : {black_model} (Groq)")
    log_info(f" Delay : {STEP_DELAY_SECONDS}s/turn  |  Timeout: {LLM_CALL_TIMEOUT}s/call")
    log_info(f" Games : {NUM_GAMES}  |  Max plies: {MAX_PLIES}")
    log_info("═" * 60)

    with httpx.Client(timeout=60.0) as http_client:
        log_info(f"\nWaiting for Chess Arena server at {ENV_URL} ...")
        if not _wait_for_server(ENV_URL, http_client):
            log_info("ERROR: env server did not become ready in time.")
            sys.exit(1)
        log_info("Server ready.\n")

        # Create two separate OpenAI clients for White (Google) and Black (Groq)
        client_white = OpenAI(base_url=google_url, api_key=google_key)
        client_black = OpenAI(base_url=groq_url, api_key=groq_key)

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

            # Show final rewards on stderr for quick visibility.
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

