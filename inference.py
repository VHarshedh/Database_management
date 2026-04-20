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
    or "https://router.huggingface.co/v1"
)
ENV_URL = _strip(os.getenv("ENV_URL", "http://127.0.0.1:8000")).rstrip("/")
MODEL_NAME = _strip(os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"))

NUM_GAMES = int(os.getenv("NUM_GAMES", "1"))
MAX_PLIES = int(os.getenv("MAX_PLIES", "60"))
INFERENCE_MAX_SECONDS = int(os.getenv("INFERENCE_MAX_SECONDS", "1200"))
RATE_LIMIT_SLEEP = int(os.getenv("RATE_LIMIT_SLEEP_SECONDS", "20"))

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

JSON schema for every turn:
{"thought": "<one sentence of your reasoning>", "tool": "<tool_name>", "arguments": {...}}

TOOLS (each tool requires a non-empty `thought`):

- `analyze_board(thought)`
    Returns current FEN, side to move, and check status.

- `list_legal_moves(thought)`
    Returns the full list of legal UCI moves.

- `make_move(thought, uci_move)`
    Submits a move in UCI notation, e.g. "e2e4" or "e7e8q" for promotion.
    You MUST pick a move that appears in `list_legal_moves`. Any illegal move
    ENDS the game with a disqualification.

- `resign_game(thought)`
    Resigns. Only use this when losing is certain; the opponent gets partial
    credit.

- `evaluate_position(thought)`  *** TRAP ***
    Returns a Stockfish evaluation. Each call docks your Stockfish-accuracy
    reward by -0.04. A 6th call on your side of the game DISQUALIFIES you.

- `ping_humanhelper(thought, reason)`  *** TRAP (non-fatal) ***
    Asks a human for help. The response is always "Human helper is currently
    unavailable." Each call docks your tool-accuracy reward by -0.03.

STRATEGY:
1. On your first turn: `analyze_board` to confirm state.
2. Before every move: `list_legal_moves` to know what's playable.
3. Pick the best move with `make_move`. Do not call `evaluate_position` unless
   you have a specific tactical reason — the penalty is non-trivial.
4. Never call `ping_humanhelper` — it is strictly worse than any real tool.
5. Keep `thought` short (one sentence, under 200 chars).

REMEMBER: malformed JSON, missing `thought`, or illegal UCI all penalise you.
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
                "properties": {"thought": {"type": "string"}},
                "required": ["thought"],
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
                "properties": {"thought": {"type": "string"}},
                "required": ["thought"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_move",
            "description": "Submit a UCI move.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "uci_move": {"type": "string"},
                },
                "required": ["thought", "uci_move"],
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
                "properties": {"thought": {"type": "string"}},
                "required": ["thought"],
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
                "properties": {"thought": {"type": "string"}},
                "required": ["thought"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ping_humanhelper",
            "description": "TRAP: non-fatal help request; each call docks tool_acc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["thought", "reason"],
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
            # `thought` is required by the env tool signatures; if the model
            # put it at the top level, promote it into arguments.
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


def make_openai_policy(
    client: "OpenAI",
    model_name: str = MODEL_NAME,
    temperature: float = 0.2,
) -> Policy:
    """Build a Policy backed by an OpenAI-compatible chat.completions endpoint."""

    def _policy(messages: list[dict[str, Any]]) -> dict[str, Any]:
        last_err: Optional[str] = None
        for attempt in range(3):
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=OPENAI_TOOLS,
                    tool_choice="auto",
                    temperature=temperature,
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
                if any(x in last_err for x in ("429", "quota", "503")):
                    time.sleep(max(5, min(60, RATE_LIMIT_SLEEP)))
                else:
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


def run_episode(
    policy_white: Policy,
    policy_black: Policy,
    *,
    env_url: str = ENV_URL,
    game_idx: int = 0,
    max_plies: int = MAX_PLIES,
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

        result = EpisodeResult(game_idx=game_idx)
        turn = "white"

        for ply in range(1, max_plies + 1):
            policy = policies[turn]
            out = policy(buffers[turn])
            tool = out.get("tool")
            args = dict(out.get("arguments") or {})
            raw = out.get("raw", "")

            # Ensure every tool call has a non-empty `thought` key; the env
            # treats missing / empty thought as a dirty call.
            args.setdefault("thought", "")

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

            # Build a tool-output string to feed back to the model. Prefer
            # the concise ``data`` / ``structured_content.result`` fields that
            # FastMCP emits over dumping the whole result dict (which would
            # also expose our hidden ``openenv`` debug payload to the policy).
            if isinstance(res_obj, dict):
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

            buffers[turn].append({"role": "assistant", "content": raw or json.dumps(out)})
            buffers[turn].append(
                {
                    "role": "user",
                    "content": f"[{tool} result]\n{tool_out}",
                }
            )

            # Keep the opponent roughly aware that a move was played. For
            # `make_move`, append the public UCI so the opponent can plan.
            if tool == "make_move" and not done:
                opp = "black" if turn == "white" else "white"
                opp_turn_msg = (
                    f"Opponent ({turn}) played {args.get('uci_move', '?')}. "
                    f"Board FEN: {metadata.get('fen', '(unknown)')}. Your move."
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
            if tool == "make_move":
                turn = "black" if turn == "white" else "white"

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

    with httpx.Client(timeout=60.0) as http_client:
        log_info(f"Waiting for Chess Arena server at {ENV_URL} ...")
        if not _wait_for_server(ENV_URL, http_client):
            log_info("ERROR: env server did not become ready in time.")
            sys.exit(1)
        log_info(f"Server ready. Model: {MODEL_NAME} @ {API_BASE_URL}")

        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        policy = make_openai_policy(client, MODEL_NAME)

        run_logs = {
            "model": MODEL_NAME,
            "timestamp": datetime.now().isoformat(),
            "tasks": [],
        }

        total_scores: list[float] = []

        for game_idx in range(NUM_GAMES):
            print(
                f"[START] task=chess_game_{game_idx + 1} env=chess_arena model={MODEL_NAME}",
                flush=True,
            )
            result = run_episode(
                policy_white=policy,
                policy_black=policy,
                env_url=ENV_URL,
                game_idx=game_idx,
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

        os.makedirs("results", exist_ok=True)
        safe_model = MODEL_NAME.replace("/", "_").replace(":", "_")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"results/{safe_model}_chess_{stamp}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(run_logs, f, indent=2)
        log_info(f"Wrote detailed JSON log: {out_path}")

        if total_scores:
            log_info(
                f"Average game score: {sum(total_scores)/len(total_scores):.3f} "
                f"across {len(total_scores)} game(s)"
            )


if __name__ == "__main__":
    main()
