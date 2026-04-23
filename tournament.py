#!/usr/bin/env python3
"""
tournament.py — 8-Model Knockout Chess Tournament Orchestrator
==============================================================

Runs an 8-model, 3-round knockout tournament using the Chess Arena's
`run_episode` and `make_openai_policy` from `inference.py`.

Bracket structure:
  Round 1 (Quarterfinals): 8 models → 4 winners
  Round 2 (Semifinals):    4 winners → 2 winners + 2 losers
  Round 3 (Finals):        Championship (winner vs winner)
                           Bronze Match (loser vs loser)

Environment:
  .env       → 4 Google/Gemini models (GOOGLE_MODEL_1..4) + Google API key/URL
  .env.local → 4 Groq models          (GROQ_MODEL_1..4)   + Groq API key/URL

Usage:
    python tournament.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

# ---------------------------------------------------------------------------
# Optional dotenv — silently skipped if not installed
# ---------------------------------------------------------------------------
try:
    from dotenv import dotenv_values
    _has_dotenv = True
except ImportError:
    _has_dotenv = False
    print("[WARN] python-dotenv not installed; reading env vars from OS only.")


# ---------------------------------------------------------------------------
# Import from inference.py
# ---------------------------------------------------------------------------
try:
    from inference import run_episode, make_openai_policy, EpisodeResult
except ImportError as exc:
    print(f"[FATAL] Cannot import from inference.py: {exc}")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("[FATAL] openai package not installed. pip install openai")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _load_env(path: str) -> dict[str, str]:
    """Load a .env file into a plain dict, falling back to empty dict."""
    if _has_dotenv and Path(path).is_file():
        return {k: v for k, v in dotenv_values(path).items() if v is not None}
    return {}


def _get(d: dict[str, str], key: str, default: str = "") -> str:
    return (d.get(key) or os.environ.get(key) or default).strip()


# ---------------------------------------------------------------------------
# Load provider credentials
# ---------------------------------------------------------------------------

_env_google = _load_env(".env")
_env_groq   = _load_env(".env.local")

GOOGLE_API_KEY  = _get(_env_google, "HF_TOKEN") or _get(_env_google, "OPENAI_API_KEY")
GOOGLE_BASE_URL = _get(_env_google, "API_BASE_URL",
                        "https://generativelanguage.googleapis.com/v1beta/openai/")

GROQ_API_KEY    = _get(_env_groq, "HF_TOKEN") or _get(_env_groq, "OPENAI_API_KEY")
GROQ_BASE_URL   = _get(_env_groq, "API_BASE_URL", "https://api.groq.com/openai/v1")

ENV_URL         = _get(_env_google, "ENV_URL", "http://127.0.0.1:8000")
STEP_DELAY      = float(_get(_env_google, "STEP_DELAY_SECONDS", "2.0"))
CALL_TIMEOUT    = float(_get(_env_google, "LLM_CALL_TIMEOUT", "60.0"))
RATE_LIMIT_SLEEP = float(_get(_env_google, "RATE_LIMIT_SLEEP_SECONDS", "15.0"))

# ---------------------------------------------------------------------------
# Read the 8 model names
# ---------------------------------------------------------------------------

GOOGLE_MODELS: list[str] = [
    _get(_env_google, f"GOOGLE_MODEL_{i}") for i in range(1, 5)
]
GROQ_MODELS: list[str] = [
    _get(_env_groq, f"GROQ_MODEL_{i}") for i in range(1, 5)
]

# Validate — surface empty strings as a helpful error
_missing = [f"GOOGLE_MODEL_{i}" for i, m in enumerate(GOOGLE_MODELS, 1) if not m] + \
           [f"GROQ_MODEL_{i}"   for i, m in enumerate(GROQ_MODELS,   1) if not m]
if _missing:
    print(
        f"[FATAL] Missing model definitions in .env / .env.local:\n  "
        + "\n  ".join(_missing)
    )
    sys.exit(1)

ALL_MODELS: list[str] = GOOGLE_MODELS + GROQ_MODELS  # 8 models total

# ---------------------------------------------------------------------------
# Provider routing
# ---------------------------------------------------------------------------

def _is_google_model(model_name: str) -> bool:
    """Heuristic: Google models come from GOOGLE_MODELS list."""
    return model_name in GOOGLE_MODELS


def _build_client(model_name: str) -> "OpenAI":
    """Return the correct OpenAI-compat client for the given model."""
    if _is_google_model(model_name):
        return OpenAI(base_url=GOOGLE_BASE_URL, api_key=GOOGLE_API_KEY)
    return OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)


def _build_policy(model_name: str) -> Callable:
    """Build a make_openai_policy for the given model, using the right client."""
    client = _build_client(model_name)
    return make_openai_policy(
        client,
        model_name=model_name,
        temperature=0.2,
        call_timeout=CALL_TIMEOUT,
        base_rate_limit_sleep=RATE_LIMIT_SLEEP,
    )


# ---------------------------------------------------------------------------
# Match dataclass
# ---------------------------------------------------------------------------

@dataclass
class MatchRecord:
    """Tracks the outcome of a single match."""
    round_num: int
    match_label: str          # e.g. "QF-1", "SF-2", "FINAL"
    model_a: str
    model_b: str
    white: str                # who played White
    black: str                # who played Black
    result: Optional[str] = None        # e.g. "checkmate_white", "draw_stalemate", "error"
    winner: Optional[str] = None
    loser: Optional[str] = None
    decided_by: str = "game"  # "game" | "rematch" | "coin_flip" | "walkover"
    task_log: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Directory & file helpers
# ---------------------------------------------------------------------------

def _safe_name(model: str) -> str:
    """Make a model name filesystem-safe."""
    return model.replace("/", "_").replace(":", "_").replace(" ", "_")


def _save_match(record: MatchRecord) -> None:
    """Persist a match's task_log as a JSON file under results/round{n}/."""
    out_dir = Path(f"results/round{record.round_num}")
    out_dir.mkdir(parents=True, exist_ok=True)
    label = record.match_label.replace(" ", "_")
    fname = (
        out_dir /
        f"{label}_{_safe_name(record.model_a)}_vs_{_safe_name(record.model_b)}.json"
    )
    payload = {
        "match_label": record.match_label,
        "round": record.round_num,
        "white": record.white,
        "black": record.black,
        "result": record.result,
        "winner": record.winner,
        "decided_by": record.decided_by,
        "task_log": record.task_log,
    }
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  📁 Saved: {fname}")


# ---------------------------------------------------------------------------
# Core match runner
# ---------------------------------------------------------------------------

def _run_one_game(
    http_client: httpx.Client,
    model_white: str,
    model_black: str,
    game_idx: int = 0,
) -> tuple[Optional[str], Optional[EpisodeResult]]:
    """
    Run a single game between two models.

    Returns (result_str, episode_result) or ("error", None) on exception.
    """
    try:
        policy_white = _build_policy(model_white)
        policy_black = _build_policy(model_black)
        ep = run_episode(
            policy_white=policy_white,
            policy_black=policy_black,
            env_url=ENV_URL,
            game_idx=game_idx,
            step_delay=STEP_DELAY,
            model_white_name=model_white,
            model_black_name=model_black,
            http_client=http_client,
        )
        return ep.result, ep
    except Exception as exc:
        print(f"  ⚠️  Game crashed: {exc}")
        return "error", None
    finally:
        # Clean up any leaked env instances + Stockfish processes from crashes.
        # Import here to avoid circular dependency at module level.
        try:
            from server.chess_environment import ChessEnvironment
            stale_ids = [
                eid for eid, env in ChessEnvironment._instances.items()
                if env.done or env._state.step_count == 0
            ]
            for eid in stale_ids:
                env = ChessEnvironment._instances.pop(eid, None)
                if env and hasattr(env, "_stockfish"):
                    env._stockfish.close()
        except Exception:
            pass


def _determine_winner_from_result(
    result: str,
    model_white: str,
    model_black: str,
) -> Optional[str]:
    """Map an OpenEnv result string to the winning model name.

    OpenEnv emits strings like:
      "checkmate_white"    → White won by checkmate
      "checkmate_black"    → Black won by checkmate
      "resign_white"       → White resigned → Black wins
      "resign_black"       → Black resigned → White wins
      "dq_illegal_white"   → White DQ'd → Black wins
      "dq_eval_abuse_black"→ Black DQ'd → White wins
      "draw_*"             → draw (stalemate / repetition / 50-move)

    Returns None for draws, errors, and unfinished games so the
    tie-breaker chain (rematch → coin flip) triggers correctly.
    """
    if not result or result == "error":
        return None

    r = result.lower()

    # Explicit draws — return None to trigger rematch / coin-flip
    if r.startswith("draw_") or r in ("1/2-1/2", "draw"):
        return None

    # White wins when the result says white is the WINNER side:
    #   checkmate_white  → White delivered checkmate
    #   resign_black     → Black resigned (White wins)
    #   dq_*_black       → Black disqualified (White wins)
    if "checkmate_white" in r or "resign_black" in r or (
        ("dq_illegal_black" in r or "dq_eval_abuse_black" in r)
    ):
        return model_white

    # Black wins symmetrically
    if "checkmate_black" in r or "resign_white" in r or (
        ("dq_illegal_white" in r or "dq_eval_abuse_white" in r)
    ):
        return model_black

    # Fallback: check which colour appears in the result string.
    # This handles any future result tags the env might add.
    if "white" in r and "black" not in r:
        return model_white
    if "black" in r and "white" not in r:
        return model_black

    return None  # unrecognised / draw-like



def play_match(
    http_client: httpx.Client,
    model_a: str,
    model_b: str,
    round_num: int,
    match_label: str,
    game_idx: int = 0,
) -> MatchRecord:
    """
    Play a match between model_a and model_b with tie-breaking logic.

    1. Randomly assign colours and play.
    2. If drawn/unfinished → swap colours and replay (rematch).
    3. If still drawn → coin flip.
    4. If either side errors → opponent gets a walkover.
    """
    # Randomly assign colours for game 1
    if random.random() < 0.5:
        white, black = model_a, model_b
    else:
        white, black = model_b, model_a

    record = MatchRecord(
        round_num=round_num,
        match_label=match_label,
        model_a=model_a,
        model_b=model_b,
        white=white,
        black=black,
    )

    print(f"\n  ♟  {match_label}: {white} (W) vs {black} (B)")

    # --- Game 1 ---
    result, ep = _run_one_game(http_client, white, black, game_idx)
    record.result = result

    if ep is not None:
        record.task_log = ep.to_task_log()

    if result == "error":
        # Both sides failed — coin flip
        record.winner = random.choice([model_a, model_b])
        record.loser  = model_b if record.winner == model_a else model_a
        record.decided_by = "coin_flip"
        print(f"  💀 Both sides errored. Coin flip → {record.winner} advances.")
        _save_match(record)
        return record

    winner = _determine_winner_from_result(result, white, black)

    if winner is not None:
        record.winner = winner
        record.loser  = black if winner == white else white
        record.decided_by = "game"
        print(f"  ✅ {winner} wins ({result}).")
        _save_match(record)
        return record

    # --- Draw → rematch with colours swapped ---
    print(f"  🤝 Draw ({result}). Playing rematch with colours swapped…")
    white2, black2 = black, white  # swap
    record.white = white2
    record.black = black2
    result2, ep2 = _run_one_game(http_client, white2, black2, game_idx + 1)

    if ep2 is not None:
        record.task_log = ep2.to_task_log()  # overwrite with rematch log

    record.result = result2

    if result2 == "error":
        record.winner = random.choice([model_a, model_b])
        record.loser  = model_b if record.winner == model_a else model_a
        record.decided_by = "coin_flip"
        print(f"  💀 Rematch errored. Coin flip → {record.winner} advances.")
        _save_match(record)
        return record

    winner2 = _determine_winner_from_result(result2, white2, black2)

    if winner2 is not None:
        record.winner = winner2
        record.loser  = black2 if winner2 == white2 else white2
        record.decided_by = "rematch"
        print(f"  ✅ {winner2} wins rematch ({result2}).")
        _save_match(record)
        return record

    # --- Both draws → coin flip ---
    record.winner = random.choice([model_a, model_b])
    record.loser  = model_b if record.winner == model_a else model_a
    record.decided_by = "coin_flip"
    print(f"  🪙 Double draw! Coin flip → {record.winner} advances.")
    _save_match(record)
    return record


# ---------------------------------------------------------------------------
# Server readiness check
# ---------------------------------------------------------------------------

def _wait_for_server(url: str, http_client: httpx.Client, retries: int = 30) -> bool:
    # Bug 4 fix: probe /health (or /docs as fallback) instead of /state.
    # On a fresh FastAPI boot, /state returns 400/404 until reset() is called,
    # which would cause the tournament to hang here indefinitely at startup.
    for _ in range(retries):
        for probe in ("/health", "/docs"):
            try:
                r = http_client.get(f"{url}{probe}", timeout=5.0)
                if r.status_code < 500:
                    return True
            except httpx.RequestError:
                pass
        time.sleep(1)
    return False


# ---------------------------------------------------------------------------
# Tournament runner
# ---------------------------------------------------------------------------

def run_tournament() -> None:
    print("═" * 62)
    print("  🏆  Chess Arena — 8-Model Knockout Tournament")
    print("═" * 62)
    print(f"  Google models : {GOOGLE_MODELS}")
    print(f"  Groq models   : {GROQ_MODELS}")
    print(f"  ENV URL       : {ENV_URL}")
    print(f"  Step delay    : {STEP_DELAY}s")
    print("═" * 62)

    # Shuffle all 8 models for a randomised bracket
    bracket = list(ALL_MODELS)
    random.shuffle(bracket)

    with httpx.Client(timeout=CALL_TIMEOUT + 10.0) as http_client:
        print(f"\nWaiting for Chess Arena server at {ENV_URL} …")
        if not _wait_for_server(ENV_URL, http_client):
            print("[FATAL] Server did not become ready. Is uvicorn running?")
            sys.exit(1)
        print("Server ready.\n")

        # ──────────────────────────────────────────────────
        # ROUND 1 — QUARTERFINALS (4 matches)
        # ──────────────────────────────────────────────────
        print("\n" + "─" * 62)
        print("  ROUND 1 — QUARTERFINALS")
        print("─" * 62)

        qf_records: list[MatchRecord] = []
        for i in range(4):
            model_a = bracket[i * 2]
            model_b = bracket[i * 2 + 1]
            rec = play_match(
                http_client, model_a, model_b,
                round_num=1, match_label=f"QF-{i+1}", game_idx=i * 2,
            )
            qf_records.append(rec)

        qf_winners = [r.winner for r in qf_records]
        qf_losers  = [r.loser  for r in qf_records]

        # ──────────────────────────────────────────────────
        # ROUND 2 — SEMIFINALS (2 matches)
        # ──────────────────────────────────────────────────
        print("\n" + "─" * 62)
        print("  ROUND 2 — SEMIFINALS")
        print("─" * 62)

        # Shuffle winners so semifinal pairings are random too
        sf_bracket = list(qf_winners)
        random.shuffle(sf_bracket)

        sf_records: list[MatchRecord] = []
        for i in range(2):
            model_a = sf_bracket[i * 2]
            model_b = sf_bracket[i * 2 + 1]
            rec = play_match(
                http_client, model_a, model_b,
                round_num=2, match_label=f"SF-{i+1}", game_idx=8 + i * 2,
            )
            sf_records.append(rec)

        sf_winners = [r.winner for r in sf_records]
        sf_losers  = [r.loser  for r in sf_records]

        # ──────────────────────────────────────────────────
        # ROUND 3 — FINALS
        # ──────────────────────────────────────────────────
        print("\n" + "─" * 62)
        print("  ROUND 3 — FINALS")
        print("─" * 62)

        # Championship
        print("\n  🥇  Championship Match")
        final_rec = play_match(
            http_client, sf_winners[0], sf_winners[1],
            round_num=3, match_label="FINAL", game_idx=12,
        )

        # Bronze match
        print("\n  🥉  Bronze Match")
        bronze_rec = play_match(
            http_client, sf_losers[0], sf_losers[1],
            round_num=3, match_label="BRONZE", game_idx=14,
        )

    # ──────────────────────────────────────────────────────
    # FINAL STANDINGS
    # ──────────────────────────────────────────────────────
    place_1st = final_rec.winner
    place_2nd = final_rec.loser
    place_3rd = bronze_rec.winner

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n")
    print("╔" + "═" * 60 + "╗")
    print("║" + "  🏆  TOURNAMENT RESULTS".center(60) + "║")
    print("║" + f"  {timestamp}".center(60) + "║")
    print("╠" + "═" * 60 + "╣")
    print("║" + f"  🥇  1st Place : {place_1st}".ljust(60) + "║")
    print("║" + f"  🥈  2nd Place : {place_2nd}".ljust(60) + "║")
    print("║" + f"  🥉  3rd Place : {place_3rd}".ljust(60) + "║")
    print("╠" + "═" * 60 + "╣")
    print("║" + "  QUARTERFINAL RESULTS".ljust(60) + "║")
    for r in qf_records:
        line = f"  {r.match_label}: {r.winner} def. {r.loser} [{r.decided_by}]"
        print("║" + line.ljust(60) + "║")
    print("║" + "  SEMIFINAL RESULTS".ljust(60) + "║")
    for r in sf_records:
        line = f"  {r.match_label}: {r.winner} def. {r.loser} [{r.decided_by}]"
        print("║" + line.ljust(60) + "║")
    print("║" + "  FINAL RESULTS".ljust(60) + "║")
    for r in (final_rec, bronze_rec):
        line = f"  {r.match_label}: {r.winner} def. {r.loser} [{r.decided_by}]"
        print("║" + line.ljust(60) + "║")
    print("╚" + "═" * 60 + "╝")

    # Persist overall summary
    summary_path = Path("results") / f"tournament_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.parent.mkdir(exist_ok=True)
    summary = {
        "timestamp": timestamp,
        "1st": place_1st,
        "2nd": place_2nd,
        "3rd": place_3rd,
        "quarterfinals": [
            {"label": r.match_label, "winner": r.winner, "loser": r.loser,
             "result": r.result, "decided_by": r.decided_by}
            for r in qf_records
        ],
        "semifinals": [
            {"label": r.match_label, "winner": r.winner, "loser": r.loser,
             "result": r.result, "decided_by": r.decided_by}
            for r in sf_records
        ],
        "final": {"winner": final_rec.winner, "loser": final_rec.loser,
                  "result": final_rec.result, "decided_by": final_rec.decided_by},
        "bronze": {"winner": bronze_rec.winner, "loser": bronze_rec.loser,
                   "result": bronze_rec.result, "decided_by": bronze_rec.decided_by},
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n📁 Tournament summary saved: {summary_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_tournament()
