#!/usr/bin/env python3
"""
self_play.py — Single-Model Self-Play
======================================

Picks one model from the 8-model pool (or accepts --model) and plays it
against itself. Both White and Black share the same LLM policy but maintain
independent context windows, so each side only sees its own reasoning history.

Usage:
    # Random model from the pool:
    python self_play.py

    # Force a specific model:
    python self_play.py --model gemini-2.5-pro

    # Multiple games:
    python self_play.py --games 3

    # Custom step delay:
    python self_play.py --delay 2.0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

try:
    from dotenv import dotenv_values
except ImportError:
    dotenv_values = None  # type: ignore

try:
    from openai import OpenAI
except ImportError:
    print("[FATAL] openai package not installed. pip install openai")
    sys.exit(1)

from inference import (
    ENV_URL,
    LLM_CALL_TIMEOUT,
    MAX_PLIES,
    NUM_GAMES,
    RATE_LIMIT_SLEEP,
    STEP_DELAY_SECONDS,
    _clamp_phase2,
    _fmt,
    _PHASE2_MIN,
    log_info,
    make_openai_policy,
    run_episode,
)


# ---------------------------------------------------------------------------
# Model pool loader (mirrors inference.py main())
# ---------------------------------------------------------------------------

def _load_model_pool() -> list[tuple[str, str, str, str]]:
    """Load all models from .env and .env.local.

    Returns a list of (model_name, provider, base_url, api_key) tuples.
    """
    here = Path(__file__).resolve().parent

    if dotenv_values is not None:
        env_google = dotenv_values(str(here / ".env"))
        env_groq = dotenv_values(str(here / ".env.local"))
    else:
        env_google = {}
        env_groq = {}

    google_url = env_google.get(
        "API_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    google_key = (
        env_google.get("HF_TOKEN")
        or env_google.get("OPENAI_API_KEY")
        or os.getenv("HF_TOKEN", "")
    )

    groq_url = env_groq.get("API_BASE_URL", "https://api.groq.com/openai/v1")
    groq_key = (
        env_groq.get("HF_TOKEN")
        or env_groq.get("OPENAI_API_KEY")
        or os.getenv("HF_TOKEN", "")
    )

    pool: list[tuple[str, str, str, str]] = []
    for i in range(1, 5):
        name = env_google.get(f"GOOGLE_MODEL_{i}")
        if name:
            pool.append((name, "google", google_url, google_key))
    for i in range(1, 5):
        name = env_groq.get(f"GROQ_MODEL_{i}")
        if name:
            pool.append((name, "groq", groq_url, groq_key))

    return pool


# ---------------------------------------------------------------------------
# Server readiness
# ---------------------------------------------------------------------------

def _wait_for_server(url: str, client: httpx.Client, retries: int = 30) -> bool:
    for _ in range(retries):
        for probe in ("/health", "/docs"):
            try:
                r = client.get(f"{url}{probe}", timeout=5.0)
                if r.status_code < 500:
                    return True
            except httpx.RequestError:
                pass
        time.sleep(1)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-model self-play using the Chess Arena environment.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name to use for self-play. If omitted, a random model is picked.",
    )
    parser.add_argument(
        "--games", type=int, default=NUM_GAMES,
        help=f"Number of self-play games to run (default: {NUM_GAMES}).",
    )
    parser.add_argument(
        "--delay", type=float, default=STEP_DELAY_SECONDS,
        help=f"Per-turn delay in seconds (default: {STEP_DELAY_SECONDS}).",
    )
    parser.add_argument(
        "--max-plies", type=int, default=MAX_PLIES,
        help=f"Maximum plies per game (default: {MAX_PLIES}).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2,
        help="LLM sampling temperature (default: 0.2).",
    )
    args = parser.parse_args()

    # Load model pool
    pool = _load_model_pool()
    if not pool:
        log_info("ERROR: no models found in .env / .env.local.")
        sys.exit(1)

    # Pick the model
    if args.model:
        # Find the requested model in the pool (for provider/key routing)
        match = [m for m in pool if m[0] == args.model]
        if not match:
            # Not in pool — assume Google provider as default
            log_info(f"⚠ Model '{args.model}' not found in pool. Assuming Google provider.")
            model_name = args.model
            provider = "google"
            base_url = pool[0][2] if pool else "https://generativelanguage.googleapis.com/v1beta/openai/"
            api_key = pool[0][3] if pool else ""
        else:
            model_name, provider, base_url, api_key = match[0]
    else:
        model_name, provider, base_url, api_key = random.choice(pool)

    provider_label = {"google": "Google/Gemini", "groq": "Groq"}.get(provider, provider)

    # Banner
    log_info("═" * 60)
    log_info("  ♟  Chess Arena — Self-Play Mode")
    log_info("═" * 60)
    log_info(f"  Model    : {model_name} ({provider_label})")
    log_info(f"  Games    : {args.games}")
    log_info(f"  Delay    : {args.delay}s/turn")
    log_info(f"  Max Plies: {args.max_plies}")
    log_info(f"  Temp     : {args.temperature}")
    log_info("═" * 60)

    # Build a single client and a single policy — both sides share it,
    # but run_episode gives each side its own message buffer.
    client = OpenAI(base_url=base_url, api_key=api_key)
    policy = make_openai_policy(
        client,
        model_name=model_name,
        temperature=args.temperature,
        call_timeout=LLM_CALL_TIMEOUT,
        base_rate_limit_sleep=RATE_LIMIT_SLEEP,
    )

    run_logs = {
        "mode": "self_play",
        "model": model_name,
        "provider": provider,
        "timestamp": datetime.now().isoformat(),
        "step_delay_seconds": args.delay,
        "temperature": args.temperature,
        "tasks": [],
    }

    total_scores: list[float] = []

    with httpx.Client(timeout=60.0) as http_client:
        log_info(f"\nWaiting for server at {ENV_URL} ...")
        if not _wait_for_server(ENV_URL, http_client):
            log_info("ERROR: server did not become ready. Is uvicorn running?")
            sys.exit(1)
        log_info("Server ready.\n")

        for game_idx in range(args.games):
            print(
                f"[START] task=self_play_{game_idx + 1} env=chess_arena "
                f"model={model_name}",
                flush=True,
            )

            result = run_episode(
                policy_white=policy,
                policy_black=policy,
                env_url=ENV_URL,
                game_idx=game_idx,
                max_plies=args.max_plies,
                step_delay=args.delay,
                model_white_name=model_name,
                model_black_name=model_name,
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

            log_info(f"\n{'─' * 60}")
            log_info(f"  🏁 Game {game_idx + 1} finished — {result.result or 'unfinished'}")
            log_info(f"  ⬜ WHITE: {_fmt(result.final_reward['white'])}")
            log_info(f"  ⬛ BLACK: {_fmt(result.final_reward['black'])}")
            log_info(f"{'─' * 60}")

    # Save results
    os.makedirs("results", exist_ok=True)
    safe_name = model_name.replace("/", "_").replace(":", "_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results/self_play_{safe_name}_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(run_logs, f, indent=2)
    log_info(f"\n📁 Wrote JSON log: {out_path}")

    if total_scores:
        avg = sum(total_scores) / len(total_scores)
        log_info(
            f"\n{'═' * 60}\n"
            f"   Average score: {avg:.3f} across {len(total_scores)} game(s)\n"
            f"{'═' * 60}"
        )


if __name__ == "__main__":
    main()
