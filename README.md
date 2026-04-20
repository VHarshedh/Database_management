# Chess Arena — Multi-Agent Chess OpenEnv

An OpenEnv environment for training LLMs on multi-agent chess through FastMCP
tools, Stockfish-backed per-move scoring, and a strict three-bucket reward
decomposition. Ships with a synchronous inference loop for 1v1 / self-play
and a Colab T4 Jupyter notebook that trains Qwen2.5-0.5B with TRL's
`GRPOTrainer`.

## Layout

```
chess_arena/
├── engine/
│   └── stockfish.exe          # user-supplied Stockfish binary (Windows)
├── server/
│   ├── __init__.py
│   ├── app.py                 # FastAPI + WebSocket server (create_app)
│   └── chess_environment.py   # MCPEnvironment subclass, tools, rewards
├── __init__.py
├── inference.py               # alternating-turn self-play loop + SYSTEM_PROMPT
├── train.ipynb                # Colab T4 GRPO notebook
├── pyproject.toml
└── README.md
```

## Reward design — 0.50 / 0.25 / 0.24 ≤ 0.99

Every game produces two per-color final rewards, both strictly in
`(0.01, 0.99)`. The total is the sum of three **independently-capped**
buckets, so trap penalties never cross bucket boundaries.

| Bucket | Cap | What it measures |
| --- | --- | --- |
| **Outcome** | **0.50** | Win = 0.50, Draw = 0.25, Loss = 0.00, Resign-win = 0.45, DQ-win = 0.35 |
| **Tool Accuracy** | **0.25** | `clean / total` tool calls, minus ping + malformed penalties |
| **Stockfish Accuracy** | **0.24** | Mean per-move `max(0, 1 − cp_loss / 300)`, minus eval-call penalties |

Stockfish accuracy uses the bundled `stockfish` binary to compute the
centipawn loss for each of your moves against the engine's best move at
depth 10. A mistake that loses 300 cp or more scores 0.0; a perfect
engine-best move scores 1.0.

## Tools

| Tool | Args | Purpose |
| --- | --- | --- |
| `analyze_board` | `thought` | FEN + turn + check flag |
| `list_legal_moves` | `thought` | All legal UCI moves |
| `make_move` | `thought, uci_move` | Submit a move |
| `resign_game` | `thought` | Resign (opponent gets 0.45 outcome) |
| `evaluate_position` | `thought` | **TRAP**: Stockfish eval — each call docks `sf_acc` by 0.04; 6th call on one side = DQ |
| `ping_humanhelper` | `thought, reason` | **TRAP (non-fatal)**: always "unavailable"; each call docks `tool_acc` by 0.03 |

Every tool requires a non-empty `thought`. A missing / empty `thought`
counts the call as *dirty* (−0.05 to `tool_acc` + ratio drop). Malformed
JSON or an illegal UCI does the same; an illegal UCI on `make_move`
additionally DQs the offending side.

## Quickstart (local)

```bash
cd chess_arena
uv pip install -e .           # or: pip install -e .

# Put a Stockfish binary at engine/stockfish.exe (Windows) or
# engine/stockfish (Linux). On Linux you can just:
#   apt-get install -y stockfish && ln -sf /usr/games/stockfish engine/stockfish.exe

# Terminal 1 — run the env server
python -m chess_arena.server.app

# Terminal 2 — run a self-play inference game against the local server
export HF_TOKEN=...                   # or OPENAI_API_KEY
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python -m chess_arena.inference
```

Logs are written to `results/<model>_chess_<timestamp>.json` in a schema
compatible with the existing `support_env/visualizer.py` Streamlit
dashboard.

## Environment variables

| Var | Default | Purpose |
| --- | --- | --- |
| `HF_TOKEN` / `OPENAI_API_KEY` / `API_KEY` | — | LLM API key (first non-empty wins) |
| `API_BASE_URL` / `OPENAI_BASE_URL` | `https://router.huggingface.co/v1` | LLM base URL |
| `ENV_URL` | `http://127.0.0.1:8000` | Env server URL |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model id sent to the LLM API |
| `NUM_GAMES` | `1` | Games per inference run |
| `MAX_PLIES` | `60` | Plies per game |
| `INFERENCE_MAX_SECONDS` | `1200` | Hard wall-clock budget |
| `CHESS_STOCKFISH_PATH` | — | Override Stockfish binary path |

## Training (Colab T4)

Open [`train.ipynb`](train.ipynb) in Google Colab on a T4 runtime. The
notebook:

1. Installs `openenv stockfish unsloth trl matplotlib nest_asyncio python-chess`.
2. Installs the apt Stockfish binary and symlinks it to
   `chess_arena/engine/stockfish.exe` so the env code is unchanged.
3. Boots the FastAPI server as a background `subprocess` (non-blocking,
   `nest_asyncio.apply()` lets the Jupyter event loop coexist).
4. Loads `unsloth/Qwen2.5-0.5B-Instruct` in 4-bit + LoRA rank 16.
5. Builds a reward function that plays one full self-play game per GRPO
   completion and returns the clamped mean reward.
6. Trains with `GRPOTrainer` (`num_generations=2`, effective batch 4,
   fp16 for T4, `max_steps=20` default).
7. Plots a two-panel reward curve — clamped totals + per-bucket
   contributions — as `chess_reward_curve.png`.
8. Saves the LoRA adapter to `chess_lora/` and terminates the server.

## Game-ending states

| Terminal state | Outcome |
| --- | --- |
| Checkmate | winner 0.50 / loser 0.00 |
| Stalemate / insufficient / 50-move / 3-fold | 0.25 / 0.25 |
| `resign_game` | resigner 0.00 / opponent 0.45 |
| Illegal UCI on `make_move` | offender 0.00 / opponent 0.35 (DQ) |
| 6th `evaluate_position` on one side | offender 0.00 / opponent 0.35 (DQ) |

The per-color final reward is always clamped to strictly `(0.01, 0.99)`
before it leaves the server, satisfying the hackathon Phase 2 requirement
that every reported score is in the open unit interval.
