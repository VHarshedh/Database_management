---
title: Chess Arena
emoji: ♟️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Chess Arena — Multi-Agent Chess OpenEnv

An OpenEnv environment for training LLMs on multi-agent chess through FastMCP tools, Stockfish-backed per-move scoring, and a strict four-bucket reward decomposition. Ships with an 8-model tournament orchestrator and a GRPO training loop.

## Layout

```text
chess_arena/
├── engine/
│   └── stockfish.exe          # user-supplied Stockfish binary
├── server/
│   ├── app.py                 # FastAPI + WebSocket server (create_app)
│   └── chess_environment.py   # MCPEnvironment subclass, tools, rewards
├── inference.py               # alternating-turn self-play loop
├── tournament.py              # 8-model knockout tournament orchestrator
├── train.ipynb                # GRPO single-step training notebook
└── README.md
```

## Reward Design — 0.50 / 0.10 / 0.15 / 0.24 ≤ 0.99

Every game produces two per-color final rewards, strictly clamped to (0.01, 0.99). The total is the sum of four independently-capped buckets. Penalties are bucket-local, meaning they never cross boundaries.

| Bucket | Cap | What it measures |
| :--- | :--- | :--- |
| **Outcome** | 0.50 | Win = 0.50, Draw = 0.25, Loss = 0.00, Resign-win = 0.45, DQ-win = 0.35 |
| **Format** | 0.10 | Schema compliance. Deductions for missing reasoning fields or illegal UCI formats. |
| **Thought Quality** | 0.15 | Deterministic grading of structured reasoning (Threat awareness, move tracing, strategic concepts). |
| **Stockfish Accuracy** | 0.24 | Mean per-move accuracy `max(0, 1 − cp_loss / 300)`. Docks points for over-relying on evaluation tools. |

## Tools & Traps

We expose standard FastMCP tools (`analyze_board`, `list_legal_moves`, `make_move`, `resign_game`) alongside two specific traps designed to test the agent's reward optimization:

* **`evaluate_position`**: **TRAP**. Returns Stockfish eval, but each call docks `sf_acc` by 0.04. 6th call on one side results in an instant DQ.
* **`ping_humanhelper`**: **TRAP (non-fatal)**. Always returns "unavailable". Each call docks the Format bucket by 0.03.

Every tool requires a mandatory 3-part structured reasoning schema (`threat_analysis`, `candidate_moves`, `justification`).

## Quickstart (Local)

```bash
uv pip install -e .

# Terminal 1 — run the env server
python -m chess_arena.server.app

# Terminal 2 — Run an 8-model knockout tournament
export HF_TOKEN=...
python tournament.py
```

## Training

The `train.ipynb` uses Hugging Face TRL's `GRPOTrainer` to optimize the model. It extracts the raw single-turn completions, evaluates them against the OpenEnv step logic, and uses the dense bucket rewards to calculate advantage.

## Results

Across self-play, inference evaluation, and our 8-model knockout tournaments, we evaluate models based on their tool usage formatting, adherence to chess rules, and strength. The following results reflect testing conducted with our framework.

### Tournament 1 (2026-04-23 — Folder: `tournament 2`)
* **1st Place**: `gemini-2.5-pro`
* **2nd Place**: `gemini-3.1-flash-lite-preview`
* **3rd Place**: `openai/gpt-oss-120b`

### Tournament 2 (2026-04-23 — Folder: `tournament 3`)
* **1st Place**: `gemini-3.1-flash-lite-preview`
* **2nd Place**: `openai/gpt-oss-20b`
* **3rd Place**: `gemini-2.5-pro`

### Tournament 3 (2026-04-24 — Folder: `results/`)
* **1st Place**: `gemini-2.5-pro`
* **2nd Place**: `gemini-3.1-pro-preview`
* **3rd Place**: `gemini-3.1-flash-lite-preview`



### Selected Matchups (Inference & Self-Play)
Extensive 1v1 play logs are captured in the `results/` folder. Notable runs include:
* **Gemini 2.5 Pro vs OpenAI GPT-OSS-120b**
* **Gemini 3.1 Pro Preview vs Gemini 2.5 Pro**
* **Gemma 4 31b IT vs Gemini 3.1 Flash Lite Preview**
* **Self-play benchmarks** on `gemini-2.5-pro` and `gemini-3.1-flash-lite-preview` to track format consistency.

Full trace logs for all games, including reasoning outputs and Stockfish evaluations per move, are accessible in the `results/` directory as JSON files.