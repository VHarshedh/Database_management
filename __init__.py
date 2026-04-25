"""Global SOC Datacenter Simulation OpenEnv package.

A multi-agent simulation where DEFENDER and ADVERSARY teams migrate workloads
across a 4D ``(region, zone, rack, pod)`` tensor. The environment is a
re-skinning adapter on top of `python-chess` + Stockfish: the LLM never sees
a chess board, FEN, or UCI move - everything is exposed as JSON datacenter
state and dictionary-shaped node coordinates.

Reward decomposition (sum <= 0.99):

  - Outcome bucket        : <= 0.50  (breach contained / compromise / DQ)
  - Format bucket          : <= 0.10  (clean structured tool calls)
  - Thought-quality bucket : <= 0.15  (deterministic reasoning score)
  - Stockfish bucket       : <= 0.24  (cp closeness to engine best)

Trap tools `query_threat_oracle` (capped) and `escalate_to_oncall` (non-fatal)
are preserved from the chess version under SOC-themed names.
"""

__version__ = "0.2.0"
