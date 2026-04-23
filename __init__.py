"""Chess Arena OpenEnv package.

Multi-agent chess environment with Stockfish-backed per-move scoring,
strict 0.50 / 0.10 / 0.15 / 0.24 reward decomposition (sum <= 0.99),
and two trap tools (`evaluate_position` with a hard cap,
`ping_humanhelper` non-fatal).
"""

__version__ = "0.1.0"
