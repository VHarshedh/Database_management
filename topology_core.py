"""Topology core wrapper.

This module is the *only* place in the datacenter workload-migration package
that imports the third-party board/engine backend. Everything else should
import this module and use its re-exported symbols.
"""

from __future__ import annotations

from typing import Any

import chess as _backend
import chess.engine as _engine

# Re-export the backend module (useful for rarely-used constants / helpers).
backend = _backend
engine = _engine

# Common types / helpers used across the env + orchestrator.
Board = _backend.Board
Move = _backend.Move
Piece = _backend.Piece
SQUARES = _backend.SQUARES

square = _backend.square
square_file = _backend.square_file
square_rank = _backend.square_rank
square_name = _backend.square_name
parse_square = _backend.parse_square

# Backend constants / exceptions we reference explicitly.
WHITE = _backend.WHITE
BLACK = _backend.BLACK
IllegalMoveError = _backend.IllegalMoveError

# A small escape hatch for odd call sites that need an attribute not
# explicitly re-exported.
def getattr_backend(name: str) -> Any:
    return getattr(_backend, name)