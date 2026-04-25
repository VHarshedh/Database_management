"""
Chess.com-style analysis visualizer for OpenEnv chess self-play JSON logs.

Features:
  - Backwards-compatible move extraction (move_history OR steps fallback)
  - Stockfish eval bar (depth 15, cached per game)
  - Game termination banner (checkmate, DQ, draw, unfinished)
  - Move list panel with SAN notation and live highlight
  - LLM reasoning panel (threat analysis, candidates, justification)
  - Navigation: ◀◀ ◀ ▶ ▶▶ + slider + auto-play
  - Appearance: System / Light / Dark (Gradio + CSS)

Usage:
    python visualizer.py
"""
from __future__ import annotations

import inspect
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.svg
import gradio as gr

# ---------------------------------------------------------------------------
# Stockfish setup
# ---------------------------------------------------------------------------
try:
    from stockfish import Stockfish as _SF  # type: ignore
    _SF_AVAILABLE = True
except ImportError:
    _SF_AVAILABLE = False

_HERE = Path(__file__).resolve().parent
# Stable root for any future assets (pieces, icons) — Gradio UI is web-based, not Pygame.
BASE_DIR = _HERE
_SF_PATHS = [
    _HERE / "engine" / "soc_threat_analyzer.exe",
    _HERE / "engine" / "soc_threat_analyzer",
    # Back-compat names:
    _HERE / "engine" / "stockfish.exe",
    _HERE / "engine" / "stockfish",
    Path("/usr/games/stockfish"),
    Path("/usr/local/bin/stockfish"),
]
SF_DEPTH = int(os.getenv("VIZ_SF_DEPTH", "15"))


def _find_stockfish() -> Optional[str]:
    import shutil
    env_path = os.getenv("SOC_THREAT_ANALYZER_PATH") or os.getenv("CHESS_STOCKFISH_PATH")
    if env_path and Path(env_path).is_file():
        return env_path
    for p in _SF_PATHS:
        if p.is_file():
            return str(p)
    resolved = shutil.which("stockfish")
    if resolved:
        return resolved
    return None


def _build_engine() -> Optional[Any]:
    if not _SF_AVAILABLE:
        return None
    path = _find_stockfish()
    if not path:
        return None
    try:
        return _SF(path=path, depth=SF_DEPTH)
    except Exception:
        return None


_ENGINE = _build_engine()

import atexit
def _shutdown_engine():
    if _ENGINE is not None:
        try:
            _ENGINE.quit()
        except Exception:
            pass
atexit.register(_shutdown_engine)


def _engine_version_str() -> str:
    if _ENGINE is None:
        return "unavailable"
    for attr in ("get_stockfish_major_version", "get_api_version", "_stockfish_version"):
        if hasattr(_ENGINE, attr):
            try:
                v = getattr(_ENGINE, attr)
                v = v() if callable(v) else v
                if v is not None:
                    return str(v)
            except Exception:
                pass
    return "ok"


def _sf_eval_cp(fen: str) -> Optional[int]:
    """Return centipawn evaluation from White's perspective. None = engine unavailable."""
    if _ENGINE is None:
        return None
    try:
        _ENGINE.set_fen_position(fen)
        ev = _ENGINE.get_evaluation()
        is_white_turn = " w " in fen
        if ev["type"] == "cp":
            val = ev["value"]
            return val if is_white_turn else -val
        if ev["type"] == "mate":
            m = ev["value"]
            val = (100000 - abs(m) * 10) * (1 if m > 0 else -1)
            return val if is_white_turn else -val
    except Exception as e:
        print(f"[visualizer] Eval error: {e}")
    return None


def _sf_eval_str(cp: Optional[int]) -> str:
    if cp is None:
        return "?"
    if abs(cp) >= 99000:
        mate_in = (100000 - abs(cp)) // 10
        sign = "#" if cp > 0 else "-#"
        return f"{sign}{mate_in}"
    pawns = cp / 100.0
    return f"{pawns:+.1f}" if pawns != 0 else "0.0"


# ---------------------------------------------------------------------------
# Board colors (tournament: terracotta / cream — not green / not indigo)
# ---------------------------------------------------------------------------
# Board SVG colors (python-chess); UI chrome follows CSS .dark — not Pygame surfaces.
BOARD_THEMES = {
    "classic": {
        "square light": "#e7dcc8",
        "square dark": "#8b6914",
        "margin": "#1a1f26",
        "coord": "#a8b0ba",
    },
    "slate": {
        "square light": "#d8e0e8",
        "square dark": "#4a6678",
        "margin": "#12181c",
        "coord": "#9aa8b2",
    },
}


# ---------------------------------------------------------------------------
# Move extraction
# ---------------------------------------------------------------------------

def _extract_moves(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    history = task.get("metadata", {}).get("move_history", [])
    if history:
        return history

    moves: List[Dict[str, Any]] = []
    for step in task.get("steps", []):
        if step.get("tool_name") != "make_move":
            continue
        args = step.get("arguments", {})
        uci = args.get("uci_move", "")
        if not uci:
            continue
        result_str = str(step.get("result", ""))
        if "ILLEGAL MOVE" in result_str:
            continue
        cp_loss = _parse_field(result_str, "cp_loss", 0)
        move_score = _parse_field(result_str, "move_score", 0.0)
        moves.append({
            "color": step.get("color", ""),
            "uci": uci,
            "cp_loss": cp_loss,
            "move_score": move_score,
        })
    return moves


def _parse_field(text: str, field: str, default: Any) -> Any:
    try:
        m = re.search(rf"{re.escape(field)}=([\d.+-]+)", text)
        if m:
            return type(default)(float(m.group(1)))
    except Exception:
        pass
    return default


# ---------------------------------------------------------------------------
# Eval bar
# ---------------------------------------------------------------------------

def _compute_evals(moves: List[Dict[str, Any]]) -> List[Optional[int]]:
    board = chess.Board()
    positions = [board.fen()]
    for mv in moves:
        uci = mv.get("uci", "")
        try:
            board.push_uci(uci)
            positions.append(board.fen())
        except Exception as exc:
            print(f"[visualizer] WARNING: Skipping bad UCI '{uci}': {exc}")
            positions.append(board.fen())
    return [_sf_eval_cp(fen) for fen in positions]


def _eval_bar_html(cp: Optional[int]) -> str:
    if cp is None:
        white_pct = 50.0
        label = "—"
    else:
        import math
        white_pct = 50 + 50 * (2 / (1 + math.exp(-cp / 400.0)) - 1)
        white_pct = max(5, min(95, white_pct))
        label = _sf_eval_str(cp)
    black_pct = 100.0 - white_pct
    return f"""
<div class="v-eval-outer" role="img" aria-label="Evaluation bar: White {label}">
  <div class="v-eval-seg v-eval-black" style="height:{black_pct:.1f}%"></div>
  <div class="v-eval-seg v-eval-white" style="height:{white_pct:.1f}%"></div>
  <span class="v-eval-label" style="top:{black_pct - 4:.1f}%">{label}</span>
</div>"""


# ---------------------------------------------------------------------------
# Result banner
# ---------------------------------------------------------------------------

_RESULT_LABELS = {
    "checkmate_white":           ("White wins by checkmate",        "#0f5132"),
    "checkmate_black":           ("Black wins by checkmate",        "#0b2d4a"),
    "resign_white":              ("Black wins — white resigned",    "#0b2d4a"),
    "resign_black":              ("White wins — black resigned",   "#0f5132"),
    "dq_illegal_white":          ("White disqualified (illegal moves)", "#842029"),
    "dq_illegal_black":          ("Black disqualified (illegal moves)", "#842029"),
    "dq_eval_abuse_white":       ("White disqualified (eval abuse)",  "#842029"),
    "dq_eval_abuse_black":       ("Black disqualified (eval abuse)",  "#842029"),
    "draw_stalemate":            ("Draw — stalemate",                 "#3d4d5c"),
    "draw_threefold_repetition": ("Draw — threefold repetition",     "#3d4d5c"),
    "draw_fifty_moves":          ("Draw — 50-move rule",             "#3d4d5c"),
    "draw_insufficient_material": ("Draw — insufficient material", "#3d4d5c"),
    "draw_variant_end":          ("Draw — position",                 "#3d4d5c"),
    "draw_repetition":           ("Draw — repetition",              "#3d4d5c"),
    "draw_50_move":              ("Draw — 50-move rule",            "#3d4d5c"),
    "unfinished":                ("Unfinished (ply limit reached)",  "#5c4a1a"),
}


def _result_banner(result_str: Optional[str], at_end: bool) -> str:
    if not at_end or not result_str:
        return ""
    label, color = _RESULT_LABELS.get(result_str, (f"End: {result_str}", "var(--v-border-strong)"))
    return f"""
<div class="v-banner" style="--banner-bg:{color};">
  {label}
</div>"""


# ---------------------------------------------------------------------------
# Move list
# ---------------------------------------------------------------------------

def _move_list_html(moves: List[Dict[str, Any]], current_ply: int) -> str:
    board = chess.Board()
    san_moves: List[str] = []
    for mv in moves:
        uci = mv.get("uci", "")
        move_obj = None
        try:
            move_obj = chess.Move.from_uci(uci)
            san = board.san(move_obj)
        except Exception:
            san = uci
        finally:
            if move_obj is not None and move_obj in board.legal_moves:
                board.push(move_obj)
        san_moves.append(san)

    rows = []
    for i in range(0, len(san_moves), 2):
        move_num = i // 2 + 1
        white_san = san_moves[i]
        black_san = san_moves[i + 1] if i + 1 < len(san_moves) else ""
        w_ply = i + 1
        b_ply = i + 2
        w_active = " is-active" if w_ply == current_ply else ""
        b_active = " is-active" if b_ply == current_ply else ""
        rows.append(
            f"<tr class='v-ml-row'>"
            f"<td class='v-ml-idx'>{move_num}.</td>"
            f"<td class='v-ml-san{w_active}' data-ply='{w_ply}'>{white_san}</td>"
            f"<td class='v-ml-san{b_active}' data-ply='{b_ply}'>{black_san}</td>"
            f"</tr>"
        )

    scroll_to_row = max(0, current_ply // 2 - 3)
    scroll_px = scroll_to_row * 40

    table = f"<table class='v-ml-table'>{''.join(rows)}</table>"
    return f"""
<div id="movelist" class="v-movelist" data-scrolltop="{scroll_px}">
  {table}
</div>"""


# ---------------------------------------------------------------------------
# Core render
# ---------------------------------------------------------------------------

def render_step(
    ply: int,
    game_name: str,
    data: Dict[str, Any],
    evals_cache: List[Optional[int]],
) -> Tuple[str, str, str, str, str, str, str, str]:
    """Render board, eval bar, reasoning, move list. Returns 8 values."""
    empty = (
        _eval_bar_html(None),
        "<div class='v-empty-inset'>No game loaded</div>",
        "<p class='v-empty-hint'>Upload a <code>*.json</code> match log to begin.</p>",
        "",
        "", "", "", "",
    )
    if not game_name or not data or ply < 0:
        return empty

    # Must use _get_task: supports `tasks[]` **or** single top-level `task_log` (no `tasks`).
    task = _get_task(str(game_name), data)
    if task is None:
        return empty

    moves = _extract_moves(task)
    total_plies = len(moves)
    at_end = ply >= total_plies and total_plies > 0

    board = chess.Board()
    current_move_info: Optional[Dict[str, Any]] = None
    for i in range(min(ply, total_plies)):
        uci = moves[i].get("uci", "")
        try:
            board.push_uci(uci)
        except Exception:
            pass
        if i == ply - 1:
            current_move_info = moves[i]

    lastmove = board.peek() if board.move_stack else None
    # One board palette (classic); dark UI mode slightly tones the SVG via CSS filter.
    _bc = BOARD_THEMES["classic"]
    board_svg = chess.svg.board(
        board=board,
        size=456,
        lastmove=lastmove,
        colors=dict(_bc),
    )
    board_html = f'<div class="v-board-wrap">{board_svg}</div>'

    cp = evals_cache[ply] if evals_cache and ply < len(evals_cache) else None
    eval_bar = _eval_bar_html(cp)

    result_str = task.get("metadata", {}).get("result")
    banner = _result_banner(result_str, at_end)
    move_list = _move_list_html(moves, ply)

    white_model = data.get("model_white", "White")
    black_model = data.get("model_black", "Black")
    header_info = f"""
    <div class="v-header-row">
        <div class="v-header-side">
            <span class="v-pawn v-pawn-w" aria-hidden="true">♔</span>
            <span class="v-header-name">{white_model}</span>
        </div>
        <span class="v-header-vs">vs</span>
        <div class="v-header-side v-header-b">
            <span class="v-header-name">{black_model}</span>
            <span class="v-pawn v-pawn-b" aria-hidden="true">♚</span>
        </div>
    </div>
    """

    status = f"Move {ply} / {total_plies}"
    threat = candidates = justification = engine_eval = ""

    if current_move_info and ply > 0:
        color = current_move_info.get("color", "").lower()
        uci = current_move_info.get("uci", "")
        model_name = data.get(f"model_{color}", "?")
        cp_loss = current_move_info.get("cp_loss", "N/A")
        mv_score = current_move_info.get("move_score", "N/A")
        status = f"Move {ply} / {total_plies} · {color} ({model_name}) · {uci}"
        engine_eval = f"CP loss: {cp_loss}  |  Move score: {mv_score}  |  Eval: {_sf_eval_str(cp)}"
        for step in reversed(task.get("steps", [])):
            if step.get("tool_name") == "make_move":
                args = step.get("arguments", {})
                if args.get("uci_move") == uci and step.get("color") == color:
                    threat = args.get("threat_analysis", "—")
                    cand_list = args.get("candidate_moves", [])
                    candidates = "\n".join(f"· {c}" for c in cand_list) if isinstance(cand_list, list) else str(cand_list)
                    justification = args.get("justification", "—")
                    break

    combined_board = f"{header_info}{banner}{board_html}"
    return eval_bar, combined_board, move_list, status, threat, candidates, justification, engine_eval


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _file_path_from_upload(file_obj: Any) -> Optional[str]:
    if file_obj is None:
        return None
    if isinstance(file_obj, (list, tuple)) and file_obj:
        file_obj = file_obj[0]
    raw: Any
    if isinstance(file_obj, str):
        raw = file_obj
    else:
        raw = (
            getattr(file_obj, "name", None)
            or getattr(file_obj, "path", None)
            or getattr(file_obj, "orig_name", None)
            or str(file_obj)
        )
    p = Path(str(raw))
    try:
        p = p.resolve()
    except (OSError, ValueError, RuntimeError):
        pass
    if p.is_file():
        return str(p)
    return None


def _parse_log_data(file_path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)


def _load_game_internals(game_name: str, data: Dict[str, Any]) -> Tuple:
    if not game_name or not data:
        return (gr.update(minimum=0, maximum=0, value=0, step=1), 0, [],) + (("",) * 8)
    task = _get_task(game_name, data)
    if task is None:
        return (gr.update(minimum=0, maximum=0, value=0, step=1), 0, [],) + (("",) * 8)
    moves = _extract_moves(task)
    total = len(moves)
    evals = _compute_evals(moves)
    slider_update = gr.update(minimum=0, maximum=total, value=0, step=1,
                              label=f"Position (0–{total})")
    render_out = render_step(0, game_name, data, evals)
    return (slider_update, total, evals) + render_out


def _get_task(game_name: str, data: Dict[str, Any]):
    # Try 'tasks' list first
    tasks = data.get("tasks", [])
    if tasks:
        try:
            idx = int(str(game_name).split()[1]) - 1
            return tasks[idx]
        except Exception:
            return None
    # Fallback to single 'task_log'
    return data.get("task_log")


def load_game(game_name: str, data: Dict[str, Any]) -> Tuple:
    """Return (load_status, …slider through engine) — 12 values for ``outputs[2:]``."""
    inner = _load_game_internals(game_name, data)
    task = _get_task(str(game_name), data) if (game_name and data) else None
    n = len(_extract_moves(task)) if task is not None else 0
    if not game_name or not data:
        status = "—"
    elif task is None:
        status = f"**Not found** · {game_name}"
    else:
        status = f"**Viewing** · {game_name} · {n} half-moves"
    return (gr.update(value=status),) + inner


def on_file_upload(
    file_obj: Any,
) -> Tuple:
    """
    Load JSON, refresh dropdown, and immediately run first game.
    """
    # game_dropdown, log_data, load_status, slider, max_plies, evals_cache, + 8 render
    def _mt(msg: str):
        return gr.update(value=msg)

    empty = (
        gr.update(choices=[], value=None),
        {},
        _mt("**No file** · choose a `*.json` log."),
        gr.update(minimum=0, maximum=0, value=0, step=1, label="Position"),
        0,
        [],
    ) + (("",) * 8)
    path = _file_path_from_upload(file_obj)
    if not path:
        return empty

    data, err = _parse_log_data(path)
    if data is None:
        print(f"[visualizer] Load error: {err}")
        return (
            gr.update(choices=[], value=None),
            {},
            _mt(f"**Load failed:** {err}"),
            gr.update(minimum=0, maximum=0, value=0, step=1, label="Position"),
            0,
            [],
        ) + (("",) * 8)

    tasks = data.get("tasks", [])
    if not tasks and "task_log" in data:
        # Single game log
        names = ["Game 1"]
    else:
        names = [f"Game {i + 1}" for i in range(len(tasks))]

    if not names:
        return (
            gr.update(choices=[], value=None),
            data,
            _mt("**No games** in this file (empty `tasks` and no `task_log`)."),
            gr.update(minimum=0, maximum=0, value=0, step=1, label="Position"),
            0,
            [],
        ) + (("",) * 8)

    first = names[0]
    out = _load_game_internals(first, data)
    base = Path(path).name
    n_games = len(names)
    msg = f"**Loaded** · `{base}` · {n_games} game(s)"
    return (
        gr.update(choices=names, value=first),
        data,
        _mt(msg),
    ) + out


def on_slider(ply: int, game_name: str, data: Dict[str, Any], evals: List) -> Tuple:
    # Critical fix: ensure ply is not None
    if ply is None:
        ply = 0
    return render_step(ply, game_name, data, evals)


# ---------------------------------------------------------------------------
# Theming: CSS variables + [data-theme=light|dark|system] (set by Gradio + JS)
# ---------------------------------------------------------------------------

MAIN_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* —— Theme tokens —— */
:root, .dark {
  --v-canvas: #f6f3ee;
  --v-surface: #ffffff;
  --v-elev: #f0ebe3;
  --v-text: #1a1d21;
  --v-text-muted: #4a5560;
  --v-border: #d1c9bc;
  --v-border-strong: #9a8f7e;
  --v-accent: #0d6e63;
  --v-eval-w: #f2e3c9;
  --v-eval-b: #1e3a3f;
  --v-ml-cur: rgba(13, 110, 99, 0.2);
  --v-ml-cur-b: #0d6e63;
}

.dark {
  --v-canvas: #0d1114;
  --v-surface: #141a1f;
  --v-elev: #1a2229;
  --v-text: #e8eaed;
  --v-text-muted: #9aa3ad;
  --v-border: #2d3a45;
  --v-border-strong: #4a5a68;
  --v-accent: #2dd4bf;
  --v-eval-w: #c9a86c;
  --v-eval-b: #0a1620;
  --v-ml-cur: rgba(45, 212, 191, 0.18);
  --v-ml-cur-b: #2dd4bf;
}

.gradio-container, body {
  background: var(--v-canvas) !important;
  color: var(--v-text) !important;
  font-family: "DM Sans", system-ui, sans-serif !important;
}
.v-root { max-width: 1280px; margin: 0 auto; padding: 0.5rem 1rem 1.5rem; }
.v-title h2 { color: var(--v-text) !important; font-weight: 800 !important; }
.v-sub, .v-sub p { color: var(--v-text-muted) !important; font-size: 0.95rem !important; }
.v-load-status, .v-load-status p, .v-load-status li { font-size: 0.9rem !important; color: var(--v-text) !important; margin: 0.35rem 0 0.75rem 0 !important; }
.v-load-status strong, .v-title strong, .v-sub strong { color: var(--v-text) !important; }
/* Board SVG: slight tone when UI is in .dark (theme is web/CSS, not Pygame fill). */
.dark .v-board-wrap svg { filter: brightness(0.92) contrast(1.05); }

/* Main area: do not stretch columns to one tall height (prevents board/HTML from overlapping side panels). */
.v-main { align-items: flex-start !important; }
.v-main > .gr-column, .v-main > .column { min-width: 0 !important; }

/* Form / block labels: theme text on elevated surface (always readable; avoids white-on-bright-teal in dark). */
.v-root .gr-box > label,
.v-root .gr-form label,
.v-root .label-wrap,
.v-root .block > label { 
  color: var(--v-text) !important; 
  background: var(--v-elev) !important;
  border: 1px solid var(--v-border) !important;
  padding: 0.25rem 0.5rem !important;
  border-radius: 6px !important;
  font-size: 0.8rem !important;
  font-weight: 600 !important;
  text-transform: none !important;
  letter-spacing: normal !important;
  margin-bottom: 0.35rem !important;
  display: inline-block !important;
}
.v-root .tabitem, .v-root [role="tabpanel"] { color: var(--v-text) !important; }
.v-root [role="tab"] { color: var(--v-text) !important; }

.v-engine-pill {
  text-align: right; font-size: 0.85rem; color: var(--v-text) !important;
  padding: 0.5rem 0.75rem; border: 1px solid var(--v-border-strong);
  border-radius: 10px; background: var(--v-elev) !important;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

#board-col {
  background: var(--v-surface) !important;
  border: 1px solid var(--v-border) !important;
  border-radius: 16px;
  padding: 1.25rem !important;
  box-shadow: 0 4px 24px rgba(0,0,0,0.06);
  min-width: 0 !important;
  position: relative;
  z-index: 1;
  overflow: visible;
}

/* Board image: never overflow or cover the nav row; stays in document flow. */
.v-board-wrap {
  max-width: 100%;
  overflow: auto;
  line-height: 0;
  position: relative;
  z-index: 0;
}
.v-board-wrap svg {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0 auto;
  vertical-align: top;
}
.v-empty-inset, .v-empty-hint, .v-empty-hint code {
  color: var(--v-text) !important;
}
.v-empty-hint code { background: var(--v-elev) !important; padding: 0.1em 0.35em; border-radius: 4px; }

/* Result banner over board */
.v-banner {
  box-sizing: border-box;
  width: 100%;
  padding: 0.5rem 0.75rem;
  border-radius: 8px;
  margin-bottom: 0.75rem;
  text-align: center;
  font-weight: 600;
  font-size: 0.9rem;
  color: #f1f5f9 !important;
  text-shadow: 0 1px 2px rgba(0,0,0,0.4);
  background: var(--banner-bg, #3d4d5c);
  border: 1px solid rgba(255,255,255,0.12);
}

.v-header-row {
  display: flex; justify-content: space-between; align-items: center; gap: 0.5rem;
  margin-bottom: 0.75rem; padding: 0.5rem 0.75rem; background: var(--v-elev);
  border: 1px solid var(--v-border); border-radius: 10px;
  font-size: 0.95rem; font-weight: 600; color: var(--v-text);
}
.v-header-vs { color: var(--v-accent); font-weight: 700; letter-spacing: 0.1em; font-size: 0.7rem; }

/* Better Input Contrast */
textarea, .gr-text-input input, .gr-text-input textarea, .gr-box, .dropdown, select, input {
  background: var(--v-surface) !important; 
  color: var(--v-text) !important;
  border: 1px solid var(--v-border-strong) !important;
  border-radius: 10px !important;
}
.v-root textarea, .v-root .gr-text-input textarea, .v-root input:disabled, .v-root textarea:disabled {
  color: var(--v-text) !important;
  opacity: 1 !important;
  -webkit-text-fill-color: var(--v-text) !important;
}

/* Eval bar column: narrow strip aligned to top, same height as board SVG target */
.v-eval-col { align-self: flex-start !important; }
.v-eval-outer {
  position: relative;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  width: 32px;
  min-width: 32px;
  height: 456px;
  max-height: min(80vh, 600px);
  border: 1px solid var(--v-border-strong) !important;
  border-radius: 8px;
  background: var(--v-elev) !important;
  overflow: hidden;
}
.v-eval-seg {
  width: 100%;
  flex: 0 0 auto;
  min-height: 0;
  box-sizing: border-box;
}
.v-eval-black { background: var(--v-eval-b) !important; }
.v-eval-white { background: var(--v-eval-w) !important; }
.v-eval-label {
  position: absolute;
  left: 50%;
  transform: translate(-50%, -50%);
  z-index: 2;
  font-size: 0.7rem;
  font-weight: 800;
  line-height: 1;
  color: #fff !important;
  text-shadow: 0 0 4px #000, 0 1px 2px #000, 0 -1px 1px #000;
  pointer-events: none;
  border-radius: 3px;
  padding: 0.1em 0.2em;
  background: rgba(0,0,0,0.28);
}

.v-movelist {
  box-sizing: border-box;
  max-height: min(55vh, 24rem);
  overflow-y: auto;
  overflow-x: auto;
  color: var(--v-text) !important;
  background: var(--v-elev) !important; 
  border: 1px solid var(--v-border-strong) !important;
  border-radius: 10px;
  padding: 0.4rem 0.5rem;
}
.v-ml-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; color: var(--v-text) !important; }
.v-ml-row { border-bottom: 1px solid var(--v-border); }
.v-ml-idx { color: var(--v-text-muted) !important; padding: 0.25rem 0.5rem 0.25rem 0; width: 2.2rem; }
.v-ml-san { color: var(--v-text) !important; padding: 0.25rem 0.4rem; }

/* Nav always above board SVG in stacking order; wrap on narrow viewports. */
.v-nav-row {
  position: relative;
  z-index: 2;
  flex-wrap: wrap !important;
  gap: 0.35rem;
  margin-top: 0.75rem;
}
.v-nav-row button { position: relative; z-index: 2; }

.v-ml-san.is-active {
  background: var(--v-ml-cur) !important; 
  border-left: 3px solid var(--v-ml-cur-b) !important;
}

::-webkit-scrollbar { width: 7px; height: 7px; }
::-webkit-scrollbar-thumb { background: var(--v-border-strong); border-radius: 5px; }
"""

_THEME_BOOT_HTML = (
    """<script>
    (function(){
      try {
        var m = localStorage.getItem("v-theme") || "system";
        if (m === "dark" || (m === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches)) {
            document.documentElement.classList.add("dark");
        } else {
            document.documentElement.classList.remove("dark");
        }
      } catch(e) {}
    })();
    </script>"""
)

# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def _make_theme() -> gr.themes.Soft:
    try:
        th = gr.themes.Soft(
            primary_hue=getattr(gr.themes.colors, "teal", "teal"),
            secondary_hue=getattr(gr.themes.colors, "stone", "stone"),
            neutral_hue=getattr(gr.themes.colors, "zinc", "zinc"),
            font=(
                gr.themes.GoogleFont("DM Sans"),
                "ui-sans-serif",
                "system-ui",
                "sans-serif",
            ),
            font_mono=(
                gr.themes.GoogleFont("IBM Plex Mono"),
                "ui-monospace",
                "monospace",
            ),
        )
    except Exception:
        th = gr.themes.Soft(primary_hue="teal", neutral_hue="zinc")
    return th.set(
        block_title_text_weight="600",
        input_border_width="1px",
    )


def build_app() -> Tuple[gr.Blocks, gr.themes.Soft]:
    """Return (Blocks, theme). Pass ``theme`` and :data:`MAIN_CSS` to ``launch()`` (Gradio 6+)."""
    theme = _make_theme()
    with gr.Blocks(
        title="OpenEnv — Chess Arena",
        elem_classes="v-root",
    ) as app:
        gr.HTML(
            "<div class='v-theme-boot'>" + _THEME_BOOT_HTML + "</div>",
            visible=True,
        )
        # --- state ---
        log_data = gr.State({})
        evals_cache = gr.State([])
        max_plies = gr.State(0)

        with gr.Row(elem_classes="v-toolbar"):
            with gr.Column(scale=3, min_width=220):
                gr.Markdown("## Chess Arena · match viewer", elem_classes="v-title")
                gr.Markdown(
                    "Upload a match JSON (self-play / tournament). **Theme** below. **← / →** prev/next · **Home** / **End** start/end · **T** cycle theme.",
                    elem_classes="v-sub",
                )
            with gr.Column(scale=1, min_width=200):
                appearance = gr.Radio(
                    label="Theme",
                    choices=[("System", "system"), ("Light", "light"), ("Dark", "dark")],
                    value="system",
                    elem_id="v-appearance",
                )
            with gr.Column(scale=0, min_width=140):
                gr.Markdown(
                    f"**Engine** · `{_engine_version_str()}`",
                    elem_classes="v-engine-pill",
                )

        with gr.Row(elem_classes="v-upload-row"):
            file_input = gr.File(
                label="Match log (JSON)",
                file_types=[".json"],
            )
            game_dropdown = gr.Dropdown(
                label="Select game",
                choices=[],
                interactive=True,
            )
        load_status = gr.Markdown("—", elem_id="v-load-status", elem_classes="v-load-status")

        with gr.Row(equal_height=False, elem_classes="v-main"):

            with gr.Column(scale=0, min_width=48):
                eval_bar_html = gr.HTML(elem_id="eval-bar", elem_classes="v-eval-col")

            with gr.Column(scale=4, min_width=320, elem_id="board-col"):
                board_html = gr.HTML()

                with gr.Row(elem_classes="nav-controls v-nav-row"):
                    btn_start = gr.Button("⏮", scale=0, min_width=48, variant="secondary", elem_id="v-btn-start")
                    btn_prev = gr.Button("◀", scale=0, min_width=48, variant="secondary", elem_id="v-btn-prev")
                    btn_next = gr.Button("▶", scale=0, min_width=48, variant="secondary", elem_id="v-btn-next")
                    btn_end = gr.Button("⏭", scale=0, min_width=48, variant="secondary", elem_id="v-btn-end")
                    btn_auto = gr.Button("Auto-play", scale=1, variant="primary")

                slider_ply = gr.Slider(
                    label="Position",
                    minimum=0,
                    maximum=0,
                    step=1,
                    value=0,
                    elem_id="v-slider-ply",
                )
                status_box = gr.Textbox(
                    label="Current position",
                    interactive=False,
                    lines=1,
                )

            with gr.Column(scale=3, min_width=240):
                with gr.Tabs():
                    with gr.Tab("Move list"):
                        move_list_html = gr.HTML()
                    with gr.Tab("Analysis"):
                        threat_box = gr.Textbox(
                            label="Threat analysis", lines=6, interactive=False
                        )
                        candidates_box = gr.Textbox(
                            label="Candidate moves", lines=2, interactive=False
                        )
                        justification_box = gr.Textbox(
                            label="Strategic justification", lines=5, interactive=False
                        )
                        engine_box = gr.Textbox(
                            label="Heuristic metrics", lines=1, interactive=False
                        )

        _outputs = [
            game_dropdown, log_data, load_status,
            slider_ply, max_plies, evals_cache,
            eval_bar_html, board_html, move_list_html,
            status_box, threat_box, candidates_box, justification_box, engine_box,
        ]
        _render_only = _outputs[6:]

        file_input.upload(
            fn=on_file_upload,
            inputs=[file_input],
            outputs=_outputs,
            show_progress="minimal",
        )

        game_dropdown.change(
            fn=load_game,
            inputs=[game_dropdown, log_data],
            outputs=_outputs[2:],
            show_progress="minimal",
        )

        _SCROLL_JS = """
        (...args) => {
            setTimeout(() => {
                const el = document.getElementById('movelist');
                if (el) {
                    const target = parseInt(el.dataset.scrolltop || '0', 10);
                    el.scrollTop = target;
                }
            }, 50);
            return args;
        }
        """
        slider_ply.change(
            fn=on_slider,
            inputs=[slider_ply, game_dropdown, log_data, evals_cache],
            outputs=_render_only,
            show_progress="hidden",
            js=_SCROLL_JS,
        )

        def go_start():
            return gr.update(value=0)
        def go_end(mp: int):
            return gr.update(value=int(mp))
        def go_prev(ply: int):
            return gr.update(value=max(0, int(ply) - 1))
        def go_next(ply: int, mp: int):
            return gr.update(value=min(int(mp), int(ply) + 1))

        btn_start.click(
            go_start, [], [slider_ply], show_progress="hidden", js=_SCROLL_JS
        )
        btn_end.click(
            go_end, [max_plies], [slider_ply], show_progress="hidden", js=_SCROLL_JS
        )
        btn_prev.click(
            go_prev, [slider_ply], [slider_ply], show_progress="hidden", js=_SCROLL_JS
        )
        btn_next.click(
            go_next, [slider_ply, max_plies], [slider_ply], show_progress="hidden", js=_SCROLL_JS
        )

        autoplay_state = gr.State(False)
        def toggle_autoplay(active: bool) -> Tuple[bool, str]:
            new = not active
            return new, ("Stop" if new else "Auto-play")
        btn_auto.click(
            toggle_autoplay, [autoplay_state], [autoplay_state, btn_auto]
        )
        def autoplay_tick(active: bool, ply: int, mp: int) -> int:
            if active and int(ply) < int(mp):
                return int(ply) + 1
            return int(ply)
        timer = gr.Timer(value=1.2, active=False)
        autoplay_state.change(
            fn=lambda a: gr.Timer(active=a), inputs=[autoplay_state], outputs=[timer]
        )
        timer.tick(
            fn=autoplay_tick,
            inputs=[autoplay_state, slider_ply, max_plies],
            outputs=[slider_ply],
        )

        _THEME_JS = (
            """(m) => {
                const mode = m || "system";
                try { localStorage.setItem("v-theme", mode); } catch (e) {}
                if (mode === "dark" || (mode === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches)) {
                    document.documentElement.classList.add("dark");
                } else {
                    document.documentElement.classList.remove("dark");
                }
            }"""
        )

        def _persist_appearance(choice: str) -> str:
            return choice

        appearance.change(
            _persist_appearance,
            inputs=[appearance],
            outputs=[appearance],
            js=_THEME_JS,
        )

        # Arrow keys + Home/End + T: mirror nav buttons and theme radio (avoids a Pygame loop).
        gr.HTML(
            """
<div style="height:0;overflow:hidden" aria-hidden="true" id="v-keybind-anchor"></div>
<script>
(function () {
  if (window.__vChessNav) { return; }
  window.__vChessNav = true;
  function clickBtn(id) {
    var w = document.getElementById(id);
    if (!w) { return; }
    var b = w.querySelector("button");
    if (b) { b.click(); }
  }
  document.addEventListener("keydown", function (e) {
    var t = e.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) {
      return;
    }
    if (e.key === "ArrowLeft") { e.preventDefault(); clickBtn("v-btn-prev"); }
    else if (e.key === "ArrowRight") { e.preventDefault(); clickBtn("v-btn-next"); }
    else if (e.key === "Home") { e.preventDefault(); clickBtn("v-btn-start"); }
    else if (e.key === "End") { e.preventDefault(); clickBtn("v-btn-end"); }
    else if (e.key === "t" || e.key === "T") {
      e.preventDefault();
      var fs = document.getElementById("v-appearance");
      if (!fs) { return; }
      var radios = fs.querySelectorAll("input[type=radio]");
      var arr = Array.prototype.slice.call(radios);
      if (!arr.length) { return; }
      var i = arr.findIndex(function (x) { return x.checked; });
      if (i < 0) { i = 0; }
      var n = (i + 1) % arr.length;
      if (arr[n]) { arr[n].click(); }
    }
  });
})();
</script>
            """,
            visible=True,
        )

    return app, theme


if __name__ == "__main__":
    demo, _theme = build_app()
    _kw: Dict[str, Any] = {
        "server_name": "127.0.0.1",
        "server_port": 7860,
        "share": False,
        "theme": _theme,
        "css": MAIN_CSS,
    }
    try:
        sig = inspect.signature(demo.launch)
        if "theme_mode" in sig.parameters:
            _kw["theme_mode"] = "system"
    except (TypeError, ValueError):
        pass
    demo.launch(**_kw)
