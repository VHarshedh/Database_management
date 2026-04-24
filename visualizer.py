"""
Chess.com-style analysis visualizer for OpenEnv chess self-play JSON logs.

Features:
  - Backwards-compatible move extraction (move_history OR steps fallback)
  - Stockfish eval bar (depth 18, cached per game)
  - Game termination banner (checkmate, DQ, draw, unfinished)
  - Move list panel with SAN notation and live highlight
  - LLM reasoning panel (threat analysis, candidates, justification)
  - Navigation: ◀◀ ◀ ▶ ▶▶ + slider + auto-play

Usage:
    python visualizer.py
"""
from __future__ import annotations

import json
import os
import re
import time
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
_SF_PATHS = [
    _HERE / "engine" / "stockfish.exe",
    _HERE / "engine" / "stockfish",
    Path("/usr/games/stockfish"),
    Path("/usr/local/bin/stockfish"),
]
SF_DEPTH = int(os.getenv("VIZ_SF_DEPTH", "15"))


def _find_stockfish() -> Optional[str]:
    import shutil
    env_path = os.getenv("CHESS_STOCKFISH_PATH")
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

# Ensure the Stockfish subprocess is cleaned up on exit to avoid zombie processes.
import atexit
def _shutdown_engine():
    if _ENGINE is not None:
        try:
            _ENGINE.quit()
        except Exception:
            pass
atexit.register(_shutdown_engine)


def _sf_eval_cp(fen: str) -> Optional[int]:
    """Return centipawn evaluation from White's perspective. None = engine unavailable."""
    if _ENGINE is None:
        return None
    try:
        _ENGINE.set_fen_position(fen)
        ev = _ENGINE.get_evaluation()
        is_white_turn = " w " in fen
        # Stockfish python lib returns CP relative to side to move.
        # Normalize to White's perspective (+ = White better).
        if ev["type"] == "cp":
            val = ev["value"]
            return val if is_white_turn else -val
        elif ev["type"] == "mate":
            m = ev["value"]
            # Assign a very high value for mate
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
# Move extraction (backwards-compatible)
# ---------------------------------------------------------------------------

def _extract_moves(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a list of move dicts from either move_history or steps fallback."""
    history = task.get("metadata", {}).get("move_history", [])
    if history:
        return history

    # Fallback: reconstruct from steps array
    moves: List[Dict[str, Any]] = []
    for step in task.get("steps", []):
        if step.get("tool_name") != "make_move":
            continue
        args = step.get("arguments", {})
        uci = args.get("uci_move", "")
        if not uci:
            continue
        
        result_str = str(step.get("result", ""))
        # Only include successful moves (skip illegal move attempts)
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
# Eval bar & evals cache
# ---------------------------------------------------------------------------

def _compute_evals(moves: List[Dict[str, Any]]) -> List[Optional[int]]:
    """Pre-compute Stockfish evals for all positions (starting + after each move)."""
    board = chess.Board()
    positions = [board.fen()]  # starting position
    for mv in moves:
        uci = mv.get("uci", "")
        try:
            board.push_uci(uci)
            positions.append(board.fen())
        except Exception as exc:
            print(f"[visualizer] WARNING: Skipping bad UCI '{uci}' at move {len(positions)}: {exc}")
            positions.append(board.fen())  # keep last valid position

    return [_sf_eval_cp(fen) for fen in positions]


def _eval_bar_html(cp: Optional[int]) -> str:
    """Render a premium vertical eval bar as HTML (White on bottom, Black on top)."""
    if cp is None:
        white_pct = 50.0
        label = "?"
    else:
        # Sigmoid mapping: cp -> 0..100%
        import math
        # 400cp = 73%, 1000cp = 92%
        white_pct = 50 + 50 * (2 / (1 + math.exp(-cp / 400)) - 1)
        white_pct = max(5, min(95, white_pct))
        label = _sf_eval_str(cp)

    black_pct = 100 - white_pct
    
    # Label styling: position it inside the larger part of the bar
    if white_pct >= 50:
        # White winning: label at bottom of black part or top of white part?
        # Let's put it at the "horizon" for small advantages, or fixed for large ones.
        label_pos = f"bottom: {white_pct - 12}%" if white_pct < 85 else "bottom: 10px"
        label_color = "#1c2b38" if white_pct > 60 else "#ffffff"
    else:
        label_pos = f"top: {black_pct - 12}%" if black_pct < 85 else "top: 10px"
        label_color = "#ffffff" if black_pct > 60 else "#1c2b38"

    return f"""
<div style="display:flex;flex-direction:column;align-items:center;height:450px;width:32px;border-radius:4px;overflow:hidden;border:1px solid #444;position:relative;background:#1c2b38;box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
  <div style="width:100%;height:{black_pct:.1f}%;background:#1c2b38;transition:height 0.4s cubic-bezier(0.4, 0, 0.2, 1);"></div>
  <div style="width:100%;height:{white_pct:.1f}%;background:#e8d5b0;transition:height 0.4s cubic-bezier(0.4, 0, 0.2, 1);position:relative;">
  </div>
  <span style="position:absolute;{label_pos};left:50%;transform:translateX(-50%);font-size:11px;font-weight:900;color:{label_color};z-index:10;pointer-events:none;text-shadow: 0 0 2px rgba(0,0,0,0.2);">{label}</span>
</div>"""


# ---------------------------------------------------------------------------
# Result banner
# ---------------------------------------------------------------------------

_RESULT_LABELS = {
    "checkmate_white":           ("♔ White wins by Checkmate",              "#1d5c2e"),
    "checkmate_black":           ("♚ Black wins by Checkmate",              "#1a2b5e"),
    "resign_white":              ("♚ Black wins — White Resigned",           "#1a2b5e"),
    "resign_black":              ("♔ White wins — Black Resigned",           "#1d5c2e"),
    "dq_illegal_white":          ("⚠️ White Disqualified — 2 Illegal Moves", "#6b2d2d"),
    "dq_illegal_black":          ("⚠️ Black Disqualified — 2 Illegal Moves", "#6b2d2d"),
    "dq_eval_abuse_white":       ("⚠️ White Disqualified — Eval Abuse",      "#6b2d2d"),
    "dq_eval_abuse_black":       ("⚠️ Black Disqualified — Eval Abuse",      "#6b2d2d"),
    # python-chess Termination enum names (lowercased) — primary keys
    "draw_stalemate":            ("½-½  Draw by Stalemate",                  "#3a4a5c"),
    "draw_threefold_repetition": ("½-½  Draw by Repetition",                "#3a4a5c"),
    "draw_fifty_moves":          ("½-½  Draw by 50-Move Rule",              "#3a4a5c"),
    "draw_insufficient_material":("½-½  Draw — Insufficient Material",      "#3a4a5c"),
    "draw_variant_end":          ("½-½  Draw (Variant End)",                "#3a4a5c"),
    # Legacy short-form aliases kept for backwards compat with older logs
    "draw_repetition":           ("½-½  Draw by Repetition",                "#3a4a5c"),
    "draw_50_move":              ("½-½  Draw by 50-Move Rule",              "#3a4a5c"),
    "unfinished":                ("⏳  Game did not finish (ply limit reached)", "#5c4a1a"),
}



def _result_banner(result_str: Optional[str], at_end: bool) -> str:
    if not at_end or not result_str:
        return ""
    label, color = _RESULT_LABELS.get(result_str, (f"Game Over: {result_str}", "#444"))
    return f"""
<div style="background:{color};color:#fff;padding:10px 18px;border-radius:6px;font-size:15px;font-weight:700;text-align:center;margin-bottom:8px;">
  {label}
</div>"""


# ---------------------------------------------------------------------------
# Move list HTML
# ---------------------------------------------------------------------------

def _move_list_html(moves: List[Dict[str, Any]], current_ply: int) -> str:
    """Render a scrollable SAN move list with the current ply highlighted."""
    board = chess.Board()
    san_moves: List[str] = []
    for mv in moves:
        uci = mv.get("uci", "")
        # Bug 3 fix: always push the move so the board stays in sync even when
        # board.san() raises (e.g. for valid-UCI but engine-illegal edge cases).
        # board.san() is called inside try; push happens in finally.
        move_obj = None
        try:
            move_obj = chess.Move.from_uci(uci)
            san = board.san(move_obj)
        except Exception:
            san = uci  # fallback to raw UCI for display only
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
        w_style = "background:#d4edda;border-radius:3px;padding:1px 4px;font-weight:bold;color:#155724;" if w_ply == current_ply else "color:#333;"
        b_style = "background:#d4edda;border-radius:3px;padding:1px 4px;font-weight:bold;color:#155724;" if b_ply == current_ply else "color:#333;"

        rows.append(
            f"<tr>"
            f"<td style='color:#6c757d;padding:2px 6px;user-select:none;'>{move_num}.</td>"
            f"<td style='padding:2px 6px;font-family:monospace;cursor:default;{w_style}'>{white_san}</td>"
            f"<td style='padding:2px 6px;font-family:monospace;cursor:default;{b_style}'>{black_san}</td>"
            f"</tr>"
        )

    # Bug 8 fix: do NOT embed onload or <script> — Gradio 4.x strips both.
    # The auto-scroll JS is wired via slider_ply.change(..., js=...) in build_app().
    # Store the target row index as a data attribute for the JS to read.
    scroll_to_row = max(0, current_ply // 2 - 3)
    scroll_px = scroll_to_row * 24

    table = f"<table style='border-collapse:collapse;width:100%;font-size:13px;'>{''.join(rows)}</table>"
    return f"""
<div id="movelist" data-scrolltop="{scroll_px}" style="height:280px;overflow-y:auto;background:#f8f9fa;border-radius:6px;padding:6px;border:1px solid #dee2e6;scrollbar-width:thin;">
  {table}
</div>"""


# ---------------------------------------------------------------------------
# Core render function
# ---------------------------------------------------------------------------

def render_step(
    ply: int,
    game_name: str,
    data: Dict[str, Any],
    evals_cache: List[Optional[int]],
) -> Tuple[str, str, str, str, str, str, str, str]:
    """Render board, eval bar, reasoning, move list. Returns 8 values."""
    empty = ("", "", "No game loaded", "Select a game to begin", "", "", "", "")
    if not game_name or not data or ply < 0:
        return empty

    tasks = data.get("tasks", [])
    try:
        game_idx = int(game_name.split()[1]) - 1
        task = tasks[game_idx]
    except (IndexError, ValueError):
        return empty

    moves = _extract_moves(task)
    total_plies = len(moves)
    at_end = ply >= total_plies and total_plies > 0

    # --- Build board ---
    board = chess.Board()
    current_move_info: Optional[Dict[str, Any]] = None
    
    # We want to show the board AFTER 'ply' moves have been played.
    # If ply=0, it's the start.
    for i in range(min(ply, total_plies)):
        uci = moves[i].get("uci", "")
        try:
            board.push_uci(uci)
        except Exception:
            pass
        if i == ply - 1:
            current_move_info = moves[i]

    lastmove = board.peek() if board.move_stack else None
    board_svg = chess.svg.board(board=board, size=450, lastmove=lastmove)
    board_html = f"<div style='display:flex;justify-content:center;filter: drop-shadow(0 10px 20px rgba(0,0,0,0.4));'>{board_svg}</div>"

    # --- Eval bar ---
    cp = evals_cache[ply] if evals_cache and ply < len(evals_cache) else None
    eval_bar = _eval_bar_html(cp)

    # --- Result banner ---
    result_str = task.get("metadata", {}).get("result")
    banner = _result_banner(result_str, at_end)

    # --- Move list ---
    move_list = _move_list_html(moves, ply)

    # --- Status & Reasoning ---
    status = f"Ply {ply} / {total_plies}  |  {'Game Start' if ply == 0 else ''}"
    threat = candidates = justification = engine_eval = ""

    if current_move_info and ply > 0:
        color = current_move_info.get("color", "").lower()
        uci = current_move_info.get("uci", "")
        model_name = data.get(f"model_{color}", "?")
        cp_loss = current_move_info.get("cp_loss", "N/A")
        mv_score = current_move_info.get("move_score", "N/A")

        status = f"Ply {ply}/{total_plies}  ·  {color.capitalize()} ({model_name})  ·  {uci}"
        engine_eval = f"CP Loss: {cp_loss}   |   Move Score: {mv_score}   |   SF Eval: {_sf_eval_str(cp)}"

        # Find the reasoning step that matches this move and ply.
        # Note: server 'ply' in steps might include illegal attempts, so we match by color + move.
        steps = task.get("steps", [])
        # Iterate backwards to find the ACTUAL successful move attempt if there were multiple.
        for step in reversed(steps):
            if step.get("tool_name") == "make_move":
                args = step.get("arguments", {})
                # Match by uci_move and color. 
                # We also check if the step's index roughly corresponds to our ply if possible.
                if args.get("uci_move") == uci and step.get("color") == color:
                    threat = args.get("threat_analysis", "No threat analysis provided.")
                    cand_list = args.get("candidate_moves", [])
                    candidates = "\n".join(f"• {c}" for c in cand_list) if isinstance(cand_list, list) else str(cand_list)
                    justification = args.get("justification", "No justification provided.")
                    break

    combined_board = f"{banner}{board_html}"
    return eval_bar, combined_board, move_list, status, threat, candidates, justification, engine_eval


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_log(file_obj) -> Tuple[Any, Dict[str, Any]]:
    if file_obj is None:
        return gr.update(choices=[], value=None), {}
    try:
        with open(file_obj.name, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return gr.update(choices=[], value=None), {}

    tasks = data.get("tasks", [])
    names = [f"Game {i + 1}" for i in range(len(tasks))]
    return gr.update(choices=names, value=names[0] if names else None), data


def _get_task(game_name: str, data: Dict[str, Any]):
    tasks = data.get("tasks", [])
    try:
        idx = int(game_name.split()[1]) - 1
        return tasks[idx]
    except Exception:
        return None


def load_game(game_name: str, data: Dict[str, Any]) -> Tuple:
    """Called when game dropdown changes. Returns slider update + initial render."""
    if not game_name or not data:
        return (gr.update(maximum=0, value=0), 0, [], *[""] * 8)

    task = _get_task(game_name, data)
    if task is None:
        return (gr.update(maximum=0, value=0), 0, [], *[""] * 8)

    moves = _extract_moves(task)
    total = len(moves)

    # Pre-compute evals (may take a few seconds)
    evals = _compute_evals(moves)

    slider_update = gr.update(minimum=0, maximum=total, value=0, step=1)
    render_out = render_step(0, game_name, data, evals)
    # Return total as max_plies so buttons (go_next/go_end) know the true limit.
    return (slider_update, total, evals, *render_out)


def on_slider(ply: int, game_name: str, data: Dict[str, Any], evals: List) -> Tuple:
    return render_step(ply, game_name, data, evals)


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="OpenEnv Chess Visualizer") as app:

        # --- Hidden state ---
        log_data = gr.State({})
        evals_cache = gr.State([])
        max_plies = gr.State(0)

        # --- Header ---
        gr.Markdown(
            "# ♟️ OpenEnv Chess — Analysis Board\n"
            "*Upload a JSON log from a self-play run. Step through moves and inspect LLM reasoning.*"
        )

        with gr.Row():
            file_input = gr.File(label="Upload JSON Log", file_types=[".json"])
            game_dropdown = gr.Dropdown(label="Select Game", choices=[], interactive=True, scale=1)
            sf_status = gr.Markdown(
                f"**Stockfish:** {'✅ Ready (depth ' + str(SF_DEPTH) + ')' if _ENGINE else '❌ Not found — eval bar disabled'}",
                elem_id="stockfish-status"
            )

        # --- Main layout ---
        with gr.Row(equal_height=True):

            # Eval bar (narrow column)
            with gr.Column(scale=0, min_width=50):
                eval_bar_html = gr.HTML(label="Eval", elem_id="eval-bar")

            # Board + controls
            with gr.Column(scale=3, elem_id="board-col"):
                board_html = gr.HTML(label="Board")

                with gr.Row():
                    btn_start = gr.Button("⏮", scale=0, min_width=48)
                    btn_prev  = gr.Button("◀", scale=0, min_width=48)
                    btn_next  = gr.Button("▶", scale=0, min_width=48)
                    btn_end   = gr.Button("⏭", scale=0, min_width=48)
                    btn_auto  = gr.Button("▶ Auto-play", scale=1)

                slider_ply = gr.Slider(
                    label="Ply",
                    minimum=0, maximum=0, step=1, value=0,
                    interactive=True,
                )
                status_box = gr.Textbox(
                    label="Move", interactive=False, lines=1,
                )

            # Move list
            with gr.Column(scale=2):
                move_list_html = gr.HTML(label="Move List")

        # --- Reasoning panel ---
        gr.Markdown("### 🧠 LLM Reasoning")
        with gr.Row():
            threat_box        = gr.Textbox(label="Threat Analysis",  lines=3, interactive=False)
            candidates_box    = gr.Textbox(label="Candidate Moves",  lines=3, interactive=False)
            justification_box = gr.Textbox(label="Justification",    lines=3, interactive=False)
        engine_box = gr.Textbox(label="Engine Evaluation", lines=1, interactive=False)

        # All render outputs in order
        _render_outputs = [
            eval_bar_html, board_html, move_list_html,
            status_box, threat_box, candidates_box, justification_box, engine_box,
        ]

        # ---- Wiring ----

        # 1. Upload → parse → populate dropdown
        file_input.upload(
            fn=parse_log,
            inputs=[file_input],
            outputs=[game_dropdown, log_data],
            show_progress="hidden",
        )

        # 2. Game select → pre-compute evals + render ply 0
        game_dropdown.change(
            fn=load_game,
            inputs=[game_dropdown, log_data],
            outputs=[slider_ply, max_plies, evals_cache, *_render_outputs],
            show_progress="hidden",
        )

        # 3. Slider → render + Bug 8 fix: use Gradio's native JS hook for auto-scroll
        # so the movelist div scrolls to the highlighted row.  Gradio 4.x strips
        # inline <script> and onload attributes from gr.HTML, but js= on event
        # handlers runs in the browser before the Python round-trip.
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
            outputs=_render_outputs,
            show_progress="hidden",
            js=_SCROLL_JS,
        )

        # 4. Navigation buttons
        def go_start(*_):
            return gr.update(value=0)

        def go_end(_, mp):
            return gr.update(value=mp)

        def go_prev(ply):
            return gr.update(value=max(0, ply - 1))

        def go_next(ply, mp):
            return gr.update(value=min(mp, ply + 1))

        btn_start.click(fn=go_start, inputs=[slider_ply, max_plies], outputs=[slider_ply], show_progress="hidden")
        btn_prev.click(fn=go_prev,  inputs=[slider_ply],             outputs=[slider_ply], show_progress="hidden")
        btn_next.click(fn=go_next,  inputs=[slider_ply, max_plies],  outputs=[slider_ply], show_progress="hidden")
        btn_end.click(fn=go_end,    inputs=[slider_ply, max_plies],  outputs=[slider_ply], show_progress="hidden")

        # 5. Auto-play using gr.Timer (Gradio 4.x) or polling fallback
        # We inject a JS-based auto-stepper that clicks the ▶ button every 1.2s
        # when "Auto-play" is active. We use a toggle state to start/stop.
        autoplay_state = gr.State(False)

        def toggle_autoplay(active: bool) -> Tuple[bool, str]:
            new_state = not active
            label = "⏹ Stop" if new_state else "▶ Auto-play"
            return new_state, label

        btn_auto.click(
            fn=toggle_autoplay,
            inputs=[autoplay_state],
            outputs=[autoplay_state, btn_auto],
            show_progress="hidden",
        )

        # Timer tick: advances ply by 1 when autoplay is on
        def autoplay_tick(active: bool, ply: int, mp: int):
            if active and ply < mp:
                return gr.update(value=ply + 1)
            elif active and ply >= mp:
                # Stop at end
                return gr.update(value=ply)
            return gr.update(value=ply)

        timer = gr.Timer(value=1.2, active=False)
        autoplay_state.change(
            fn=lambda a: gr.Timer(value=1.2, active=a),
            inputs=[autoplay_state],
            outputs=[timer],
        )
        timer.tick(
            fn=autoplay_tick,
            inputs=[autoplay_state, slider_ply, max_plies],
            outputs=[slider_ply],
            show_progress="hidden",
        )



    return app


if __name__ == "__main__":
    _css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&family=Outfit:wght@400;600;700&display=swap');
    
    body { background: #0f172a; font-family: 'Inter', sans-serif; color: #f8fafc; }
    .gradio-container { background: #0f172a !important; color: #f8fafc !important; }
    
    /* Ensure all text on dark background is pure white */
    .gradio-container, .gradio-container p, .gradio-container span, .gradio-container .markdown-text, .gradio-container .prose { 
        color: #ffffff !important; 
    }
    
    /* LABELS & TITLES IN LIGHT BOXES - Force Black Text */
    .gradio-container .label, 
    .gradio-container .label span, 
    .gradio-container .block-title, 
    .gradio-container .block-title span, 
    .gradio-container .gr-label,
    .gradio-container .gr-box-label {
        color: #000000 !important;
        background: #e2e8f0 !important; /* Slightly more solid bluish-white */
        font-weight: 900 !important;
        text-transform: uppercase;
        font-size: 11px;
    }

    /* Stockfish Status specific fix */
    #stockfish-status, #stockfish-status p, #stockfish-status span {
        color: #4ade80 !important; /* Brighter green */
        font-weight: 600 !important;
    }
    
    /* Title and headers */
    .gradio-container h1, .gradio-container h2, .gradio-container h3 { 
        color: #ffffff !important; 
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    
    #board-col { 
        display: flex; 
        flex-direction: column; 
        align-items: center; 
        padding: 24px; 
        background: rgba(30, 41, 59, 0.8); 
        border-radius: 16px; 
        border: 1px solid rgba(255,255,255,0.15); 
        backdrop-filter: blur(16px);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
    }
    
    textarea, input[type=text] { 
        background: #1e293b !important; 
        color: #ffffff !important; 
        border: 1px solid #475569 !important; 
        border-radius: 8px !important;
    }
    
    .gradio-container .form { background: #1e293b !important; border: 1px solid #475569 !important; }
    
    button { 
        background: #334155 !important; 
        color: #ffffff !important; 
        border: 1px solid #475569 !important; 
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
    }
    button:hover { background: #475569 !important; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
    button:active { transform: translateY(0); }
    
    /* Auto-play pulse effect */
    .btn-active { animation: pulse 2s infinite; background: #15803d !important; border-color: #22c55e !important; }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(34, 197, 94, 0); }
        100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }
    """
    _theme = gr.themes.Soft(
        primary_hue="indigo",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#0f172a",
        block_background_fill="#1e293b",
        block_border_width="1px",
        block_title_text_color="#f8fafc",
        input_background_fill="#0f172a",
    )
    
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=_theme,
        css=_css,
    )
