import json
import sys
import time
import os
from pathlib import Path
from typing import Any, Optional, Dict, List
from dataclasses import dataclass

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.columns import Columns
    from rich.align import Align
    from rich.theme import Theme
except ImportError:
    print("Error: 'rich' library not found. Please install it with 'pip install rich'.")
    sys.exit(1)

# Custom theme for SOC Aesthetics
SOC_THEME = Theme({
    "layer1": "bold green",
    "layer3": "bold yellow",
    "layer5": "bold magenta",
    "layer6": "bold red",
    "defender": "cyan",
    "adversary": "red",
    "threat_high": "bold red",
    "threat_med": "yellow",
    "threat_low": "green",
    "status_secured": "bold green",
    "status_compromised": "bold red",
    "status_active": "cyan",
})

console = Console(theme=SOC_THEME)

def get_latest_result() -> Path:
    results_dir = Path("results")
    if not results_dir.exists():
        raise FileNotFoundError("Results directory not found.")
    
    files = list(results_dir.glob("soc_orchestrator_*.json"))
    if not files:
        raise FileNotFoundError("No SOC result files found in results/.")
    
    # Sort by modification time
    return max(files, key=os.path.getmtime)

def format_node(node: Any) -> str:
    if not isinstance(node, dict):
        return "Unknown"
    return f"{node.get('region','?')}/{node.get('zone','?')}/{node.get('rack','?')}/{node.get('pod','?')}"

def get_layer_info(action: dict) -> str:
    tool = action.get("tool", "")
    if tool == "scan_topology":
        return "[layer1]Layer 1: Sensors Active[/]"
    if tool == "enumerate_authorized_migrations":
        return "[layer2]Layer 2: Policy Mapping[/]"
    if tool == "escalate_to_sysadmin":
        return "[layer5]Layer 5: HITL Escalation[/]"
    
    # Layer 6 Traps (Shadow Nodes)
    # This might be in the arguments or errors if we had it
    return ""

def render_region_bar(regions: List[dict]) -> Table:
    table = Table.grid(expand=True)
    for i in range(len(regions)):
        table.add_column(justify="center", ratio=1)
    
    status_panels = []
    for r in regions:
        name = r.get("region_name", r.get("region_id", "Unknown"))
        threat = r.get("scores", {}).get("adversary_threat_level", 0.0)
        done = r.get("done", False)
        result = r.get("result", "")
        
        threat_color = "threat_low"
        if threat > 0.7: threat_color = "threat_high"
        elif threat > 0.4: threat_color = "threat_med"
        
        status_text = "ACTIVE"
        status_style = "status_active"
        
        if done:
            if "adversary" in result.lower():
                status_text = "COMPROMISED"
                status_style = "status_compromised"
            else:
                status_text = "SECURED"
                status_style = "status_secured"
        
        panel_content = Text.assemble(
            (f"{name.upper()}\n", "bold"),
            (f"STATUS: {status_text}\n", status_style),
            (f"THREAT: {threat*100:.1f}%", threat_color)
        )
        status_panels.append(Panel(Align.center(panel_content), border_style=status_style))
    
    table.add_row(*status_panels)
    return table

def render_cycle_dashboard(cycle_idx: int, regions: List[dict], log_entries: List[Text]) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="status", size=6),
        Layout(name="log", minimum_size=10)
    )
    
    layout["header"].update(Panel(Align.center(f"GLOBAL SOC SIMULATION - CYCLE {cycle_idx}"), style="bold white on blue"))
    layout["status"].update(render_region_bar(regions))
    
    log_table = Table(box=None, expand=True)
    log_table.add_column("TELEMETRY LOG", style="dim")
    for entry in log_entries[-15:]: # Show last 15 entries
        log_table.add_row(entry)
    
    layout["log"].update(Panel(log_table, title="8-LAYER SIMULATION FEED", border_style="blue"))
    
    return layout

def visualize(file_path: Optional[str] = None, speed: float = 1.0):
    if file_path:
        path = Path(file_path)
    else:
        path = get_latest_result()
    
    console.print(f"[bold cyan]LOADING SOC DATA FROM:[/] {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    regions_snapshot = data.get("regions", [])
    audit_trail = data.get("audit_trail", [])
    
    log_entries: List[Text] = []
    
    # We want to iterate cycle by cycle, but audit_trail is half-moves.
    # Group them or just iterate? The user wants "mimicking a live replay".
    # I'll iterate through the audit_trail directly.
    
    with Live(console=console, auto_refresh=False) as live:
        for record in audit_trail:
            cycle_idx = record.get("cycle_index", 0)
            region_id = record.get("region_id", "unknown")
            
            # Find the region to update threat level in our snapshot (mocking live updates)
            # In a real replay we'd want the threat level at that specific moment, 
            # but the snapshot in JSON is usually the FINAL state. 
            # However, for the visualizer we'll just show the cycle info.
            
            # Handle Defender Actions
            for action in record.get("defender_swarm", []):
                profile = action.get("profile", "defender")
                tool = action.get("tool", "")
                args = action.get("arguments", {})
                reward = action.get("reward", 0.0)
                
                layer_info = get_layer_info(action)
                entry = Text.assemble(
                    (f"[{region_id}] ", "dim"),
                    (f"DEFENDER ({profile}): ", "defender"),
                    (f"{tool} ", "bold"),
                )
                
                if tool == "migrate_workload":
                    src = format_node(args.get("source_node"))
                    dst = format_node(args.get("target_node"))
                    entry.append(f"{src} -> {dst}", style="bold cyan")
                
                if layer_info:
                    entry.append(" ")
                    entry.append(Text.from_markup(layer_info))
                
                log_entries.append(entry)
                
                # Protocol Red Alert
                if tool == "escalate_to_sysadmin":
                    live.update(render_cycle_dashboard(cycle_idx, regions_snapshot, log_entries))
                    live.refresh()
                    time.sleep(0.5)
                    alert_panel = Panel(
                        Align.center("\n[bold blink red]PROTOCOL RED - LAYER 5 HITL ESCALATION[/]\n"),
                        border_style="bold red",
                        padding=(1, 2)
                    )
                    console.print(alert_panel)
                    time.sleep(2.0)

            # Handle Adversary Actions
            for action in record.get("adversary_swarm", []):
                profile = action.get("profile", "adversary")
                tool = action.get("tool", "")
                args = action.get("arguments", {})
                reward = action.get("reward", 0.0)
                
                entry = Text.assemble(
                    (f"[{region_id}] ", "dim"),
                    (f"ADVERSARY ({profile}): ", "adversary"),
                    (f"{tool} ", "bold"),
                )
                
                if tool == "migrate_workload":
                    src = format_node(args.get("source_node"))
                    dst = format_node(args.get("target_node"))
                    entry.append(f"{src} -> {dst}", style="bold red")
                
                log_entries.append(entry)

            # Highlight Errors/Traps
            # Adversary candidates might have errors
            for cand in record.get("adversary_all_candidates", []):
                err = cand.get("error", "")
                if err:
                    if "shadow" in err.lower() or "signature" in err.lower():
                        log_entries.append(Text(f"!! [Layer 6] SHADOW NODE TRAP DETECTED: {err}", style="layer6"))
                    elif "rate" in err.lower():
                        log_entries.append(Text(f"!! [Layer 3] RATE LIMIT EJECTION: {err}", style="layer3"))

            live.update(render_cycle_dashboard(cycle_idx, regions_snapshot, log_entries))
            live.refresh()
            time.sleep(speed)

    console.print("\n[bold green]VISUALIZATION COMPLETE.[/]")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SOC Datacenter Simulation Visualizer")
    parser.add_argument("file", nargs="?", help="Path to the JSON result file")
    parser.add_argument("--speed", type=float, default=0.5, help="Playback speed (seconds per move)")
    
    args = parser.parse_args()
    try:
        visualize(args.file, args.speed)
    except Exception as e:
        console.print(f"[bold red]FATAL ERROR:[/] {e}")
        # import traceback; traceback.print_exc()
