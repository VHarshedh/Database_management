import json
import os
from pathlib import Path
from typing import Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

def load_latest_results(results_dir: Path) -> dict[str, Any]:
    files = list(results_dir.glob("soc_orchestrator_*.json"))
    if not files:
        raise FileNotFoundError(f"No result files found in {results_dir}")
    latest_file = max(files, key=os.path.getmtime)
    with open(latest_file, "r", encoding="utf-8") as f:
        return json.load(f)

def summarize():
    console = Console()
    results_dir = Path("results")
    
    try:
        data = load_latest_results(results_dir)
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        return

    audit_trail = data.get("audit_trail", [])
    regions = data.get("regions", [])

    # 1. Headline
    who_won = "Stalemate"
    for r in regions:
        if r.get("done"):
            res = r.get("result", "")
            if "dq_illegal_defender" in res or "resign_defender" in res:
                who_won = "Adversary Swarm Victory"
                break
            elif "dq_illegal_adversary" in res or "resign_adversary" in res:
                who_won = "Defender Victory"
                break

    console.print(Panel(f"[bold cyan]Headline:[/] [bold green]{who_won}[/]", title="Executive SOC Incident Report", expand=False))

    # 2. Tier Analysis
    profile_rewards = {}
    profile_counts = {}

    for cycle in audit_trail:
        # Include both defender and adversary actions
        for swarm_key in ["defender_swarm", "adversary_swarm"]:
            for action in cycle.get(swarm_key, []):
                profile = action.get("profile", "unknown")
                reward = action.get("reward", 0.0)
                profile_rewards[profile] = profile_rewards.get(profile, 0.0) + reward
                profile_counts[profile] = profile_counts.get(profile, 0) + 1

    table = Table(title="Tier Analysis: Performance Gradient", header_style="bold magenta")
    table.add_column("Agent Role", style="cyan")
    table.add_column("Avg Reward", justify="right", style="green")
    table.add_column("Actions", justify="right", style="dim")

    for profile in sorted(profile_rewards.keys()):
        avg = profile_rewards[profile] / profile_counts[profile]
        table.add_row(profile, f"{avg:.4f}", str(profile_counts[profile]))

    console.print(table)

    # 3. Layer Failures
    layer_4_hallucinations = 0
    layer_6_shadow_traps = 0

    for cycle in audit_trail:
        for cand_key in ["defender_all_candidates", "adversary_all_candidates"]:
            for cand in cycle.get(cand_key, []):
                # Layer 4: Hallucinated Nodes (invalid target_node or resolution failure)
                if cand.get("error") == "invalid target_node or canonical resolution failure":
                    layer_4_hallucinations += 1
                # Layer 6: Shadow Traps (damage_score == -100.0)
                if cand.get("damage_score") == -100.0:
                    layer_6_shadow_traps += 1

    failure_table = Table(title="Security Violation Metrics")
    failure_table.add_column("Violation Type", style="yellow")
    failure_table.add_column("Count", justify="right", style="red")
    failure_table.add_row("Layer 4: Hallucinated Nodes", str(layer_4_hallucinations))
    failure_table.add_row("Layer 6: Shadow Traps", str(layer_6_shadow_traps))
    
    console.print(failure_table)

    # 4. Hero Move
    hero_move = None
    max_reward = -1.0

    for cycle in audit_trail:
        for swarm_key in ["defender_swarm", "adversary_swarm"]:
            for action in cycle.get(swarm_key, []):
                reward = action.get("reward", 0.0)
                if reward > max_reward:
                    max_reward = reward
                    hero_move = action

    if hero_move:
        args = hero_move.get("arguments", {})
        console.print(Panel(
            f"[bold cyan]Profile:[/] {hero_move.get('profile')}\n"
            f"[bold cyan]Reward:[/] [bold green]{hero_move.get('reward'):.2f}[/]\n"
            f"[bold cyan]Threat Analysis:[/] {args.get('threat_analysis')}\n"
            f"[bold cyan]Justification:[/] {args.get('justification')}",
            title="The 'Hero' Move",
            border_style="gold1"
        ))
    else:
        console.print("[dim italic]No hero moves recorded (all rewards was floor).[/]")

if __name__ == "__main__":
    summarize()
