import os
import json
import glob
from collections import defaultdict

def generate_readme_summary(results_dir="results", output_file="results.txt"):
    """
    Parses Datacenter SOC Orchestrator JSON trace logs, aggregates
    performance metrics, and outputs a formatted Markdown summary.
    """
    # Look for the soc_orchestrator json files in the target directory
    search_pattern = os.path.join(results_dir, "soc_orchestrator_*.json")
    json_files = glob.glob(search_pattern)
    
    if not json_files:
        print(f"No SOC Orchestrator JSON files found matching '{search_pattern}'.")
        return

    total_regions_simulated = 0
    defender_wins = 0
    adversary_wins = 0
    draws = 0
    
    total_def_eff = 0.0
    total_adv_threat = 0.0
    
    dq_counts = defaultdict(int)

    print(f"Parsing {len(json_files)} SOC match logs...")

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            regions = data.get("regions", [])
            for region in regions:
                total_regions_simulated += 1
                
                # Extract scores
                scores = region.get("scores", {})
                def_eff = scores.get("defender_efficiency", 0.0)
                adv_threat = scores.get("adversary_threat_level", 0.0)
                
                total_def_eff += def_eff
                total_adv_threat += adv_threat
                
                # Extract outcomes safely (handle explicit nulls in JSON)
                result = region.get("result")
                if result is None:
                    result = "unknown"
                
                if "dq_violation_defender" in result or "adversary_win" in result or "dq_defender" in result:
                    adversary_wins += 1
                elif "dq_violation_adversary" in result or "defender_win" in result or "dq_adversary" in result:
                    defender_wins += 1
                else:
                    # Fallback heuristic based on final scores
                    if def_eff > adv_threat:
                        defender_wins += 1
                    elif adv_threat > def_eff:
                        adversary_wins += 1
                    else:
                        draws += 1
                        
                # Tally specific disqualifications (like the one in your uploaded JSON)
                if "dq_" in result:
                    dq_counts[result] += 1

        except Exception as e:
            print(f"Warning: Failed to parse {file_path} - {e}")

    if total_regions_simulated == 0:
        print("No valid region data found in logs.")
        return

    # Calculate Averages & Win Rates
    avg_def_eff = total_def_eff / total_regions_simulated
    avg_adv_threat = total_adv_threat / total_regions_simulated
    
    def_win_rate = (defender_wins / total_regions_simulated) * 100
    adv_win_rate = (adversary_wins / total_regions_simulated) * 100

    # Generate Markdown Output
    md_lines = []
    md_lines.append("### Automated Datacenter SOC Benchmark Results")
    md_lines.append("")
    md_lines.append("The following metrics are aggregated directly from the raw `soc_orchestrator` JSON traces. Metrics reflect the continuous struggle between the Defender (SOC Architect) and the Adversary Swarm across all simulated datacenter regions.")
    md_lines.append("")
    md_lines.append(f"**Total Regions Simulated:** {total_regions_simulated}")
    md_lines.append("")
    md_lines.append("| Metric | Score / Rate |")
    md_lines.append("| :--- | :---: |")
    md_lines.append(f"| **Defender Win Rate** | {def_win_rate:.1f}% |")
    md_lines.append(f"| **Adversary Win Rate** | {adv_win_rate:.1f}% |")
    md_lines.append(f"| **Avg Defender Efficiency** | {avg_def_eff:.3f} |")
    md_lines.append(f"| **Avg Adversary Threat Level** | {avg_adv_threat:.3f} |")
    
    # Optional DQ Table if any agents hallucinated or broke rules
    if dq_counts:
        md_lines.append("")
        md_lines.append("#### Disqualification (DQ) Breakdown")
        md_lines.append("| Reason | Count |")
        md_lines.append("| :--- | :---: |")
        for reason, count in sorted(dq_counts.items(), key=lambda x: x[1], reverse=True):
            md_lines.append(f"| `{reason}` | {count} |")

    md_lines.append("")
    md_lines.append("*Metrics calculated automatically via `summarize_results.py`.*")

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))

    print(f"\n✅ Successfully aggregated {total_regions_simulated} region simulations.")
    print(f"✅ Results written to '{output_file}'. You can now copy this into your README.md.")

if __name__ == "__main__":
    # Point this to wherever your JSON logs are dumped
    generate_readme_summary(results_dir="results")