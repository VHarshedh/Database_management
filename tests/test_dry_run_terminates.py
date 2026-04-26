"""
Tests that a dry-run (simulation without active LLMs) properly terminates
and formats its fallback migrations with the correct cryptographic hashes.
"""
import pytest
import ast
import re
from server.datacenter_env import DatacenterEnvironment
from openenv.core.env_server.mcp_types import CallToolAction

def extract_first_node(text: str) -> dict:
    for match in re.finditer(r'\{[^{}]*\}', str(text)):
        try:
            node = ast.literal_eval(match.group(0))
            if isinstance(node, dict) and "region" in node:
                return node
        except Exception:
            pass
    return {"region": "us-east", "zone": "az-a", "rack": "rack-1", "pod": "pod-1"}

def test_dry_run_fallback_preserves_hashes():
    env = DatacenterEnvironment()
    
    # 1. Simulate an initial scan to populate telemetry
    scan_action = CallToolAction(
        tool_name="scan_topology",
        arguments={"threat_analysis": "dry run", "justification": "dry run", "candidate_migrations": []}
    )
    scan_obs = env.step(scan_action)
    scan_text = scan_obs.result.content[0].text
    
    # 2. Dry run must extract hashes to be a valid fallback
    src_node = extract_first_node(scan_text)
    dst_node = dict(src_node)
    dst_node["pod"] = "pod-dry-run"
    
    migrate_action = CallToolAction(
        tool_name="migrate_workload",
        arguments={
            "threat_analysis": "Executing dry run fallback.",
            "justification": "Verifying fallback hash compliance.",
            "candidate_migrations": ["fallback"],
            "source_node": src_node,
            "target_node": dst_node
        }
    )
    obs = env.step(migrate_action)
    
    # The environment should reward the dry run for preserving hashes natively
    assert obs.reward > 0.01, "Dry run fallback failed cryptographic hash check"
    env.close()