import os
import sys
import pytest
import ast
import re

# PATH INJECTION: Allow imports from sibling 'server' folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from server.datacenter_env import DatacenterEnvironment
    from openenv.core.env_server.mcp_types import CallToolAction
except ImportError:
    print("Error: Could not find 'server' module. Ensure you are running from the project root.")
    sys.exit(1)

def extract_all_nodes(text: str) -> list[dict]:
    """Extracts all valid 4D node dictionaries found in the text, including dynamic hashes."""
    nodes = []
    # Match dictionary-like structures
    for match in re.finditer(r'\{[^{}]*\}', str(text)):
        try:
            node = ast.literal_eval(match.group(0))
            if isinstance(node, dict) and "region" in node:
                nodes.append(node)
        except Exception:
            pass
    return nodes

class ChaosStubAgent:
    def __init__(self, role="defender"):
        self.role = role
        self.id = f"chaos_{role}"
        self.name = f"Chaos{role.capitalize()}"

    async def get_action(self, state):
        # Extract all nodes from the state telemetry
        nodes = extract_all_nodes(str(state))
        
        if len(nodes) >= 2:
            # Pick two real nodes to ensure they exist in the live topology grid
            src = nodes[0]
            dst = nodes[-1]
        else:
            # Fallback if telemetry is sparse
            src = nodes[0] if nodes else {"region": "us-east", "zone": "az-a", "rack": "rack-1", "pod": "pod-1"}
            dst = dict(src)
            # Try to stay within the same rack to increase chance of valid node existence if we have to guess
            dst["pod"] = "pod-2" 

        return {
            "tool": "migrate_workload",
            "arguments": {
                "threat_analysis": "Chaos testing in progress...",
                "justification": "Verifying stability under pseudo-random conditions.",
                "candidate_migrations": [f"{src.get('pod')}->{dst.get('pod')}"],
                "source_node": src,
                "target_node": dst
            }
        }

@pytest.mark.asyncio
async def test_chaos_agent_avoids_penalty():
    print("\n[TEST] Initializing Chaos Environment...")
    env = DatacenterEnvironment()
    
    # CRITICAL FIX: Manually register the instance so tools like migrate_workload can find it
    DatacenterEnvironment._latest_instance = env 
    
    agent = ChaosStubAgent()
    
    # 1. Feed the agent the raw environment state
    print("[TEST] Getting agent action...")
    action_dict = await agent.get_action(env.state)
    
    # 2. Execute against environment
    print("[TEST] Executing tool call...")
    action = CallToolAction(
        tool_name=action_dict["tool"],
        arguments=action_dict["arguments"]
    )
    obs = env.step(action)
    
    # Log the result for debugging
    print(f"[RESULT] Reward: {obs.reward}")
    if obs.reward == 0.01:
        print(f"[DEBUG] Observation Metadata: {obs.metadata}")
        
    # Should not get the 0.01 floor now that the agent uses real nodes from the grid
    assert obs.reward > 0.01, f"Agent hit the floor! Message: {obs.metadata.get('message', 'No error message')}"
    
    env.close()
    print("[PASS] Chaos agent test successful.")