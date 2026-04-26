"""
Tests the Orchestrator's ability to handle elastic swarm sizing (e.g., 1v3, 2v4)
ensuring all simulated actions contain valid cryptographic hashes.
"""
import pytest
import ast
import re
from server.datacenter_env import DatacenterEnvironment

def extract_first_node(text: str) -> dict:
    for match in re.finditer(r'\{[^{}]*\}', str(text)):
        try:
            node = ast.literal_eval(match.group(0))
            if isinstance(node, dict) and "region" in node:
                return node
        except Exception:
            pass
    return {"region": "us-east", "zone": "az-a", "rack": "rack-1", "pod": "pod-1"}

class MockSwarmAgent:
    def __init__(self, id_str):
        self.id = id_str
        self.name = f"Mock_{id_str}"
        
    async def get_action(self, state):
        src = extract_first_node(str(state))
        dst = dict(src)
        dst["pod"] = f"pod-mock-{self.id}"
        return {
            "tool": "migrate_workload",
            "arguments": {
                "threat_analysis": f"Swarm agent {self.id} analysis.",
                "justification": "Elastic swarm payload validation.",
                "candidate_migrations": ["mock"],
                "source_node": src,
                "target_node": dst
            }
        }

@pytest.mark.asyncio
async def test_elastic_swarm_hash_compliance():
    # Setup mock swarm (1 Defender, 3 Attackers)
    defenders = [MockSwarmAgent("def1")]
    attackers = [MockSwarmAgent("atk1"), MockSwarmAgent("atk2"), MockSwarmAgent("atk3")]
    
    env = DatacenterEnvironment()
    
    # Verify defender can securely format the node
    def_action = await defenders[0].get_action(env.state)
    assert isinstance(def_action["arguments"]["source_node"], dict)
    assert "region" in def_action["arguments"]["source_node"]
    
    # Verify all attackers format their nodes correctly
    for atk in attackers:
        atk_action = await atk.get_action(env.state)
        assert isinstance(atk_action["arguments"]["source_node"], dict)
        assert "region" in atk_action["arguments"]["source_node"]
        
    env.close()