
import ast
import re
from openenv.core.env_server.mcp_types import CallToolAction
from server.datacenter_env import DatacenterEnvironment

def extract_first_node(text: str) -> dict:
    for match in re.finditer(r'\{[^{}]*\}', text):
        try:
            node = ast.literal_eval(match.group(0))
            if isinstance(node, dict) and "region" in node:
                return node
        except Exception:
            pass
    return None

def test_hash_penalty_logic():
    print("Initializing Datacenter Environment for Penalty Testing...")
    env = DatacenterEnvironment()
    
    # 1. Scan Topology to get a valid node with hashes
    action = CallToolAction(
        tool_name="scan_topology",
        arguments={
            "threat_analysis": "Initial recon",
            "candidate_migrations": [],
            "justification": "Mapping grid."
        }
    )
    obs = env.step(action)
    text = obs.result.content[0].text
    print(f"[DEBUG] Topology Text Sample: {text[:500]}...")
    
    # Extract nodes from the 'active_workloads' section
    all_nodes = []
    # Search for "node": { or 'node': {
    for match in re.finditer(r'[\'"]node[\'"]:\s*(\{.*?\})', text, re.DOTALL):
        try:
            node_str = match.group(1)
            # Standardise to single quotes for ast.literal_eval if needed
            # Actually, JSON usually uses double quotes, so we can try json.loads if it looks like JSON
            # or just use ast.literal_eval if it's Python-esque.
            # FastMCP usually produces standard JSON strings.
            import json
            try:
                node = json.loads(node_str)
            except:
                node = ast.literal_eval(node_str)
            if isinstance(node, dict) and "region" in node:
                all_nodes.append(node)
        except Exception:
            pass
            
    if len(all_nodes) < 2:
        print(f"[ERROR] Could not find enough nodes in topology scan. Found: {len(all_nodes)}")
        return

    full_src = all_nodes[0]
    full_tgt = all_nodes[1]
    
    print(f"[DEBUG] Full Src: {full_src}")
    print(f"[DEBUG] Full Tgt: {full_tgt}")
    
    # Create stripped versions
    stripped_src = {k: full_src[k] for k in ["region", "zone", "rack", "pod"]}
    stripped_tgt = {k: full_tgt[k] for k in ["region", "zone", "rack", "pod"]}
    
    # --- CASE A: Full Hashes ---
    print("\n--- CASE A: Full Hashes ---")
    action_a = CallToolAction(
        tool_name="migrate_workload",
        arguments={
            "threat_analysis": "Migration with hashes.",
            "candidate_migrations": ["A->B"],
            "justification": "Testing full compliance.",
            "source_node": full_src,
            "target_node": full_tgt
        }
    )
    obs_a = env.step(action_a)
    reward_a = obs_a.reward
    print(f"Result Reward A: {reward_a}")

    # Reset environment
    env.reset()
    obs = env.step(action) # Re-scan
    text = obs.result.content[0].text
    all_nodes = []
    import json
    for match in re.finditer(r'[\'"]node[\'"]:\s*(\{.*?\})', text, re.DOTALL):
        try:
            node_str = match.group(1)
            try:
                node = json.loads(node_str)
            except:
                node = ast.literal_eval(node_str)
            if isinstance(node, dict) and "region" in node:
                all_nodes.append(node)
        except Exception:
            pass

    full_src = all_nodes[0]
    full_tgt = all_nodes[1]
    stripped_src = {k: full_src[k] for k in ["region", "zone", "rack", "pod"]}
    stripped_tgt = {k: full_tgt[k] for k in ["region", "zone", "rack", "pod"]}

    # --- CASE B: Stripped Hashes ---
    print("\n--- CASE B: Stripped Hashes ---")
    action_b = CallToolAction(
        tool_name="migrate_workload",
        arguments={
            "threat_analysis": "Migration with stripped hashes.",
            "candidate_migrations": ["A->B"],
            "justification": "Testing penalty application.",
            "source_node": stripped_src,
            "target_node": stripped_tgt
        }
    )
    obs_b = env.step(action_b)
    reward_b = obs_b.reward
    print(f"Result Reward B: {reward_b}")
    
    # Verify penalty: B should be significantly lower than A
    # (Assuming the physical outcomes are identical)
    diff = reward_a - reward_b
    print(f"Reward difference: {diff:.3f}")
    if diff > 0.05: # Allowing for some rounding/float noise, but 0.1 is the target
        print("[OK] Penalty detected!")
    else:
        print("[FAIL] Penalty NOT detected.")

    # --- CASE C: Hallucination (Expect 0.01 floor) ---
    print("\n--- CASE C: Hallucination ---")
    hallucinated_node = {"region": "mars-1", "zone": "az-red", "rack": "rack-99", "pod": "pod-void"}
    action_c = CallToolAction(
        tool_name="migrate_workload",
        arguments={
            "threat_analysis": "Hallucinating a node.",
            "candidate_migrations": ["X->Y"],
            "justification": "Testing floor reward.",
            "source_node": hallucinated_node,
            "target_node": full_tgt
        }
    )
    obs_c = env.step(action_c)
    print(f"Result Reward C: {obs_c.reward}")
    if obs_c.reward == 0.01:
        print("[OK] Hallucination floor (0.01) triggered.")
    else:
        print("[FAIL] Hallucination floor NOT triggered.")

    env.close()

if __name__ == "__main__":
    test_hash_penalty_logic()
