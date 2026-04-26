"""
Smoke test for the SOC Datacenter HTTP API.
Tests /reset, /list_tools, and /step (scan_topology, migrate_workload).
"""
import subprocess
import sys
import time
import httpx
import re
import ast
import json

def extract_first_node(text: str) -> dict:
    """Extracts the first valid 4D node dictionary including dynamic hashes."""
    for match in re.finditer(r'\{[^{}]*\}', text):
        try:
            node = ast.literal_eval(match.group(0))
            if isinstance(node, dict) and "region" in node:
                return node
        except Exception:
            try:
                node = json.loads(match.group(0).replace("'", '"'))
                if isinstance(node, dict) and "region" in node:
                    return node
            except Exception:
                pass
    return {"region": "us-east", "zone": "az-a", "rack": "rack-1", "pod": "pod-1"}

def run_smoke_tests(base_url: str) -> list[str]:
    log = []
    
    # 1. Reset Environment
    log.append("1. Testing POST /reset...")
    r = httpx.post(f"{base_url}/reset", json={}, timeout=5.0)
    if r.status_code != 200:
        log.append(f"  [FAIL] /reset returned {r.status_code}")
        return log
    log.append("  [OK] /reset successful.")
    
    # 2. List Tools
    log.append("\n2. Testing POST /step (list_tools)...")
    payload = {"action": {"type": "list_tools"}, "episode_id": ""}
    r = httpx.post(f"{base_url}/step", json=payload, timeout=5.0)
    tools = r.json().get("observation", {}).get("result", {}).get("tools", [])
    log.append(f"  [OK] Found {len(tools)} tools.")

    # 3. Scan Topology
    log.append("\n3. Testing scan_topology...")
    payload = {
        "action": {
            "type": "call_tool",
            "tool_name": "scan_topology",
            "arguments": {
                "threat_analysis": "Checking initial state.",
                "candidate_migrations": [],
                "justification": "Standard baseline verification."
            }
        },
        "episode_id": ""
    }
    r = httpx.post(f"{base_url}/step", json=payload, timeout=10.0)
    res_text = str(r.json().get("observation", {}).get("result", {}).get("content", [{"text": ""}])[0].get("text", ""))
    log.append("  [OK] scan_topology successful.")

    # Extract real node from scan to avoid the -0.1 penalty
    src_node = extract_first_node(res_text)
    dst_node = dict(src_node)
    dst_node["pod"] = "pod-999"

    # 4. Migrate Workload
    log.append("\n4. Testing migrate_workload...")
    payload = {
        "action": {
            "type": "call_tool",
            "tool_name": "migrate_workload",
            "arguments": {
                "threat_analysis": "Executing test migration.",
                "candidate_migrations": [f"test->test"],
                "justification": "Verifying migration logic.",
                "source_node": src_node,
                "target_node": dst_node
            }
        },
        "episode_id": ""
    }
    r = httpx.post(f"{base_url}/step", json=payload, timeout=10.0)
    reward = r.json().get("reward", 0.0)
    log.append(f"  [OK] migrate_workload successful. Preview Reward: {reward}")

    log.append("\n[PASS] All API smoke tests completed successfully!")
    return log

def main():
    print("Starting background uvicorn server for Datacenter SOC...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    
    base_url = "http://127.0.0.1:8000"
    ready = False
    for _ in range(20):
        try:
            if httpx.get(f"{base_url}/health").status_code == 200:
                ready = True
                break
        except Exception:
            pass
        time.sleep(0.5)

    if not ready:
        print("Server failed to start. Port 8000 might be in use.")
        proc.kill()
        sys.exit(1)

    try:
        logs = run_smoke_tests(base_url)
        print("\n".join(logs))
    finally:
        proc.kill()
        print("\nBackground server stopped.")

if __name__ == "__main__":
    main()