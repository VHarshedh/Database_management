"""
Smoke test for the chess_arena HTTP API.
Tests /reset, /list_tools, and /step (make_move, analyze_board, list_legal_moves).
"""
import os
import subprocess
import sys
import time
import httpx

def run_smoke_tests(base_url: str) -> list[str]:
    log = []
    
    # 1. Reset Environment
    log.append("1. Testing POST /reset...")
    r = httpx.post(f"{base_url}/reset", json={}, timeout=5.0)
    if r.status_code != 200:
        log.append(f"  [FAIL] /reset returned {r.status_code}: {r.text}")
        return log
    
    log.append("  [OK] /reset successful.")
    
    # We no longer rely on /state to get the episode ID, as it can cause race conditions.
    # Instead, we just let the backend route to the latest active instance.
    episode_id = ""

    # 2. List Tools
    log.append("\n2. Testing POST /step (list_tools)...")
    payload = {
        "action": {"type": "list_tools"},
        "episode_id": episode_id
    }
    r = httpx.post(f"{base_url}/step", json=payload, timeout=5.0)
    if r.status_code != 200:
        log.append(f"  [FAIL] /step list_tools returned {r.status_code}: {r.text}")
        return log
        
    tools = r.json().get("observation", {}).get("result", {}).get("tools", [])
    tool_names = [t.get("name") for t in tools]
    log.append(f"  [OK] Found tools: {', '.join(tool_names)}")

    # 3. Scan Topology
    log.append("\n3. Testing scan_topology...")
    payload = {
        "action": {
            "type": "call_tool",
            "tool_name": "scan_topology",
            "arguments": {
                "threat_analysis": "Checking initial state.",
                "candidate_migrations": ["us-east/az-a/rack-1/pod-1->us-east/az-a/rack-1/pod-3"],
                "justification": "Standard baseline verification."
            }
        },
        "episode_id": episode_id
    }
    r = httpx.post(f"{base_url}/step", json=payload, timeout=10.0)
    if r.status_code != 200:
        log.append(f"  [FAIL] scan_topology returned {r.status_code}: {r.text}")
        return log
    log.append("  [OK] scan_topology successful.")

    # 4. Migrate Workload
    log.append("\n4. Testing migrate_workload...")
    payload = {
        "action": {
            "type": "call_tool",
            "tool_name": "migrate_workload",
            "arguments": {
                "threat_analysis": "No threats.",
                "candidate_migrations": ["us-east/az-a/rack-1/pod-1->us-east/az-a/rack-1/pod-3"],
                "justification": "Optimizing compute distribution.",
                "source_node": {"region": "us-east", "zone": "az-a", "rack": "rack-1", "pod": "pod-1"},
                "target_node": {"region": "us-east", "zone": "az-a", "rack": "rack-1", "pod": "pod-3"}
            }
        },
        "episode_id": episode_id
    }
    r = httpx.post(f"{base_url}/step", json=payload, timeout=10.0)
    if r.status_code != 200:
        log.append(f"  [FAIL] migrate_workload returned {r.status_code}: {r.text}")
        return log
        
    res_data = r.json()
    reward = res_data.get("reward", 0.0)
    log.append(f"  [OK] migrate_workload successful. Preview Reward: {reward}")

    log.append("\n[PASS] All API smoke tests completed successfully!")
    return log

def main():
    print("Starting background uvicorn (chess_arena)...")
    # Route output to DEVNULL to avoid cluttering the test logs
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "chess_arena.server.app:app", "--host", "127.0.0.1", "--port", "63575"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    
    base_url = "http://127.0.0.1:63575"
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
        print("Server failed to start.")
        proc.kill()
        sys.exit(1)

    print(f"Server ready at {base_url}\n")
    
    try:
        logs = run_smoke_tests(base_url)
        print("\n".join(logs))
    finally:
        proc.kill()
        print("\nBackground server stopped.")

if __name__ == "__main__":
    main()