"""
Verify that the HTTP server correctly resets the chess board state
when the /reset endpoint is called.
"""
import subprocess
import sys
import time
import httpx

INITIAL_STATE_HASH = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

def step(base_url: str, tool: str, **kwargs) -> dict:
    payload = {
        "action": {
            "type": "call_tool",
            "tool_name": tool,
            "arguments": kwargs
        },
        "episode_id": "" # Let backend route to latest
    }
    r = httpx.post(f"{base_url}/step", json=payload, timeout=10.0)
    r.raise_for_status()
    return r.json()

def main():
    print("Starting background uvicorn...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "chess_arena.server.app:app", "--host", "127.0.0.1", "--port", "63576"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = "http://127.0.0.1:63576"
    
    ready = False
    for _ in range(20):
        try:
            if httpx.get(f"{base}/health").status_code == 200:
                ready = True
                break
        except Exception:
            pass
        time.sleep(0.5)

    if not ready:
        print("Server failed to start.")
        proc.kill()
        sys.exit(1)

    try:
        # 1. Reset
        httpx.post(f"{base}/reset", json={}, timeout=5.0).raise_for_status()
        print("[OK] Reset 1 successful.")

        # 2. Migrate Workload
        step(
            base, 
            "migrate_workload", 
            threat_analysis="Initial deployment.", 
            candidate_migrations=["us-east/az-a/rack-1/pod-1->us-east/az-a/rack-1/pod-3"], 
            justification="Load balancing.", 
            source_node={"region": "us-east", "zone": "az-a", "rack": "rack-1", "pod": "pod-1"},
            target_node={"region": "us-east", "zone": "az-a", "rack": "rack-1", "pod": "pod-3"}
        )
        print("[OK] Workload migrated successfully.")

        # 3. Scan topology
        res = step(
            base, 
            "scan_topology", 
            threat_analysis="Verifying state.", 
            candidate_migrations=["us-east/az-a/rack-1/pod-3->us-east/az-a/rack-2/pod-1"], 
            justification="Telemetry check."
        )
        text = str(res.get("observation", {}).get("result", {}).get("data", ""))
        assert "turn=black" in text, f"Expected black's turn, got {text}"
        print("[OK] Topology state verified (Adversary's turn).")

        # 4. Reset AGAIN
        httpx.post(f"{base}/reset", json={}, timeout=5.0).raise_for_status()
        print("[OK] Reset 2 successful.")

        # 5. Scan topology again (Should be back to Defender's turn!)
        res = step(
            base, 
            "scan_topology", 
            threat_analysis="Post-reset baseline.", 
            candidate_migrations=[], 
            justification="Verification."
        )
        text = str(res.get("observation", {}).get("result", {}).get("data", ""))
        assert "turn=white" in text, "Expected white's turn after reset!"
        assert INITIAL_STATE_HASH in text, "Expected initial state hash after reset!"
        print("[OK] Topology state verified (Back to start).")

        print("\n[PASS] All HTTP reset tests passed!")

    except Exception as e:
        print(f"\n[FAIL] Test crashed: {e}")
    finally:
        proc.kill()
        print("Server stopped.")

if __name__ == "__main__":
    main()