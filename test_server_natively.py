import time
from client import DatacenterEnv

def test_native_client():
    """Test the renamed DatacenterEnv client against the live server."""
    print("Connecting to live SOC server...")
    try:
        with DatacenterEnv(base_url="http://127.0.0.1:8000") as env:
            print("Sending Reset...")
            obs = env.reset()
            print(f"Observation Received (Reward: {obs.reward})")
            
            print("Scanning Topology...")
            topo = env.step({"tool": "scan_topology", "arguments": {}})
            print("Scan Successful.")
            
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Ensure 'python -m uvicorn server.app:app' is running.")

if __name__ == "__main__":
    test_native_client()
