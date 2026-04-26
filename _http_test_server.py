import subprocess
import sys
import time
import requests

def test_http_reset():
    """Verify that the HTTP server correctly resets the SOC state."""
    print("Starting SOC Server...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", "63576"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    time.sleep(3)  # Wait for startup
    
    try:
        print("Sending /reset request...")
        resp = requests.post("http://127.0.0.1:63576/reset", json={})
        data = resp.json()
        print(f"Server Response: {resp.status_code}")
        assert resp.status_code == 200
        assert "observation" in data
    finally:
        print("Stopping SOC Server...")
        proc.terminate()

if __name__ == "__main__":
    test_http_reset()
