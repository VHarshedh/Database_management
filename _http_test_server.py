# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Shared helper: spin up the Datacenter SOC FastAPI in a subprocess so the
# adjacent ``test_ws.py`` / ``test_api.py`` smoke tests can be run
# standalone (``python test_ws.py``) without manually launching uvicorn.
#
# The subprocess gets its own copy of ``DatacenterEnvironment._instances``
# and its own Stockfish UCI engine, so the test process can be torn down
# cleanly without affecting any other state in the workspace.

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx


# Directory containing ``server.app:app`` (this package root).
_DCM_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _DCM_ROOT.parent


def pick_free_port() -> int:
    """Return an OS-assigned ephemeral TCP port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_until_ready(base_url: str, timeout_s: float = 45.0) -> None:
    """Block until ``GET /health`` returns 200 (or raise on timeout).

    The Stockfish UCI handshake on cold-start can take a couple of
    seconds on slow VMs, so the default ceiling is intentionally
    generous.
    """
    deadline = time.monotonic() + timeout_s
    health = f"{base_url.rstrip('/')}/health"
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(health, timeout=1.0)
            if r.status_code == 200:
                return
        except Exception as exc:
            last_err = exc
        time.sleep(0.25)
    raise RuntimeError(
        f"Server at {base_url} did not become ready within {timeout_s}s "
        f"(last error: {last_err!r})"
    )


def _build_subprocess_env() -> dict[str, str]:
    """``server.app`` imports ``openenv.*`` (from ``src/``) and lives under
    this package directory. We have to make both visible on ``PYTHONPATH``
    in the spawned uvicorn worker -- the parent shell may not have done
    so."""
    env = os.environ.copy()
    extra_paths = [str(_REPO_ROOT / "src"), str(_REPO_ROOT / "envs"), str(_DCM_ROOT)]
    existing = env.get("PYTHONPATH", "")
    if existing:
        env["PYTHONPATH"] = os.pathsep.join([*extra_paths, existing])
    else:
        env["PYTHONPATH"] = os.pathsep.join(extra_paths)
    return env


def start_background_server(
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    silent: bool = True,
) -> tuple[subprocess.Popen, str]:
    """Spawn ``uvicorn server.app:app`` in a subprocess and wait for /health.

    Returns ``(process, base_url)``. The caller is responsible for handing
    ``process`` back to :func:`stop_background_server` once tests finish.
    """
    p = port if port is not None else pick_free_port()
    stdout = subprocess.DEVNULL if silent else None
    stderr = subprocess.DEVNULL if silent else None
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "server.app:app",
            "--host", host,
            "--port", str(p),
        ],
        cwd=str(_DCM_ROOT),
        env=_build_subprocess_env(),
        stdout=stdout,
        stderr=stderr,
    )
    base = f"http://{host}:{p}"
    try:
        wait_until_ready(base)
    except Exception:
        # Don't leave a zombie uvicorn behind if /health never came up.
        stop_background_server(proc)
        raise
    return proc, base


def stop_background_server(proc: subprocess.Popen | None) -> None:
    """Best-effort terminate of the uvicorn subprocess.

    Sends SIGTERM first; escalates to SIGKILL after 8s. Safe to call with
    ``None`` or with an already-exited process.
    """
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
