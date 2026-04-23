# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI + WebSocket server for the Chess Arena OpenEnv environment."""

import os
from fastapi import Request
import json

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
)

try:
    from .chess_environment import ChessEnvironment, _current_episode_id
except ImportError:
    from server.chess_environment import ChessEnvironment, _current_episode_id


app = create_app(
    ChessEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="chess_arena",
)


@app.middleware("http")
async def episode_id_extractor(request: Request, call_next):
    """Pin the per-request ContextVar so FastMCP tool fns can route correctly."""
    if request.url.path.endswith("/step") and request.method == "POST":
        body = await request.body()
        try:
            data = json.loads(body)
            ep_id = data.get("episode_id")
            if ep_id:
                _current_episode_id.set(ep_id)
        except Exception:
            pass

        async def receive():
            return {"type": "http.request", "body": body}

        request._receive = receive

    return await call_next(request)


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """Liveness probe: 200 OK for Docker HEALTHCHECK + the hackathon validator."""
    return {"status": "healthy"}


# --- BUG 3 FIX: OVERRIDE /state TO PREVENT 500 ERRORS ---
# 1. Remove the broken /state route provided by openenv-core
for i, route in enumerate(app.routes):
    if getattr(route, "path", "") == "/state" and "GET" in getattr(route, "methods", []):
        app.routes.pop(i)
        break

# 2. Replace it with a safe version that returns the active episode's state
@app.get("/state")
async def get_state_override():
    from server.chess_environment import ChessEnvironment
    if getattr(ChessEnvironment, "_latest_instance", None) is not None:
        return ChessEnvironment._latest_instance.state
    return {"error": "No active episode. Call /reset first."}


def main() -> None:
    """Run the server directly via `python -m chess_arena.server.app`."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()