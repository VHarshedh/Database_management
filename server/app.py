# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI + WebSocket server for the Chess Arena OpenEnv environment."""

from fastapi import Request
import json

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import (
        CallToolAction,
        CallToolObservation,
    )
    from .chess_environment import ChessEnvironment, _current_episode_id
except ImportError:  # standalone layout (openenv installed from pip)
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import (
        CallToolAction,
        CallToolObservation,
    )
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


def main() -> None:
    """Run the server directly via `python -m chess_arena.server.app`."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
