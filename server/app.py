# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI + WebSocket server for the Datacenter Workload Migration env.

The environment is a multi-agent SOC simulation: an LLM agent acts as either
DEFENDER or ADVERSARY in a Global SOC, migrating workloads across a 4D
``(region, zone, rack, pod)`` tensor.
"""

import json
import os  # noqa: F401  (kept for parity with the previous server module)

from fastapi import Request

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
)

try:
    from .datacenter_env import DatacenterEnvironment, _current_episode_id
except ImportError:
    from server.datacenter_env import DatacenterEnvironment, _current_episode_id


app = create_app(
    DatacenterEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="datacenter_arena",
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


# Replace the framework's default /state endpoint with one that returns the
# active episode's state instead of throwing 500 when no episode is loaded.
for i, route in enumerate(app.routes):
    if getattr(route, "path", "") == "/state" and "GET" in getattr(route, "methods", []):
        app.routes.pop(i)
        break


@app.get("/state")
async def get_state_override():
    if getattr(DatacenterEnvironment, "_latest_instance", None) is not None:
        return DatacenterEnvironment._latest_instance.state
    return {"error": "No active engagement. Call /reset first."}


@app.post("/finalize")
async def finalize_episode_endpoint(request: Request):
    """Early finalization for stuck/errored engagements."""
    data = await request.json()
    ep_id = data.get("episode_id")
    reason = data.get("reason", "stuck_unknown")

    active = None
    if ep_id and ep_id in DatacenterEnvironment._instances:
        active = DatacenterEnvironment._instances[ep_id]
    elif DatacenterEnvironment._latest_instance:
        active = DatacenterEnvironment._latest_instance

    if active:
        active._finalize_episode(result=reason)
        DatacenterEnvironment._instances.pop(active._state.episode_id, None)
        return active.snapshot()
    return {"error": "No active engagement found."}


def main() -> None:
    """Run the server directly via `python -m server.app`."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
