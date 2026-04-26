#!/usr/bin/env python3
"""Datacenter SOC Environment Client.

This module provides the client for connecting to a Datacenter SOC environment server.
DatacenterEnv extends MCPToolClient to provide tool-calling style interactions for workload migration.
"""

from openenv.core.client.mcp_client import MCPToolClient


class DatacenterEnv(MCPToolClient):
    """Client for the Datacenter SOC Environment.

    This client allows agents to interact with the multi-region
    Datacenter SOC via MCP tools. It inherits all functionality
    from MCPToolClient.

    Example:
        >>> with DatacenterEnv(base_url="http://localhost:8000") as env:
        ...     obs = env.reset()
        ...     print(obs.observation)
    """

    pass
