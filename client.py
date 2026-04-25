# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Chess Arena Environment Client.

This module provides the client for connecting to a Chess Arena environment server.
ChessArenaEnv extends MCPToolClient to provide tool-calling style interactions for chess play.

Example:
    >>> with ChessArenaEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...     tools = env.list_tools()
    ...     result = env.call_tool("make_move", uci_move="e2e4")
    ...     print(result)
"""

from openenv.core.mcp_client import MCPToolClient


class ChessArenaEnv(MCPToolClient):
    """
    Client for the Chess Arena Environment.

    This client provides a simple interface for interacting with the
    Chess Arena via MCP tools. It inherits all functionality
    from MCPToolClient:
    - `list_tools()`: Discover available tools (make_move, evaluate_position, etc.)
    - `call_tool(name, **kwargs)`: Call a tool by name
    - `reset(**kwargs)`: Reset the environment
    - `step(action)`: Execute an action (for advanced use)

    Example:
        >>> with ChessArenaEnv(base_url="http://localhost:8000") as env:
        ...     env.reset()
        ...
        ...     # List available tools
        ...     tools = env.list_tools()
        ...     for tool in tools:
        ...         print(f"{tool.name}: {tool.description}")
        ...
        ...     # Make a move
        ...     move_result = env.call_tool("make_move", uci_move="e2e4")
        ...     print(move_result)
        ...
        ...     # Get board evaluation
        ...     eval_result = env.call_tool("evaluate_position")
        ...     print(eval_result)
    """

    pass  # MCPToolClient provides all needed functionality
