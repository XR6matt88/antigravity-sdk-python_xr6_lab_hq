# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bridge between MCP services and the SDK ToolRunner."""

# pylint: disable=g-importing-member

from collections.abc import Mapping, Sequence
import contextvars
from datetime import timedelta
import re
from typing import Any, Callable
from mcp.client import stdio
from mcp.client.session_group import ClientSessionGroup
from mcp.client.session_group import SseServerParameters
from mcp.client.session_group import StreamableHttpParameters
from google.antigravity import types
from google.antigravity.tools.tool_runner import ToolWithSchema


_current_server_cfg_var = contextvars.ContextVar[types.McpServerConfig | None](
    "_current_server_cfg_var", default=None
)


async def get_mcp_tools(
    session_group: ClientSessionGroup,
) -> list[ToolWithSchema]:
  """Fetches tools from session_group and returns them as ToolWithSchema.

  Args:
    session_group: The ClientSessionGroup to fetch tools from.

  Returns:
    A list of ToolWithSchema objects.
  """
  tools = []
  for name, tool_info in session_group.tools.items():

    def make_wrapper(tool_name: str, doc: str | None) -> Callable[..., Any]:
      async def wrapper(**kwargs: Any) -> Any:
        return await session_group.call_tool(tool_name, kwargs)

      wrapper.__name__ = tool_name
      if doc:
        wrapper.__doc__ = doc
      return wrapper

    wrapper_fn = make_wrapper(name, tool_info.description)
    tool_with_schema = ToolWithSchema(wrapper_fn, tool_info.inputSchema)
    tools.append(tool_with_schema)

  return tools


def _component_name_hook(name: str, server_info: Any) -> str:
  """Renames tools to prefix them with the server name.

  Args:
    name: Original tool name.
    server_info: Server implementation details.

  Returns:
    The namespaced prefixed tool name.
  """
  server_cfg = _current_server_cfg_var.get()

  if server_cfg:
    # Custom server name is pre-validated to match ^[a-zA-Z0-9_-]+$.
    prefix = server_cfg.name.lower()
  else:
    # Fallback to server-reported name and sanitize it.
    raw_prefix = server_info.name if server_info else ""
    # Replace non-alphanumeric/hyphen/underscore with underscore.
    prefix = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw_prefix).strip("_").lower()

  if prefix:
    return f"mcp_{prefix}_{name}"
  return f"mcp_{name}"


class McpBridge:
  """Simplifies the lifecycle of MCP Client Sessions."""

  def __init__(self):
    """Initializes the McpBridge instance."""
    self._session_group: ClientSessionGroup | None = None
    self._tools: list[ToolWithSchema] = []

  @property
  def tools(self) -> list[ToolWithSchema]:
    """The MCP tools discovered from connected servers."""
    return list(self._tools)

  async def connect(self, server_cfg: types.McpServerConfig):
    """Connects to an MCP server based on its configuration.

    Args:
      server_cfg: The configuration for the MCP server.

    Raises:
      ValueError: If the server configuration type is unsupported.
    """
    if server_cfg.type == "stdio":
      await self.connect_stdio(
          server_cfg.command, server_cfg.args, server_cfg=server_cfg
      )
    elif server_cfg.type == "sse":
      await self.connect_sse(
          server_cfg.url, server_cfg.headers, server_cfg=server_cfg
      )
    elif server_cfg.type == "http":
      await self.connect_streamable_http(
          url=server_cfg.url,
          headers=server_cfg.headers,
          timeout=server_cfg.timeout,
          sse_read_timeout=server_cfg.sse_read_timeout,
          terminate_on_close=server_cfg.terminate_on_close,
          server_cfg=server_cfg,
      )
    else:
      raise ValueError(f"Unsupported MCP server type: {server_cfg.type}")

  async def connect_stdio(
      self,
      command: str,
      args: Sequence[str],
      server_cfg: types.McpServerConfig | None = None,
  ):
    """Connects to a local MCP server over stdio.

    Args:
      command: The command to run to start the server.
      args: Arguments to pass to the command.
      server_cfg: Optional server configuration.
    """
    params = stdio.StdioServerParameters(command=command, args=list(args))
    await self._connect(params, server_cfg)

  async def connect_sse(
      self,
      url: str,
      headers: Mapping[str, str] | None = None,
      server_cfg: types.McpServerConfig | None = None,
  ):
    """Connects to a remote MCP server over SSE.

    Args:
      url: The URL of the SSE endpoint.
      headers: Optional headers to send with the connection request.
      server_cfg: Optional server configuration.
    """
    params = SseServerParameters(
        url=url, headers=dict(headers) if headers is not None else None
    )
    await self._connect(params, server_cfg)

  async def connect_streamable_http(
      self,
      url: str,
      headers: Mapping[str, str] | None = None,
      timeout: float = 30.0,
      sse_read_timeout: float = 300.0,
      terminate_on_close: bool = True,
      server_cfg: types.McpServerConfig | None = None,
  ):
    """Connects to a remote MCP server over Streamable HTTP.

    Args:
      url: The URL of the HTTP endpoint.
      headers: Optional headers to send with the connection request.
      timeout: Connection timeout in seconds.
      sse_read_timeout: SSE read timeout in seconds.
      terminate_on_close: Whether to terminate the connection on close.
      server_cfg: Optional server configuration.
    """
    params = StreamableHttpParameters(
        url=url,
        headers=dict(headers) if headers is not None else None,
        timeout=timedelta(seconds=timeout),
        sse_read_timeout=timedelta(seconds=sse_read_timeout),
        terminate_on_close=terminate_on_close,
    )
    await self._connect(params, server_cfg)

  async def _connect(
      self,
      params: (
          stdio.StdioServerParameters
          | SseServerParameters
          | StreamableHttpParameters
      ),
      server_cfg: types.McpServerConfig | None = None,
  ) -> None:
    """Establishes connection using ClientSessionGroup and registers tools."""
    if not self._session_group:
      self._session_group = ClientSessionGroup(
          component_name_hook=_component_name_hook
      )
      # Direct __aenter__ call because McpBridge manages the session
      # lifecycle itself (connect/stop) rather than being used as an
      # async context manager. __aexit__ is called in stop().
      await self._session_group.__aenter__()

    token = _current_server_cfg_var.set(server_cfg)
    try:
      await self._session_group.connect_to_server(params)
    finally:
      _current_server_cfg_var.reset(token)

    self._tools = await get_mcp_tools(self._session_group)

  async def stop(self):
    """Cleans up all active MCP sessions and releases resources."""
    if self._session_group:
      await self._session_group.__aexit__(None, None, None)
      self._session_group = None
