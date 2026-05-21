<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->

# MCP Integration in Google Antigravity SDK

Reference guide for connecting and using Model Context Protocol (MCP) servers in
the Google Antigravity SDK.

## Overview

The Model Context Protocol (MCP) allows agents to connect to external servers
that expose tools, resources, and prompts. The Google Antigravity SDK supports
integrating MCP servers to extend the capabilities of your custom agents.

For a concrete code example of setting up and using MCP, see
[mcp_tools.md](../../examples/getting_started/mcp_tools.md).

## Configuration Modes

Google Antigravity SDK supports two main ways to connect to MCP servers:

1.  **Stdio Transport**: The SDK launches and manages the MCP server process,
    communicating over standard input/output.
2.  **SSE Transport**: The SDK connects to a remote MCP server running as a web
    service using Server-Sent Events (SSE).

## Stdio Transport Configuration

Use Stdio transport when you want to manage the lifecycle of the MCP server
locally. This is configured in `LocalAgentConfig` using `mcp_servers`.

### Example

```python
from google.antigravity import Agent, LocalAgentConfig, types

mcp_servers = [
    types.McpStdioServer(
        name="my_stdio_server",
        command="python3",
        args=["mcp_server.py"],
    )
]

config = LocalAgentConfig(mcp_servers=mcp_servers)

async with Agent(config) as agent:
    response = await agent.chat("Use the MCP server to perform a task.")
    print(await response.text())
```

## SSE Transport Configuration

Use SSE transport when you want to connect to a remote MCP server running as a
web service.

### Example

```python
from google.antigravity import Agent, LocalAgentConfig, types

mcp_servers = [
    types.McpSseServer(
        name="my_sse_server",
        url="https://example.com/mcp/sse",
        headers={"Authorization": "Bearer your-token-here"},  # Optional headers
    )
]

config = LocalAgentConfig(mcp_servers=mcp_servers)

async with Agent(config) as agent:
    response = await agent.chat("Ask the remote MCP server to perform a task.")
    print(await response.text())
```

## Accessing Tools

Tools exposed by the MCP server are automatically registered with the agent. To
allow the agent to use these tools, you must ensure your safety policy grants
permission.

### Permissions

By default, the SDK's default policy (`confirm_run_command()`) is permissive and
**allows all MCP tools** (it only blocks or asks for confirmation on
`run_command`).

However, if you configure a strict **deny-by-default** setup (using
`policy.deny_all()`), you must explicitly allow your MCP tools by their exact
namespaced names.

To prevent name collisions across multiple MCP servers, the SDK automatically
namespaces and prefixes all MCP tools with the pattern:
`mcp_{sanitized_server_name}_{original_tool_name}`

For example, if a server is configured with `name="google-workspace"` and
exposes a tool named `read_email`, it will be registered as
`mcp_google-workspace_read_email`.

To allowlist this tool in a deny-by-default setup, you must use this namespaced
name:

```python
# In your safety policy configuration
policies = [
    policy.deny_all(),
    policy.allow("mcp_google-workspace_read_email"),  # Must use the exact namespaced tool name
]
```

See [Safety Policies](safety_policies.md) for more details on how to configure
policies.

## Gotchas

> [!WARNING] **Identifier Safety**: FastMCP (used under the hood) requires tool
> and parameter names to be valid Python identifiers. Ensure your MCP server
> adheres to this.

> [!IMPORTANT] **Permissions**: Failing to grant permissions for MCP tools will
> prevent the agent from using them, even if the server is correctly connected.

> [!NOTE] **Timeouts**: External processes can cause timeouts if they block the
> customization server. Ensure your MCP server is responsive.
