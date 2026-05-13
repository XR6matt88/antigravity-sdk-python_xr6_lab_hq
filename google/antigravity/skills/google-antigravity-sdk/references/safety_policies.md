# Safety Policies in Antigravity SDK

Reference guide for configuring access control and safety policies in the
Antigravity SDK.

## Overview

The Antigravity SDK provides a declarative policy system to control which tools
an agent can execute. Policies are evaluated using a priority-based model to
ensure safety and prevent unauthorized actions.

## Mandatory Requirement

> [!IMPORTANT] When **write tools** or **MCP servers** are enabled, you **MUST**
> specify a safety policy or register a custom `PreToolCallDecideHook`. Failing
> to do so will result in a `ValueError` at agent startup.

Write tools are any tools that are not read-only (e.g., `view_file` is
read-only, but `run_command` and `write_to_file` are not).

## Policy Resolution Order

Policies are evaluated in the following order of precedence (highest to lowest):

1. **Specific Deny**: `policy.deny("tool_name", ...)`
2. **Specific Ask**: `policy.ask_user("tool_name", ...)`
3. **Specific Allow**: `policy.allow("tool_name", ...)`
4. **Wildcard Deny**: `policy.deny("*", ...)`
5. **Wildcard Ask**: `policy.ask_user("*", ...)`
6. **Wildcard Allow**: `policy.allow("*", ...)`

Within each priority group, the **first match wins** (short-circuit evaluation).

## Configuration

Use the `google.antigravity.hooks.policy` module to define policies.

### Allow

Approves tool calls without confirmation.

```python
from google.antigravity.hooks import policy

# Allow all calls to view_file

policy.allow("view_file")
```

### Deny

Blocks tool calls immediately.

```python
from google.antigravity.hooks import policy

# Deny all calls to run_command

policy.deny("run_command")
```

### Ask User

Requires user confirmation before execution. Must provide a handler.

```python
from google.antigravity.hooks import policy

async def my_approval_handler(tool_call):
  # Custom logic to ask user or auto-approve
  # Return True to allow, False to deny
  return True

policy.ask_user("run_command", handler=my_approval_handler)
```

### Wildcards

-   `policy.allow_all()`: Approves all tool calls. Equivalent to `allow("*")`.
-   `policy.deny_all()`: Denies all tool calls. Equivalent to `deny("*")`.

## Predicates (Argument Checking)

You can use the `when` parameter to restrict policies based on tool arguments.
The predicate receives the tool arguments as a dictionary.

```python
from google.antigravity.hooks import policy

# Deny run_command if it contains 'rm'
policy.deny(
    "run_command",
    when=lambda args: "rm" in args.get("CommandLine", ""),
    name="deny_rm",
)
```

> [!CAUTION] If a predicate raises an exception during evaluation, the policy
> **fails closed** and treats it as a match (i.e., the decision for that policy
> applies).

## Minimal Safe Templates

### Deny by Default (Recommended)

Start by denying everything and selectively allow safe tools.

```python
from google.antigravity import Agent, LocalAgentConfig, CapabilitiesConfig
from google.antigravity.hooks import policy

policies = [
    policy.deny_all(),
    policy.allow("view_file"),
    policy.allow("code_search"),
    policy.ask_user("run_command", handler=my_approval_handler),
]

config = LocalAgentConfig(
    system_instructions="You are a helpful assistant.",
    capabilities=CapabilitiesConfig(),  # Enables write tools
    policies=policies,
)
```

### Allow All (Dangerous)

Use only for local development where safety is not a concern.

```python
from google.antigravity import Agent, LocalAgentConfig, CapabilitiesConfig
from google.antigravity.hooks import policy

config = LocalAgentConfig(
    system_instructions="You are a helpful assistant.",
    capabilities=CapabilitiesConfig(),
    policies=[policy.allow_all()],
)
```
