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

"""Tool call policy system for the Google Antigravity SDK.

Provides a declarative API for expressing tool call policies (APPROVE, DENY,
ASK_USER) that are enforced via the hooks system. Policies are evaluated using
a priority-based model where specificity and safety determine precedence:

  Specific Deny > Specific Ask > Specific Allow >
  Wildcard Deny > Wildcard Ask > Wildcard Allow

Within each priority group, first match wins, enabling short-circuit evaluation.

Default Behavior:

  ``LocalAgentConfig`` uses ``confirm_run_command()`` as its default policy.
  This denies ``run_command`` (the most dangerous tool) while allowing all
  other tools.  To enable autonomous shell access, explicitly pass
  ``policies=[policy.allow_all()]``.

Policy Denial vs. Disabling Tools:

  Policies operate at the hook layer: a denied tool is still *visible* to the
  model in its tool list. If the model calls a policy-denied tool, the SDK
  rejects the call and returns a denial message. The model may then retry or
  choose another approach, but each attempt costs tokens.

  To remove a tool from the model's context entirely — so it never sees the
  tool and never wastes tokens on it — use ``CapabilitiesConfig.disabled_tools``
  (or ``enabled_tools``) instead.

  **Use policies** when the restriction is conditional or context-dependent
  (e.g., denying ``run_command`` only for dangerous arguments, or requiring
  user approval for certain operations).

  **Use CapabilitiesConfig** when the tool is simply irrelevant to the agent's
  purpose and should not appear in its context at all.

Usage:
  from google.antigravity.hooks import policy

  policies = [
      policy.deny("*"),                     # Block everything by default
      policy.allow("read_file"),            # Except reading files
      policy.deny("run_command",            # Block dangerous commands
          when=lambda args: "rm" in args.get("CommandLine", "")),
      policy.ask_user("run_command",        # Ask for other commands
          handler=my_approval_fn),
  ]

  hook = policy.enforce(policies)
  # Register hook with HookRunner's pre_tool_call_decide_hooks
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
import dataclasses
import enum
import functools
import inspect
import logging
import os
import pathlib
import sys
import typing
from typing import Any, Union, overload

import pydantic

from google.antigravity import types
from google.antigravity.hooks import hooks


_logger = logging.getLogger(__name__)

# A predicate receives the tool call's argument dict (or a Pydantic model,
# if the predicate's first parameter is annotated with a BaseModel subclass)
# and returns whether the policy applies. Supports both sync and async.
Predicate = Callable[..., bool | Awaitable[bool]]

# An ask_user handler receives the full ToolCall and returns whether the
# user approved execution. Supports both sync and async callables.
AskUserHandler = Callable[[types.ToolCall], bool | Awaitable[bool]]

_WILDCARD = "*"


class Decision(enum.Enum):
  """Outcome a policy can produce."""

  APPROVE = "APPROVE"
  DENY = "DENY"
  ASK_USER = "ASK_USER"


@dataclasses.dataclass(frozen=True)
class Policy:
  """A single tool call policy rule.

  Attributes:
    tool: Tool name this policy targets, or "*" for all tools.
    decision: The outcome when this policy matches.
    when: Optional predicate on the tool call's arguments. If None the policy
      matches any call to the named tool.
    ask_user: Handler invoked when decision is ASK_USER. Must be provided for
      ASK_USER policies (validated at enforce() time).
    name: Human-readable label used in logging and deny reasons.
  """

  tool: str
  decision: Decision
  when: Predicate | None = None
  ask_user: AskUserHandler | None = None
  name: str = ""


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _mcp_policies(
    decision: Decision,
    mcp_config: types.BaseMcpServerConfig,
    mcp_tools: Sequence[str] | None = None,
    *,
    when: Predicate | None = None,
    name: str = "",
    handler: AskUserHandler | None = None,
) -> list[Policy]:
  """Generates MCP-specific policies.

  Commonly used by allow(), deny(), and ask_user() builders. Translates the
  config and tools into the structured 'server/tool' or 'server/*' target
  formats expected by the runtime hook.

  Args:
    decision: The Decision outcome when the policy matches.
    mcp_config: The BaseMcpServerConfig of the target MCP server.
    mcp_tools: Optional sequence of tool names to allow/deny/ask for. If None,
      applies to all tools on this server.
    when: Optional argument predicate.
    name: Optional human-readable label.
    handler: Optional AskUserHandler for ASK_USER policies.

  Returns:
    A list of Policy objects.
  """
  server = mcp_config.name

  if isinstance(mcp_tools, str):
    raise ValueError(
        f"mcp_tools must be a sequence of strings (e.g., ['{mcp_tools}']), "
        "not a single string."
    )

  if mcp_tools is None:
    # Server-wide wildcard policy (covers all tools of this MCP server).
    # Lowercase prefix matching is purely for debug-logging consistency.
    policy_name = name or f"{decision.value.lower()}_{server}_all"
    return [
        Policy(
            tool=f"{server}/*",
            decision=decision,
            when=when,
            name=policy_name,
            ask_user=handler,
        )
    ]

  policies = []
  for t in mcp_tools:
    policy_name = (
        f"{name}_{t}" if name else f"{decision.value.lower()}_{server}_{t}"
    )
    policies.append(
        Policy(
            tool=f"{server}/{t}",
            decision=decision,
            when=when,
            name=policy_name,
            ask_user=handler,
        )
    )
  return policies


@overload
def allow(
    tool: str,
    *,
    when: Predicate | None = None,
    name: str = "",
) -> Policy:
  ...


@overload
def allow(
    tool: types.BaseMcpServerConfig,
    mcp_tools: Sequence[str] | None = None,
    *,
    when: Predicate | None = None,
    name: str = "",
) -> list[Policy]:
  ...


def allow(
    tool: str | types.BaseMcpServerConfig,
    mcp_tools: Sequence[str] | None = None,
    *,
    when: Predicate | None = None,
    name: str = "",
) -> Any:
  """Creates an APPROVE policy.

  Args:
    tool: Tool name, "*" for all tools, or BaseMcpServerConfig.
    mcp_tools: Optional list of tool names if BaseMcpServerConfig is provided.
    when: Optional argument predicate.
    name: Human-readable label.

  Returns:
    A Policy or a list of Policies.
  """
  if isinstance(tool, str):
    if mcp_tools is not None:
      raise ValueError("mcp_tools cannot be specified when tool is a string.")
    return Policy(tool=tool, decision=Decision.APPROVE, when=when, name=name)  # pytype: disable=bad-return-type  # False positive: pytype fails to narrow overloaded types in implementation body.

  return _mcp_policies(Decision.APPROVE, tool, mcp_tools, when=when, name=name)  # pytype: disable=bad-return-type  # False positive: pytype fails to narrow overloaded types in implementation body.


@overload
def deny(
    tool: str,
    *,
    when: Predicate | None = None,
    name: str = "",
) -> Policy:
  ...


@overload
def deny(
    tool: types.BaseMcpServerConfig,
    mcp_tools: Sequence[str] | None = None,
    *,
    when: Predicate | None = None,
    name: str = "",
) -> list[Policy]:
  ...


def deny(
    tool: str | types.BaseMcpServerConfig,
    mcp_tools: Sequence[str] | None = None,
    *,
    when: Predicate | None = None,
    name: str = "",
) -> Any:
  """Creates a DENY policy.

  Args:
    tool: Tool name, "*" for all tools, or BaseMcpServerConfig.
    mcp_tools: Optional list of tool names if BaseMcpServerConfig is provided.
    when: Optional argument predicate.
    name: Human-readable label.

  Returns:
    A Policy or a list of Policies.
  """
  if isinstance(tool, str):
    if mcp_tools is not None:
      raise ValueError("mcp_tools cannot be specified when tool is a string.")
    return Policy(tool=tool, decision=Decision.DENY, when=when, name=name)  # pytype: disable=bad-return-type  # False positive: pytype fails to narrow overloaded types in implementation body.

  return _mcp_policies(Decision.DENY, tool, mcp_tools, when=when, name=name)  # pytype: disable=bad-return-type  # False positive: pytype fails to narrow overloaded types in implementation body.


@overload
def ask_user(
    tool: str,
    *,
    handler: AskUserHandler,
    when: Predicate | None = None,
    name: str = "",
) -> Policy:
  ...


@overload
def ask_user(
    tool: types.BaseMcpServerConfig,
    mcp_tools: Sequence[str] | None = None,
    *,
    handler: AskUserHandler,
    when: Predicate | None = None,
    name: str = "",
) -> list[Policy]:
  ...


def ask_user(
    tool: str | types.BaseMcpServerConfig,
    mcp_tools: Sequence[str] | None = None,
    *,
    handler: AskUserHandler,
    when: Predicate | None = None,
    name: str = "",
) -> Any:
  """Creates an ASK_USER policy.

  Args:
    tool: Tool name, "*" for all tools, or BaseMcpServerConfig.
    mcp_tools: Optional list of tool names if BaseMcpServerConfig is provided.
    handler: Callable invoked to obtain user approval.
    when: Optional argument predicate.
    name: Human-readable label.

  Returns:
    A Policy or a list of Policies.
  """
  if isinstance(tool, str):
    if mcp_tools is not None:
      raise ValueError("mcp_tools cannot be specified when tool is a string.")
    return Policy(  # pytype: disable=bad-return-type  # False positive: pytype fails to narrow overloaded types in implementation body.
        tool=tool,
        decision=Decision.ASK_USER,
        when=when,
        ask_user=handler,
        name=name,
    )

  return _mcp_policies(  # pytype: disable=bad-return-type  # False positive: pytype fails to narrow overloaded types in implementation body.
      Decision.ASK_USER,
      tool,
      mcp_tools,
      when=when,
      name=name,
      handler=handler,
  )


def allow_all() -> Policy:
  """Creates a policy that approves all tool calls without confirmation.

  Intended for autonomous agents and local development where interactive
  confirmation is not needed. Equivalent to ``allow("*")``.

  Returns:
    A Policy that approves every tool call.
  """
  return allow(_WILDCARD, name="allow_all")


def safe_defaults(handler: AskUserHandler) -> list[Policy]:
  """Creates a set of safe default policies.

  Allows all read-only tools and asks the user for any other tool calls.

  Args:
    handler: The handler to invoke for ASK_USER decisions.

  Returns:
    A list of Policies.
  """
  return [allow(t.value) for t in types.BuiltinTools.read_only()] + [
      ask_user("*", handler=handler)
  ]


def deny_all() -> Policy:
  """Creates a policy that denies all tool calls.

  Use as a base rule with specific ``allow()`` overrides for a
  deny-by-default posture. Specific policies always take priority over
  wildcard policies, so ``[deny_all(), allow("view_file")]`` will allow
  only ``view_file`` and deny everything else.

  Returns:
    A Policy that denies every tool call.
  """
  return deny(_WILDCARD, name="deny_all")


def confirm_run_command(
    handler: AskUserHandler | None = None,
) -> list[Policy]:
  """Safe default: allows all tools, denies or confirms run_command.

  When no handler is given, ``run_command`` is denied outright — the agent
  sees the tool but calls are rejected with a clear message explaining
  how to enable it.  When a handler is given, ``run_command`` calls
  trigger an ASK_USER flow instead.

  All other tools (file read/write, subagents, image generation, etc.)
  are allowed.

  This is the default policy for ``LocalAgentConfig``.

  Args:
    handler: Optional handler for ASK_USER on run_command. If None, run_command
      is denied.

  Returns:
    A list of Policies.
  """
  if handler is not None:
    return [
        ask_user(
            types.BuiltinTools.RUN_COMMAND.value,
            handler=handler,
            name="confirm_run_command",
        ),
        allow(_WILDCARD, name="confirm_run_command"),
    ]
  return [
      deny(types.BuiltinTools.RUN_COMMAND.value, name="confirm_run_command"),
      allow(_WILDCARD, name="confirm_run_command"),
  ]


PathOrStr = Union[str, os.PathLike[str]]


def _secure_normalize_path(path: PathOrStr) -> pathlib.Path:
  """Symmetrically canonicalizes paths, resolving symlinks and junctions.

  Raises OSError if the path cannot be securely canonicalized (Fail-Closed).
  """
  # We do NOT specify strict=True because new files to be created do not exist yet.
  # Instead, we use resolve(strict=False) which resolves existing symlinks.
  # We let OSErrors bubble up so the caller fails closed.
  return pathlib.Path(path).resolve()


@functools.lru_cache(maxsize=256)
def _is_case_insensitive(path: pathlib.Path) -> bool:
  """Dynamically checks if the filesystem at the given path is case-insensitive.

  Employs LRU caching to prevent excessive filesystem disk stats.
  """
  try:
    if not path.exists():
      return sys.platform in ("win32", "darwin")
  except OSError:
    return sys.platform in ("win32", "darwin")

  parent = path.parent
  name = path.name
  if not name:
    return sys.platform in ("win32", "darwin")

  # Invert character casing to check if the OS resolves to the same file
  swapped_name = "".join(c.swapcase() for c in name)
  if swapped_name == name:
    # No alphabetic characters in name — recursively probe parent directory
    if parent and parent != path:
      return _is_case_insensitive(parent)
    return sys.platform in ("win32", "darwin")

  try:
    return path.samefile(parent / swapped_name)
  except OSError:
    return False


def _is_path_in_workspace(
    target_path: PathOrStr, workspace_path: PathOrStr
) -> bool:
  """Returns True if the canonicalized target_path lies strictly within workspace_path."""
  try:
    norm_target = _secure_normalize_path(target_path)
    norm_ws = _secure_normalize_path(workspace_path)
  except OSError:
    # Security Fallback: Fail-closed if normalization fails
    return False

  if _is_case_insensitive(norm_ws):
    # Unicode-compliant case folding for robust case-insensitive matching
    t_parts = [p.casefold() for p in norm_target.parts]
    w_parts = [p.casefold() for p in norm_ws.parts]
  else:
    t_parts = list(norm_target.parts)
    w_parts = list(norm_ws.parts)

  if len(t_parts) < len(w_parts):
    return False

  # Structural containment comparison (avoids trailing separator slicing vulnerabilities)
  return t_parts[: len(w_parts)] == w_parts


def workspace_only(workspaces: Sequence[PathOrStr]) -> list[Policy]:
  """Restricts file tools to the given workspace directories.

  File read/write/create operations targeting paths outside any of the
  configured workspace directories are denied. Other tools are unaffected.

  Args:
    workspaces: Absolute paths of allowed workspace directories.

  Returns:
    A list of Policies.
  """
  file_tools = [t.value for t in types.BuiltinTools.file_tools()]

  def _outside_workspace(tc: types.ToolCall) -> bool:
    """Returns True when the target path is outside all workspaces."""
    path = tc.canonical_path or ""
    if not path:
      # Allow omit-path edge cases (e.g. list_dir with no args uses cwd)
      return False

    return not any(_is_path_in_workspace(path, ws) for ws in workspaces)

  return [
      deny(tool, when=_outside_workspace, name="workspace_only")
      for tool in file_tools
  ]


# ---------------------------------------------------------------------------
# Priority bucket indices (lower = higher priority)
# ---------------------------------------------------------------------------

_LEVEL_SPECIFIC_DENY = 0
_LEVEL_SPECIFIC_ASK = 1
_LEVEL_SPECIFIC_ALLOW = 2

_LEVEL_PREFIX_DENY = 3
_LEVEL_PREFIX_ASK = 4
_LEVEL_PREFIX_ALLOW = 5

_LEVEL_GLOBAL_DENY = 6
_LEVEL_GLOBAL_ASK = 7
_LEVEL_GLOBAL_ALLOW = 8

_NUM_LEVELS = 9

_DECISION_TO_SPECIFIC_LEVEL = {
    Decision.DENY: _LEVEL_SPECIFIC_DENY,
    Decision.ASK_USER: _LEVEL_SPECIFIC_ASK,
    Decision.APPROVE: _LEVEL_SPECIFIC_ALLOW,
}

_DECISION_TO_PREFIX_LEVEL = {
    Decision.DENY: _LEVEL_PREFIX_DENY,
    Decision.ASK_USER: _LEVEL_PREFIX_ASK,
    Decision.APPROVE: _LEVEL_PREFIX_ALLOW,
}

_DECISION_TO_GLOBAL_LEVEL = {
    Decision.DENY: _LEVEL_GLOBAL_DENY,
    Decision.ASK_USER: _LEVEL_GLOBAL_ASK,
    Decision.APPROVE: _LEVEL_GLOBAL_ALLOW,
}


def _is_global_wildcard(tool: str) -> bool:
  return tool == _WILDCARD


def _is_prefix_wildcard(tool: str) -> bool:
  # Prefix wildcards (e.g. "server/*") are strictly supported for MCP tools.
  return tool.endswith("/*")


def _bucket_index(p: Policy) -> int:
  """Returns the priority bucket for a policy."""
  if _is_global_wildcard(p.tool):
    return _DECISION_TO_GLOBAL_LEVEL[p.decision]
  if _is_prefix_wildcard(p.tool):
    return _DECISION_TO_PREFIX_LEVEL[p.decision]
  return _DECISION_TO_SPECIFIC_LEVEL[p.decision]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _matches_target(policy_tool: str, call_target: str, is_mcp: bool) -> bool:
  """Matches policy tool definition against parsed call target."""
  if policy_tool == _WILDCARD:
    return True

  if is_mcp:
    # policy_tool can be "server/*" or "server/tool"
    if _is_prefix_wildcard(policy_tool):
      policy_server = policy_tool[:-2]
      call_server, _ = call_target.split("/", 1)
      return policy_server == call_server
    return policy_tool == call_target

  return policy_tool == call_target


async def _evaluate_predicate(
    policy: Policy, tool_call: types.ToolCall
) -> bool:
  """Evaluates a policy's predicate.

  If the predicate is None, the policy always matches.
  Exceptions are propagated to the caller.

  Args:
    policy: The policy being evaluated.
    tool_call: The ToolCall instance.

  Returns:
    True if the predicate matches, False otherwise.
  """
  if policy.when is None:
    return True

  sig = inspect.signature(policy.when)
  params = list(sig.parameters.values())

  if params:
    first_param = params[0]
    # Resolve string annotations if future annotations are active
    try:
      hints = typing.get_type_hints(policy.when)
      annotation = hints.get(first_param.name, first_param.annotation)
    except (TypeError, NameError):
      annotation = first_param.annotation

    if isinstance(annotation, type) and issubclass(
        annotation, pydantic.BaseModel
    ):
      if issubclass(annotation, types.ToolCall):
        raw_result = policy.when(tool_call)
      else:
        typed_args = annotation.model_validate(tool_call.args)
        raw_result = policy.when(typed_args)
    else:
      raw_result = policy.when(tool_call.args)
  else:
    raw_result = policy.when()

  result = await raw_result if inspect.isawaitable(raw_result) else raw_result
  return bool(result)


async def _execute_ask_user(policy: Policy, tool_call: types.ToolCall) -> bool:
  """Invokes the policy's ask_user handler, propagating exceptions."""
  assert policy.ask_user is not None  # Validated at enforce() time.
  result = policy.ask_user(tool_call)
  if inspect.isawaitable(result):
    result = await result
  return bool(result)


# ---------------------------------------------------------------------------
# Hook implementation
# ---------------------------------------------------------------------------


class _PolicyDecideHook(hooks.PreToolCallDecideHook):
  """PreToolCallDecideHook that enforces a set of policies.

  Created by enforce(). Policies are pre-sorted into priority buckets at
  construction time; evaluation walks buckets high-to-low and short-circuits
  on the first matching policy.
  """

  def __init__(
      self,
      buckets: Sequence[Sequence[Policy]],
      *,
      server_names: Sequence[str] | None = None,
  ):
    self._buckets = buckets
    # Sort descending by length to guarantee longest-match parsing (security)
    self._server_names = sorted(server_names or [], key=len, reverse=True)

  def _parse_mcp_tool(self, tool_name: str) -> tuple[str, str] | None:
    """Securely parses 'mcp_server_tool' into ('server', 'tool') using known servers."""
    if not tool_name.startswith("mcp_"):
      return None

    rest = tool_name[4:]
    for server in self._server_names:
      if rest.startswith(f"{server}_"):
        return server, rest[len(server) + 1 :]
    return None

  async def _evaluate_policy(
      self, p: Policy, tool_call: types.ToolCall
  ) -> hooks.HookResult | None:
    """Evaluates a single policy against the tool call.

    Args:
      p: The policy to evaluate.
      tool_call: The tool call data.

    Returns:
      A HookResult if the policy matches and a decision is made, or None
      if the policy does not match. Propagates exceptions during evaluation.
    """
    mcp_info = self._parse_mcp_tool(tool_call.name)
    if mcp_info:
      server, tool = mcp_info
      call_target = f"{server}/{tool}"
      is_mcp = True
    else:
      call_target = tool_call.name
      is_mcp = False

    if not _matches_target(p.tool, call_target, is_mcp):
      return None

    try:
      if not await _evaluate_predicate(p, tool_call):
        return None

      # First match in this bucket wins.
      return await self._apply(p, tool_call)
    except Exception as e:  # pylint: disable=broad-exception-caught
      _logger.error(
          "Exception during policy %r evaluation — failing closed.",
          p.name or p.tool,
          exc_info=True,
      )
      return hooks.HookResult(
          allow=False,
          message=(
              f"Policy evaluation failed for policy '{p.name or p.tool}':"
              f" {repr(e)}"
          ),
      )

  async def run(
      self, context: hooks.HookContext, data: types.ToolCall
  ) -> hooks.HookResult:
    """Evaluates policies against the tool call.

    Args:
      context: The hook context.
      data: A ToolCall instance.

    Returns:
      HookResult allowing or denying the tool call.
    """
    tool_call = data
    try:
      for bucket in self._buckets:
        for p in bucket:
          result = await self._evaluate_policy(p, tool_call)
          if result is not None:
            return result
    except Exception as e:  # pylint: disable=broad-exception-caught
      _logger.error(
          "Unexpected exception in policy hook — failing closed.",
          exc_info=True,
      )
      return hooks.HookResult(
          allow=False, message=f"Internal policy error: {repr(e)}"
      )

    # No policy matched — default open.
    return hooks.HookResult(allow=True)

  async def _apply(
      self, p: Policy, tool_call: types.ToolCall
  ) -> hooks.HookResult:
    """Applies the matched policy's decision."""
    label = p.name or p.tool

    if p.decision == Decision.DENY:
      _logger.info("Policy %r denied tool %r.", label, tool_call.name)
      return hooks.HookResult(
          allow=False,
          message=f"Denied by policy '{label}'.",
      )

    if p.decision == Decision.APPROVE:
      _logger.info("Policy %r approved tool %r.", label, tool_call.name)
      return hooks.HookResult(allow=True)

    # ASK_USER
    _logger.info(
        "Policy %r requesting user approval for tool %r.",
        label,
        tool_call.name,
    )
    approved = await _execute_ask_user(p, tool_call)
    if approved:
      return hooks.HookResult(allow=True)
    return hooks.HookResult(
        allow=False,
        message=f"User denied tool '{tool_call.name}' (policy '{label}').",
    )


# ---------------------------------------------------------------------------
# Private helpers for hook construction
# ---------------------------------------------------------------------------


def _flatten_policies(
    policies: Sequence[Policy | Sequence[Policy]],
) -> list[Policy]:
  """Flattens nested sequences of policies into a flat list.

  This allows combining single Policy objects and lists of Policies (e.g.
  returned by overloaded builders) seamlessly in config declarations.

  Args:
    policies: A sequence of Policy objects or nested sequences of Policies.

  Returns:
    A flat list of Policy objects.
  """
  flat = []
  for p in policies:
    if isinstance(p, Policy):
      flat.append(p)
    elif isinstance(p, Sequence) and not isinstance(p, (str, bytes)):
      # Safely verify that all sub-elements are indeed Policy instances
      for sub_p in p:
        if not isinstance(sub_p, Policy):
          raise ValueError(f"Expected Policy, got {type(sub_p)}")
      flat.extend(p)
    else:
      raise ValueError(
          f"Expected Policy or Sequence of Policies, got {type(p)}"
      )
  return flat


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def enforce(
    policies: Sequence[Policy | Sequence[Policy]],
    *,
    mcp_servers: Sequence[types.BaseMcpServerConfig] | None = None,
) -> hooks.PreToolCallDecideHook:
  """Creates a PreToolCallDecideHook that enforces the given policies.

  Validates policies at construction time:
  - Every ASK_USER policy must have a handler.
  - MCP policies must have mcp_servers provided.

  Policies are bucketed by priority so that evaluation can short-circuit.

  Args:
    policies: The policies to enforce (can be nested).
    mcp_servers: Optional registered MCP server configurations.

  Returns:
    A PreToolCallDecideHook ready for registration with HookRunner.

  Raises:
    ValueError: If any ASK_USER policy is missing a handler, or if MCP
      policies are used without mcp_servers.
  """
  flat_policies = _flatten_policies(policies)

  # Validate MCP policies against mcp_servers (Fail-Closed Security Guard)
  has_mcp_policy = any(
      ("/" in p.tool and p.tool != _WILDCARD) for p in flat_policies
  )
  if has_mcp_policy and not mcp_servers:
    raise ValueError(
        "MCP policies (containing '/') were detected, but 'mcp_servers' was not"
        " provided to enforce(). You must pass the registered MCP servers to"
        " enable secure policy matching and prevent silent bypasses."
    )

  # Startup validation.
  for p in flat_policies:
    if p.decision == Decision.ASK_USER and p.ask_user is None:
      raise ValueError(
          f"ASK_USER policy '{p.name or p.tool}' is missing an ask_user"
          " handler. Provide one via policy.ask_user(tool, handler=...)."
      )

  # Build priority buckets, preserving registration order within each.
  buckets: list[list[Policy]] = [[] for _ in range(_NUM_LEVELS)]
  for p in flat_policies:
    buckets[_bucket_index(p)].append(p)

  server_names = [s.name for s in mcp_servers] if mcp_servers else []
  return _PolicyDecideHook(buckets, server_names=server_names)
