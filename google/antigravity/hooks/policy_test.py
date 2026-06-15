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

"""Tests for the tool call policy system.

Covers:
- Builder functions (allow, deny, ask_user, allow_all, deny_all)
- Startup validation (missing ASK_USER handler)
- Priority-based evaluation order across all 6 levels
- Short-circuit behavior (first match wins within a group)
- Sync and async predicates, including exception fail-closed
- ASK_USER handler invocation (approve, deny, async, exception)
- Default behavior when no policies match
- Edge cases (empty policy list, policy name in deny reason)
"""

from collections.abc import Mapping
import os
import pathlib
import sys
from typing import Any
import unittest

from absl.testing import absltest
import pydantic

from google.antigravity import types
from google.antigravity.hooks import hooks
from google.antigravity.hooks import policy


class RunCommandArgs(pydantic.BaseModel):
  """Arguments for run_command tool."""

  command_line: str


def _make_tool_call(name: str = "run_command", **args: Any) -> types.ToolCall:
  # Simulate Connection layer path normalization for tests
  canonical_path = None
  for path_key in ("path", "file_path", "TargetFile", "directory_path"):
    if path_key in args and isinstance(args[path_key], str):
      canonical_path = args[path_key]
      break
  return types.ToolCall(name=name, args=args, canonical_path=canonical_path)


class BuilderTest(unittest.TestCase):
  """Verifies that builder functions construct Policy objects correctly."""

  def test_allow_creates_approve_policy(self):
    """allow() must produce a Policy with decision=APPROVE."""
    p = policy.allow("read_file", name="allow-read")
    self.assertEqual(p.tool, "read_file")
    self.assertEqual(p.decision, policy.Decision.APPROVE)
    self.assertIsNone(p.when)
    self.assertIsNone(p.ask_user)
    self.assertEqual(p.name, "allow-read")

  def test_deny_creates_deny_policy(self):
    """deny() must produce a Policy with decision=DENY."""
    p = policy.deny("run_command", name="block-cmd")
    self.assertEqual(p.tool, "run_command")
    self.assertEqual(p.decision, policy.Decision.DENY)
    self.assertEqual(p.name, "block-cmd")

  def test_ask_user_creates_ask_user_policy(self):
    """ask_user() must produce a Policy with decision=ASK_USER and handler."""
    def handler(_):
      return True

    p = policy.ask_user("run_command", handler=handler, name="confirm-cmd")
    self.assertEqual(p.decision, policy.Decision.ASK_USER)
    self.assertIs(p.ask_user, handler)

  def test_deny_with_predicate(self):
    """deny() with a when clause stores the predicate."""
    def pred(args):
      return "rm" in args.get("CommandLine", "")

    p = policy.deny("run_command", when=pred)
    self.assertIs(p.when, pred)

  def test_allow_all_creates_wildcard_approve(self):
    """allow_all() must produce a wildcard APPROVE policy."""
    p = policy.allow_all()
    self.assertEqual(p.tool, "*")
    self.assertEqual(p.decision, policy.Decision.APPROVE)
    self.assertEqual(p.name, "allow_all")

  def test_deny_all_creates_wildcard_deny(self):
    """deny_all() must produce a wildcard DENY policy."""
    p = policy.deny_all()
    self.assertEqual(p.tool, "*")
    self.assertEqual(p.decision, policy.Decision.DENY)
    self.assertEqual(p.name, "deny_all")


class ValidationTest(unittest.TestCase):
  """Verifies startup validation in enforce()."""

  def test_enforce_rejects_ask_user_without_handler(self):
    """enforce() must raise ValueError when ASK_USER has no handler."""
    bad_policy = policy.Policy(
        tool="run_command", decision=policy.Decision.ASK_USER, name="oops"
    )
    with self.assertRaises(ValueError) as ctx:
      policy.enforce([bad_policy])
    self.assertIn("oops", str(ctx.exception))
    self.assertIn("missing an ask_user handler", str(ctx.exception))

  def test_enforce_rejects_ask_user_without_handler_unnamed(self):
    """enforce() error message includes tool name when policy has no name."""
    bad_policy = policy.Policy(
        tool="my_tool", decision=policy.Decision.ASK_USER
    )
    with self.assertRaises(ValueError) as ctx:
      policy.enforce([bad_policy])
    self.assertIn("my_tool", str(ctx.exception))


class PriorityEvaluationTest(unittest.IsolatedAsyncioTestCase):
  """Verifies the 6-level priority evaluation model."""

  async def test_specific_deny_overrides_wildcard_allow(self):
    """Level 1 (specific deny) beats Level 6 (wildcard allow)."""
    hook = policy.enforce([
        policy.allow("*"),
        policy.deny("dangerous_tool"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("dangerous_tool"))
    self.assertFalse(result.allow)

  async def test_specific_deny_overrides_specific_allow(self):
    """Level 1 (specific deny) beats Level 3 (specific allow)."""
    hook = policy.enforce([
        policy.allow("run_command"),
        policy.deny("run_command"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)

  async def test_specific_ask_overrides_wildcard_deny(self):
    """Level 2 (specific ask) beats Level 4 (wildcard deny)."""
    hook = policy.enforce([
        policy.deny("*"),
        policy.ask_user("run_command", handler=lambda tc: True),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    # ask_user handler returns True → approved
    self.assertTrue(result.allow)

  async def test_specific_allow_overrides_wildcard_deny(self):
    """Level 3 (specific allow) beats Level 4 (wildcard deny).

    This is the critical "deny all except X" pattern.
    """
    hook = policy.enforce([
        policy.deny("*"),
        policy.allow("read_file"),
    ])
    ctx = hooks.HookContext()

    result = await hook.run(ctx, _make_tool_call("read_file"))
    self.assertTrue(result.allow)

    # Other tools should still be denied by the wildcard
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)

  async def test_wildcard_deny_blocks_unmatched_tools(self):
    """Level 4 (wildcard deny) blocks tools with no specific policy."""
    hook = policy.enforce([
        policy.deny("*"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("anything"))
    self.assertFalse(result.allow)

  async def test_wildcard_ask_user(self):
    """Level 5 (wildcard ask) applies to all tools."""
    hook = policy.enforce([
        policy.ask_user("*", handler=lambda tc: False),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("any_tool"))
    self.assertFalse(result.allow)

  async def test_wildcard_allow(self):
    """Level 6 (wildcard allow) allows all tools."""
    hook = policy.enforce([
        policy.allow("*"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("any_tool"))
    self.assertTrue(result.allow)


class ShortCircuitTest(unittest.IsolatedAsyncioTestCase):
  """Verifies first-match-wins within a priority group."""

  async def test_first_match_wins_within_deny_group(self):
    """When two specific deny policies match, only the first is evaluated."""
    call_count = 0

    def counting_predicate(unused_args: Mapping[str, Any]) -> bool:
      nonlocal call_count
      call_count += 1
      return True

    hook = policy.enforce([
        policy.deny("run_command", when=counting_predicate, name="first"),
        policy.deny("run_command", when=counting_predicate, name="second"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)
    # Only the first deny's predicate should have been called.
    self.assertEqual(call_count, 1)

  async def test_first_match_wins_within_allow_group(self):
    """When two specific allow policies match, only the first is evaluated."""
    call_count = 0

    def counting_predicate(unused_args: Mapping[str, Any]) -> bool:
      nonlocal call_count
      call_count += 1
      return True

    hook = policy.enforce([
        policy.allow("read_file", when=counting_predicate),
        policy.allow("read_file", when=counting_predicate),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("read_file"))
    self.assertTrue(result.allow)
    self.assertEqual(call_count, 1)

  async def test_skips_non_matching_predicate(self):
    """A policy whose predicate returns False is skipped; next one wins."""
    hook = policy.enforce([
        policy.deny("run_command", when=lambda args: False, name="skip-me"),
        policy.deny("run_command", when=lambda args: True, name="catch-me"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)
    self.assertIn("catch-me", result.message)


class PredicateTest(unittest.IsolatedAsyncioTestCase):
  """Verifies sync, async, and failing predicates."""

  async def test_sync_predicate_true(self):
    """Sync predicate returning True causes the policy to match."""
    hook = policy.enforce([
        policy.deny(
            "run_command",
            when=lambda args: args.get("CommandLine", "").startswith("rm"),
        ),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(
        ctx, _make_tool_call("run_command", CommandLine="rm -rf /")
    )
    self.assertFalse(result.allow)

  async def test_sync_predicate_false(self):
    """Sync predicate returning False skips the policy."""
    hook = policy.enforce([
        policy.deny(
            "run_command",
            when=lambda args: args.get("CommandLine", "").startswith("rm"),
        ),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(
        ctx, _make_tool_call("run_command", CommandLine="echo hi")
    )
    self.assertTrue(result.allow)

  async def test_async_predicate_true(self):
    """Async predicate returning True causes the policy to match."""

    async def is_dangerous(args: Mapping[str, Any]) -> bool:
      return "rm" in args.get("CommandLine", "")

    hook = policy.enforce([
        policy.deny("run_command", when=is_dangerous),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(
        ctx, _make_tool_call("run_command", CommandLine="rm -rf")
    )
    self.assertFalse(result.allow)

  async def test_async_predicate_false(self):
    """Async predicate returning False skips the policy."""

    async def is_dangerous(args: Mapping[str, Any]) -> bool:
      return "rm" in args.get("CommandLine", "")

    hook = policy.enforce([
        policy.deny("run_command", when=is_dangerous),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(
        ctx, _make_tool_call("run_command", CommandLine="echo")
    )
    self.assertTrue(result.allow)

  async def test_predicate_exception_matches_fail_closed(self):
    """Exception in predicate → policy matches (fail-closed).

    This is the critical safety property: a deny policy with a broken
    predicate still denies, preventing accidental allow-through.
    """

    def exploding_predicate(_: Mapping[str, Any]) -> bool:
      raise RuntimeError("boom")

    hook = policy.enforce([
        policy.deny("run_command", when=exploding_predicate, name="broken"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)
    self.assertIn("broken", result.message)
    self.assertIn("boom", result.message)

  async def test_parameterless_predicate(self):
    """Predicate with no arguments should be called without arguments."""

    def no_args_predicate():
      return True

    hook = policy.enforce([
        policy.deny("run_command", when=no_args_predicate, name="no-args"),
    ])
    ctx = hooks.HookContext()

    # This calls no_args_predicate() with 0 arguments.
    # It succeeds and returns True (match), leading to denial.
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)
    self.assertEqual(result.message, "Denied by policy 'no-args'.")

  async def test_parameterless_predicate_allow(self):
    """Parameterless predicate in ALLOW policy works and allows/denies correctly."""
    is_allowed = False

    def my_predicate():
      return is_allowed

    hook = policy.enforce([
        policy.allow("run_command", when=my_predicate, name="paramless-allow"),
        policy.deny("*"),
    ])
    ctx = hooks.HookContext()

    # When predicate returns False -> should deny (via deny("*"))
    is_allowed = False
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)

    # When predicate returns True -> should allow
    is_allowed = True
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertTrue(result.allow)

  async def test_typed_predicate(self):

    """Predicate expecting a Pydantic model receives the parsed object."""

    def my_typed_predicate(args: RunCommandArgs) -> bool:
      return "rm" in args.command_line

    hook = policy.enforce([
        policy.deny("run_command", when=my_typed_predicate),
    ])
    ctx = hooks.HookContext()

    # Matches
    result = await hook.run(
        ctx, _make_tool_call("run_command", command_line="rm -rf")
    )
    self.assertFalse(result.allow)

    # Doesn't match
    result = await hook.run(
        ctx, _make_tool_call("run_command", command_line="echo hi")
    )
    self.assertTrue(result.allow)

  async def test_allow_predicate_exception_denies(self):
    """Exception in allow policy predicate must deny (fail-closed)."""

    def exploding_predicate(_: Mapping[str, Any]) -> bool:
      raise RuntimeError("boom")

    hook = policy.enforce([
        policy.allow(
            "run_command", when=exploding_predicate, name="broken-allow"
        ),
        policy.allow("*"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)
    self.assertIn("broken-allow", result.message)
    self.assertIn("boom", result.message)


class AskUserTest(unittest.IsolatedAsyncioTestCase):
  """Verifies ASK_USER handler invocation."""

  async def test_handler_approve(self):
    """Handler returning True → tool is allowed."""
    hook = policy.enforce([
        policy.ask_user("run_command", handler=lambda tc: True),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertTrue(result.allow)

  async def test_handler_deny(self):
    """Handler returning False → tool is denied."""
    hook = policy.enforce([
        policy.ask_user("run_command", handler=lambda tc: False),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)
    self.assertIn("User denied", result.message)

  async def test_handler_async(self):
    """Async handler is awaited correctly."""

    async def async_handler(tc: types.ToolCall) -> bool:
      return tc.args.get("safe", False)

    hook = policy.enforce([
        policy.ask_user("run_command", handler=async_handler),
    ])
    ctx = hooks.HookContext()

    result = await hook.run(ctx, _make_tool_call("run_command", safe=True))
    self.assertTrue(result.allow)

    result = await hook.run(ctx, _make_tool_call("run_command", safe=False))
    self.assertFalse(result.allow)

  async def test_handler_exception_denies(self):
    """Handler exception is caught and denies the tool call."""

    def broken_handler(_: types.ToolCall) -> bool:
      raise RuntimeError("handler broke")

    hook = policy.enforce([
        policy.ask_user(
            "run_command", handler=broken_handler, name="broken-ask"
        ),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)
    self.assertIn("broken-ask", result.message)
    self.assertIn("handler broke", result.message)

  async def test_handler_receives_tool_call(self):
    """Handler receives the full ToolCall object, not just args."""
    received = []

    def capturing_handler(tc: types.ToolCall) -> bool:
      received.append(tc)
      return True

    hook = policy.enforce([
        policy.ask_user("run_command", handler=capturing_handler),
    ])
    ctx = hooks.HookContext()
    tc = _make_tool_call("run_command", CommandLine="echo hi")
    await hook.run(ctx, tc)
    self.assertEqual(len(received), 1)
    self.assertIs(received[0], tc)


class DefaultBehaviorTest(unittest.IsolatedAsyncioTestCase):
  """Verifies behavior when no policies match."""

  async def test_no_matching_policy_allows(self):
    """When no policy matches, the tool call is allowed (open system)."""
    hook = policy.enforce([
        policy.deny("other_tool"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("unrelated_tool"))
    self.assertTrue(result.allow)

  async def test_empty_policies_allows_all(self):
    """An empty policy list allows everything."""
    hook = policy.enforce([])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("any_tool"))
    self.assertTrue(result.allow)


class ConvenienceBuilderTest(unittest.IsolatedAsyncioTestCase):
  """Verifies allow_all() and deny_all() evaluate correctly through enforce."""

  async def test_allow_all_approves_any_tool(self):
    """allow_all() approves arbitrary tool calls."""
    hook = policy.enforce([policy.allow_all()])
    ctx = hooks.HookContext()
    for tool in ("run_command", "view_file", "create_file", "unknown_tool"):
      result = await hook.run(ctx, _make_tool_call(tool))
      self.assertTrue(result.allow, f"{tool} should be allowed")

  async def test_deny_all_denies_any_tool(self):
    """deny_all() denies arbitrary tool calls."""
    hook = policy.enforce([policy.deny_all()])
    ctx = hooks.HookContext()
    for tool in ("run_command", "view_file", "create_file"):
      result = await hook.run(ctx, _make_tool_call(tool))
      self.assertFalse(result.allow, f"{tool} should be denied")

  async def test_deny_all_with_specific_allow_override(self):
    """deny_all() + allow(tool) creates deny-by-default with exceptions."""
    hook = policy.enforce([
        policy.deny_all(),
        policy.allow("view_file"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("view_file"))
    self.assertTrue(result.allow)
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)


class DenyReasonTest(unittest.IsolatedAsyncioTestCase):
  """Verifies that deny reasons include useful context."""

  async def test_named_policy_in_deny_reason(self):
    """Policy name appears in the deny reason message."""
    hook = policy.enforce([
        policy.deny("run_command", name="no-commands"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertIn("no-commands", result.message)

  async def test_unnamed_policy_uses_tool_name(self):
    """When a policy has no name, the tool name is used in the reason."""
    hook = policy.enforce([
        policy.deny("run_command"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertIn("run_command", result.message)


class IntegrationWithHookRunnerTest(unittest.IsolatedAsyncioTestCase):
  """Verifies the policy hook integrates with HookRunner dispatch."""

  async def test_policy_hook_in_hook_runner(self):
    """Policy hook works when dispatched through HookRunner.

    This confirms the hook is a proper PreToolCallDecideHook subclass
    that the HookRunner can dispatch.
    """
    from google.antigravity.hooks import hook_runner  # pylint: disable=g-import-not-at-top

    hook = policy.enforce([
        policy.deny("*"),
        policy.allow("read_file"),
    ])

    runner = hook_runner.HookRunner(pre_tool_call_decide_hooks=[hook])
    turn_context = hooks.TurnContext(runner.session_context)

    # read_file should be allowed
    result, _, _ = await runner.dispatch_pre_tool_call(
        turn_context, _make_tool_call("read_file")
    )
    self.assertTrue(result.allow)

    # run_command should be denied
    result, _, _ = await runner.dispatch_pre_tool_call(
        turn_context, _make_tool_call("run_command")
    )
    self.assertFalse(result.allow)


class SafeDefaultsTest(unittest.IsolatedAsyncioTestCase):
  """Verifies safe_defaults() preset."""

  async def test_safe_defaults_allows_read_only_tools(self):
    """safe_defaults() must allow read-only tools."""

    def handler(_):
      return False

    policies = policy.safe_defaults(handler=handler)
    hook = policy.enforce(policies)
    ctx = hooks.HookContext()

    for tool in (
        "list_directory",
        "search_directory",
        "find_file",
        "view_file",
        "finish",
    ):
      result = await hook.run(ctx, _make_tool_call(tool))
      self.assertTrue(result.allow, f"{tool} should be allowed")

  async def test_safe_defaults_asks_for_other_tools(self):
    """safe_defaults() must ask for non-read-only tools."""
    handler_called = False

    def handler(_):
      nonlocal handler_called
      handler_called = True
      return True

    policies = policy.safe_defaults(handler=handler)
    hook = policy.enforce(policies)
    ctx = hooks.HookContext()

    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertTrue(result.allow)
    self.assertTrue(handler_called)


class ConfirmCommandsTest(unittest.IsolatedAsyncioTestCase):
  """Verifies the confirm_run_command() preset — the default for LocalAgentConfig.

  confirm_run_command() is the safe-by-default policy: it denies run_command
  while allowing all other tools.  When a handler is provided, run_command
  is upgraded to ASK_USER instead of DENY.
  """

  async def test_denies_run_command_by_default(self):
    """Without a handler, run_command is denied with a clear message."""
    policies = policy.confirm_run_command()
    hook = policy.enforce(policies)
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)
    self.assertIn("confirm_run_command", result.message)

  async def test_allows_other_tools_by_default(self):
    """Without a handler, all non-run_command tools are allowed."""
    policies = policy.confirm_run_command()
    hook = policy.enforce(policies)
    ctx = hooks.HookContext()
    for tool in types.BuiltinTools:
      if tool == types.BuiltinTools.RUN_COMMAND:
        continue
      result = await hook.run(ctx, _make_tool_call(tool.value))
      self.assertTrue(result.allow, f"{tool.value} should be allowed")

  async def test_with_handler_asks_user_for_run_command(self):
    """With a handler, run_command triggers ASK_USER instead of DENY."""
    handler_calls = []

    def handler(tc: types.ToolCall) -> bool:
      handler_calls.append(tc)
      return True

    policies = policy.confirm_run_command(handler=handler)
    hook = policy.enforce(policies)
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertTrue(result.allow)
    self.assertEqual(len(handler_calls), 1)

  async def test_with_handler_deny_propagates(self):
    """Handler returning False denies the tool call."""
    policies = policy.confirm_run_command(handler=lambda tc: False)
    hook = policy.enforce(policies)
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)
    self.assertIn("User denied", result.message)

  async def test_with_handler_allows_other_tools(self):
    """With a handler, non-run_command tools are still auto-allowed."""
    policies = policy.confirm_run_command(handler=lambda tc: False)
    hook = policy.enforce(policies)
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("view_file"))
    self.assertTrue(result.allow)

  def test_returns_list_of_policies(self):
    """confirm_run_command() always returns a list of Policy objects."""
    for policies in (
        policy.confirm_run_command(),
        policy.confirm_run_command(handler=lambda tc: True),
    ):
      self.assertIsInstance(policies, list)
      self.assertGreaterEqual(len(policies), 2)
      for p in policies:
        self.assertIsInstance(p, policy.Policy)


class WorkspaceOnlyTest(unittest.IsolatedAsyncioTestCase):
  """Verifies workspace_only() — restricts file tools to workspace dirs.

  File tools targeting paths outside configured workspaces are denied.
  Non-file tools and calls without path arguments are unaffected.
  """

  async def test_allows_files_inside_workspace(self):
    """File tool with path inside workspace is allowed."""
    policies = policy.workspace_only(["/tmp/workspace"])
    hook = policy.enforce(policies)
    ctx = hooks.HookContext()
    result = await hook.run(
        ctx, _make_tool_call("view_file", path="/tmp/workspace/foo.py")
    )
    self.assertTrue(result.allow)

  async def test_denies_files_outside_workspace(self):
    """File tool with path outside workspace is denied."""
    policies = policy.workspace_only(["/tmp/workspace"])
    hook = policy.enforce(policies)
    ctx = hooks.HookContext()
    result = await hook.run(
        ctx, _make_tool_call("view_file", path="/etc/secrets/key.pem")
    )
    self.assertFalse(result.allow)
    self.assertIn("workspace_only", result.message)

  async def test_denies_create_outside_workspace(self):
    """create_file outside workspace is denied."""
    policies = policy.workspace_only(["/tmp/workspace"])
    hook = policy.enforce(policies)
    ctx = hooks.HookContext()
    result = await hook.run(
        ctx, _make_tool_call("create_file", TargetFile="/etc/malicious.sh")
    )
    self.assertFalse(result.allow)

  async def test_denies_edit_outside_workspace(self):
    """edit_file outside workspace is denied."""
    policies = policy.workspace_only(["/tmp/workspace"])
    hook = policy.enforce(policies)
    ctx = hooks.HookContext()
    result = await hook.run(
        ctx, _make_tool_call("edit_file", file_path="/etc/passwd")
    )
    self.assertFalse(result.allow)

  async def test_allows_non_file_tools(self):
    """Non-file tools are unaffected — no matching policy, default allows."""
    policies = policy.workspace_only(["/tmp/workspace"])
    hook = policy.enforce(policies)
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertTrue(result.allow)

  async def test_allows_when_no_path_arg(self):
    """File tool with no path argument is allowed (don't break edge cases)."""
    policies = policy.workspace_only(["/tmp/workspace"])
    hook = policy.enforce(policies)
    ctx = hooks.HookContext()
    # A view_file call with no path arg — the predicate returns False,
    # so the deny policy doesn't match and the call is allowed.
    result = await hook.run(ctx, _make_tool_call("view_file"))
    self.assertTrue(result.allow)

  async def test_multiple_workspaces(self):
    """Paths in any configured workspace are allowed."""
    policies = policy.workspace_only(["/tmp/ws1", "/tmp/ws2"])
    hook = policy.enforce(policies)
    ctx = hooks.HookContext()
    result = await hook.run(
        ctx, _make_tool_call("view_file", path="/tmp/ws1/a.py")
    )
    self.assertTrue(result.allow)
    result = await hook.run(
        ctx, _make_tool_call("view_file", path="/tmp/ws2/b.py")
    )
    self.assertTrue(result.allow)

  async def test_prevents_path_prefix_attack(self):
    """Path /workspace-evil/file.txt must NOT match workspace /workspace.

    Uses os.sep boundary check to prevent prefix-based traversal.
    """
    policies = policy.workspace_only(["/tmp/workspace"])
    hook = policy.enforce(policies)
    ctx = hooks.HookContext()
    result = await hook.run(
        ctx, _make_tool_call("view_file", path="/tmp/workspace-evil/file.txt")
    )
    self.assertFalse(result.allow)

  async def test_exact_workspace_path_allowed(self):
    """A path that is exactly the workspace directory itself is allowed."""
    policies = policy.workspace_only(["/tmp/workspace"])
    hook = policy.enforce(policies)
    ctx = hooks.HookContext()
    result = await hook.run(
        ctx, _make_tool_call("view_file", path="/tmp/workspace")
    )
    self.assertTrue(result.allow)


class PolicyPathScopingDirectTest(absltest.TestCase):
  """Direct unit tests for path normalization and workspace scoping."""

  def setUp(self):
    super().setUp()
    # Symmetrically resolve base directory using standard buildenv temp dirs
    self.temp_dir_path = pathlib.Path(self.create_tempdir().full_path).resolve()

  def test_secure_normalize_path_resolves_existing_symlinks(self):
    """_secure_normalize_path must follow and resolve existing symlinks."""
    # Create real folder and symlink pointing to it
    real_dir = self.temp_dir_path / "real_dir"
    real_dir.mkdir(exist_ok=True)
    symlink_dir = self.temp_dir_path / "symlink_dir"

    try:
      os.symlink(real_dir, symlink_dir)
    except OSError:
      self.skipTest("Symbolic links are not supported in this environment.")

    resolved_path = policy._secure_normalize_path(str(symlink_dir / "file.txt"))

    # Assert that the symlinked parent was resolved to the canonical real_dir
    self.assertEqual(resolved_path, real_dir / "file.txt")

  def test_is_case_insensitive_prober(self):
    """_is_case_insensitive must dynamically check OS filesystem case sensitivity."""
    # Probe our active hermetic temp directory
    is_ci = policy._is_case_insensitive(self.temp_dir_path)

    # Validate platform expectations
    expected_ci = sys.platform in ("win32", "darwin")
    self.assertEqual(is_ci, expected_ci)

  def test_is_path_in_workspace_structural_containment(self):
    """is_path_in_workspace must securely check path containment component-wise."""
    ws = self.temp_dir_path / "my_workspace"
    ws.mkdir(exist_ok=True)

    self.assertTrue(
        policy._is_path_in_workspace(str(ws / "sub/file.txt"), str(ws))
    )

    self.assertTrue(policy._is_path_in_workspace(str(ws), str(ws)))

    evil_ws = self.temp_dir_path / "my_workspace-evil"
    self.assertFalse(
        policy._is_path_in_workspace(str(evil_ws / "file.txt"), str(ws))
    )

    self.assertFalse(
        policy._is_path_in_workspace(
            str(self.temp_dir_path / "outside.txt"), str(ws)
        )
    )

  def test_is_path_in_workspace_case_folding(self):
    """is_path_in_workspace must fold casing symmetrically on case-insensitive drives."""
    ws = self.temp_dir_path / "WorkspaceDir"
    ws.mkdir(exist_ok=True)

    # Query filesystem casing to verify case folding behavior
    if policy._is_case_insensitive(ws):
      # On case-insensitive APFS/Windows, lowercased paths must match
      lower_target = str(ws).lower() + "/sub/file.txt"
      self.assertTrue(policy._is_path_in_workspace(lower_target, str(ws)))
    else:
      # On case-sensitive EXT4 Linux, mismatched casing must be blocked
      upper_target = str(ws).upper() + "/sub/file.txt"
      self.assertFalse(policy._is_path_in_workspace(upper_target, str(ws)))


class McpPolicyTest(unittest.IsolatedAsyncioTestCase):
  """Unit tests for overloaded MCP policy methods and structured target alignment."""

  def setUp(self):
    super().setUp()
    self.mcp_config = types.McpStdioServer(name="math", command="npx")
    self.mcp_config_adv = types.McpStdioServer(
        name="math_advanced", command="npx"
    )

  def test_allow_mcp_builder_wildcard(self):
    """allow(mcp_config) must produce a single wildcard policy in structured target format."""
    policies = policy.allow(self.mcp_config)
    self.assertIsInstance(policies, list)
    (p,) = policies  # Implicitly asserts length is exactly 1
    self.assertEqual(p.tool, "math/*")
    self.assertEqual(p.decision, policy.Decision.APPROVE)

  def test_allow_mcp_builder_specific_tools(self):
    """allow(mcp_config, tools) must produce policies in 'server/tool' format."""
    policies = policy.allow(self.mcp_config, ["calc", "multiply"])
    self.assertIsInstance(policies, list)
    p1, p2 = policies  # Implicitly asserts length is exactly 2
    self.assertEqual(p1.tool, "math/calc")
    self.assertEqual(p2.tool, "math/multiply")
    self.assertEqual(p1.name, "approve_math_calc")

  def test_deny_mcp_builder_specific_tools(self):
    """deny(mcp_config, tools) must produce policies in 'server/tool' format."""
    policies = policy.deny(self.mcp_config, ["calc"])
    self.assertIsInstance(policies, list)
    (p,) = policies  # Implicitly asserts length is exactly 1
    self.assertEqual(p.tool, "math/calc")
    self.assertEqual(p.decision, policy.Decision.DENY)

  def test_ask_user_mcp_builder_specific_tools(self):
    """ask_user(mcp_config, tools) must produce policies in 'server/tool' format with handler."""

    def dummy_handler(tc):
      return True

    policies = policy.ask_user(self.mcp_config, ["calc"], handler=dummy_handler)
    self.assertIsInstance(policies, list)
    (p,) = policies  # Implicitly asserts length is exactly 1
    self.assertEqual(p.tool, "math/calc")
    self.assertEqual(p.decision, policy.Decision.ASK_USER)
    self.assertIs(p.ask_user, dummy_handler)

  def test_builder_custom_name_unique(self):
    """Builders must append tool suffix to custom name for unique logging."""
    policies = policy.allow(self.mcp_config, ["calc"], name="custom")
    (p,) = policies  # Implicitly asserts length is exactly 1
    self.assertEqual(p.name, "custom_calc")

  def test_builder_rejects_string_with_mcp_tools(self):
    """Builders must raise ValueError if mcp_tools is provided for a string tool name."""
    with self.assertRaises(ValueError) as ctx:
      policy.allow("read_file", ["tool1"])
    self.assertIn("mcp_tools cannot be specified", str(ctx.exception))

  def test_mcp_tools_string_type_guard(self):
    """Builders must raise ValueError if mcp_tools is passed as a raw string."""
    with self.assertRaises(ValueError) as ctx:
      policy.allow(self.mcp_config, "calc")
    self.assertIn("mcp_tools must be a sequence of strings", str(ctx.exception))

  def test_enforce_fails_closed_on_missing_servers(self):
    """enforce() must raise ValueError if MCP policies exist but mcp_servers is missing."""
    policies = policy.allow(self.mcp_config)
    with self.assertRaises(ValueError) as ctx:
      policy.enforce(policies)  # mcp_servers omitted!
    self.assertIn(
        "'mcp_servers' was not provided to enforce()", str(ctx.exception)
    )

  def test_enforce_flattens_nested_policies(self):
    """enforce() must successfully flatten mixed nested lists of policies."""
    policies = [
        policy.allow("read_file"),
        policy.allow(self.mcp_config),  # Returns list[Policy]
    ]
    hook = policy.enforce(policies, mcp_servers=[self.mcp_config])
    self.assertIsInstance(hook, hooks.PreToolCallDecideHook)

  async def test_secure_longest_match_matching(self):
    """math prefix must not eagerly match math_advanced tools."""
    # Register both servers (math_advanced has longer name)
    policies = [
        policy.allow(self.mcp_config),  # math/*
        policy.deny(self.mcp_config_adv),  # math_advanced/*
    ]
    hook = policy.enforce(
        policies, mcp_servers=[self.mcp_config, self.mcp_config_adv]
    )
    ctx = hooks.HookContext()

    result = await hook.run(ctx, _make_tool_call("mcp_math_advanced_calc"))
    self.assertFalse(result.allow)
    self.assertIn("math_advanced_all", result.message)

    result = await hook.run(ctx, _make_tool_call("mcp_math_calc"))
    self.assertTrue(result.allow)

  async def test_9_level_priority_specific_allow_beats_prefix_deny(self):
    """Specific Allow (level 2) must beat Prefix Deny (level 3)."""
    policies = [
        policy.allow(
            self.mcp_config, ["calc"]
        ),  # math/calc -> Specific Allow (level 2)
        policy.deny(self.mcp_config),  # math/* -> Prefix Deny (level 3)
    ]
    hook = policy.enforce(policies, mcp_servers=[self.mcp_config])
    ctx = hooks.HookContext()

    result = await hook.run(ctx, _make_tool_call("mcp_math_calc"))
    self.assertTrue(result.allow)

    result = await hook.run(ctx, _make_tool_call("mcp_math_multiply"))
    self.assertFalse(result.allow)

  def test_enforce_rejects_non_policy_in_sequence(self):
    """enforce() must raise ValueError if a non-Policy is found in a nested sequence."""
    bad_policies = [
        policy.allow("read_file"),
        ["not_a_policy"],  # Invalid element in nested list
    ]
    with self.assertRaises(ValueError) as ctx:
      policy.enforce(bad_policies, mcp_servers=[self.mcp_config])
    self.assertIn("Expected Policy, got <class 'str'>", str(ctx.exception))

  def test_enforce_rejects_invalid_policy_type(self):
    """enforce() must raise ValueError if a direct invalid type is passed."""
    bad_policies = [
        policy.allow("read_file"),
        123,  # Invalid direct type
    ]
    with self.assertRaises(ValueError) as ctx:
      policy.enforce(bad_policies, mcp_servers=[self.mcp_config])
    self.assertIn(
        "Expected Policy or Sequence of Policies, got <class 'int'>",
        str(ctx.exception),
    )

  async def test_matches_target_unknown_server_prefix(self):
    """Prefixed calls with unknown server name must be treated as standard tools."""
    policies = [
        policy.allow("math/*"),  # math/*
    ]
    # 'unknown' is NOT in mcp_servers
    hook = policy.enforce(policies, mcp_servers=[self.mcp_config])
    ctx = hooks.HookContext()

    # Since 'unknown' is unregistered, 'mcp_unknown_calc' is treated as a standard tool.
    # It should NOT match the 'math/*' prefix wildcard.
    result = await hook.run(ctx, _make_tool_call("mcp_unknown_calc"))
    self.assertTrue(result.allow)  # Default open since no policy matches


if __name__ == "__main__":
  absltest.main()
