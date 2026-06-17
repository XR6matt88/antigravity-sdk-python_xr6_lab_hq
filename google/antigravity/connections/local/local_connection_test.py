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

"""Unit tests for LocalConnection."""

import asyncio
import datetime
import importlib
import io
import json
import os
import pathlib
import struct
import subprocess
import tempfile
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import pydantic
import websockets

from google.antigravity import types
from google.antigravity.connections.local import local_connection
from google.antigravity.connections.local import local_connection_config
from google.antigravity.connections.local import localharness_pb2
from google.antigravity.connections.local import test_utils
from google.antigravity.hooks import hook_runner
from google.antigravity.hooks import hooks as hooks_base
from google.antigravity.hooks import policy
from google.antigravity.models import DEFAULT_MODEL
from google.antigravity.tools import tool_runner
from google.antigravity.types import QuestionResponse


class LocalConnectionTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock(spec=subprocess.Popen)
    self.tool_runner = tool_runner.ToolRunner()

  def _make_harness(self, hook_runner=None, initial_history=None):
    return test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hook_runner,
        initial_history=initial_history,
    )

  async def test_initial_handshake_synchronization(self):
    """Verifies that L2 connections receive restored historical steps synchronously during initialization and start fully idle."""
    hist_step = local_connection.LocalConnectionStep(
        step_index=1,
        content="Historical text",
        status=types.StepStatus.DONE,
        source=types.StepSource.MODEL,
    )
    harness = self._make_harness(initial_history=[hist_step])

    # 1. Confirm connection starts 100% fully idle by default
    self.assertTrue(harness.conn.is_idle)

    # 2. Confirm historical steps are exposed immediately on _initial_history
    self.assertEqual(len(harness.conn._initial_history), 1)
    self.assertEqual(
        harness.conn._initial_history[0].content, "Historical text"
    )

  async def test_receive_steps_basic(self):
    harness = self._make_harness()
    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            text="Hello world",
            state=localharness_pb2.StepUpdate.STATE_ACTIVE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
        )
    )

    await harness.send_event(event)
    await harness.close_from_harness_side()

    # Simulate that a turn is active (send clears this in reality)
    harness.conn._is_idle.clear()

    steps = []
    async for step in harness.conn.receive_steps():
      steps.append(step)

    self.assertEqual(len(steps), 1)
    self.assertEqual(steps[0].content, "Hello world")
    self.assertEqual(steps[0].status, types.StepStatus.ACTIVE)
    self.assertEqual(steps[0].source, types.StepSource.MODEL)

  async def test_receive_steps_system_error(self):
    harness = self._make_harness()
    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            error=localharness_pb2.ActionError(
                error_message="Fatal system failure",
                http_code=400,
            ),
            state=localharness_pb2.StepUpdate.STATE_ERROR,
            source=localharness_pb2.StepUpdate.SOURCE_SYSTEM,
        )
    )

    await harness.send_event(event)
    await harness.close_from_harness_side()
    harness.conn._is_idle.clear()

    # receive_steps should raise AntigravityConnectionError when it
    # encounters the system error step.
    with self.assertRaisesRegex(
        types.AntigravityConnectionError, "Fatal system failure"
    ):
      async for _ in harness.conn.receive_steps():
        pass

  async def test_receive_steps_system_error_401(self):
    harness = self._make_harness()
    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            error=localharness_pb2.ActionError(
                error_message="Unauthorized access",
                http_code=401,
            ),
            state=localharness_pb2.StepUpdate.STATE_ERROR,
            source=localharness_pb2.StepUpdate.SOURCE_SYSTEM,
        )
    )

    await harness.send_event(event)
    await harness.close_from_harness_side()
    harness.conn._is_idle.clear()

    # receive_steps should raise AntigravityConnectionError when it
    # encounters the system error step.
    with self.assertRaisesRegex(
        types.AntigravityConnectionError, "Unauthorized access"
    ):
      async for _ in harness.conn.receive_steps():
        pass

  async def test_receive_steps_system_error_429(self):
    harness = self._make_harness()
    mock_trajectory_id = "my_cascade"

    await harness.conn.send("Hello")
    init_data = await harness.wait_for_response()
    self.assertEqual(init_data.get("userInput"), "Hello")

    # Set the cascade ID and send the 429 error step.
    event1 = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id=mock_trajectory_id,
            trajectory_id=mock_trajectory_id,
            step_index=1,
            error=localharness_pb2.ActionError(
                error_message="Resource exhausted",
                http_code=429,
            ),
            state=localharness_pb2.StepUpdate.STATE_ERROR,
            source=localharness_pb2.StepUpdate.SOURCE_SYSTEM,
        )
    )
    await harness.send_event(event1)

    # Send idle update with error.
    event2 = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            trajectory_id=mock_trajectory_id,
            state=localharness_pb2.TrajectoryStateUpdate.STATE_IDLE,
            error="executor run failed: Resource exhausted",
        )
    )
    await harness.send_event(event2)
    await harness.close_from_harness_side()

    steps = []
    with self.assertRaisesRegex(
        types.AntigravityExecutionError,
        "executor run failed: Resource exhausted",
    ):
      async for step in harness.conn.receive_steps():
        steps.append(step)

    self.assertEqual(len(steps), 1)
    self.assertEqual(steps[0].error, "Resource exhausted")
    self.assertEqual(steps[0].status, types.StepStatus.ERROR)

  async def test_receive_steps_trajectory_error(self):
    harness = self._make_harness()

    await harness.conn.send("Hello")
    init_data = await harness.wait_for_response()
    self.assertEqual(init_data.get("userInput"), "Hello")

    # Set the cascade ID.
    event1 = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="my_cascade",
            trajectory_id="my_cascade",
            step_index=1,
            state=localharness_pb2.StepUpdate.STATE_ACTIVE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            text="I'm working",
        )
    )
    await harness.send_event(event1)

    # Send an error.
    event2 = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            trajectory_id="my_cascade",
            state=localharness_pb2.TrajectoryStateUpdate.STATE_IDLE,
            error="Trajectory execution failed",
        )
    )
    await harness.send_event(event2)
    await harness.close_from_harness_side()

    steps = []
    with self.assertRaisesRegex(
        types.AntigravityExecutionError, "Trajectory execution failed"
    ):
      async for step in harness.conn.receive_steps():
        steps.append(step)

    self.assertEqual(len(steps), 1)
    self.assertEqual(steps[0].content, "I'm working")

  def test_local_connection_step_from_dict(self):
    """Tests that LocalConnectionStep maps fields correctly."""
    step_dict = {
        "step_index": 1,
        "text": "Hello world",
        "state": "STATE_ACTIVE",
        "source": "SOURCE_MODEL",
        "target": "TARGET_USER",
    }
    step = local_connection.LocalConnectionStep.from_dict(step_dict)
    self.assertEqual(step.id, "1")
    self.assertEqual(step.content, "Hello world")
    self.assertEqual(step.status, types.StepStatus.ACTIVE)
    self.assertEqual(step.source, types.StepSource.MODEL)
    self.assertEqual(step.target, "TARGET_USER")

  def test_local_connection_step_from_dict_thinking(self):
    """Tests that thinking field is correctly populated from step dict."""
    step_dict = {
        "step_index": 1,
        "text": "",
        "thinking": "Let me analyze this step by step.",
        "state": "STATE_DONE",
        "source": "SOURCE_MODEL",
    }
    step = local_connection.LocalConnectionStep.from_dict(step_dict)
    self.assertEqual(step.thinking, "Let me analyze this step by step.")
    self.assertEqual(step.content, "")

  def test_local_connection_step_from_dict_thinking_empty_by_default(self):
    """Tests that thinking defaults to empty string when not present."""
    step_dict = {
        "step_index": 1,
        "text": "Hello",
        "state": "STATE_DONE",
        "source": "SOURCE_MODEL",
    }
    step = local_connection.LocalConnectionStep.from_dict(step_dict)
    self.assertEqual(step.thinking, "")
    self.assertEqual(step.content, "Hello")

  async def test_receive_steps_thinking_populated(self):
    """Tests that thinking field flows from proto through to SDK Step."""
    harness = self._make_harness()
    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            text="",
            thinking="Internal reasoning about the problem.",
            state=localharness_pb2.StepUpdate.STATE_DONE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
        )
    )

    await harness.send_event(event)
    await harness.close_from_harness_side()
    harness.conn._is_idle.clear()

    steps = []
    async for step in harness.conn.receive_steps():
      steps.append(step)

    self.assertEqual(len(steps), 1)
    self.assertEqual(steps[0].thinking, "Internal reasoning about the problem.")
    self.assertEqual(steps[0].content, "")

  async def test_receive_steps_thinking_and_text_independent(self):
    """Tests that thinking and text are independent, non-exclusive fields.

    This is the key behavioral invariant: the translator must populate both
    fields from the same model response. A regression to mutually exclusive
    branches would zero out one of the two.
    """
    harness = self._make_harness()
    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            text="Here is my answer.",
            thinking="Let me reason through this carefully.",
            state=localharness_pb2.StepUpdate.STATE_DONE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
        )
    )

    await harness.send_event(event)
    await harness.close_from_harness_side()
    harness.conn._is_idle.clear()

    steps = []
    async for step in harness.conn.receive_steps():
      steps.append(step)

    self.assertEqual(len(steps), 1)
    self.assertEqual(steps[0].content, "Here is my answer.")
    self.assertEqual(steps[0].thinking, "Let me reason through this carefully.")

  async def test_thinking_only_step_is_target_user_not_complete(self):
    """Tests that thinking-only steps are TARGET_USER but not is_complete_response.

    Thinking is user-visible output (TARGET_USER), but a step with only
    thinking and no text must not be flagged as a complete response —
    otherwise the SDK would prematurely treat the turn as finished.
    """
    step_dict = {
        "step_index": 1,
        "text": "",
        "thinking": "Internal reasoning about the problem.",
        "state": "STATE_DONE",
        "source": "SOURCE_MODEL",
        "target": "TARGET_USER",
    }
    step = local_connection.LocalConnectionStep.from_dict(step_dict)
    self.assertEqual(step.thinking, "Internal reasoning about the problem.")
    self.assertEqual(step.target, "TARGET_USER")
    self.assertFalse(step.is_complete_response)

  def test_local_connection_step_from_dict_content_delta(self):
    """Tests that content_delta is correctly parsed from text_delta."""
    step_dict = {
        "step_index": 1,
        "text": "Hello world",
        "text_delta": " world",
        "state": "STATE_DONE",
        "source": "SOURCE_MODEL",
    }
    step = local_connection.LocalConnectionStep.from_dict(step_dict)
    self.assertEqual(step.content, "Hello world")
    self.assertEqual(step.content_delta, " world")

  def test_local_connection_step_from_dict_thinking_delta(self):
    """Tests that thinking_delta is correctly parsed."""
    step_dict = {
        "step_index": 1,
        "text": "",
        "thinking": "Step 1. Step 2.",
        "thinking_delta": " Step 2.",
        "state": "STATE_DONE",
        "source": "SOURCE_MODEL",
    }
    step = local_connection.LocalConnectionStep.from_dict(step_dict)
    self.assertEqual(step.thinking, "Step 1. Step 2.")
    self.assertEqual(step.thinking_delta, " Step 2.")

  def test_local_connection_step_from_dict_deltas_default_empty(self):
    """Tests that delta fields default to empty when not present."""
    step_dict = {
        "step_index": 1,
        "text": "Hello",
        "state": "STATE_DONE",
        "source": "SOURCE_MODEL",
    }
    step = local_connection.LocalConnectionStep.from_dict(step_dict)
    self.assertEqual(step.content_delta, "")
    self.assertEqual(step.thinking_delta, "")

  async def test_turn_hook_deny(self):
    hr = hook_runner.HookRunner()

    @hooks_base.pre_turn
    async def denying_turn(data):
      return hooks_base.HookResult(allow=False, message="Denied by hook")

    hr.register_hook(denying_turn)

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    await harness.conn.send("Hello")

    steps = []
    async for step in harness.conn.receive_steps():
      steps.append(step)

    self.assertEqual(len(steps), 1)
    self.assertEqual(steps[0].status, types.StepStatus.CANCELED)
    self.assertEqual(steps[0].error, "Denied by hook")

  async def test_send_none_dispatches_turn_hook_with_empty_string(self):
    hr = hook_runner.HookRunner()
    captured = []

    @hooks_base.pre_turn
    async def capturing_turn(data: str) -> hooks_base.HookResult:
      captured.append(data)
      return hooks_base.HookResult(allow=True)

    hr.register_hook(capturing_turn)

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    await harness.conn.send(None)
    self.assertEqual(captured, [""])

  async def test_tool_hook_deny(self):
    hr = hook_runner.HookRunner()

    @hooks_base.pre_tool_call_decide
    async def denying_tool(data):
      return hooks_base.HookResult(allow=False, message="Denied tool")

    hr.register_hook(denying_tool)

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    event = localharness_pb2.OutputEvent(
        tool_call=localharness_pb2.ToolCall(
            id="call_1",
            name="some_tool",
            arguments_json="{}",
        )
    )

    await harness.send_event(event)

    # Verify that ToolResponse was sent back to harness denying it
    sent_data = await harness.wait_for_response()
    self.assertIn("toolResponse", sent_data)
    resp = sent_data["toolResponse"]
    self.assertEqual(resp["id"], "call_1")
    self.assertIn("Denied tool", resp["responseJson"])

  async def test_tool_confirmation_request_integration(self):
    hr = hook_runner.HookRunner()

    @hooks_base.pre_tool_call_decide
    async def denying_tool(data):
      return hooks_base.HookResult(allow=False)

    hr.register_hook(denying_tool)

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            trajectory_id="test_traj",
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            tool_confirmation_request=localharness_pb2.ToolConfirmationRequest(),
            view_file=localharness_pb2.ActionViewFile(file_path="/foo/bar"),
        )
    )

    await harness.send_event(event)

    sent_data = await harness.wait_for_response()
    self.assertIn("toolConfirmation", sent_data)
    self.assertEqual(sent_data["toolConfirmation"]["trajectoryId"], "test_traj")
    self.assertFalse(sent_data["toolConfirmation"]["accepted"])

  async def test_mcp_tool_confirmation_request_allowed(self):
    hr = hook_runner.HookRunner()
    captured_tool_calls = []

    @hooks_base.pre_tool_call_decide
    async def policy_hook(data: types.ToolCall) -> hooks_base.HookResult:
      captured_tool_calls.append(data)
      return hooks_base.HookResult(allow=True)

    hr.register_hook(policy_hook)

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    allowed_event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            trajectory_id="test_traj",
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            tool_confirmation_request=localharness_pb2.ToolConfirmationRequest(),
            mcp_tool=localharness_pb2.ActionMcpTool(
                server_name="calc",
                tool_name="math_add",
                arguments_json='{"x": 10, "y": 20}',
            ),
        )
    )
    await harness.send_event(allowed_event)
    sent_allowed = await harness.wait_for_response()
    self.assertTrue(sent_allowed["toolConfirmation"]["accepted"])
    self.assertEqual(len(captured_tool_calls), 1)
    self.assertEqual(
        captured_tool_calls[0].name,
        local_connection._get_mcp_tool_name("calc", "math_add"),
    )
    self.assertEqual(captured_tool_calls[0].args, {"x": 10, "y": 20})

  async def test_mcp_tool_confirmation_request_denied(self):
    hr = hook_runner.HookRunner()
    captured_tool_calls = []

    @hooks_base.pre_tool_call_decide
    async def policy_hook(data: types.ToolCall) -> hooks_base.HookResult:
      captured_tool_calls.append(data)
      return hooks_base.HookResult(allow=False)

    hr.register_hook(policy_hook)

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    denied_event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=2,
            trajectory_id="test_traj",
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            tool_confirmation_request=localharness_pb2.ToolConfirmationRequest(),
            mcp_tool=localharness_pb2.ActionMcpTool(
                server_name="calc",
                tool_name="math_multiply",
                arguments_json='{"x": 5, "y": 4}',
            ),
        )
    )
    await harness.send_event(denied_event)
    sent_denied = await harness.wait_for_response()
    self.assertFalse(sent_denied["toolConfirmation"]["accepted"])
    self.assertEqual(len(captured_tool_calls), 1)
    self.assertEqual(
        captured_tool_calls[0].name,
        local_connection._get_mcp_tool_name("calc", "math_multiply"),
    )

  async def test_tool_confirmation_request_has_id(self):
    hr = hook_runner.HookRunner()
    hook_event = asyncio.Event()
    captured_ids = []

    @hooks_base.pre_tool_call_decide
    async def hook(data):
      captured_ids.append(data.id)
      hook_event.set()
      return hooks_base.HookResult(allow=True)

    hr.register_hook(hook)

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=5,
            trajectory_id="test_traj",
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            tool_confirmation_request=localharness_pb2.ToolConfirmationRequest(),
            view_file=localharness_pb2.ActionViewFile(file_path="/foo/bar"),
        )
    )

    await harness.send_event(event)
    await harness.wait_for_event(hook_event)

    self.assertEqual(captured_ids, ["test_traj:5"])

  async def test_tool_confirmation_uses_enum_value_for_view_file(self):
    """Verifies that hooks receive the BuiltinTools enum value as the tool name.

    Why: Hooks should see stable, semantic names (e.g. "view_file") rather
    than raw proto field names. For view_file these happen to match, but the
    test locks in the contract.
    """
    hook_event = asyncio.Event()
    captured_tool_names = []

    @hooks_base.pre_tool_call_decide
    async def capturing_tool(data):
      captured_tool_names.append(data.name)
      hook_event.set()
      return hooks_base.HookResult(allow=True)

    hr = hook_runner.HookRunner()
    hr.register_hook(capturing_tool)

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            trajectory_id="test_traj",
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            tool_confirmation_request=localharness_pb2.ToolConfirmationRequest(),
            view_file=localharness_pb2.ActionViewFile(file_path="/foo/bar"),
        )
    )
    await harness.send_event(event)
    await harness.wait_for_event(hook_event)

    self.assertEqual(captured_tool_names, [types.BuiltinTools.VIEW_FILE.value])

  async def test_tool_confirmation_uses_enum_value_for_find_file(self):
    """Verifies that a find_file step update is correctly recognized.

    Why: find_file is a harness builtin tool that must be correctly identified
    in _BUILTIN_TOOL_PROTO_FIELDS so hooks receive the right name.
    """
    hook_event = asyncio.Event()
    captured_tool_names = []

    @hooks_base.pre_tool_call_decide
    async def capturing_tool(data):
      captured_tool_names.append(data.name)
      hook_event.set()
      return hooks_base.HookResult(allow=True)

    hr = hook_runner.HookRunner()
    hr.register_hook(capturing_tool)

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            trajectory_id="test_traj",
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            tool_confirmation_request=localharness_pb2.ToolConfirmationRequest(),
            find_file=localharness_pb2.ActionFindFile(
                directory_path="file:///home/user",
                query="*.py",
            ),
        )
    )
    await harness.send_event(event)
    await harness.wait_for_event(hook_event)

    self.assertEqual(captured_tool_names, [types.BuiltinTools.FIND_FILE.value])

  async def test_question_hook_integration(self):
    hr = hook_runner.HookRunner()

    @hooks_base.on_interaction
    async def auto_answer(data):
      return hooks_base.QuestionHookResult(
          responses=[
              QuestionResponse(selected_option_ids=["1"]),
          ]
      )

    hr.register_hook(auto_answer)

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            trajectory_id="test_traj",
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            questions_request=localharness_pb2.UserQuestionsRequest(
                questions=[
                    localharness_pb2.UserQuestion(
                        multiple_choice=localharness_pb2.MultipleChoice(
                            question="Do you agree?",
                            choices=["Yes", "No"],
                        )
                    )
                ]
            ),
        )
    )

    await harness.send_event(event)

    sent_data = await harness.wait_for_response()
    self.assertIn("questionResponse", sent_data)
    self.assertEqual(sent_data["questionResponse"]["trajectoryId"], "test_traj")

  async def test_question_hook_integration_unhandled_question(self):
    hr = hook_runner.HookRunner()

    @hooks_base.on_interaction
    async def auto_answer(data):
      return hooks_base.QuestionHookResult(
          responses=[
              QuestionResponse(selected_option_ids=["1"]),
          ]
      )

    hr.register_hook(auto_answer)

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            trajectory_id="test_traj",
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            questions_request=localharness_pb2.UserQuestionsRequest(
                questions=[
                    localharness_pb2.UserQuestion(
                        multiple_choice=localharness_pb2.MultipleChoice(
                            question="Do you agree?",
                            choices=["Yes", "No"],
                        )
                    ),
                    localharness_pb2.UserQuestion(),  # Unhandled question type (empty)
                ]
            ),
        )
    )

    await harness.send_event(event)

    sent_data = await harness.wait_for_response()
    self.assertIn("questionResponse", sent_data)
    self.assertEqual(sent_data["questionResponse"]["trajectoryId"], "test_traj")

    resp = sent_data["questionResponse"]["response"]
    self.assertIn("answers", resp)
    self.assertEqual(len(resp["answers"]), 2)

    # First answer should be from hook (selected option 1)
    self.assertIn("multipleChoiceAnswer", resp["answers"][0])
    self.assertEqual(
        resp["answers"][0]["multipleChoiceAnswer"]["selectedChoiceIndices"], [0]
    )

    # Second answer should be unanswered
    self.assertTrue(resp["answers"][1].get("unanswered"))

  async def test_question_hook_integration_empty_questions(self):
    harness = self._make_harness()

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            trajectory_id="test_traj",
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            questions_request=localharness_pb2.UserQuestionsRequest(
                questions=[]
            ),
        )
    )

    await harness.send_event(event)

    sent_data = await harness.wait_for_response()
    self.assertIn("questionResponse", sent_data)
    self.assertEqual(sent_data["questionResponse"]["trajectoryId"], "test_traj")

    resp = sent_data["questionResponse"]["response"]
    self.assertEqual(resp, {})

  async def test_deduplication_of_wait_requests(self):
    """Verifies that multiple updates for the same wait state don't duplicate."""
    hr = hook_runner.HookRunner()
    hook_event = asyncio.Event()

    call_count = [0]

    @hooks_base.pre_tool_call_decide
    async def counting_hook(data):
      call_count[0] += 1
      hook_event.set()
      return hooks_base.HookResult(allow=True)

    hr.register_hook(counting_hook)

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            trajectory_id="test_traj",
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            tool_confirmation_request=localharness_pb2.ToolConfirmationRequest(),
            view_file=localharness_pb2.ActionViewFile(file_path="/foo/bar"),
        )
    )

    # Send the exact same wait event three times (e.g. keepalives)
    await harness.send_event(event)
    await harness.send_event(event)
    await harness.send_event(event)

    # Wait for the response to ensure at least one event was processed
    await harness.wait_for_response()

    # Hook should only be called ONCE despite 3 events, thanks to _handled_waits
    self.assertEqual(call_count[0], 1)
    self.assertEqual(len(harness.ws.sent_messages), 1)

  async def test_async_non_blocking_dispatch(self):
    """Verifies that wait handlers run concurrently without blocking loop."""
    hr = hook_runner.HookRunner()
    started_event = asyncio.Event()
    finish_event = asyncio.Event()

    class BlockingHook(hooks_base.PreToolCallDecideHook):

      def __init__(self):
        self.started = False
        self.finished = False

      async def run(self, context, data):  # pylint: disable=unused-argument
        del context, data
        self.started = True
        started_event.set()
        await finish_event.wait()
        self.finished = True
        return hooks_base.HookResult(allow=True)

    hook_instance = BlockingHook()
    hr.register_hook(hook_instance)

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    wait_event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            trajectory_id="traj_1",
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            tool_confirmation_request=localharness_pb2.ToolConfirmationRequest(),
            view_file=localharness_pb2.ActionViewFile(file_path="/foo"),
        )
    )

    # An event from another subagent that should not be blocked
    active_event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            trajectory_id="traj_2",
            state=localharness_pb2.StepUpdate.STATE_ACTIVE,
            text="I am another agent running concurrently",
        )
    )

    await harness.send_event(wait_event)
    await harness.send_event(active_event)

    # Wait for the hook to start
    await harness.wait_for_event(started_event)

    # The hook should have started, but not finished
    self.assertTrue(hook_instance.started)
    self.assertFalse(hook_instance.finished)

    # The reader loop SHOULD NOT be blocked! It should have processed traj_2
    # and put both events into the step queue.
    step1 = await harness.conn._step_queue.get()
    step2 = await harness.conn._step_queue.get()

    self.assertEqual(step1.trajectory_id, "traj_1")
    self.assertEqual(step2.trajectory_id, "traj_2")
    self.assertEqual(step2.content, "I am another agent running concurrently")

    # Cleanup: Allow hook to finish
    finish_event.set()

  async def test_state_transition_clears_handled_requests(self):
    """Verifies WAITING -> ACTIVE -> WAITING transitions re-trigger handlers."""
    hr = hook_runner.HookRunner()
    hook_event = asyncio.Event()

    class CountingHook(hooks_base.PreToolCallDecideHook):

      def __init__(self):
        self.call_count = 0

      async def run(self, context, data):  # pylint: disable=unused-argument
        del context, data
        self.call_count += 1
        hook_event.set()
        return hooks_base.HookResult(allow=True)

    hook_instance = CountingHook()
    hr.register_hook(hook_instance)

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    def create_wait_event():
      return localharness_pb2.OutputEvent(
          step_update=localharness_pb2.StepUpdate(
              step_index=1,
              trajectory_id="test_traj",
              state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
              tool_confirmation_request=localharness_pb2.ToolConfirmationRequest(),
              view_file=localharness_pb2.ActionViewFile(file_path="/foo/bar"),
          )
      )

    active_event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            trajectory_id="test_traj",
            state=localharness_pb2.StepUpdate.STATE_ACTIVE,
        )
    )

    # 1. First wait
    await harness.send_event(create_wait_event())
    await harness.wait_for_event(hook_event)
    self.assertEqual(hook_instance.call_count, 1)

    # Reset event for next wait
    hook_event.clear()

    # 2. Transition back to active
    await harness.send_event(active_event)

    # 3. Second wait on the SAME step
    await harness.send_event(create_wait_event())
    await harness.wait_for_event(hook_event)

    # The hook should be called a second time!
    self.assertEqual(hook_instance.call_count, 2)
    self.assertEqual(len(harness.ws.sent_messages), 2)

  async def test_yielding_wait_state_to_queue(self):
    """Verifies that wait states are correctly yielded to the step queue for the UI to render."""
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
    )

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=5,
            trajectory_id="ui_traj",
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            text="Waiting for confirmation",
        )
    )

    await harness.send_event(event)

    # We should be able to retrieve this step from the queue
    step_obj = await asyncio.wait_for(
        harness.conn._step_queue.get(), timeout=2.0
    )
    self.assertEqual(step_obj.trajectory_id, "ui_traj")
    self.assertEqual(step_obj.id, "ui_traj:5")
    self.assertEqual(step_obj.status, types.StepStatus.WAITING_FOR_USER)
    self.assertEqual(step_obj.content, "Waiting for confirmation")

  async def test_cancel_e2e_raises_cancelled_error(self):
    """Verifies programmatic cancel raises AntigravityCancelledError."""
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
    )

    # Start the turn
    await harness.conn.send("Hello")
    init_data = await harness.wait_for_response()
    self.assertEqual(init_data.get("userInput"), "Hello")

    # Simulate an active generation step from the harness
    event1 = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="my_cascade",
            trajectory_id="my_cascade",
            step_index=1,
            state=localharness_pb2.StepUpdate.STATE_ACTIVE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            text="I'm working",
        )
    )
    await harness.send_event(event1)

    # Consume the steps in a background task to capture yielded output
    steps = []
    receive_error = None

    async def consume() -> None:
      nonlocal receive_error
      try:
        async for step in harness.conn.receive_steps():
          steps.append(step)
      except asyncio.CancelledError as e:
        receive_error = e

    consume_task = asyncio.create_task(consume())

    # Let the background consumer loop spin once
    await asyncio.sleep(0.1)

    # Programmatically cancel the turn
    await harness.conn.cancel()

    # Verify that a halt_request was sent to the backend
    sent_data = await harness.wait_for_response()
    self.assertTrue(sent_data.get("haltRequest"))

    # Simulate the harness transitioning to idle after halting
    event2 = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            trajectory_id="my_cascade",
            state=localharness_pb2.TrajectoryStateUpdate.STATE_IDLE,
        )
    )
    await harness.send_event(event2)

    # Await background consumer task completion
    await consume_task

    # Verify yielded step and ensure AntigravityCancelledError propagates
    self.assertEqual(len(steps), 1)
    self.assertEqual(steps[0].content, "I'm working")
    self.assertIsInstance(receive_error, types.AntigravityCancelledError)

  async def test_handle_tool_call_queues_step(self):
    """Tests ensuring _handle_tool_call manually queues the ToolCall step in _step_queue."""
    harness = self._make_harness()
    conn = harness.conn

    # Mock tool_call protobuf message from WebSocket
    raw_tool_call = localharness_pb2.ToolCall(
        id="call_123",
        name="view_file",
        arguments_json='{"path": "README.md"}',
    )

    # Trigger connection event dispatch
    await conn._handle_tool_call(raw_tool_call)
    await asyncio.sleep(0.1)

    self.assertFalse(conn._step_queue.empty())
    step_obj = await conn._step_queue.get()

    actual_properties = {
        "id": step_obj.id,
        "type": step_obj.type,
        "source": step_obj.source,
        "target": step_obj.target,
        "status": step_obj.status,
        "tool_calls": [
            {"name": tc.name, "args": tc.args} for tc in step_obj.tool_calls
        ],
    }

    expected_properties = {
        "id": "call_123",
        "type": types.StepType.TOOL_CALL,
        "source": types.StepSource.MODEL,
        "target": types.StepTarget.ENVIRONMENT,
        "status": types.StepStatus.ACTIVE,
        "tool_calls": [{"name": "view_file", "args": {"path": "README.md"}}],
    }

    self.assertEqual(actual_properties, expected_properties)

  async def test_wait_for_idle_does_not_deadlock(self):
    """Verifies that wait_for_idle completes when the connection goes idle.

    This test reproduces a bug where wait_for_idle blocks indefinitely if the
    connection becomes idle while receive_steps is awaiting the step queue.
    It also verifies that wait_for_idle supports multiple concurrent callers.
    """
    harness = self._make_harness()
    harness.conn._cascade_id = "parent_traj"
    harness.conn._is_idle.clear()
    harness.conn._parent_idle = False

    # 1. Send an active step update
    await harness.send_event(
        localharness_pb2.OutputEvent(
            step_update=localharness_pb2.StepUpdate(
                cascade_id="parent_traj",
                trajectory_id="parent_traj",
                step_index=1,
                text="Hello",
                state=localharness_pb2.StepUpdate.STATE_ACTIVE,
                source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            )
        )
    )

    # Start multiple wait_for_idle tasks concurrently to verify they all unblock
    wait_task_1 = asyncio.create_task(harness.conn.wait_for_idle())
    wait_task_2 = asyncio.create_task(harness.conn.wait_for_idle())

    # Give tasks time to block
    await asyncio.sleep(0.1)

    # 2. Send trajectory_state_update indicating parent went idle
    await harness.send_event(
        localharness_pb2.OutputEvent(
            trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
                trajectory_id="parent_traj",
                state=localharness_pb2.TrajectoryStateUpdate.State.STATE_IDLE,
            )
        )
    )

    # 3. Wait for all wait_tasks to finish with a timeout.
    try:
      await asyncio.wait_for(
          asyncio.gather(wait_task_1, wait_task_2), timeout=2.0
      )
    except asyncio.TimeoutError:
      self.fail("wait_for_idle deadlocked!")

  async def test_concurrent_receive_steps_raises_runtime_error(self):
    """Verifies that concurrent receive_steps() calls raise RuntimeError.

    This test ensures that the SDK prevents multiple consumers from iterating
    over receive_steps() simultaneously. Because steps are drained from a
    single FIFO queue, concurrent iterations would steal steps from one
    another and corrupt conversation history. The active reader guard
    guarantees that a second consumer fails fast with an explicit exception.
    """
    harness = self._make_harness()
    harness.conn._is_idle.clear()

    async def consume_partially() -> None:
      async for _ in harness.conn.receive_steps():
        break

    # Start first consumer in a background task
    consumer_task = asyncio.create_task(consume_partially())
    await asyncio.sleep(0.05)

    # Attempting to start second consumer concurrently raises RuntimeError
    with self.assertRaisesRegex(
        RuntimeError, r"Concurrent receive_steps\(\) calls are not supported"
    ):
      async for _ in harness.conn.receive_steps():
        pass

    # Clean up background task
    await harness.send_event(
        localharness_pb2.OutputEvent(
            trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
                trajectory_id="traj_1",
                state=localharness_pb2.TrajectoryStateUpdate.STATE_IDLE,
            )
        )
    )
    await asyncio.wait_for(consumer_task, timeout=1.0)

  async def test_connection_normalizes_file_uri_arguments(self):
    """Verifies that file:// URIs in tool confirmations are normalized before hooks."""
    hr = hook_runner.HookRunner()
    captured_tc = None

    @hooks_base.pre_tool_call_decide
    async def capturing_hook(data):
      nonlocal captured_tc
      captured_tc = data
      return hooks_base.HookResult(allow=True)

    hr.register_hook(capturing_hook)

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            trajectory_id="test_traj",
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            tool_confirmation_request=localharness_pb2.ToolConfirmationRequest(),
            view_file=localharness_pb2.ActionViewFile(
                file_path="file:///dev/shm/workspace/foo.py"
            ),
        )
    )

    await harness.send_event(event)
    await harness.wait_for_response()

    self.assertIsNotNone(captured_tc)
    self.assertEqual(
        captured_tc.args.get("file_path"),
        "/dev/shm/workspace/foo.py",
    )
    self.assertNotIn("canonical_path", captured_tc.args)
    self.assertEqual(
        captured_tc.canonical_path,
        "/dev/shm/workspace/foo.py",
    )


class LocalConnectionStepFromDictTest(unittest.TestCase):
  """Tests for LocalConnectionStep.from_dict derivation logic.

  Specifically targets the is_complete_response calculation and edge cases in
  step type detection.
  """

  def test_is_complete_response_true(self):
    """Verifies is_complete_response is True when source=MODEL, state=DONE, target=TARGET_USER, and text is present.

    Why: This is the canonical "agent finished speaking" signal that callers
    rely on to surface the final answer. All four conditions must hold:
    source is MODEL, status is DONE, text is present, and target is USER.
    """
    step = local_connection.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_DONE",
        "text": "Here is my answer.",
        "target": "TARGET_USER",
    })
    self.assertTrue(step.is_complete_response)

  def test_is_complete_response_false_when_source_not_model(self):
    """Verifies is_complete_response is False when source is not MODEL.

    Why: System or user steps that are done and have text should not be
    treated as a completed model response.
    """
    step = local_connection.LocalConnectionStep.from_dict({
        "source": "SOURCE_USER",
        "state": "STATE_DONE",
        "text": "Some user text.",
    })
    self.assertFalse(step.is_complete_response)

  def test_is_complete_response_false_when_not_done(self):
    """Verifies is_complete_response is False when state is not DONE.

    Why: An active model step is still streaming; it should not be treated
    as complete until the harness marks it done.
    """
    step = local_connection.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_ACTIVE",
        "text": "Partial response...",
    })
    self.assertFalse(step.is_complete_response)

  def test_is_complete_response_false_when_no_text(self):
    """Verifies is_complete_response is False when text is empty.

    Why: A done model step with no text is a structural step (e.g. tool use
    completion), not a completed textual response.
    """
    step = local_connection.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_DONE",
    })
    self.assertFalse(step.is_complete_response)

  def test_is_complete_response_false_when_error_state(self):
    """Verifies is_complete_response is False when state is ERROR."""
    step = local_connection.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_ERROR",
        "text": "Something went wrong",
        "error_message": "internal error",
    })
    self.assertFalse(step.is_complete_response)

  def test_is_complete_response_false_when_target_environment(self):
    """Verifies is_complete_response is False for TARGET_ENVIRONMENT steps.

    Why: Tool execution steps (view_file, run_command, etc.) are targeted at
    the environment, not the user. Even when they are source=MODEL, state=DONE,
    and have text (e.g. "Requesting permission to make tool call"), they must
    not be treated as a completed model response.
    """
    step = local_connection.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_DONE",
        "text": "Requesting permission to make tool call",
        "target": "TARGET_ENVIRONMENT",
    })
    self.assertFalse(step.is_complete_response)

  def test_step_type_tool_call_with_builtin(self):
    """Verifies that a step with a builtin tool proto field is typed TOOL_CALL and parses details."""
    step = local_connection.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_ACTIVE",
        "view_file": {"file_path": "/foo"},
    })
    self.assertEqual(step.type, types.StepType.TOOL_CALL)

    self.assertEqual(len(step.tool_calls), 1)
    self.assertEqual(step.tool_calls[0].name, "view_file")
    self.assertEqual(step.tool_calls[0].args, {"file_path": "/foo"})
    self.assertEqual(step.tool_calls[0].canonical_path, "/foo")

  def test_structured_output_extracted_from_finish(self):
    """Verifies that structured output is extracted when finish payload is present.

    Why: The connection layer is responsible for extracting and parsing
    the final structured output from the wire format so Layer 2 and E2E tests
    can access it natively.
    """
    step = local_connection.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_DONE",
        "finish": {
            "output_string": (
                '{"total_revenue": 386.0, "top_selling_product": "Widget A"}'
            ),
        },
    })
    self.assertEqual(
        step.structured_output,
        {"total_revenue": 386.0, "top_selling_product": "Widget A"},
    )

  def test_structured_output_extracted_from_finish_handles_invalid_json(self):
    """Verifies that invalid JSON in finish payload defaults to None.

    Why: The connection layer should handle malformed JSON payloads gracefully
    by returning None instead of raising a fatal exception.
    """
    step = local_connection.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_DONE",
        "finish": {
            "output_string": (  # Invalid JSON
                '{"total_revenue": 386.0, "top_selling_product": }'
            ),
        },
    })
    self.assertIsNone(step.structured_output)

  def test_step_from_dict_normalizes_file_uri_arguments(self):
    """Verifies that LocalConnectionStep.from_dict normalizes file:// URIs."""
    step = local_connection.LocalConnectionStep.from_dict({
        "step_index": 1,
        "trajectory_id": "traj_1",
        "state": "STATE_WAITING_FOR_USER",
        "view_file": {"file_path": "file:///dev/shm/workspace/foo.py"},
    })
    self.assertEqual(len(step.tool_calls), 1)
    self.assertEqual(
        step.tool_calls[0].args.get("file_path"), "/dev/shm/workspace/foo.py"
    )
    self.assertNotIn("canonical_path", step.tool_calls[0].args)
    self.assertEqual(
        step.tool_calls[0].canonical_path,
        "/dev/shm/workspace/foo.py",
    )

  def test_step_from_dict_normalizes_cns_uri_arguments(self):
    """Verifies that LocalConnectionStep.from_dict normalizes cns:// URIs.

    Why: The CNS-backed filesystem uses cns:// URIs as path representations.
    The workspace_only policy compares canonical_path against /cns/... paths
    provided by the user, so cns:// must be translated to /cns/... for
    policy matching to work correctly.
    """
    step = local_connection.LocalConnectionStep.from_dict({
        "step_index": 1,
        "trajectory_id": "traj_1",
        "state": "STATE_WAITING_FOR_USER",
        "create_file": {"path": "cns://el-d/home/user/workspace/kittens.md"},
    })
    self.assertEqual(len(step.tool_calls), 1)
    self.assertEqual(
        step.tool_calls[0].args.get("path"),
        "/cns/el-d/home/user/workspace/kittens.md",
    )
    self.assertNotIn("canonical_path", step.tool_calls[0].args)
    self.assertEqual(
        step.tool_calls[0].canonical_path,
        "/cns/el-d/home/user/workspace/kittens.md",
    )


class LocalConnectionToolCallNoRunnerTest(unittest.IsolatedAsyncioTestCase):
  """Tests for tool call handling when no ToolRunner is configured."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()

  async def test_tool_call_without_runner_yields_step(self):
    """Verifies that a tool call with no ToolRunner queues a step for the user.

    Why: When no ToolRunner is configured, the connection should surface the
    tool call to the caller so they can handle it manually, rather than
    silently dropping it.
    """
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=None,
    )

    event = localharness_pb2.OutputEvent(
        tool_call=localharness_pb2.ToolCall(
            id="call_99",
            name="custom_tool",
            arguments_json='{"key": "value"}',
        )
    )

    await harness.send_event(event)

    step_obj = await asyncio.wait_for(
        harness.conn._step_queue.get(), timeout=1.0
    )
    self.assertEqual(step_obj.type, types.StepType.TOOL_CALL)
    self.assertEqual(step_obj.tool_calls[0].name, "custom_tool")
    self.assertEqual(step_obj.tool_calls[0].args, {"key": "value"})
    self.assertEqual(step_obj.tool_calls[0].id, "call_99")
    # No messages should have been sent back to the harness.
    self.assertEqual(len(harness.ws.sent_messages), 0)


class LocalConnectionStrategyConfigTest(parameterized.TestCase):
  """Tests for config-to-proto translation in LocalConnectionStrategy.

  These tests exercise _build_harness_config() directly, without mocking
  any internal logic. Only the strategy constructor and config builder run;
  no subprocess or websocket I/O is triggered.
  """

  def setUp(self):
    super().setUp()
    self.patcher = mock.patch(
        "google.antigravity.connections.local.local_connection._get_default_binary_path",
        return_value="/fake/binary",
    )
    self.patcher.start()
    self.addCleanup(self.patcher.stop)

  def _make_strategy(self, **kwargs):
    """Creates a LocalConnectionStrategy with the given kwargs."""
    return local_connection.LocalConnectionStrategy(**kwargs)

  def test_default_config_produces_valid_harness_config(self):
    """Verifies that a strategy with all defaults produces a well-formed proto.

    Why: The default path is the most common case. Callers should be able to
    construct a strategy with only binary_path and get a valid HarnessConfig.
    How: Build the config and assert the proto has expected default structure.
    """
    strategy = self._make_strategy()
    config = strategy._build_harness_config()
    self.assertIsInstance(config, localharness_pb2.HarnessConfig)
    # Default: all harness side tools enabled.
    self.assertTrue(config.harness_side_tools.subagents.enabled)
    self.assertTrue(config.harness_side_tools.user_questions.enabled)
    self.assertTrue(config.harness_side_tools.run_command.enabled)
    self.assertTrue(config.harness_side_tools.find.enabled)
    self.assertTrue(config.harness_side_tools.generate_image.enabled)
    # No models, system instructions, workspaces, or skills by default.
    self.assertEmpty(config.models)
    self.assertFalse(config.HasField("system_instructions"))
    self.assertEqual(len(config.workspaces), 0)
    self.assertEqual(len(config.skills_paths), 0)

  def test_legacy_shorthands_api_key_produces_valid_proto(self):
    """Verifies that the legacy api_key shorthand translates to the models proto."""
    cfg = local_connection_config.LocalAgentConfig(
        model="gemini-2.5-flash",
        api_key="shorthand-key",
    )
    strategy = self._make_strategy(
        models=cfg.models,
    )
    config = strategy._build_harness_config()
    self.assertLen(config.models, 2)
    self.assertEqual(config.models[0].name, "gemini-2.5-flash")
    self.assertTrue(config.models[0].HasField("gemini_api_endpoint"))
    self.assertEqual(
        config.models[0].gemini_api_endpoint.api_key, "shorthand-key"
    )
    self.assertEqual(config.models[0].types, [localharness_pb2.MODEL_TYPE_TEXT])

  def test_legacy_shorthands_vertex_produces_valid_proto(self):
    """Verifies that the legacy vertex shorthands translate to the models proto."""
    cfg = local_connection_config.LocalAgentConfig(
        model="gemini-2.5-flash",
        vertex=True,
        project="vertex-project",
        location="us-east4",
    )
    strategy = self._make_strategy(
        models=cfg.models,
    )
    config = strategy._build_harness_config()
    self.assertLen(config.models, 2)
    self.assertEqual(config.models[0].name, "gemini-2.5-flash")
    self.assertTrue(config.models[0].HasField("vertex_endpoint"))
    self.assertEqual(config.models[0].vertex_endpoint.project, "vertex-project")
    self.assertEqual(config.models[0].vertex_endpoint.location, "us-east4")

  def test_capabilities_config_finish_tool_schema_json_to_proto(self):
    """Verifies capabilities config propagates finish tool schema to the proto config.

    Why: The user's custom schema must be delivered to the Go harness so it can
    be appropriately injected into the finish tool declaration.
    """
    strategy = self._make_strategy(
        capabilities_config=types.CapabilitiesConfig(
            finish_tool_schema_json='{"type": "object"}',
        )
    )
    config = strategy._build_harness_config()
    self.assertEqual(config.finish_tool_schema_json, '{"type": "object"}')

  def test_gemini_config_to_proto(self):
    """Verifies GeminiConfig fields translate to the correct proto fields.

    Why: The proto's field names must match the Pydantic model's semantics
    exactly, or the Go harness will receive incorrect configuration.
    How: Set all GeminiConfig fields and assert proto field values.
    """
    strategy = self._make_strategy(
        models=[
            types.ModelTarget(
                name="gemini-2.5-pro",
                types=[types.ModelType.TEXT],
                endpoint=types.GeminiAPIEndpoint(api_key="test-key"),
            )
        ]
    )
    config = strategy._build_harness_config()
    self.assertEqual(config.models[0].gemini_api_endpoint.api_key, "test-key")
    self.assertEqual(config.models[0].name, "gemini-2.5-pro")

  def test_gemini_config_none_fields_omitted(self):
    """Verifies that None fields on ModelConfig are not set on the proto."""
    models = [
        types.ModelTarget(
            name="gemini-3.5-flash",
            types=[types.ModelType.TEXT],
            endpoint=types.GeminiAPIEndpoint(),
        )
    ]
    strategy = self._make_strategy(models=models)
    config = strategy._build_harness_config()
    self.assertEqual(config.models[0].name, "gemini-3.5-flash")
    # api_key should not be set (proto default empty string).
    self.assertEqual(config.models[0].gemini_api_endpoint.api_key, "")

  def test_models_default_model_name(self):
    """Verifies the default model name propagates correctly."""
    models = [
        types.ModelTarget(
            name=None,
            types=[types.ModelType.TEXT],
            endpoint=types.GeminiAPIEndpoint(),
        )
    ]
    strategy = self._make_strategy(models=models)
    config = strategy._build_harness_config()
    self.assertEqual(config.models[0].name, "")

  def test_system_instructions_string_shorthand(self):
    """Verifies that a plain string normalizes to AppendedSystemInstructions.

    Why: The str shorthand is an ergonomic convenience. It defaults to
    appending.
    How: Pass a string, build proto, and assert the appended field is set.
    """
    strategy = self._make_strategy(system_instructions="Be concise.")
    config = strategy._build_harness_config()
    self.assertEqual(
        len(config.system_instructions.appended.appended_sections), 1
    )
    self.assertEqual(
        config.system_instructions.appended.appended_sections[0].content,
        "Be concise.",
    )
    self.assertEqual(
        config.system_instructions.appended.appended_sections[0].title,
        "user_system_instructions",
    )

  def test_system_instructions_model_custom(self):
    """Verifies that CustomSystemInstructions sets custom on the proto."""
    strategy = self._make_strategy(
        system_instructions=types.CustomSystemInstructions(
            text="Override everything."
        )
    )
    config = strategy._build_harness_config()
    self.assertEqual(
        config.system_instructions.custom.part[0].text, "Override everything."
    )

  def test_system_instructions_model_templated(self):
    """Verifies that TemplatedSystemInstructions sets appended on the proto."""
    section = types.SystemInstructionSection(
        title="extra", content="More instructions"
    )
    strategy = self._make_strategy(
        system_instructions=types.TemplatedSystemInstructions(
            identity="New Identity", sections=[section]
        )
    )
    config = strategy._build_harness_config()
    self.assertEqual(
        config.system_instructions.appended.custom_identity, "New Identity"
    )
    self.assertEqual(
        len(config.system_instructions.appended.appended_sections), 1
    )
    self.assertEqual(
        config.system_instructions.appended.appended_sections[0].title, "extra"
    )

  def test_system_instructions_model_templated_only_identity(self):
    """Verifies that TemplatedSystemInstructions with only identity maps correctly."""
    strategy = self._make_strategy(
        system_instructions=types.TemplatedSystemInstructions(
            identity="Only Identity"
        )
    )
    config = strategy._build_harness_config()
    self.assertEqual(
        config.system_instructions.appended.custom_identity, "Only Identity"
    )
    self.assertEqual(
        len(config.system_instructions.appended.appended_sections), 0
    )

  def test_system_instructions_model_templated_only_sections(self):
    """Verifies that TemplatedSystemInstructions with only sections maps correctly."""
    section = types.SystemInstructionSection(
        title="extra", content="More instructions"
    )
    strategy = self._make_strategy(
        system_instructions=types.TemplatedSystemInstructions(
            sections=[section]
        )
    )
    config = strategy._build_harness_config()
    self.assertEqual(config.system_instructions.appended.custom_identity, "")
    self.assertEqual(
        len(config.system_instructions.appended.appended_sections), 1
    )
    self.assertEqual(
        config.system_instructions.appended.appended_sections[0].title, "extra"
    )

  def test_system_instructions_none(self):
    """Verifies that no system_instructions field is set when not provided.

    Why: The harness should use its own defaults when no instructions are given.
    How: Build with system_instructions=None and assert no proto field is set.
    """
    strategy = self._make_strategy()
    config = strategy._build_harness_config()
    self.assertFalse(config.HasField("system_instructions"))

  def test_workspaces_to_proto(self):
    """Verifies workspace paths translate to Workspace protos correctly.

    Why: The harness uses a structured Workspace proto with FilesystemWorkspace;
    plain strings must be wrapped correctly.
    How: Pass two paths via session_config, build proto, and assert each
    workspace directory.
    """
    strategy = self._make_strategy(
        workspaces=["/home/user/project", "/tmp/scratch"]
    )
    config = strategy._build_harness_config()
    self.assertEqual(len(config.workspaces), 2)
    self.assertEqual(
        config.workspaces[0].filesystem_workspace.directory,
        "/home/user/project",
    )
    self.assertEqual(
        config.workspaces[1].filesystem_workspace.directory,
        "/tmp/scratch",
    )

  def test_workspaces_none(self):
    """Verifies that no workspaces are set when not provided.

    Why: The harness should not receive spurious workspace entries.
    How: Build with default session_config and assert empty repeated field.
    """
    strategy = self._make_strategy()
    config = strategy._build_harness_config()
    self.assertEqual(len(config.workspaces), 0)

  def test_empty_workspaces_list(self):
    """Verifies that an empty list produces an empty repeated field.

    Why: workspaces=[] is a valid explicit choice meaning 'no workspaces',
    distinct from None (which also means no workspaces but is implicit).
    How: Pass empty list via session_config and assert empty repeated field.
    """
    strategy = self._make_strategy(workspaces=[])
    config = strategy._build_harness_config()
    self.assertEqual(len(config.workspaces), 0)

  def test_skills_paths_to_proto(self):
    """Verifies skills_paths translate directly to the proto repeated field.

    Why: Skills paths are simple strings that map 1:1 to the proto field.
    How: Pass a list and assert proto field contents.
    """
    strategy = self._make_strategy(skills_paths=["/skills/a", "/skills/b"])
    config = strategy._build_harness_config()
    self.assertEqual(list(config.skills_paths), ["/skills/a", "/skills/b"])

  def test_capabilities_config_disabled_tools(self):
    """Verifies that disabling tools produces the correct proto.

    Why: Each BuiltinTool with a proto toggle should map to its config field.
    How: Disable RUN_COMMAND and ASK_QUESTION and assert each sub-proto's
    enabled field, plus check that other tools remain enabled.
    """
    strategy = self._make_strategy(
        capabilities_config=types.CapabilitiesConfig(
            disabled_tools=[
                types.BuiltinTools.RUN_COMMAND,
                types.BuiltinTools.ASK_QUESTION,
                types.BuiltinTools.GENERATE_IMAGE,
            ],
        )
    )
    config = strategy._build_harness_config()
    self.assertFalse(config.harness_side_tools.run_command.enabled)
    self.assertFalse(config.harness_side_tools.user_questions.enabled)
    self.assertFalse(config.harness_side_tools.generate_image.enabled)
    # Subagents were not disabled; should still be enabled by default.
    self.assertTrue(config.harness_side_tools.subagents.enabled)
    # Tools that were not disabled should still be enabled.
    self.assertTrue(config.harness_side_tools.find.enabled)
    self.assertTrue(config.harness_side_tools.file_edit.enabled)
    self.assertTrue(config.harness_side_tools.view_file.enabled)
    self.assertTrue(config.harness_side_tools.write_to_file.enabled)
    self.assertTrue(config.harness_side_tools.grep_search.enabled)
    self.assertTrue(config.harness_side_tools.list_dir.enabled)
    self.assertTrue(config.harness_side_tools.search_web.enabled)

  def test_capabilities_config_enabled_tools(self):
    """Verifies that enabled_tools allowlist excludes non-listed tools.

    Why: When an explicit allowlist is provided, only those tools should be
    active; all others should be disabled at the proto level.
    How: Enable only VIEW_FILE and assert all other tools are disabled.
    """
    strategy = self._make_strategy(
        capabilities_config=types.CapabilitiesConfig(
            enabled_tools=[types.BuiltinTools.VIEW_FILE],
        )
    )
    config = strategy._build_harness_config()

    expected_harness_side_tools = localharness_pb2.HarnessSideTools(
        view_file=localharness_pb2.ViewFileToolConfig(enabled=True),
        subagents=localharness_pb2.SubagentsConfig(enabled=False),
        user_questions=localharness_pb2.UserQuestionsConfig(enabled=False),
        run_command=localharness_pb2.RunCommandToolConfig(enabled=False),
        find=localharness_pb2.FindToolConfig(enabled=False),
        generate_image=localharness_pb2.GenerateImageToolConfig(enabled=False),
        file_edit=localharness_pb2.FileEditToolConfig(enabled=False),
        write_to_file=localharness_pb2.WriteToFileToolConfig(enabled=False),
        grep_search=localharness_pb2.GrepSearchToolConfig(enabled=False),
        list_dir=localharness_pb2.ListDirToolConfig(enabled=False),
        search_web=localharness_pb2.SearchWebToolConfig(enabled=False),
    )

    self.assertEqual(config.harness_side_tools, expected_harness_side_tools)

  def test_capabilities_config_compaction_threshold(self):
    """Verifies compaction_threshold maps to HarnessConfig.compaction_threshold.

    Why: This controls context window compaction behavior in the harness.
    How: Set a threshold and assert it appears on the proto.
    """
    strategy = self._make_strategy(
        capabilities_config=types.CapabilitiesConfig(compaction_threshold=50000)
    )
    config = strategy._build_harness_config()
    self.assertEqual(config.compaction_threshold, 50000)

  def test_capabilities_config_none_uses_defaults(self):
    """Verifies that capabilities_config=None produces default-enabled tools.

    Why: The most common case is no explicit CapabilitiesConfig; all tools
    should be enabled and compaction_threshold unset.
    How: Build with no capabilities_config and assert defaults.
    """
    strategy = self._make_strategy()
    config = strategy._build_harness_config()
    self.assertTrue(config.harness_side_tools.subagents.enabled)
    self.assertTrue(config.harness_side_tools.user_questions.enabled)
    self.assertTrue(config.harness_side_tools.run_command.enabled)
    self.assertTrue(config.harness_side_tools.find.enabled)
    self.assertEqual(config.compaction_threshold, 0)

  def test_cascade_id_passed_through(self):
    """Verifies that session_config.conversation_id maps to HarnessConfig.cascade_id.

    Why: cascade_id is used for session resumption; if it's lost, the
    harness creates a new session instead of resuming.
    How: Set conversation_id via session_config and assert it appears
    on the proto.
    """
    strategy = self._make_strategy(conversation_id="resume-123")
    config = strategy._build_harness_config()
    self.assertEqual(config.cascade_id, "resume-123")

  def test_cascade_id_default_empty(self):
    """Verifies that cascade_id defaults to empty string when no conversation_id set.

    Why: The harness treats an empty cascade_id as a fresh session.
    How: Build with default session_config and assert empty cascade_id.
    """
    strategy = self._make_strategy()
    config = strategy._build_harness_config()
    self.assertEqual(config.cascade_id, "")

  def test_storage_directory_from_save_dir(self):
    """Verifies save_dir maps to InputConfig.storage_directory.

    Why: The harness writes trajectory data to storage_directory. If
    save_dir is silently dropped, session state is never persisted and
    resumption breaks.
    How: Set save_dir via session_config and assert it appears on
    the strategy's stored config for InputConfig construction.
    """
    strategy = self._make_strategy(save_dir="/tmp/state")
    self.assertEqual(strategy._save_dir, "/tmp/state")

  def test_storage_directory_defaults_to_none(self):
    """Verifies save_dir is None when not specified.

    Why: A None save_dir signals an ephemeral session. The or "" fallback
    in __aenter__ must produce an empty string for the proto.
    How: Build with default session_config and assert save_dir is None.
    """
    strategy = self._make_strategy()
    self.assertIsNone(strategy._save_dir)

  def test_workspaces_default_empty(self):
    """Verifies no workspace protos when session_config has no workspaces.

    Why: The or [] fallback prevents iterating over None. If removed,
    the list comprehension raises TypeError on None.
    How: Build with default session_config and assert empty workspaces.
    """
    strategy = self._make_strategy()
    config = strategy._build_harness_config()
    self.assertEqual(len(config.workspaces), 0)

  def test_models_thinking_level_set(self):
    """Verifies that thinking_level on ModelTarget maps to the proto field."""
    strategy = self._make_strategy(
        models=[
            types.ModelTarget(
                name=DEFAULT_MODEL,
                types=[types.ModelType.TEXT],
                endpoint=types.GeminiAPIEndpoint(
                    options=types.GeminiModelOptions(
                        thinking_level=types.ThinkingLevel.HIGH,
                    ),
                ),
            )
        ]
    )
    config = strategy._build_harness_config()
    self.assertEqual(
        config.models[0].gemini_api_endpoint.options.thinking_level, "high"
    )

  def test_models_thinking_level_none_omitted(self):
    """Verifies that thinking_level=None leaves the proto field at its default."""
    strategy = self._make_strategy(
        models=[
            types.ModelTarget(
                name=DEFAULT_MODEL,
                types=[types.ModelType.TEXT],
                endpoint=types.GeminiAPIEndpoint(
                    options=types.GeminiModelOptions(thinking_level=None),
                ),
            )
        ]
    )
    config = strategy._build_harness_config()
    self.assertFalse(
        config.models[0].gemini_api_endpoint.HasField("options")
    )

  def test_models_thinking_level_all_values(self):
    """Verifies all ThinkingLevel enum values produce correct proto strings."""
    for level in types.ThinkingLevel:
      strategy = self._make_strategy(
          models=[
              types.ModelTarget(
                  name=DEFAULT_MODEL,
                  types=[types.ModelType.TEXT],
                  endpoint=types.GeminiAPIEndpoint(
                      options=types.GeminiModelOptions(thinking_level=level),
                  ),
              )
          ]
      )
      config = strategy._build_harness_config()
      self.assertEqual(
          config.models[0].gemini_api_endpoint.options.thinking_level,
          level.value,
          f"ThinkingLevel.{level.name} should produce proto string"
          f" '{level.value}'",
      )

  def test_vertex_config_propagates(self):
    """Verifies that Vertex configuration fields propagate to proto."""
    models = [
        types.ModelTarget(
            name="gemini-3.5-flash",
            types=[types.ModelType.TEXT],
            endpoint=types.VertexEndpoint(
                project="my-project",
                location="us-central1",
            ),
        )
    ]
    strategy = self._make_strategy(models=models)
    config = strategy._build_harness_config()
    self.assertTrue(config.models[0].HasField("vertex_endpoint"))
    self.assertEqual(config.models[0].vertex_endpoint.project, "my-project")
    self.assertEqual(config.models[0].vertex_endpoint.location, "us-central1")

  def test_models_list_propagates(self):
    """Verifies that the new models list propagates to HarnessConfig."""
    models = [
        types.ModelTarget(
            name="gemini-2.5-pro",
            types=[types.ModelType.TEXT],
            endpoint=types.GeminiAPIEndpoint(
                options=types.GeminiModelOptions(
                    thinking_level=types.ThinkingLevel.HIGH
                )
            ),
        ),
        types.ModelTarget(
            name="imagen-3-custom",
            types=[types.ModelType.IMAGE],
            endpoint=types.GeminiAPIEndpoint(),
        ),
    ]
    strategy = self._make_strategy(models=models)
    config = strategy._build_harness_config()

    # Text model assertions
    self.assertEqual(config.models[0].name, "gemini-2.5-pro")
    self.assertEqual(
        config.models[0].gemini_api_endpoint.options.thinking_level, "high"
    )

    # Image model assertions
    self.assertEqual(config.models[1].name, "imagen-3-custom")
    self.assertEqual(
        config.models[1].types, [localharness_pb2.MODEL_TYPE_IMAGE]
    )

  def test_models_list_custom_endpoints_propagate(self):
    """Verifies that custom endpoints in the models list propagate to proto."""
    # Test VertexEndpoint
    vertex_endpoint = types.VertexEndpoint(
        project="vertex-proj", location="europe-west1"
    )
    models_vertex = [
        types.ModelTarget(
            name="gemini-ultra",
            types=[types.ModelType.TEXT],
            endpoint=vertex_endpoint,
        )
    ]
    strategy = self._make_strategy(models=models_vertex)
    config = strategy._build_harness_config()
    self.assertTrue(config.models[0].HasField("vertex_endpoint"))
    self.assertEqual(config.models[0].vertex_endpoint.project, "vertex-proj")
    self.assertEqual(config.models[0].vertex_endpoint.location, "europe-west1")

    # Test GeminiAPIEndpoint
    api_endpoint = types.GeminiAPIEndpoint(api_key="api-key-xyz")
    models_api = [
        types.ModelTarget(
            name="gemini-flash",
            types=[types.ModelType.TEXT],
            endpoint=api_endpoint,
        )
    ]
    strategy = self._make_strategy(models=models_api)
    config = strategy._build_harness_config()
    self.assertTrue(config.models[0].HasField("gemini_api_endpoint"))
    self.assertEqual(
        config.models[0].gemini_api_endpoint.api_key, "api-key-xyz"
    )

  def test_models_stored_directly_on_strategy(self):
    """Verifies that the models list is stored as self._models."""
    models = [
        types.ModelTarget(
            name="gemini-2.5-pro",
            types=[types.ModelType.TEXT],
        ),
    ]
    strategy = self._make_strategy(models=models)
    self.assertIsNotNone(strategy._models)
    self.assertLen(strategy._models, 1)
    self.assertEqual(strategy._models[0].name, "gemini-2.5-pro")

  def test_session_config_save_dir_stored(self):
    """Verifies that session_config.save_dir is preserved on the strategy.

    Why: save_dir maps to InputConfig.storage_directory during __aenter__.
    The strategy must store it so the startup sequence can use it.
    How: Set save_dir via session_config and assert strategy attribute.
    """
    strategy = self._make_strategy(save_dir="/data/sessions")
    self.assertEqual(strategy._save_dir, "/data/sessions")

  def test_session_config_save_dir_default_none(self):
    """Verifies that save_dir defaults to None when not provided.

    Why: When no save_dir is set, InputConfig.storage_directory should be
    empty and persistence is disabled.
    How: Build with default session_config and assert save_dir is None.
    """
    strategy = self._make_strategy()
    self.assertIsNone(strategy._save_dir)

  def test_full_session_config_to_proto(self):
    """Verifies that a full session_config produces correct proto fields.

    Why: This is the canonical resumption case — all three session fields
    must map correctly to their proto counterparts.
    How: Set all session_config fields, build proto, and assert each mapping.
    """
    strategy = self._make_strategy(
        conversation_id="session-789",
        save_dir="/state/dir",
        workspaces=["/ws/a"],
    )
    config = strategy._build_harness_config()
    self.assertEqual(config.cascade_id, "session-789")
    self.assertEqual(len(config.workspaces), 1)
    self.assertEqual(
        config.workspaces[0].filesystem_workspace.directory, "/ws/a"
    )
    # save_dir is wired in __aenter__, not _build_harness_config;
    # verify storage.
    self.assertEqual(strategy._save_dir, "/state/dir")

  def test_app_data_dir_specified(self):
    strategy = self._make_strategy(app_data_dir="/custom/app/data")
    config = strategy._build_harness_config()
    self.assertEqual(config.app_data_dir, "/custom/app/data")

  def test_app_data_dir_default_empty(self):
    strategy = self._make_strategy()
    config = strategy._build_harness_config()
    self.assertEqual(config.app_data_dir, "")

  @parameterized.named_parameters(
      dict(
          testcase_name="via_disabled_tools",
          capabilities_config=types.CapabilitiesConfig(
              enable_subagents=True,
              disabled_tools=[types.BuiltinTools.START_SUBAGENT],
          ),
      ),
      dict(
          testcase_name="via_enable_subagents_false",
          capabilities_config=types.CapabilitiesConfig(
              enable_subagents=False,
          ),
      ),
  )
  def test_capabilities_config_subagents_disabled(self, capabilities_config):
    """Verifies that subagents are disabled based on capabilities_config."""
    strategy = self._make_strategy(capabilities_config=capabilities_config)
    config = strategy._build_harness_config()
    self.assertFalse(config.harness_side_tools.subagents.enabled)

  def test_strategy_normalizes_configured_workspaces(self):
    """Verifies that workspace configurations using file:// URIs are canonicalized."""
    strategy = self._make_strategy(
        workspaces=["file:///dev/shm/workspace", "/tmp/clean-path"]
    )
    self.assertEqual(
        strategy._workspaces, ["/dev/shm/workspace", "/tmp/clean-path"]
    )


class LocalConnectionStrategyApiKeyTest(unittest.IsolatedAsyncioTestCase):
  """Tests for API key validation in LocalConnectionStrategy."""

  def setUp(self):
    super().setUp()
    self.patcher = mock.patch(
        "google.antigravity.connections.local.local_connection._get_default_binary_path",
        return_value="/fake/binary",
    )
    self.patcher.start()
    self.addCleanup(self.patcher.stop)

  def _make_strategy(self, **kwargs):
    """Creates a LocalConnectionStrategy with the given kwargs."""
    return local_connection.LocalConnectionStrategy(**kwargs)

  @mock.patch.dict("os.environ", {}, clear=True)
  async def test_raises_without_api_key(self):
    """Verifies entry raises when no API key is available.

    Why: The Go localharness binary silently returns empty responses when no
    API key is provided. An explicit error at startup is much more actionable.
    How: Create a strategy with a model target and assert
    AntigravityValidationError.
    """
    models = [
        types.ModelTarget(
            name="gemini-3.5-flash",
            types=[types.ModelType.TEXT],
        )
    ]
    strategy = self._make_strategy(models=models)
    with self.assertRaises(types.AntigravityValidationError) as ctx:
      async with strategy:
        pass
    self.assertIn("must have an endpoint configured", str(ctx.exception))

  @mock.patch.dict("os.environ", {}, clear=True)
  async def test_raises_with_empty_endpoint_api_key(self):
    """Verifies entry raises when GeminiAPIEndpoint has no api_key and env is unset.

    Why: GeminiAPIEndpoint(api_key=None) must not be fooled by empty values.
    """
    models = [
        types.ModelTarget(
            name="gemini-3.5-flash",
            types=[types.ModelType.TEXT],
            endpoint=types.GeminiAPIEndpoint(api_key=None),
        )
    ]
    strategy = self._make_strategy(models=models)
    with self.assertRaises(types.AntigravityValidationError) as ctx:
      async with strategy:
        pass
    self.assertIn("A Gemini API key is required", str(ctx.exception))

  @mock.patch.dict("os.environ", {}, clear=True)
  async def test_raises_without_auth_in_vertex_mode(self):
    """Verifies strategy raises validation error when Vertex is set but no project/location or api_key provided."""
    models = [
        types.ModelTarget(
            name="gemini-3.5-flash",
            types=[types.ModelType.TEXT],
            endpoint=types.VertexEndpoint(project=None, location=None),
        )
    ]
    strategy = self._make_strategy(models=models)
    with self.assertRaises(types.AntigravityValidationError) as ctx:
      async with strategy:
        pass
    self.assertIn("project and location, or an API key", str(ctx.exception))

  @mock.patch.dict("os.environ", {}, clear=True)
  @mock.patch("subprocess.Popen")
  async def test_accepts_vertex_config_with_project_location(self, mock_popen):
    """Verifies entry does not raise when Vertex is enabled and project/location are provided."""
    mock_proc = mock.MagicMock()
    mock_proc.stdin = mock.MagicMock()
    mock_proc.stdout = mock.MagicMock()
    mock_proc.stderr = mock.MagicMock()
    mock_proc.stdout.read.return_value = b""
    mock_popen.return_value = mock_proc

    models = [
        types.ModelTarget(
            name="gemini-3.5-flash",
            types=[types.ModelType.TEXT],
            endpoint=types.VertexEndpoint(
                project="my-project",
                location="us-central1",
            ),
        )
    ]
    strategy = self._make_strategy(models=models)
    with self.assertRaises(RuntimeError):
      async with strategy:
        pass

  @mock.patch.dict("os.environ", {"GEMINI_API_KEY": "env-key"}, clear=True)
  @mock.patch("subprocess.Popen")
  async def test_accepts_env_var_api_key(self, mock_popen):
    """Verifies entry does not raise when GEMINI_API_KEY env var is set.

    Why: The env var fallback is the most common path for 3P developers.
    How: Set GEMINI_API_KEY, enter the context manager, and verify it proceeds
    past the validation check (it will fail later at subprocess I/O, which is
    expected).

    Args:
      mock_popen: Mocked subprocess.Popen to prevent actual process launch.
    """
    mock_proc = mock.MagicMock()
    mock_proc.stdin = mock.MagicMock()
    mock_proc.stdout = mock.MagicMock()
    mock_proc.stderr = mock.MagicMock()
    mock_proc.stdout.read.return_value = b""
    mock_popen.return_value = mock_proc
    strategy = self._make_strategy()
    # Should not raise AntigravityValidationError; it will raise RuntimeError
    # from the subprocess read failure, which proves we passed the check.
    with self.assertRaises(RuntimeError):
      async with strategy:
        pass

  @mock.patch.dict("os.environ", {}, clear=True)
  @mock.patch("subprocess.Popen")
  async def test_accepts_models_api_key(self, mock_popen):
    """Verifies entry does not raise when model endpoint api_key is set.

    Why: Explicit API key in config is the recommended path.
    How: Set api_key in GeminiAPIEndpoint, enter context manager.

    Args:
      mock_popen: Mocked subprocess.Popen to prevent actual process launch.
    """
    mock_proc = mock.MagicMock()
    mock_proc.stdin = mock.MagicMock()
    mock_proc.stdout = mock.MagicMock()
    mock_proc.stderr = mock.MagicMock()
    mock_proc.stdout.read.return_value = b""
    mock_popen.return_value = mock_proc
    models = [
        types.ModelTarget(
            name="gemini-3.5-flash",
            types=[types.ModelType.TEXT],
            endpoint=types.GeminiAPIEndpoint(api_key="explicit-key"),
        )
    ]
    strategy = self._make_strategy(models=models)
    with self.assertRaises(RuntimeError):
      async with strategy:
        pass


_get_default_binary_path = local_connection._get_default_binary_path


class GetDefaultBinaryPathTest(unittest.TestCase):

  @mock.patch.dict("os.environ", {"ANTIGRAVITY_HARNESS_PATH": "/env/path"})
  def test_returns_env_path(self):
    path = _get_default_binary_path()
    self.assertEqual(path, "/env/path")

  @mock.patch.dict("os.environ", {}, clear=True)
  @mock.patch("importlib.metadata.distribution")
  @mock.patch("os.path.exists")
  def test_returns_metadata_distribution_path(self, mock_exists, mock_dist):
    mock_file = mock.MagicMock()
    mock_file.__str__.return_value = "google/antigravity/bin/localharness"
    mock_file.locate.return_value = (
        "/site-packages/google/antigravity/bin/localharness"
    )

    mock_distribution = mock.MagicMock()
    mock_distribution.files = [mock_file]
    mock_dist.return_value = mock_distribution
    mock_exists.return_value = True

    path = _get_default_binary_path()
    self.assertEqual(path, "/site-packages/google/antigravity/bin/localharness")
    mock_dist.assert_called_once_with("google-antigravity")
    mock_file.locate.assert_called_once()

  @mock.patch.dict("os.environ", {}, clear=True)
  @mock.patch("importlib.metadata.distribution")
  def test_returns_internal_pyglib_resource_path(self, mock_dist):
    mock_resources = mock.MagicMock()
    mock_resources.GetResourceFilename.return_value = (
        "/g3/runfiles/localharness"
    )

    with mock.patch.object(local_connection, "resources", mock_resources):
      path = local_connection._get_default_binary_path()
      self.assertEqual(path, "/g3/runfiles/localharness")
      mock_resources.GetResourceFilename.assert_called_once_with(
          "antigravity_harness"
      )
      mock_dist.assert_not_called()

  @mock.patch.dict("os.environ", {}, clear=True)
  @mock.patch.object(local_connection, "resources", None)
  @mock.patch("importlib.metadata.distribution")
  @mock.patch("importlib.resources.files")
  @mock.patch("os.path.exists")
  def test_returns_external_wheel_path(
      self, mock_exists, mock_files, mock_dist
  ):
    mock_dist.side_effect = importlib.metadata.PackageNotFoundError
    mock_path = mock.MagicMock()
    mock_path.joinpath.return_value.__str__.return_value = "/wheel/path"
    mock_files.return_value = mock_path
    mock_exists.return_value = True

    path = _get_default_binary_path()
    self.assertEqual(path, "/wheel/path")

  @mock.patch.dict("os.environ", {}, clear=True)
  @mock.patch("importlib.metadata.distribution")
  @mock.patch("importlib.resources.files")
  @mock.patch("shutil.which")
  def test_returns_system_path(self, mock_which, mock_files, mock_dist):
    mock_dist.side_effect = importlib.metadata.PackageNotFoundError
    mock_files.side_effect = ImportError
    mock_which.return_value = "/system/path"

    path = _get_default_binary_path()
    self.assertEqual(path, "/system/path")
    mock_which.assert_called_once_with("localharness")

  @mock.patch.dict("os.environ", {}, clear=True)
  @mock.patch("importlib.metadata.distribution")
  @mock.patch("importlib.resources.files")
  @mock.patch("shutil.which")
  def test_raises_when_not_found(self, mock_which, mock_files, mock_dist):
    mock_dist.side_effect = importlib.metadata.PackageNotFoundError
    mock_files.side_effect = ImportError
    mock_which.return_value = None

    with self.assertRaises(RuntimeError) as ctx:
      _get_default_binary_path()
    self.assertIn(
        "Could not find default localharness binary", str(ctx.exception)
    )


class LocalConnectionSessionHooksTest(unittest.IsolatedAsyncioTestCase):
  """Tests for session start/end hook dispatch."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()
    self.tool_runner = tool_runner.ToolRunner()

  async def test_strategy_dispatches_session_start(self):
    """Verifies the connection correctly delegates OnSessionStart hook execution to the HookRunner whenever requested by the harness engine."""
    called = []
    event = asyncio.Event()

    class SessionStartHook(hooks_base.OnSessionStartHook):

      async def run(self, context, data):  # pylint: disable=unused-argument
        called.append("started")
        event.set()

    hr = hook_runner.HookRunner(on_session_start_hooks=[SessionStartHook()])

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    req = localharness_pb2.CallHookRequest(
        request_id="req_1",
        name="OnSessionStart",
        type=localharness_pb2.LIFECYCLE_HOOK_ON_SESSION_START,
    )
    await harness.send_event(
        localharness_pb2.OutputEvent(call_hook_request=req)
    )
    await asyncio.wait_for(event.wait(), timeout=1.0)
    self.assertEqual(called, ["started"])

  async def test_session_end_hook_dispatched_on_disconnect(self):
    """Verifies OnSessionEndHook fires when disconnect() is called."""
    called = []
    event = asyncio.Event()

    class SessionEndHook(hooks_base.OnSessionEndHook):

      async def run(self, context: hooks_base.HookContext, data: None):  # pylint: disable=unused-argument
        called.append("ended")
        event.set()

    hr = hook_runner.HookRunner()
    hr.register_hook(SessionEndHook())

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    await harness.conn.disconnect()
    await asyncio.wait_for(event.wait(), timeout=1.0)
    self.assertEqual(called, ["ended"])


class LocalConnectionPostTurnHookTest(unittest.IsolatedAsyncioTestCase):
  """Tests for post-turn hook dispatch."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()
    self.tool_runner = tool_runner.ToolRunner()

  async def test_post_turn_hook_dispatched_on_final_step(self):
    """Verifies PostTurnHook fires when a terminal model step is received."""
    captured = []

    class PostTurnHook(hooks_base.PostTurnHook):

      async def run(self, context: hooks_base.HookContext, data: str):  # pylint: disable=unused-argument
        captured.append(data)

    hr = hook_runner.HookRunner()
    hr.register_hook(PostTurnHook())

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    # Simulate a send to create turn context.
    await harness.conn.send("hello")

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="test_traj",
            trajectory_id="test_traj",
            step_index=1,
            text="Final answer",
            state=localharness_pb2.StepUpdate.STATE_DONE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            target=localharness_pb2.StepUpdate.TARGET_USER,
        )
    )

    await harness.send_event(event)

    # The real harness sends STATE_IDLE after the final step. The
    # connection waits for this before returning from receive_steps().
    idle_event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            trajectory_id="test_traj",
            state=localharness_pb2.TrajectoryStateUpdate.STATE_IDLE,
        )
    )
    await harness.send_event(idle_event)

    # Drain receive_steps to trigger terminal detection + hook dispatch.
    steps = []
    async for step in harness.conn.receive_steps():
      steps.append(step)

    self.assertEqual(len(steps), 1)
    self.assertEqual(captured, ["Final answer"])

  async def test_receive_steps_includes_target_environment(self):
    """Verifies TARGET_ENVIRONMENT steps are yielded by receive_steps()."""
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
    )

    # Simulate a send to create turn context.
    await harness.conn.send("hello")

    # Step 1: A TARGET_ENVIRONMENT step (tool execution).
    env_event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="test_traj",
            trajectory_id="test_traj",
            step_index=1,
            text="Requesting permission to make tool call",
            state=localharness_pb2.StepUpdate.STATE_DONE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            target=localharness_pb2.StepUpdate.TARGET_ENVIRONMENT,
        )
    )

    # Step 2: A TARGET_USER step (the final answer).
    user_event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="test_traj",
            trajectory_id="test_traj",
            step_index=2,
            text="Here is my answer.",
            state=localharness_pb2.StepUpdate.STATE_DONE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            target=localharness_pb2.StepUpdate.TARGET_USER,
        )
    )

    idle_event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            trajectory_id="test_traj",
            state=localharness_pb2.TrajectoryStateUpdate.STATE_IDLE,
        )
    )

    await harness.send_event(env_event)
    await harness.send_event(user_event)
    await harness.send_event(idle_event)

    steps = []
    async for step in harness.conn.receive_steps():
      steps.append(step)

    # Both steps must be yielded (the old filter would have dropped step 1).
    self.assertEqual(len(steps), 2)

    # Step 1: environment step — yielded but NOT a final response.
    self.assertEqual(
        steps[0].content, "Requesting permission to make tool call"
    )
    self.assertEqual(steps[0].target, "TARGET_ENVIRONMENT")
    self.assertFalse(steps[0].is_complete_response)

    # Step 2: user step — the real final response.
    self.assertEqual(steps[1].content, "Here is my answer.")
    self.assertEqual(steps[1].target, "TARGET_USER")
    self.assertTrue(steps[1].is_complete_response)

  async def test_post_turn_hook_not_fired_for_environment_step(self):
    """Verifies PostTurnHook does NOT fire for TARGET_ENVIRONMENT steps."""
    captured = []

    class PostTurnHook(hooks_base.PostTurnHook):

      async def run(self, context, data):  # pylint: disable=unused-argument
        captured.append(data)

    hr = hook_runner.HookRunner()
    hr.register_hook(PostTurnHook())

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    await harness.conn.send("hello")

    # A terminal environment step that should NOT trigger the hook.
    env_event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="test_traj",
            trajectory_id="test_traj",
            step_index=1,
            text="Requesting permission to make tool call",
            state=localharness_pb2.StepUpdate.STATE_DONE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            target=localharness_pb2.StepUpdate.TARGET_ENVIRONMENT,
        )
    )

    # The real final response that SHOULD trigger the hook.
    user_event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="test_traj",
            trajectory_id="test_traj",
            step_index=2,
            text="Final answer",
            state=localharness_pb2.StepUpdate.STATE_DONE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            target=localharness_pb2.StepUpdate.TARGET_USER,
        )
    )

    idle_event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            trajectory_id="test_traj",
            state=localharness_pb2.TrajectoryStateUpdate.STATE_IDLE,
        )
    )

    await harness.send_event(env_event)
    await harness.send_event(user_event)
    await harness.send_event(idle_event)

    steps = []
    async for step in harness.conn.receive_steps():
      steps.append(step)

    # Both steps yielded.
    self.assertEqual(len(steps), 2)

    # Hook fired exactly once, with the TARGET_USER step's content.
    self.assertEqual(captured, ["Final answer"])


class LocalConnectionCompactionHookTest(unittest.IsolatedAsyncioTestCase):
  """Tests for compaction hook dispatch."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()

  async def test_compaction_step_dispatches_hook(self):
    """Verifies OnCompactionHook fires when a compaction step is received."""
    captured = []
    event = asyncio.Event()

    class CompactionHook(hooks_base.OnCompactionHook):

      async def run(self, context, data):  # pylint: disable=unused-argument
        captured.append(data)
        event.set()

    hr = hook_runner.HookRunner()
    hr.register_hook(CompactionHook())

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        hook_runner=hr,
    )

    output_event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            text="Context compaction",
            state=localharness_pb2.StepUpdate.STATE_DONE,
            source=localharness_pb2.StepUpdate.SOURCE_SYSTEM,
            target=localharness_pb2.StepUpdate.TARGET_USER,
            compaction=localharness_pb2.ActionCompaction(),
        )
    )

    await harness.send_event(output_event)
    await asyncio.wait_for(event.wait(), timeout=1.0)

    self.assertEqual(len(captured), 1)
    self.assertIsInstance(captured[0], local_connection.LocalConnectionStep)
    self.assertEqual(captured[0].content, "Context compaction")


class LocalConnectionSubagentHookTest(unittest.IsolatedAsyncioTestCase):
  """Tests for subagent hook dispatch via tool hooks.

  Subagent invocations are treated as tool calls with the name
  START_SUBAGENT. Pre- and post-tool-call hooks receive the subagent
  data using standard tool hook dispatch.
  """

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()
    self.mock_ws = test_utils.TestWebSocket()

  async def test_invoke_subagent_step_classified_as_tool_call(self):
    """Verifies invoke_subagent steps are classified as TOOL_CALL."""
    hr = hook_runner.HookRunner()

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        hook_runner=hr,
    )

    await harness.conn.send("hello")

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="main",
            trajectory_id="main",
            step_index=1,
            text="Invoking subagent",
            state=localharness_pb2.StepUpdate.STATE_ACTIVE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            invoke_subagent=localharness_pb2.ActionInvokeSubagent(),
        )
    )

    await harness.send_event(event)

    # Drain the queue to inspect the step.
    step = await asyncio.wait_for(harness.conn._step_queue.get(), timeout=2.0)
    self.assertEqual(step.type, types.StepType.TOOL_CALL)

  async def test_post_tool_hook_on_subagent_trajectory_idle(self):
    """Verifies post-tool-call hook fires when a non-main trajectory goes idle."""
    hook_event = asyncio.Event()
    captured = []

    class PostToolHook(hooks_base.PostToolCallHook):

      async def run(self, context, data):  # pylint: disable=unused-argument
        captured.append(data)
        hook_event.set()

    hr = hook_runner.HookRunner()
    hr.register_hook(PostToolHook())

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        hook_runner=hr,
    )

    # Establish the cascade_id via a parent trajectory step
    # (cascade_id == trajectory_id).
    main_step = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="main_traj",
            step_index=0,
            trajectory_id="main_traj",
            text="Main step",
            state=localharness_pb2.StepUpdate.STATE_ACTIVE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
        )
    )
    await harness.send_event(main_step)
    # Wait for it to be processed by draining queue
    await asyncio.wait_for(harness.conn._step_queue.get(), timeout=2.0)

    self.assertEqual(harness.conn._cascade_id, "main_traj")

    # Simulate a subagent model step with text (may arrive as ACTIVE first).
    sub_active = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="main_traj",
            trajectory_id="sub_traj",
            step_index=0,
            text="Here is a poem about nature.",
            state=localharness_pb2.StepUpdate.STATE_ACTIVE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            target=localharness_pb2.StepUpdate.TARGET_USER,
        )
    )
    await harness.send_event(sub_active)
    # Wait for it to be processed by draining queue
    await asyncio.wait_for(harness.conn._step_queue.get(), timeout=2.0)

    # Now simulate the subagent trajectory going idle.
    idle_event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            trajectory_id="sub_traj",
            state=localharness_pb2.TrajectoryStateUpdate.STATE_IDLE,
        )
    )
    await harness.send_event(idle_event)
    await harness.wait_for_event(hook_event)

    self.assertEqual(len(captured), 1)
    self.assertIsInstance(captured[0], types.ToolResult)
    self.assertEqual(captured[0].name, types.BuiltinTools.START_SUBAGENT.value)
    self.assertEqual(captured[0].result, "Here is a poem about nature.")

    # Main trajectory idle should NOT fire post-tool hook for subagent.
    main_idle = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            trajectory_id="main_traj",
            state=localharness_pb2.TrajectoryStateUpdate.STATE_IDLE,
        )
    )
    await harness.send_event(main_idle)

    # Wait a tiny bit to ensure it didn't fire
    await asyncio.sleep(0.01)

    # Still only 1 capture.
    self.assertEqual(len(captured), 1)

  async def test_ws_reader_parses_usage_metadata(self):
    """Verifies that _ws_reader_loop parses and attaches usage_metadata to steps."""
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="main",
            trajectory_id="main",
            step_index=1,
            text="response with usage",
            state=localharness_pb2.StepUpdate.STATE_ACTIVE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
        ),
        usage_metadata=localharness_pb2.UsageMetadata(
            prompt_token_count=150,
            cached_content_token_count=50,
            candidates_token_count=75,
            thoughts_token_count=25,
            total_token_count=250,
        ),
    )

    await harness.send_event(event)

    step_obj = await asyncio.wait_for(
        harness.conn._step_queue.get(), timeout=1.0
    )

    self.assertEqual(
        step_obj.usage_metadata,
        types.UsageMetadata(
            prompt_token_count=150,
            cached_content_token_count=50,
            candidates_token_count=75,
            thoughts_token_count=25,
            total_token_count=250,
        ),
    )

  async def test_subagent_running_tracked(self):
    """Verifies STATE_RUNNING adds subagent to active set."""
    hr = hook_runner.HookRunner()
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        hook_runner=hr,
    )

    # Establish cascade_id.
    main_step = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="main",
            trajectory_id="main",
            step_index=0,
            text="hi",
            state=localharness_pb2.StepUpdate.STATE_ACTIVE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
        )
    )
    await harness.send_event(main_step)
    # Wait for it to be processed
    await asyncio.wait_for(harness.conn._step_queue.get(), timeout=2.0)

    # Subagent starts running.
    running_event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            trajectory_id="sub_1",
            state=(localharness_pb2.TrajectoryStateUpdate.STATE_RUNNING),
        )
    )
    await harness.send_event(running_event)

    # Poll for the state change to be processed
    async def poll_subagent_tracked():
      while "sub_1" not in harness.conn._active_subagent_ids:
        await asyncio.sleep(0.01)
      return True

    await asyncio.wait_for(poll_subagent_tracked(), timeout=2.0)

    self.assertIn("sub_1", harness.conn._active_subagent_ids)

  async def test_connection_waits_for_subagents_before_idle(self):
    """Verifies receive_steps blocks until subagents complete."""
    hr = hook_runner.HookRunner()
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        hook_runner=hr,
    )

    await harness.conn.send("hello")

    # Establish cascade_id + a step.
    main_step = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="main",
            trajectory_id="main",
            step_index=0,
            text="response",
            state=localharness_pb2.StepUpdate.STATE_ACTIVE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
        )
    )
    await harness.send_event(main_step)
    # Wait for it to be processed
    await asyncio.wait_for(harness.conn._step_queue.get(), timeout=2.0)

    # Subagent starts.
    await harness.send_event(
        localharness_pb2.OutputEvent(
            trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
                trajectory_id="sub_1",
                state=(localharness_pb2.TrajectoryStateUpdate.STATE_RUNNING),
            )
        )
    )

    # Parent goes idle, but subagent still running.
    await harness.send_event(
        localharness_pb2.OutputEvent(
            trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
                trajectory_id="main",
                state=(localharness_pb2.TrajectoryStateUpdate.STATE_IDLE),
            )
        )
    )

    # Poll for parent_idle to be True
    async def poll_parent_idle():
      while not harness.conn._parent_idle:
        await asyncio.sleep(0.01)
      return True

    await asyncio.wait_for(poll_parent_idle(), timeout=2.0)

    # _is_idle should NOT be set yet.
    self.assertFalse(harness.conn._is_idle.is_set())

    # Subagent completes.
    await harness.send_event(
        localharness_pb2.OutputEvent(
            trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
                trajectory_id="sub_1",
                state=(localharness_pb2.TrajectoryStateUpdate.STATE_IDLE),
            )
        )
    )

    # Wait for _is_idle to be set
    await asyncio.wait_for(harness.conn._is_idle.wait(), timeout=2.0)

    # NOW idle should be set.
    self.assertTrue(harness.conn._is_idle.is_set())

  async def test_send_resets_subagent_tracking(self):
    """Verifies send() clears subagent tracking state."""
    hr = hook_runner.HookRunner()
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        hook_runner=hr,
    )

    # Pollute tracking state.
    harness.conn._active_subagent_ids.add("leftover")
    harness.conn._subagent_responses["leftover"] = "stale response"
    harness.conn._parent_idle = True

    await harness.conn.send("new turn")

    self.assertEqual(harness.conn._active_subagent_ids, set())
    self.assertEqual(harness.conn._subagent_responses, {})
    self.assertFalse(harness.conn._parent_idle)
    self.assertFalse(harness.conn._is_idle.is_set())


class LocalConnectionToolCallHooksTest(unittest.IsolatedAsyncioTestCase):
  """Tests for post-tool-call and on-tool-error hooks."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()
    self.mock_ws = test_utils.TestWebSocket()

  async def test_post_tool_call_hook_dispatched(self):
    """Verifies PostToolCallHook fires after successful tool execution."""
    hook_event = asyncio.Event()
    captured_results = []

    class PostToolHook(hooks_base.PostToolCallHook):

      async def run(self, context, data):  # pylint: disable=unused-argument
        captured_results.append(data)
        hook_event.set()

    tr = tool_runner.ToolRunner()

    async def echo_handler(**kwargs):
      return json.dumps({"echo": kwargs})

    tr.register(echo_handler, "echo_tool")

    hr = hook_runner.HookRunner()
    hr.register_hook(PostToolHook())

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=tr,
        hook_runner=hr,
    )

    event = localharness_pb2.OutputEvent(
        tool_call=localharness_pb2.ToolCall(
            id="call_1",
            name="echo_tool",
            arguments_json='{"msg": "hi"}',
        )
    )

    await harness.send_event(event)
    await harness.wait_for_event(hook_event)

    self.assertEqual(len(captured_results), 1)
    self.assertIsInstance(captured_results[0], types.ToolResult)
    self.assertEqual(captured_results[0].name, "echo_tool")
    self.assertEqual(captured_results[0].result, '{"echo": {"msg": "hi"}}')

  async def test_on_tool_error_hook_with_recovery(self):
    """Verifies OnToolErrorHook can provide recovery values on tool failure."""

    class RecoveringErrorHook(hooks_base.OnToolErrorHook):

      async def run(self, context, data):  # pylint: disable=unused-argument
        return "recovered_value"

    tr = tool_runner.ToolRunner()

    async def failing_handler(**kwargs):
      raise RuntimeError("Intentional failure")

    tr.register(failing_handler, "failing_tool")

    hr = hook_runner.HookRunner()
    hr.register_hook(RecoveringErrorHook())

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=tr,
        hook_runner=hr,
    )

    event = localharness_pb2.OutputEvent(
        tool_call=localharness_pb2.ToolCall(
            id="call_fail",
            name="failing_tool",
            arguments_json="{}",
        )
    )

    await harness.send_event(event)

    # The recovery value should have been sent back.
    sent_data = await harness.wait_for_response()
    self.assertIn("toolResponse", sent_data)
    self.assertIn("recovered_value", sent_data["toolResponse"]["responseJson"])

  async def test_on_tool_error_hook_receives_original_exception_type(self):
    """Verifies OnToolErrorHook receives the original exception, not wrapped.

    Regression test for b/508736962: the hook should receive the original
    ValueError (not a RuntimeError wrapping the error string) so that
    isinstance-based dispatch works in hook implementations.
    """
    hook_event = asyncio.Event()
    captured_errors = []

    class CapturingErrorHook(hooks_base.OnToolErrorHook):

      async def run(self, context, data):  # pylint: disable=unused-argument
        captured_errors.append(data)
        hook_event.set()
        return "recovered"

    tr = tool_runner.ToolRunner()

    async def value_error_tool(**kwargs):
      raise ValueError("bad input")

    tr.register(value_error_tool, "value_error_tool")

    hr = hook_runner.HookRunner()
    hr.register_hook(CapturingErrorHook())

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=tr,
        hook_runner=hr,
    )

    event = localharness_pb2.OutputEvent(
        tool_call=localharness_pb2.ToolCall(
            id="call_typed",
            name="value_error_tool",
            arguments_json="{}",
        )
    )

    await harness.send_event(event)
    await harness.wait_for_event(hook_event)

    self.assertEqual(len(captured_errors), 1)
    # The hook must receive the original ValueError, not RuntimeError.
    self.assertIsInstance(captured_errors[0], ValueError)
    self.assertNotIsInstance(captured_errors[0], RuntimeError)
    self.assertIn("bad input", str(captured_errors[0]))


class LocalConnectionBuiltinDecideHookTest(unittest.IsolatedAsyncioTestCase):
  """Verifies Decide hooks run for built-in tool confirmations."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()
    self.mock_ws = test_utils.TestWebSocket()

  async def test_decide_hooks_run_for_builtin_tools(self):
    """Verifies PreToolCallDecideHook runs and can deny builtin tools."""

    class DenyAll(hooks_base.PreToolCallDecideHook):

      async def run(self, context, data):
        return hooks_base.HookResult(allow=False, message="Denied")

    hr = hook_runner.HookRunner(pre_tool_call_decide_hooks=[DenyAll()])
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        hook_runner=hr,
    )

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="traj",
            trajectory_id="traj",
            step_index=0,
            text='Requesting permission to call tool "run_command"',
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            target=localharness_pb2.StepUpdate.TARGET_ENVIRONMENT,
            tool_confirmation_request=(
                localharness_pb2.ToolConfirmationRequest()
            ),
            run_command=localharness_pb2.ActionRunCommand(
                command_line="rm -rf /",
            ),
        )
    )
    await harness.send_event(event)

    sent = await harness.wait_for_response()
    self.assertFalse(sent["toolConfirmation"]["accepted"])


class LocalConnectionHookAcceptanceTest(unittest.IsolatedAsyncioTestCase):
  """Verifies that previously-unsupported hooks are now accepted."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()
    self.mock_ws = test_utils.TestWebSocket()

  async def test_subagent_tool_hooks_accepted(self):
    """Subagent lifecycle is handled by tool hooks; no special subagent lists."""

    class DummyHook(hooks_base.PreToolCallDecideHook):

      async def run(self, context, data):  # pylint: disable=unused-argument
        return hooks_base.HookResult(allow=True)

    hr = hook_runner.HookRunner()
    hr.register_hook(DummyHook())

    # Should NOT raise.
    test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        hook_runner=hr,
    )

  async def test_compaction_hooks_no_longer_raise(self):
    """Compaction hooks should be accepted now."""

    class DummyHook(hooks_base.OnCompactionHook):

      async def run(self, context, data):  # pylint: disable=unused-argument
        pass

    hr = hook_runner.HookRunner()
    hr.register_hook(DummyHook())

    # Should NOT raise.
    test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        hook_runner=hr,
    )


class LocalConnectionStderrReaderTest(unittest.IsolatedAsyncioTestCase):
  """Tests for the background stderr reader thread."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()
    self.mock_ws = test_utils.TestWebSocket()

  async def test_start_stderr_reader_drains_lines(self):
    """Verifies that _start_stderr_reader captures stderr lines.

    Why: The Go harness writes diagnostic messages to stderr.  If the
    pipe buffer fills, the harness blocks and cannot save trajectory state
    at shutdown.  The reader thread prevents this by draining continuously.
    How: Write lines to a pipe, start the reader, and assert the deque
    contains all written lines.
    """

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )

    stream = io.BytesIO(b"line1\nline2\nline3\n")
    harness.conn._start_stderr_reader(stream)
    harness.conn._stderr_thread.join(timeout=2)

    self.assertEqual(
        list(harness.conn._stderr_lines), ["line1", "line2", "line3"]
    )

  async def test_stderr_reader_respects_maxlen(self):
    """Verifies the deque drops old lines when it exceeds maxlen.

    Why: Unbounded buffering could consume excessive memory during
    long-running sessions.  The deque is bounded at 100 lines.
    How: Write 105 lines and confirm only the last 100 remain.
    """

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )

    lines = "".join(f"line{i}\n" for i in range(105))
    stream = io.BytesIO(lines.encode())
    harness.conn._start_stderr_reader(stream)
    harness.conn._stderr_thread.join(timeout=2)

    self.assertEqual(len(harness.conn._stderr_lines), 100)
    self.assertEqual(harness.conn._stderr_lines[0], "line5")
    self.assertEqual(harness.conn._stderr_lines[-1], "line104")

  async def test_stderr_reader_handles_closed_stream(self):
    """Verifies the reader thread exits cleanly when the stream closes.

    Why: On process exit the stderr pipe closes.  The thread must not
    crash or log errors; it should simply stop.
    How: Pass an already-closed stream and verify the thread exits without
    raising.
    """
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )

    stream = io.BytesIO(b"")
    harness.conn._start_stderr_reader(stream)
    harness.conn._stderr_thread.join(timeout=2)
    self.assertFalse(harness.conn._stderr_thread.is_alive())

  async def test_stderr_reader_thread_is_daemon(self):
    """Verifies the stderr reader thread is a daemon thread.

    Why: The stderr reader must not prevent process exit.  If it were a
    non-daemon thread, a hung harness could keep the Python process alive
    indefinitely.
    How: Start the reader and check the thread's daemon attribute.
    """
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )

    stream = io.BytesIO(b"line1\n")
    harness.conn._start_stderr_reader(stream)
    self.assertTrue(harness.conn._stderr_thread.daemon)
    harness.conn._stderr_thread.join(timeout=2)


class LocalConnectionDisconnectTest(unittest.IsolatedAsyncioTestCase):
  """Tests for the disconnect shutdown sequence."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()
    self.mock_process.stdin = mock.MagicMock()
    self.mock_process.wait.return_value = 0
    self.mock_ws = test_utils.TestWebSocket()

  async def test_disconnect_sets_disconnecting_flag(self):
    """Verifies _disconnecting is set before any cleanup runs.

    Why: The reader loop uses this flag to distinguish expected closures
    from harness crashes.  It must be set early in disconnect().
    How: Call disconnect and check the flag is True.
    """
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )
    await harness.disconnect_sdk()
    self.assertTrue(harness.conn._disconnecting)

  async def test_disconnect_closes_stdin(self):
    """Verifies stdin is closed during disconnect to trigger harness save.

    Why: The Go harness monitors stdin for EOF.  On EOF it runs
    cleanupAllAgents which persists trajectory state to disk.  Without
    closing stdin, the trajectory is never saved.
    How: Call disconnect and verify stdin.close() was called.
    """
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )
    await harness.disconnect_sdk()
    self.mock_process.stdin.close.assert_called_once()

  async def test_disconnect_waits_for_process(self):
    """Verifies disconnect waits for the harness process to exit.

    Why: The harness needs time to flush trajectory state after stdin
    closes.  Killing it immediately would lose the trajectory.
    How: Call disconnect and verify process.wait(timeout=5) was called.
    """
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )
    await harness.disconnect_sdk()
    self.mock_process.wait.assert_called_with(timeout=5)

  async def test_disconnect_terminates_on_timeout(self):
    """Verifies SIGTERM is sent when the process doesn't exit in time.

    Why: If the harness hangs during cleanup, the SDK must not block
    indefinitely.  SIGTERM is the first escalation.
    How: Make wait() raise TimeoutExpired on the first call, then verify
    terminate() is called.
    """
    self.mock_process.wait.side_effect = [
        subprocess.TimeoutExpired("cmd", 5),  # First wait times out.
        0,  # After terminate, process exits.
    ]
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )
    await harness.disconnect_sdk()
    self.mock_process.terminate.assert_called_once()

  async def test_disconnect_kills_on_double_timeout(self):
    """Verifies SIGKILL is sent when SIGTERM also fails.

    Why: If the process ignores SIGTERM, SIGKILL is the last resort.
    How: Make wait() raise TimeoutExpired twice, then verify kill() is called.
    """
    self.mock_process.wait.side_effect = [
        subprocess.TimeoutExpired("cmd", 5),  # First wait.
        subprocess.TimeoutExpired("cmd", 1),  # After terminate.
        0,  # After kill.
    ]
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )
    await harness.disconnect_sdk()
    self.mock_process.terminate.assert_called_once()
    self.mock_process.kill.assert_called_once()

  async def test_disconnect_closes_ws_before_stdin(self):
    """Verifies the WebSocket is closed before stdin.

    Why: The Go HTTP handler's defer saves the trajectory when the handler
    returns.  agent.Close() blocks on <-runChan, which requires the Run
    goroutine to exit.  Run exits when the WS input loop breaks.  So the
    WS must close first to unblock agent.Close().  Stdin close triggers
    os.Exit(0), so it must come after the defer has had time to save.
    How: Record the call order of ws.close and stdin.close.
    """
    call_order = []

    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )

    original_close = harness.ws.close

    async def track_ws_close():
      call_order.append("ws_close")
      await original_close()

    harness.ws.close = track_ws_close
    self.mock_process.stdin.close.side_effect = lambda: call_order.append(
        "stdin_close"
    )

    await harness.disconnect_sdk()
    self.assertEqual(call_order, ["ws_close", "stdin_close"])


class LocalConnectionUnexpectedCloseTest(unittest.IsolatedAsyncioTestCase):
  """Tests for error surfacing when the harness crashes mid-session."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()

  async def test_unexpected_ws_close_surfaces_stderr(self):
    """Verifies harness stderr is surfaced when the WS closes unexpectedly.

    Why: When the harness crashes (e.g., model error, OOM), the WebSocket
    closes with code 1006.  The user needs the harness stderr to diagnose
    the failure.  Previously, this was silently logged and swallowed.
    How: Simulate a ConnectionClosed exception in the reader loop and
    verify an AntigravityConnectionError with stderr content is queued.
    """

    # Create a FakeWebSocket that raises ConnectionClosed immediately.
    class CrashingWebSocket:

      def __init__(self):
        self.sent_messages = []

      async def send(self, message):
        self.sent_messages.append(message)

      def __aiter__(self):
        async def _gen():
          raise websockets.ConnectionClosed(rcvd=None, sent=None)
          yield  # Make it a generator.  pylint: disable=unreachable

        return _gen()

      async def close(self):
        pass

    ws = CrashingWebSocket()
    conn = local_connection.LocalConnection(
        process=self.mock_process,
        ws=ws,
    )
    # Seed some stderr context.
    conn._stderr_lines.append("Failed to call model: quota exceeded")

    # The step queue should contain the error, then the sentinel None.
    item = await asyncio.wait_for(conn._step_queue.get(), timeout=2)
    self.assertIsInstance(item, types.AntigravityConnectionError)
    self.assertIn("quota exceeded", str(item))
    self.assertIn("WS close code", str(item))

  async def test_expected_ws_close_does_not_surface_error(self):
    """Verifies no error is queued when disconnect() initiated the close.

    Why: When the user calls disconnect(), the WebSocket close is expected
    and should not be reported as an error.
    How: Set _disconnecting=True, trigger a ConnectionClosed, and verify
    only the sentinel (None) is in the queue.
    """

    class DisconnectingWebSocket:

      def __init__(self):
        self.sent_messages = []

      async def send(self, message):
        self.sent_messages.append(message)

      def __aiter__(self):
        async def _gen():
          raise websockets.ConnectionClosed(rcvd=None, sent=None)
          yield  # pylint: disable=unreachable

        return _gen()

      async def close(self):
        pass

    ws = DisconnectingWebSocket()
    conn = local_connection.LocalConnection(
        process=self.mock_process,
        ws=ws,
    )
    conn._disconnecting = True

    # Should only see the sentinel, not an error.
    item = await asyncio.wait_for(conn._step_queue.get(), timeout=2)
    self.assertIsNone(item)


class LocalConnectionSendTest(unittest.IsolatedAsyncioTestCase):
  """Validates multi-modal coercion and InputEvent serialization inside LocalConnection.send()."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()
    self.mock_ws = test_utils.TestWebSocket()

  async def test_send_flat_string_populates_user_input(self):
    """Verifies that a standard string prompt maps to the user_input proto field."""
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )
    await harness.conn.send("Standard text prompt")

    sent_data = await harness.wait_for_response()
    self.assertEqual(sent_data.get("userInput"), "Standard text prompt")
    self.assertNotIn("complexUserInput", sent_data)

  async def test_send_none_prompt_populates_blank_string(self):
    """Verifies that passing a prompt of None maps to a blank userInput string frame."""
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )
    await harness.conn.send(None)

    sent_data = await harness.wait_for_response()

    # Assert it sets userInput to a blank string and does not use complex inputs
    self.assertEqual(sent_data.get("userInput"), "")
    self.assertNotIn("complexUserInput", sent_data)

  async def test_send_single_media_content_populates_complex_user_input(self):
    """Verifies that a single rich Content primitive maps to the complex_user_input parts list."""
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )
    image_content = types.Image(
        mime_type="image/png",
        data=b"fake_png",
        description="logo image",
    )
    await harness.conn.send(image_content)

    sent_data = await harness.wait_for_response()

    self.assertNotIn("userInput", sent_data)
    self.assertIn("complexUserInput", sent_data)

    parts = sent_data["complexUserInput"]["parts"]
    self.assertEqual(len(parts), 1)
    self.assertIn("media", parts[0])
    media = parts[0]["media"]
    self.assertEqual(media["mimeType"], "image/png")
    self.assertEqual(media["description"], "logo image")
    # Protobuf JSON automatically base64-encodes binary bytes
    self.assertEqual(media["data"], "ZmFrZV9wbmc=")  # b"fake_png"

  async def test_send_mixed_list_populates_multiple_complex_content(self):
    """Verifies that a list containing both strings and rich Content primitives compiles correctly to spec."""
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )
    mixed_prompt = [
        "Context text instruction.",
        types.Document(mime_type="application/pdf", data=b"fake_pdf"),
    ]
    await harness.conn.send(mixed_prompt)

    sent_data = await harness.wait_for_response()

    self.assertNotIn("userInput", sent_data)
    self.assertIn("complexUserInput", sent_data)

    parts = sent_data["complexUserInput"]["parts"]
    self.assertEqual(len(parts), 2)

    self.assertEqual(parts[0]["text"], "Context text instruction.")

    self.assertEqual(parts[1]["media"]["mimeType"], "application/pdf")
    self.assertEqual(parts[1]["media"]["data"], "ZmFrZV9wZGY=")  # b"fake_pdf"

  async def test_send_slash_command_populates_complex_user_input(self):
    """Verifies that a SlashCommand primitive maps to complex_user_input slash_command field."""
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )
    slash_command = types.SlashCommand(
        name=types.BuiltinSlashCommandName.PLAN,
    )
    await harness.conn.send(slash_command)

    sent_data = await harness.wait_for_response()

    self.assertNotIn("userInput", sent_data)
    self.assertIn("complexUserInput", sent_data)

    parts = sent_data["complexUserInput"]["parts"]
    self.assertEqual(len(parts), 1)
    self.assertIn("slashCommand", parts[0])
    sc = parts[0]["slashCommand"]
    self.assertEqual(sc["name"], "plan")

  async def test_concurrent_receive_steps_raises(self):
    """Verifies that a second concurrent receive_steps() call raises RuntimeError.

    The connection sets _is_receiving on entry and clears it on exit.
    A second caller must fail immediately with a clear message rather than
    silently racing on the shared step queue.
    """
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )
    # Put the connection into a non-idle state so receive_steps blocks
    # waiting for steps.
    harness.conn._is_idle.clear()

    first_started = asyncio.Event()

    async def _first_receiver():
      first_started.set()
      async for _ in harness.conn.receive_steps():
        pass  # Will block on the queue until idle or close sentinel.

    task = asyncio.create_task(_first_receiver())
    await first_started.wait()
    # Give the first receiver a moment to enter the iterator body.
    await asyncio.sleep(0.05)

    with self.assertRaises(RuntimeError) as ctx:
      async for _ in harness.conn.receive_steps():
        pass

    self.assertIn(
        "Concurrent receive_steps() calls are not supported", str(ctx.exception)
    )

    # Clean up: signal idle so the first receiver can exit.
    harness.conn._is_idle.set()
    await harness.conn._step_queue.put(local_connection._IDLE_SENTINEL)
    task.cancel()
    try:
      await task
    except asyncio.CancelledError:
      pass

  async def test_trigger_notification_succeeds_while_busy(self):
    """Verifies send_trigger_notification() and send() work during an active turn.

    There is no connection-level guard on sending while a receive is in
    progress. Callers (e.g. scheduled triggers) must be able to inject
    new prompts even when steps are still being consumed.
    """
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )
    # Start a turn so the connection is non-idle.
    await harness.conn.send("initial prompt")
    initial_msg = await harness.wait_for_response()
    self.assertEqual(initial_msg.get("userInput"), "initial prompt")

    # send_trigger_notification should succeed even though we are mid-turn.
    await harness.conn.send_trigger_notification("trigger content")
    trigger_msg = await harness.wait_for_response()
    self.assertIn("automatedTrigger", trigger_msg)
    self.assertEqual(trigger_msg["automatedTrigger"], "trigger content")

    # A regular send() should also succeed (no send-side guard).
    await harness.conn.send("follow-up prompt")
    followup_msg = await harness.wait_for_response()
    self.assertEqual(followup_msg.get("userInput"), "follow-up prompt")


class LocalAgentConfigTest(absltest.TestCase):

  def test_create_strategy(self):
    config = local_connection_config.LocalAgentConfig(
        system_instructions="test instructions",
        model="gemini-2.5-pro",
    )

    mock_tool_runner = mock.create_autospec(
        tool_runner.ToolRunner, instance=True
    )
    mock_hook_runner = mock.create_autospec(
        hook_runner.HookRunner, instance=True
    )

    strategy = config.create_strategy(
        tool_runner=mock_tool_runner,
        hook_runner=mock_hook_runner,
    )

    self.assertIsInstance(strategy, local_connection.LocalConnectionStrategy)
    self.assertIsNotNone(strategy._models)
    text_models = [
        m for m in strategy._models if types.ModelType.TEXT in m.types
    ]
    self.assertLen(text_models, 1)
    self.assertEqual(text_models[0].name, "gemini-2.5-pro")

  def test_merge_models_only_defaults(self):
    config = local_connection_config.LocalAgentConfig()
    self.assertLen(config.models, 2)
    self.assertEqual(config.models[0].name, DEFAULT_MODEL)
    self.assertEqual(config.models[0].types, [types.ModelType.TEXT])
    self.assertEqual(
        config.models[1].name,
        local_connection_config.DEFAULT_IMAGE_GENERATION_MODEL,
    )
    self.assertEqual(config.models[1].types, [types.ModelType.IMAGE])

  def test_merge_models_shorthand_only(self):
    config = local_connection_config.LocalAgentConfig(model="custom-text-model")
    self.assertLen(config.models, 2)
    self.assertEqual(config.models[0].name, "custom-text-model")
    self.assertEqual(config.models[0].types, [types.ModelType.TEXT])
    self.assertEqual(
        config.models[1].name,
        local_connection_config.DEFAULT_IMAGE_GENERATION_MODEL,
    )
    self.assertEqual(config.models[1].types, [types.ModelType.IMAGE])

  def test_merge_models_explicit_only(self):
    custom_image = types.ModelTarget(
        name="custom-image-model", types=[types.ModelType.IMAGE]
    )
    config = local_connection_config.LocalAgentConfig(models=[custom_image])
    self.assertLen(config.models, 2)
    self.assertEqual(config.models[0].name, "custom-image-model")
    self.assertEqual(config.models[0].types, [types.ModelType.IMAGE])
    self.assertEqual(config.models[1].name, DEFAULT_MODEL)
    self.assertEqual(config.models[1].types, [types.ModelType.TEXT])

  def test_merge_models_explicit_and_shorthand(self):
    custom_image = types.ModelTarget(
        name="custom-image-model", types=[types.ModelType.IMAGE]
    )
    config = local_connection_config.LocalAgentConfig(
        model="custom-text-model", models=[custom_image]
    )
    self.assertLen(config.models, 2)
    self.assertEqual(config.models[0].name, "custom-image-model")
    self.assertEqual(config.models[0].types, [types.ModelType.IMAGE])
    self.assertEqual(config.models[1].name, "custom-text-model")
    self.assertEqual(config.models[1].types, [types.ModelType.TEXT])

  def test_constructor_parameters_fully_typed(self):
    """Verifies all subclass fields are accepted by the constructor under pytype."""
    config = local_connection_config.LocalAgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(enable_subagents=True),
        tools=[],
        policies=[],
        hooks=[],
        triggers=[],
        mcp_servers=[],
        workspaces=["/tmp/ws"],
        conversation_id="123",
        save_dir="/tmp/save",
        app_data_dir="/tmp/app",
        response_schema="{}",
        skills_paths=["/tmp/skills"],

        model="gemini-2.5-pro",
        api_key="fake_api_key",
        vertex=True,
        project="my_project",
        location="us-central1",
    )
    self.assertEqual(config.model, "gemini-2.5-pro")
    self.assertEqual(config.api_key, "fake_api_key")
    self.assertTrue(config.vertex)
    self.assertEqual(config.project, "my_project")
    self.assertEqual(config.location, "us-central1")
    self.assertEqual(config.conversation_id, "123")

  def test_safe_defaults(self):
    """LocalAgentConfig defaults to confirm_run_command() — deny run_command."""
    config = local_connection_config.LocalAgentConfig(
        system_instructions="test",
        workspaces=[],
    )
    self.assertIsNone(config.capabilities.enabled_tools)
    self.assertIsNone(config.capabilities.disabled_tools)
    # confirm_run_command() produces 2 policies: deny(run_command) + allow(*)
    self.assertLen(config.policies, 2)
    deny_policy = config.policies[0]
    self.assertEqual(deny_policy.tool, "run_command")
    self.assertEqual(deny_policy.decision, policy.Decision.DENY)
    self.assertEqual(deny_policy.name, "confirm_run_command")
    allow_policy = config.policies[1]
    self.assertEqual(allow_policy.tool, "*")
    self.assertEqual(allow_policy.decision, policy.Decision.APPROVE)

  def test_safe_defaults_with_default_workspace(self):
    """LocalAgentConfig defaults to CWD workspace when not specified."""
    config = local_connection_config.LocalAgentConfig(
        system_instructions="test",
    )
    self.assertEqual(config.workspaces, [os.getcwd()])
    # workspace_only produces 3 deny policies (view_file, create_file,
    # edit_file), followed by the 2 confirm_run_command policies.
    self.assertLen(config.policies, 5)
    for i in range(3):
      self.assertEqual(config.policies[i].decision, policy.Decision.DENY)
      self.assertEqual(config.policies[i].name, "workspace_only")
    self.assertEqual(config.policies[3].tool, "run_command")
    self.assertEqual(config.policies[4].tool, "*")

  def test_workspace_policies_auto_prepended(self):
    """workspace_only() policies are auto-prepended when workspaces are set."""
    config = local_connection_config.LocalAgentConfig(
        system_instructions="test",
        workspaces=["/tmp/ws"],
    )
    # workspace_only produces 3 deny policies (view_file, create_file,
    # edit_file), followed by the 2 confirm_run_command policies.
    self.assertLen(config.policies, 5)
    # First 3 should be workspace_only deny policies for file tools.
    for i in range(3):
      self.assertEqual(config.policies[i].decision, policy.Decision.DENY)
      self.assertEqual(config.policies[i].name, "workspace_only")
    # Last 2 should be confirm_run_command.
    self.assertEqual(config.policies[3].tool, "run_command")
    self.assertEqual(config.policies[4].tool, "*")

  def test_explicit_allow_all_overrides_default(self):
    """Explicit allow_all() replaces the confirm_run_command default."""
    config = local_connection_config.LocalAgentConfig(
        system_instructions="test",
        policies=[policy.allow_all()],
        workspaces=[],
    )
    self.assertLen(config.policies, 1)
    self.assertEqual(config.policies[0].tool, "*")
    self.assertEqual(config.policies[0].decision, policy.Decision.APPROVE)

  def test_create_strategy_app_data_dir(self):
    config = local_connection_config.LocalAgentConfig(
        system_instructions="test instructions",
        app_data_dir="/foo/bar",
    )

    mock_tool_runner = mock.create_autospec(
        tool_runner.ToolRunner, instance=True
    )
    mock_hook_runner = mock.create_autospec(
        hook_runner.HookRunner, instance=True
    )

    strategy = config.create_strategy(
        tool_runner=mock_tool_runner,
        hook_runner=mock_hook_runner,
    )

    self.assertIsInstance(strategy, local_connection.LocalConnectionStrategy)
    self.assertEqual(strategy._app_data_dir, "/foo/bar")

  def test_app_data_dir_relative_path_raises(self):
    with self.assertRaises(pydantic.ValidationError):
      local_connection_config.LocalAgentConfig(
          system_instructions="test",
          app_data_dir="relative/path",
      )

  def test_create_strategy_with_mcp_servers(self):
    stdio_cfg = types.McpStdioServer(
        name="my-stdio",
        command="npx",
        args=["math"],
        env={"FOO": "bar"},
        enabled_tools=["add", "sub"],
    )
    sse_cfg = types.McpStreamableHttpServer(
        name="my-sse",
        url="https://sse.example.com",
    )
    config = local_connection_config.LocalAgentConfig(
        system_instructions="test",
        mcp_servers=[stdio_cfg, sse_cfg],
        api_key="fake",
    )

    mock_tool_runner = mock.create_autospec(
        tool_runner.ToolRunner, instance=True
    )
    mock_hook_runner = mock.create_autospec(
        hook_runner.HookRunner, instance=True
    )

    strategy = config.create_strategy(
        tool_runner=mock_tool_runner,
        hook_runner=mock_hook_runner,
    )

    harness_pb = strategy._build_harness_config()

    self.assertLen(harness_pb.mcp_servers, 2)

    stdio_pb = harness_pb.mcp_servers[0]
    self.assertEqual(stdio_pb.name, "my-stdio")
    self.assertEqual(stdio_pb.enabled_tools, ["add", "sub"])
    self.assertEqual(stdio_pb.stdio.command, "npx")
    self.assertEqual(stdio_pb.stdio.args, ["math"])
    self.assertEqual(dict(stdio_pb.stdio.env), {"FOO": "bar"})

    sse_pb = harness_pb.mcp_servers[1]
    self.assertEqual(sse_pb.name, "my-sse")
    self.assertEqual(sse_pb.http.url, "https://sse.example.com")


class LocalAgentConfigWorkspaceTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):
  """Tests for workspace scoping policy with app_data_dir inclusion."""

  @parameterized.named_parameters(
      dict(
          testcase_name="allowed_in_workspace",
          app_data_dir_factory=lambda temp_dir: str(
              temp_dir / "my_custom_app_data"
          ),
          path_factory=lambda temp_dir: str(temp_dir / "my_workspace/file.txt"),
          expected_allowed=True,
          msg="Target inside workspace should be allowed",
      ),
      dict(
          testcase_name="allowed_in_custom_app_data_dir",
          app_data_dir_factory=lambda temp_dir: str(
              temp_dir / "my_custom_app_data"
          ),
          path_factory=lambda temp_dir: str(
              temp_dir / "my_custom_app_data/brain/123/artifact.md"
          ),
          expected_allowed=True,
          msg="Target inside custom app_data_dir should be allowed",
      ),
      dict(
          testcase_name="allowed_in_default_app_data_dir",
          app_data_dir_factory=lambda _: None,
          path_factory=lambda temp_dir: str(
              temp_dir / "my_default_app_data/brain/123/artifact.md"
          ),
          expected_allowed=True,
          msg=(
              "Target inside default app_data_dir should be allowed when config"
              " is None"
          ),
      ),
      dict(
          testcase_name="denied_outside_both",
          app_data_dir_factory=lambda temp_dir: str(
              temp_dir / "my_custom_app_data"
          ),
          path_factory=lambda temp_dir: str(temp_dir / "outside/passwd"),
          expected_allowed=False,
          msg="Target outside both workspace and app_data_dir should be denied",
      ),
  )
  async def test_workspace_policy_scenarios(
      self,
      app_data_dir_factory,
      path_factory,
      expected_allowed: bool,
      msg: str,
  ):
    # Create dynamic, hermetic temporary directory
    temp_dir_path = pathlib.Path(self.create_tempdir().full_path)

    workspace_dir = temp_dir_path / "my_workspace"
    default_app_data_dir = temp_dir_path / "my_default_app_data"

    # Mock the module-level constant to use our hermetic default app data dir
    with mock.patch.object(
        local_connection_config,
        "DEFAULT_APP_DATA_DIR",
        str(default_app_data_dir),
    ):
      app_data_dir = app_data_dir_factory(temp_dir_path)
      path = path_factory(temp_dir_path)

      config = local_connection_config.LocalAgentConfig(
          system_instructions="test",
          workspaces=[str(workspace_dir)],
          app_data_dir=app_data_dir,
      )

      # workspace_only policies are the first 3
      policies = config.policies[:3]
      hook = policy.enforce(policies)
      ctx = hooks_base.HookContext()

      tc = types.ToolCall(
          name="view_file",
          args={"path": path},
          canonical_path=path,
      )
      res = await hook.run(ctx, tc)
      self.assertEqual(res.allow, expected_allowed, msg=msg)

  async def test_workspace_policy_denies_symlink_traversal(self):
    """Tests that the workspace scoping policy correctly blocks symlinks pointing outside."""
    temp_dir_path = pathlib.Path(self.create_tempdir().full_path)

    # Define safe workspace and unsafe outer target
    workspace_dir = temp_dir_path / "my_workspace"
    workspace_dir.mkdir(exist_ok=True)

    outer_dir = temp_dir_path / "outer"
    outer_dir.mkdir(exist_ok=True)
    outer_file = outer_dir / "secret.txt"
    outer_file.write_text("sensitive data")

    # Create a symbolic link inside the workspace pointing to the outer file
    symlink_path = workspace_dir / "escape_link.txt"
    os.symlink(outer_file, symlink_path)

    config = local_connection_config.LocalAgentConfig(
        system_instructions="test",
        workspaces=[str(workspace_dir)],
        app_data_dir=None,
    )

    # workspace_only policies are the first 3
    policies = config.policies[:3]
    hook = policy.enforce(policies)
    ctx = hooks_base.HookContext()

    # Dispatch a tool call targeting the symlink path
    tc = types.ToolCall(
        name="view_file",
        args={"path": str(symlink_path)},
        canonical_path=str(symlink_path),
    )
    res = await hook.run(ctx, tc)

    # Assert that the policy correctly resolves the symlink and BLOCKS the
    # access
    self.assertFalse(
        res.allow,
        msg="Workspace policy must resolve symlinks and block traversal",
    )


class LocalConnectionBuiltinToolHooksTest(unittest.IsolatedAsyncioTestCase):
  """Tests for PostToolCallHook / OnToolErrorHook dispatch for built-in tools.

  Built-in tools (run_command, list_directory, etc.) execute inside the Go
  harness and report results via StepUpdate proto messages over WebSocket.
  The SDK tracks approved tool calls via _pending_builtin_tool_calls and
  dispatches PostToolCallHook (on STATE_DONE) or OnToolErrorHook (on
  STATE_ERROR) with structured results extracted by _extract_tool_result().

  These tests simulate the full confirmation→completion lifecycle for each
  built-in tool type, verifying that hooks receive the correct typed result.
  """

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock(spec=subprocess.Popen)
    self.tool_runner = tool_runner.ToolRunner()

  def _make_harness(self, hr):
    """Creates a TestLocalHarness with the given HookRunner."""
    return test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

  async def _confirm_and_complete(self, harness, confirm_event, done_event):
    """Sends a confirmation request and then a completion event.

    Args:
      harness: The TestLocalHarness instance.
      confirm_event: The OutputEvent with STATE_WAITING_FOR_USER.
      done_event: The OutputEvent with STATE_DONE or STATE_ERROR.

    Returns:
      The confirmation response dict from the SDK.
    """
    await harness.send_event(confirm_event)
    sent_data = await harness.wait_for_response()
    await harness.send_event(done_event)
    return sent_data

  def _make_confirm_event(self, step_index, traj_id, **action_kwargs):
    """Builds a STATE_WAITING_FOR_USER OutputEvent with an action field."""
    return localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=step_index,
            trajectory_id=traj_id,
            cascade_id=traj_id,
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            tool_confirmation_request=localharness_pb2.ToolConfirmationRequest(),
            **action_kwargs,
        )
    )

  def _make_done_event(self, step_index, traj_id, **action_kwargs):
    """Builds a STATE_DONE OutputEvent with an action field."""
    return localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=step_index,
            trajectory_id=traj_id,
            cascade_id=traj_id,
            state=localharness_pb2.StepUpdate.STATE_DONE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            **action_kwargs,
        )
    )

  async def _run_post_hook_test(
      self,
      confirm_kwargs,
      done_kwargs,
      expected_name,
      expected_type,
      assertions_fn,
  ):
    """Runs the confirm→done→assert pattern for a PostToolCallHook test.

    Args:
      confirm_kwargs: kwargs for _make_confirm_event (action fields).
      done_kwargs: kwargs for _make_done_event (action + result fields).
      expected_name: Expected ToolResult.name value.
      expected_type: Expected type of ToolResult.result.
      assertions_fn: Callable(result) for type-specific assertions.
    """
    hook_event = asyncio.Event()
    captured = []

    class PostHook(hooks_base.PostToolCallHook):  # pylint: disable=unused-argument

      async def run(self, context, data):
        captured.append(data)
        hook_event.set()

    hr = hook_runner.HookRunner()
    hr.register_hook(PostHook())
    harness = self._make_harness(hr)

    confirm = self._make_confirm_event(0, "traj", **confirm_kwargs)
    done = self._make_done_event(0, "traj", **done_kwargs)
    sent_data = await self._confirm_and_complete(harness, confirm, done)
    self.assertTrue(sent_data["toolConfirmation"]["accepted"])

    await harness.wait_for_event(hook_event)

    self.assertEqual(len(captured), 1)
    result = captured[0]
    self.assertEqual(result.name, expected_name)
    if expected_type is not None:
      self.assertIsInstance(result.result, expected_type)
    assertions_fn(result)

  # ---- Per-tool-type tests ----

  async def test_tool_result_for_run_command(self):
    """Verifies PostToolCallHook receives RunCommandResult for run_command.

    What: PostToolCallHook receives a RunCommandResult with combined_output.
    Why: run_command is the most common built-in tool; its stdout/stderr must
         be available to hooks for logging, auditing, and policy enforcement.
    How: Simulate approval + STATE_DONE with combined_output set; assert the
         hook receives RunCommandResult with the correct output string.
    """
    from google.antigravity.connections.local import types as local_types  # pylint: disable=g-import-not-at-top

    await self._run_post_hook_test(
        confirm_kwargs=dict(
            run_command=localharness_pb2.ActionRunCommand(
                command_line="echo hello",
            ),
        ),
        done_kwargs=dict(
            run_command=localharness_pb2.ActionRunCommand(
                command_line="echo hello",
                combined_output="hello\n",
            ),
        ),
        expected_name=types.BuiltinTools.RUN_COMMAND.value,
        expected_type=local_types.RunCommandResult,
        assertions_fn=lambda r: self.assertEqual(r.result.output, "hello\n"),
    )

  async def test_tool_result_for_list_directory(self):
    """Verifies PostToolCallHook receives ListDirectoryResult for list_dir.

    What: PostToolCallHook receives a ListDirectoryResult with structured
          directory entries.
    Why: list_directory returns structured entry data (name, is_directory,
         file_size) via Result sub-messages, unlike tools that return raw text.
         This tests the most complex extraction branch in _extract_tool_result.
    How: Simulate approval + STATE_DONE with two Result entries; assert the
         hook receives ListDirectoryResult with correctly parsed entries.
    """
    from google.antigravity.connections.local import types as local_types  # pylint: disable=g-import-not-at-top

    await self._run_post_hook_test(
        confirm_kwargs=dict(
            list_directory=localharness_pb2.ActionListDirectory(
                directory_path="/tmp/test",
            ),
        ),
        done_kwargs=dict(
            list_directory=localharness_pb2.ActionListDirectory(
                directory_path="/tmp/test",
                results=[
                    localharness_pb2.ActionListDirectory.Result(
                        name="foo.py", file_size=100
                    ),
                    localharness_pb2.ActionListDirectory.Result(
                        name="bar", is_directory=True
                    ),
                ],
            ),
        ),
        expected_name=types.BuiltinTools.LIST_DIR.value,
        expected_type=local_types.ListDirectoryResult,
        assertions_fn=lambda r: (
            self.assertEqual(len(r.result.entries), 2),
            self.assertEqual(r.result.entries[0].name, "foo.py"),
        ),
    )

  async def test_tool_result_for_find_file(self):
    """Verifies PostToolCallHook receives FindFileResult for find_file.

    What: PostToolCallHook receives a FindFileResult with raw find output.
    Why: find_file returns a newline-separated list of matching file paths.
         Hooks may use this for auditing which files were discovered.
    How: Simulate approval + STATE_DONE with output set; assert the hook
         receives FindFileResult with the correct output string.
    """
    from google.antigravity.connections.local import types as local_types  # pylint: disable=g-import-not-at-top

    await self._run_post_hook_test(
        confirm_kwargs=dict(
            find_file=localharness_pb2.ActionFindFile(
                directory_path="/tmp/searchdir",
                query="target.txt",
            ),
        ),
        done_kwargs=dict(
            find_file=localharness_pb2.ActionFindFile(
                directory_path="/tmp/searchdir",
                query="target.txt",
                output="/tmp/searchdir/target.txt",
            ),
        ),
        expected_name=types.BuiltinTools.FIND_FILE.value,
        expected_type=local_types.FindFileResult,
        assertions_fn=lambda r: self.assertEqual(
            r.result.output, "/tmp/searchdir/target.txt"
        ),
    )

  async def test_tool_result_for_search_directory(self):
    """Verifies PostToolCallHook receives SearchDirectoryResult for grep_search.

    What: PostToolCallHook receives a SearchDirectoryResult with num_results.
    Why: search_directory (grep_search) returns a count of matching results.
         Hooks may use this for observability (e.g. "search found N results").
    How: Simulate approval + STATE_DONE with num_results=3; assert the hook
         receives SearchDirectoryResult with the correct count.
    """
    from google.antigravity.connections.local import types as local_types  # pylint: disable=g-import-not-at-top

    await self._run_post_hook_test(
        confirm_kwargs=dict(
            search_directory=localharness_pb2.ActionSearchDirectory(
                directory_path="/tmp",
                query="hello",
            ),
        ),
        done_kwargs=dict(
            search_directory=localharness_pb2.ActionSearchDirectory(
                directory_path="/tmp",
                query="hello",
                num_results=3,
            ),
        ),
        expected_name=types.BuiltinTools.SEARCH_DIR.value,
        expected_type=local_types.SearchDirectoryResult,
        assertions_fn=lambda r: self.assertEqual(r.result.num_results, 3),
    )

  async def test_tool_result_for_edit_file(self):
    """Verifies PostToolCallHook receives EditFileResult for edit_file.

    What: PostToolCallHook receives an EditFileResult with a text summary.
    Why: edit_file returns diff blocks; _extract_tool_result checks for the
         presence of diff_block and falls back to step_update.text for the
         summary. This tests the diff_block detection branch.
    How: Simulate approval + STATE_DONE with a diff_block set and text
         containing the summary; assert EditFileResult has the summary.
    """
    from google.antigravity.connections.local import types as local_types  # pylint: disable=g-import-not-at-top

    await self._run_post_hook_test(
        confirm_kwargs=dict(
            edit_file=localharness_pb2.ActionEditFile(
                file_path="/tmp/file.py",
            ),
        ),
        done_kwargs=dict(
            text="Applied 2 edits to /tmp/file.py",
            edit_file=localharness_pb2.ActionEditFile(
                file_path="/tmp/file.py",
                diff_block=[
                    localharness_pb2.ActionEditFile.DiffBlock(
                        start_line=0,
                        end_line=1,
                        lines=[
                            localharness_pb2.ActionEditFile.DiffLine(
                                text="+ new line",
                                action=localharness_pb2.ActionEditFile.DiffLine.LINE_ACTION_INSERT,
                            ),
                        ],
                    ),
                ],
            ),
        ),
        expected_name=types.BuiltinTools.EDIT_FILE.value,
        expected_type=local_types.EditFileResult,
        assertions_fn=lambda r: self.assertIn(
            "Applied 2 edits", r.result.summary
        ),
    )

  async def test_tool_result_for_generate_image(self):
    """Verifies PostToolCallHook receives GenerateImageResult for generate_image.

    What: PostToolCallHook receives a GenerateImageResult with image_name.
    Why: generate_image returns the name of the generated image file. Hooks
         may use this for asset tracking or post-processing.
    How: Simulate approval + STATE_DONE with image_name set; assert the hook
         receives GenerateImageResult with the correct image name.
    """
    from google.antigravity.connections.local import types as local_types  # pylint: disable=g-import-not-at-top

    await self._run_post_hook_test(
        confirm_kwargs=dict(
            generate_image=localharness_pb2.ActionGenerateImage(
                prompt="sunset photo",
            ),
        ),
        done_kwargs=dict(
            generate_image=localharness_pb2.ActionGenerateImage(
                prompt="sunset photo",
                image_name="sunset_photo",
                aspect_ratio="16:9",
            ),
        ),
        expected_name=types.BuiltinTools.GENERATE_IMAGE.value,
        expected_type=local_types.GenerateImageResult,
        assertions_fn=lambda r: (
            self.assertEqual(r.result.image_name, "sunset_photo"),
            self.assertEqual(r.result.aspect_ratio, "16:9"),
        ),
    )

  async def test_tool_result_for_search_web(self):
    """Verifies PostToolCallHook receives SearchWebResult for search_web.

    What: PostToolCallHook receives a SearchWebResult with summary.
    Why: search_web returns a summary of web search results. Hooks
         may use this for logging or auditing search queries.
    How: Simulate approval + STATE_DONE with summary set; assert the hook
         receives SearchWebResult with the correct summary and verify string
         representation.
    """
    from google.antigravity.connections.local import types as local_types  # pylint: disable=g-import-not-at-top

    await self._run_post_hook_test(
        confirm_kwargs=dict(
            search_web=localharness_pb2.ActionSearchWeb(
                query="google news",
            ),
        ),
        done_kwargs=dict(
            search_web=localharness_pb2.ActionSearchWeb(
                query="google news",
                summary="google news search results",
            ),
        ),
        expected_name=types.BuiltinTools.SEARCH_WEB.value,
        expected_type=local_types.SearchWebResult,
        assertions_fn=lambda r: (
            self.assertEqual(r.result.summary, "google news search results"),
            self.assertEqual(str(r.result), "google news search results"),
        ),
    )

  async def test_tool_result_fallback_for_view_file(self):
    """Verifies PostToolCallHook falls back to step text for view_file.

    What: PostToolCallHook receives step text (not a structured result) for
          tools without structured result fields (e.g. view_file).
    Why: view_file has no result-bearing fields in the proto (ActionViewFile
         only has file_path and line range). _extract_tool_result returns None,
         so the dispatch falls back to step_obj.content (the step text).
    How: Simulate approval + STATE_DONE with text but no structured result
         field; assert the hook receives the text string as the result.
    """
    await self._run_post_hook_test(
        confirm_kwargs=dict(
            view_file=localharness_pb2.ActionViewFile(
                file_path="/tmp/file.py",
            ),
        ),
        done_kwargs=dict(
            text="File viewing",
            view_file=localharness_pb2.ActionViewFile(
                file_path="/tmp/file.py",
            ),
        ),
        expected_name=types.BuiltinTools.VIEW_FILE.value,
        expected_type=None,  # Falls back to text string, not typed result.
        assertions_fn=lambda r: self.assertIsInstance(r.result, str),
    )

  # ---- Error path ----

  async def test_on_tool_error_dispatched_for_builtin_error(self):
    """Verifies OnToolErrorHook fires when a builtin tool transitions to STATE_ERROR.

    What: OnToolErrorHook receives a RuntimeError with the harness error
    message.
    Why: Users need observability into builtin tool failures for logging and
         recovery. The harness reports errors via STATE_ERROR transitions.
    How: Simulate approval + STATE_ERROR with an error_message; assert the hook
         receives a RuntimeError containing the message text.
    """
    hook_event = asyncio.Event()
    captured_errors = []

    class CapturingErrorHook(hooks_base.OnToolErrorHook):

      async def run(self, context, data):  # pylint: disable=unused-argument
        captured_errors.append(data)
        hook_event.set()
        return None

    hr = hook_runner.HookRunner()
    hr.register_hook(CapturingErrorHook())
    harness = self._make_harness(hr)

    confirm = self._make_confirm_event(
        0,
        "traj_err",
        run_command=localharness_pb2.ActionRunCommand(
            command_line="failing_cmd",
        ),
    )
    error = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=0,
            trajectory_id="traj_err",
            cascade_id="traj_err",
            state=localharness_pb2.StepUpdate.STATE_ERROR,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            error_message="Permission denied",
        )
    )
    await self._confirm_and_complete(harness, confirm, error)
    await harness.wait_for_event(hook_event)

    self.assertEqual(len(captured_errors), 1)
    self.assertIsInstance(captured_errors[0], RuntimeError)
    self.assertIn("Permission denied", str(captured_errors[0]))

  # ---- Guard tests ----

  async def test_denied_builtin_not_tracked(self):
    """Verifies denied builtin tools are not tracked for post-tool dispatch.

    What: PostToolCallHook does NOT fire for denied built-in tool calls.
    Why: If a Decide hook denies a builtin tool, the harness rejects it and
         there is no execution to observe. Tracking it would cause stale
         entries in _pending_builtin_tool_calls or spurious dispatches.
    How: Deny via Decide hook, send a STATE_DONE for the same step, and
         verify PostToolCallHook was not called.
    """
    hook_fired = asyncio.Event()

    class DenyHook(hooks_base.PreToolCallDecideHook):

      async def run(self, context, data):  # pylint: disable=unused-argument
        return hooks_base.HookResult(allow=False, message="Denied")

    class PostHook(hooks_base.PostToolCallHook):

      async def run(self, context, data):  # pylint: disable=unused-argument
        hook_fired.set()

    hr = hook_runner.HookRunner()
    hr.register_hook(DenyHook())
    hr.register_hook(PostHook())
    harness = self._make_harness(hr)

    # Confirmation request — will be denied.
    await harness.send_event(
        self._make_confirm_event(
            0,
            "traj_deny",
            view_file=localharness_pb2.ActionViewFile(file_path="/foo"),
        )
    )
    sent_data = await harness.wait_for_response()
    self.assertFalse(sent_data["toolConfirmation"]["accepted"])

    # Even if the step completes, PostToolCallHook must NOT fire.
    await harness.send_event(self._make_done_event(0, "traj_deny"))
    await asyncio.sleep(0.1)
    self.assertFalse(hook_fired.is_set())

  async def test_no_spurious_hook_for_non_builtin_step(self):
    """Verifies post-tool hooks don't fire for normal model response steps.

    What: PostToolCallHook does NOT fire for STATE_DONE model response steps.
    Why: Only steps that were tracked via ToolConfirmation should trigger
         PostToolCallHook. A model response step that happens to be STATE_DONE
         must not be confused with a completed builtin tool.
    How: Send a model response step (no prior confirmation) and verify
         PostToolCallHook was not called.
    """
    hook_fired = asyncio.Event()

    class PostHook(hooks_base.PostToolCallHook):

      async def run(self, context, data):  # pylint: disable=unused-argument
        hook_fired.set()

    hr = hook_runner.HookRunner()
    hr.register_hook(PostHook())
    harness = self._make_harness(hr)

    # A normal model step (not a builtin tool) that is DONE.
    await harness.send_event(
        localharness_pb2.OutputEvent(
            step_update=localharness_pb2.StepUpdate(
                cascade_id="traj",
                trajectory_id="traj",
                step_index=5,
                text="Final model response",
                state=localharness_pb2.StepUpdate.STATE_DONE,
                source=localharness_pb2.StepUpdate.SOURCE_MODEL,
                target=localharness_pb2.StepUpdate.TARGET_USER,
            )
        )
    )
    await asyncio.sleep(0.1)
    self.assertFalse(hook_fired.is_set())


class LocalConnectionExceptionSafetyTest(unittest.IsolatedAsyncioTestCase):
  """Tests verifying that handler exceptions don't deadlock the harness.

  Each background handler (_handle_question_request,
  _handle_tool_confirmation_request, _handle_tool_call) must catch
  exceptions and send an informative error response rather than dying
  silently and leaving the Go harness blocked.
  """

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock(spec=subprocess.Popen)
    self.tool_runner = tool_runner.ToolRunner()

  def _make_harness(self, hr=None):
    return test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

  async def test_question_handler_crash_sends_error(self):
    """Verifies a crashing interaction hook sends the error message.

    When the on_interaction hook raises, the handler must still respond
    to the harness to prevent deadlock. The error is sent as a single
    freeform_response answer so the model sees what happened.
    """
    hr = hook_runner.HookRunner()

    @hooks_base.on_interaction
    async def crashing_hook(data):
      _ = data
      raise RuntimeError("Intentional interaction hook crash")

    hr.register_hook(crashing_hook)
    harness = self._make_harness(hr)

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            trajectory_id="test_traj",
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            questions_request=localharness_pb2.UserQuestionsRequest(
                questions=[
                    localharness_pb2.UserQuestion(
                        multiple_choice=localharness_pb2.MultipleChoice(
                            question="Do you agree?",
                            choices=["Yes", "No"],
                        )
                    )
                ]
            ),
        )
    )

    await harness.send_event(event)

    sent_data = await harness.wait_for_response()
    self.assertIn("questionResponse", sent_data)
    resp = sent_data["questionResponse"]["response"]
    answers = resp["answers"]
    # Single answer with the error in freeform_response.
    self.assertEqual(len(answers), 1)
    freeform = answers[0]["multipleChoiceAnswer"]["freeformResponse"]
    self.assertIn("SDK error", freeform)
    self.assertIn("Intentional interaction hook crash", freeform)

  async def test_tool_confirmation_crash_sends_rejection(self):
    """Verifies a crashing pre-tool hook sends accepted=False.

    When the pre_tool_call_decide hook raises, the handler must reject
    the tool confirmation to prevent the tool from executing in a broken
    state. The harness transitions the step to STATE_ERROR.
    """
    hr = hook_runner.HookRunner()

    @hooks_base.pre_tool_call_decide
    async def crashing_hook(data):
      _ = data
      raise RuntimeError("Intentional pre-tool hook crash")

    hr.register_hook(crashing_hook)
    harness = self._make_harness(hr)

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            trajectory_id="test_traj",
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            tool_confirmation_request=localharness_pb2.ToolConfirmationRequest(),
            view_file=localharness_pb2.ActionViewFile(file_path="/foo/bar"),
        )
    )

    await harness.send_event(event)

    sent_data = await harness.wait_for_response()
    self.assertIn("toolConfirmation", sent_data)
    self.assertEqual(sent_data["toolConfirmation"]["trajectoryId"], "test_traj")
    self.assertFalse(sent_data["toolConfirmation"]["accepted"])

  async def test_tool_call_crash_sends_error_result(self):
    """Verifies a crashing pre-tool hook sends error in ToolResponse.

    When the pre_tool_call_decide hook raises during a host tool call,
    the handler must send a ToolResponse with the error so the model
    sees what went wrong and can adapt.
    """
    hr = hook_runner.HookRunner()

    @hooks_base.pre_tool_call_decide
    async def crashing_hook(data):
      _ = data
      raise RuntimeError("Intentional tool execution hook crash")

    hr.register_hook(crashing_hook)
    harness = self._make_harness(hr)

    event = localharness_pb2.OutputEvent(
        tool_call=localharness_pb2.ToolCall(
            id="call_123",
            name="some_tool",
            arguments_json="{}",
        )
    )

    await harness.send_event(event)

    sent_data = await harness.wait_for_response()
    self.assertIn("toolResponse", sent_data)
    resp = sent_data["toolResponse"]
    self.assertEqual(resp["id"], "call_123")
    self.assertIn("Intentional tool execution hook crash", resp["responseJson"])


class LocalConnectionSerializationTest(unittest.IsolatedAsyncioTestCase):
  """Tests verifying Pydantic-based normalization in _tool_result_to_dict.

  The SDK uses pydantic.TypeAdapter(Any) to normalize tool outputs
  into JSON-safe primitives before json.dumps(). This prevents
  serialization errors (the root cause of the deadlock bug) when tools
  return complex Python types like sets, datetimes, or bytes.
  """

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock(spec=subprocess.Popen)
    self.tool_runner = tool_runner.ToolRunner()

  def _make_harness(self):
    return test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
        tool_runner=self.tool_runner,
    )

  async def test_normalizes_set_to_list(self):
    """Verifies _tool_result_to_dict normalizes sets into JSON lists.

    This is the exact type that triggered the original deadlock: a tool
    returning a set caused json.dumps to raise TypeError, killing the
    background task and leaving the harness waiting forever.
    """
    conn = self._make_harness().conn
    tr = types.ToolResult(id="1", name="t", result={"tags": {"python", "sdk"}})
    res_dict = conn._tool_result_to_dict(tr)
    self.assertIsInstance(res_dict["tags"], list)
    self.assertCountEqual(res_dict["tags"], ["python", "sdk"])

  async def test_normalizes_datetime_to_iso_string(self):
    """Verifies _tool_result_to_dict normalizes datetimes into ISO strings."""
    conn = self._make_harness().conn
    dt = datetime.datetime(2026, 5, 15, 2, 30, 0)
    tr = types.ToolResult(id="1", name="t", result={"time": dt})
    self.assertEqual(
        conn._tool_result_to_dict(tr)["time"], "2026-05-15T02:30:00"
    )

  async def test_normalizes_bytes_to_string(self):
    """Verifies _tool_result_to_dict normalizes bytes into UTF-8 strings."""
    conn = self._make_harness().conn
    tr = types.ToolResult(id="1", name="t", result={"data": b"hello"})
    self.assertEqual(conn._tool_result_to_dict(tr)["data"], "hello")

  async def test_preserves_pydantic_custom_serializer(self):
    """Verifies _tool_result_to_dict respects custom @field_serializer.

    When a tool returns a Pydantic model with a custom serializer (e.g.
    to mask secrets), model_dump(mode="json") must be used instead of
    model_dump() to ensure the serializer runs.
    """
    conn = self._make_harness().conn

    class CustomModel(pydantic.BaseModel):
      secret: str

      @pydantic.field_serializer("secret")
      def mask_secret(
          self, secret: str, info: pydantic.FieldSerializationInfo
      ) -> str:
        del secret, info
        return "xxxx"

    tr = types.ToolResult(
        id="call_1",
        name="test_tool",
        result=CustomModel(secret="my_super_secret_key"),
    )

    res_dict = conn._tool_result_to_dict(tr)
    self.assertEqual(res_dict["secret"], "xxxx")


class LocalConnectionSubagentsTest(unittest.IsolatedAsyncioTestCase):
  """Tests verifying that static subagent configs are built into HarnessConfig."""

  def setUp(self):
    super().setUp()
    self.temp_dir = self.enterContext(tempfile.TemporaryDirectory())
    self.workspace = pathlib.Path(self.temp_dir) / "workspace"
    self.workspace.mkdir()

  def test_builds_subagents_proto_correctly(self):
    def my_custom_tool():
      """A test tool."""
      pass

    def another_one():
      """Another test tool."""
      pass

    subagent = types.SubagentConfig(
        name="test_helper",
        description="A helpful subagent for testing",
        system_instructions="Always say hello.",
        capabilities=types.SubagentCapabilities(
            enabled_tools=[
                types.BuiltinTools.EDIT_FILE,
            ],
        ),
        # Test mixing callable tools and string tools
        tools=[my_custom_tool, "another_one"],
    )

    tr = tool_runner.ToolRunner(tools=[my_custom_tool, another_one])

    strategy = local_connection.LocalConnectionStrategy(
        subagents=[subagent],
        workspaces=[str(self.workspace)],
        tool_runner=tr,
    )

    harness_config = strategy._build_harness_config()

    self.assertEqual(len(harness_config.custom_subagents), 1)
    custom_agent = harness_config.custom_subagents[0]
    self.assertEqual(custom_agent.name, "test_helper")
    self.assertEqual(custom_agent.description, "A helpful subagent for testing")
    self.assertTrue(custom_agent.harness_side_tools.file_edit.enabled)
    self.assertFalse(custom_agent.harness_side_tools.view_file.enabled)
    self.assertFalse(custom_agent.harness_side_tools.subagents.enabled)
    self.assertEqual(
        [t.name for t in custom_agent.tools],
        ["my_custom_tool", "another_one"],
    )
    sections = custom_agent.system_instructions.appended.appended_sections
    self.assertEqual(sections[0].title, "System")
    self.assertEqual(sections[0].content, "Always say hello.")

  def test_builds_subagents_proto_with_sections(self):
    subagent = types.SubagentConfig(
        name="test_helper",
        description="A helpful subagent for testing",
        system_instructions=[
            types.SystemInstructionSection(
                title="Identity", content="You are a helper agent."
            ),
            types.SystemInstructionSection(
                title="Guidelines", content="Keep responses short."
            ),
        ],
    )

    strategy = local_connection.LocalConnectionStrategy(
        subagents=[subagent],
        workspaces=[str(self.workspace)],
    )

    harness_config = strategy._build_harness_config()

    self.assertEqual(len(harness_config.custom_subagents), 1)
    custom_agent = harness_config.custom_subagents[0]
    self.assertEqual(custom_agent.name, "test_helper")
    sections = custom_agent.system_instructions.appended.appended_sections
    self.assertEqual(len(sections), 2)
    self.assertEqual(sections[0].title, "Identity")
    self.assertEqual(sections[0].content, "You are a helper agent.")
    self.assertEqual(sections[1].title, "Guidelines")
    self.assertEqual(sections[1].content, "Keep responses short.")

  def test_subagent_tool_not_registered_raises(self):
    def unregistered_tool():
      """Not added to parent."""
      pass

    subagent = types.SubagentConfig(
        name="test_helper",
        description="A helpful subagent",
        tools=[unregistered_tool],
    )

    strategy = local_connection.LocalConnectionStrategy(
        subagents=[subagent],
        workspaces=[str(self.workspace)],
    )

    with self.assertRaisesRegex(
        ValueError,
        "Subagent tool 'unregistered_tool' is not registered on the main agent"
        " config",
    ):
      strategy._build_harness_config()

  def test_subagent_harness_tools_as_strings_raise_if_not_registered(self):
    subagent = types.SubagentConfig(
        name="test_helper",
        description="A helpful subagent",
        tools=["view_file", "code_search"],
    )

    strategy = local_connection.LocalConnectionStrategy(
        subagents=[subagent],
        workspaces=[str(self.workspace)],
    )

    with self.assertRaisesRegex(
        ValueError,
        "Subagent tool 'view_file' is not registered on the main agent config",
    ):
      strategy._build_harness_config()

  def test_subagent_tools_stripped_and_warned(self):
    subagent = types.SubagentConfig(
        name="nested_helper",
        description="A subagent trying to use subagents",
        system_instructions="Spawn subagents.",
        capabilities=types.SubagentCapabilities(
            enabled_tools=[types.BuiltinTools.START_SUBAGENT],
        ),
    )

    strategy = local_connection.LocalConnectionStrategy(
        subagents=[subagent],
        workspaces=[str(self.workspace)],
    )

    with self.assertLogs(level="WARNING") as log_capture:
      harness_config = strategy._build_harness_config()

    # Verify warning was logged
    self.assertTrue(
        any(
            "Nested subagents are currently not supported" in msg
            for msg in log_capture.output
        )
    )

    self.assertEqual(len(harness_config.custom_subagents), 1)
    custom_agent = harness_config.custom_subagents[0]
    self.assertFalse(custom_agent.harness_side_tools.subagents.enabled)
    self.assertFalse(custom_agent.harness_side_tools.file_edit.enabled)


if __name__ == "__main__":
  absltest.main()
