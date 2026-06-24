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

"""Unit tests for HookRouter."""

import asyncio
from typing import Any
from absl.testing import absltest
from google.antigravity import types
from google.antigravity.connections.local import localharness_pb2
from google.antigravity.connections.local import types as local_types
from google.antigravity.connections.local.hook_router import HookRouter
from google.antigravity.connections.local.local_connection import _extract_tool_result
from google.antigravity.hooks import hook_runner as h_runner
from google.antigravity.hooks import hooks


class HookRouterTest(absltest.TestCase):

  def test_handle_on_session_start(self):

    async def _test():
      fired = asyncio.Event()

      @hooks.on_session_start
      async def my_hook():
        fired.set()

      hook_runner = h_runner.HookRunner(
          on_session_start_hooks=[my_hook],
      )

      sent_events = []

      async def mock_send(event: localharness_pb2.InputEvent):
        sent_events.append(event)

      router = HookRouter(hook_runner, mock_send)

      req = localharness_pb2.CallHookRequest(
          request_id="test_req_1",
          name="OnSessionStart",
          type=localharness_pb2.LIFECYCLE_HOOK_ON_SESSION_START,
      )

      await router.handle(req)

      self.assertTrue(fired.is_set())
      self.assertLen(sent_events, 1)
      self.assertTrue(sent_events[0].HasField("call_hook_response"))
      resp = sent_events[0].call_hook_response
      self.assertEqual(resp.request_id, "test_req_1")
      self.assertTrue(resp.HasField("empty_result"))

    asyncio.run(_test())

  def test_handle_on_session_end(self):

    async def _test():
      fired = asyncio.Event()

      @hooks.on_session_end
      async def my_hook():
        fired.set()

      hook_runner = h_runner.HookRunner(
          on_session_end_hooks=[my_hook],
      )

      sent_events = []

      async def mock_send(event: localharness_pb2.InputEvent):
        sent_events.append(event)

      router = HookRouter(hook_runner, mock_send)

      req = localharness_pb2.CallHookRequest(
          request_id="test_req_end",
          name="OnSessionEnd",
          type=localharness_pb2.LIFECYCLE_HOOK_ON_SESSION_END,
      )

      await router.handle(req)

      self.assertTrue(fired.is_set())
      self.assertLen(sent_events, 1)
      self.assertTrue(sent_events[0].HasField("call_hook_response"))
      resp = sent_events[0].call_hook_response
      self.assertEqual(resp.request_id, "test_req_end")
      self.assertTrue(resp.HasField("empty_result"))

    asyncio.run(_test())

  def test_handle_unknown_hook(self):

    async def _test():
      hook_runner = h_runner.HookRunner()
      sent_events = []

      async def mock_send(event: localharness_pb2.InputEvent):
        sent_events.append(event)

      router = HookRouter(hook_runner, mock_send)

      req = localharness_pb2.CallHookRequest(
          request_id="test_req_unknown",
          name="UnknownHook",
          type=localharness_pb2.LIFECYCLE_HOOK_UNSPECIFIED,
      )

      await router.handle(req)

      self.assertLen(sent_events, 1)
      self.assertTrue(sent_events[0].HasField("call_hook_response"))
      resp = sent_events[0].call_hook_response
      self.assertEqual(resp.request_id, "test_req_unknown")
      self.assertTrue(resp.HasField("empty_result"))

    asyncio.run(_test())

  def test_handle_pre_turn_allow(self):

    async def _test():
      fired = asyncio.Event()

      @hooks.pre_turn
      async def my_hook(prompt: Any):
        fired.set()
        return hooks.HookResult(allow=True)

      hook_runner = h_runner.HookRunner(pre_turn_hooks=[my_hook])
      sent_events = []

      async def mock_send(event: localharness_pb2.InputEvent):
        sent_events.append(event)

      router = HookRouter(hook_runner, mock_send)
      req = localharness_pb2.CallHookRequest(
          request_id="test_pre_turn",
          name="PreTurn",
          type=localharness_pb2.LIFECYCLE_HOOK_PRE_TURN,
      )

      await router.handle(req)

      self.assertTrue(fired.is_set())
      self.assertLen(sent_events, 1)
      resp = sent_events[0].call_hook_response
      self.assertEqual(resp.request_id, "test_pre_turn")
      self.assertTrue(resp.HasField("pre_turn_result"))
      self.assertEqual(
          resp.pre_turn_result.decision,
          localharness_pb2.PreTurnResult.Decision.ALLOW,
      )

    asyncio.run(_test())

  def test_handle_pre_turn_multipart_content(self):

    async def _test():
      captured_input = []

      @hooks.pre_turn
      async def my_hook(prompt: Any):
        captured_input.append(prompt)
        return hooks.HookResult(allow=True)

      hook_runner = h_runner.HookRunner(pre_turn_hooks=[my_hook])
      router = HookRouter(hook_runner, lambda _: asyncio.sleep(0))
      req = localharness_pb2.CallHookRequest(
          request_id="test_multi",
          type=localharness_pb2.LIFECYCLE_HOOK_PRE_TURN,
          pre_turn_args=localharness_pb2.PreTurnArgs(
              user_input=localharness_pb2.UserInput(
                  parts=[
                      localharness_pb2.UserInput.Part(text="hello"),
                      localharness_pb2.UserInput.Part(text="world"),
                  ]
              )
          ),
      )

      await router.handle(req)

      self.assertLen(captured_input, 1)
      self.assertEqual(captured_input[0], ["hello", "world"])

    asyncio.run(_test())

  def test_handle_post_turn(self):

    async def _test():
      fired = asyncio.Event()
      received_data = []

      @hooks.post_turn
      async def my_hook(data: str):
        fired.set()
        received_data.append(data)

      hook_runner = h_runner.HookRunner(post_turn_hooks=[my_hook])
      sent_events = []

      async def mock_send(event: localharness_pb2.InputEvent):
        sent_events.append(event)

      router = HookRouter(hook_runner, mock_send)
      req = localharness_pb2.CallHookRequest(
          request_id="test_post_turn",
          name="PostTurn",
          type=localharness_pb2.LIFECYCLE_HOOK_POST_TURN,
          post_turn_args=localharness_pb2.PostTurnArgs(response_text="final"),
      )

      await router.handle(req)

      self.assertTrue(fired.is_set())
      self.assertEqual(received_data, ["final"])
      self.assertLen(sent_events, 1)
      resp = sent_events[0].call_hook_response
      self.assertEqual(resp.request_id, "test_post_turn")
      self.assertTrue(resp.HasField("empty_result"))

    asyncio.run(_test())

  def test_handle_post_tool(self):

    async def _test():
      fired = asyncio.Event()
      received_data: list[Any] = []

      @hooks.post_tool_call
      async def my_hook(data: Any):
        fired.set()
        received_data.append(data)

      hook_runner = h_runner.HookRunner(post_tool_call_hooks=[my_hook])
      sent_events = []

      async def mock_send(event: localharness_pb2.InputEvent):
        sent_events.append(event)

      router = HookRouter(hook_runner, mock_send)
      req = localharness_pb2.CallHookRequest(
          request_id="test_post_tool",
          name="PostTool",
          type=localharness_pb2.LIFECYCLE_HOOK_POST_TOOL,
          post_tool_args=localharness_pb2.PostToolArgs(
              tool_name="view_file",
              result="file content here",
          ),
      )

      await router.handle(req)

      self.assertTrue(fired.is_set())
      self.assertLen(received_data, 1)
      tool_result = received_data[0]
      self.assertEqual(tool_result.name, "view_file")
      self.assertEqual(tool_result.result, "file content here")
      self.assertIsNone(tool_result.error)
      self.assertLen(sent_events, 1)
      resp = sent_events[0].call_hook_response
      self.assertEqual(resp.request_id, "test_post_tool")
      self.assertTrue(resp.HasField("empty_result"))

    asyncio.run(_test())

  def test_handle_post_tool_with_error(self):

    async def _test():
      fired = asyncio.Event()
      received_data: list[Any] = []

      @hooks.post_tool_call
      async def my_hook(data: Any):
        fired.set()
        received_data.append(data)

      hook_runner = h_runner.HookRunner(post_tool_call_hooks=[my_hook])
      sent_events = []

      async def mock_send(event: localharness_pb2.InputEvent):
        sent_events.append(event)

      router = HookRouter(hook_runner, mock_send)
      req = localharness_pb2.CallHookRequest(
          request_id="test_post_tool_err",
          name="PostTool",
          type=localharness_pb2.LIFECYCLE_HOOK_POST_TOOL,
          post_tool_args=localharness_pb2.PostToolArgs(
              tool_name="run_command",
              error="command not found",
          ),
      )

      await router.handle(req)

      self.assertTrue(fired.is_set())
      self.assertLen(received_data, 1)
      tool_result = received_data[0]
      self.assertEqual(tool_result.name, "run_command")
      self.assertIsNone(tool_result.result)
      self.assertEqual(tool_result.error, "command not found")

    asyncio.run(_test())


class HookRouterStructuredExtractionTest(absltest.TestCase):
  """Verifies structured Pydantic extraction across built-in tool types."""

  def _run_extraction_test(
      self,
      tool_name: str,
      action_kwargs: dict[str, Any],
      assertion_fn: Any,
      expected_name: str | None = None,
  ):
    async def _test():
      fired = asyncio.Event()
      received_data: list[Any] = []

      @hooks.post_tool_call
      async def my_hook(data: Any):
        fired.set()
        received_data.append(data)

      hook_runner = h_runner.HookRunner(post_tool_call_hooks=[my_hook])
      router = HookRouter(
          hook_runner,
          lambda event: asyncio.sleep(0),
          result_extractor=_extract_tool_result,
      )
      req = localharness_pb2.CallHookRequest(
          request_id="test_struct",
          name="PostTool",
          type=localharness_pb2.LIFECYCLE_HOOK_POST_TOOL,
          post_tool_args=localharness_pb2.PostToolArgs(
              tool_name=tool_name,
              step_update=localharness_pb2.StepUpdate(**action_kwargs),
          ),
      )

      await router.handle(req)
      self.assertTrue(fired.is_set())
      self.assertLen(received_data, 1)
      self.assertEqual(received_data[0].name, expected_name or tool_name)
      assertion_fn(received_data[0])

    asyncio.run(_test())

  def test_extract_invoke_subagent(self):
    self._run_extraction_test(
        tool_name="invoke_subagent",
        action_kwargs=dict(
            invoke_subagent=localharness_pb2.ActionInvokeSubagent()
        ),
        assertion_fn=lambda res: self.assertEqual(res.result, ""),
        expected_name=types.BuiltinTools.START_SUBAGENT.value,
    )

  def test_extract_run_command(self):
    self._run_extraction_test(
        tool_name="run_command",
        action_kwargs=dict(
            run_command=localharness_pb2.ActionRunCommand(
                command_line="echo hi",
                combined_output="hi\n",
            )
        ),
        assertion_fn=lambda res: (
            self.assertIsInstance(res.result, local_types.RunCommandResult),
            self.assertEqual(res.result.output, "hi\n"),
        ),
    )

  def test_extract_list_directory(self):
    self._run_extraction_test(
        tool_name="list_directory",
        action_kwargs=dict(
            list_directory=localharness_pb2.ActionListDirectory(
                directory_path="/tmp",
                results=[
                    localharness_pb2.ActionListDirectory.Result(
                        name="foo.py", file_size=100
                    ),
                ],
            )
        ),
        assertion_fn=lambda res: (
            self.assertIsInstance(res.result, local_types.ListDirectoryResult),
            self.assertEqual(len(res.result.entries), 1),
            self.assertEqual(res.result.entries[0].name, "foo.py"),
        ),
    )

  def test_extract_find_file(self):
    self._run_extraction_test(
        tool_name="find_file",
        action_kwargs=dict(
            find_file=localharness_pb2.ActionFindFile(
                directory_path="/tmp",
                output="/tmp/a.txt",
            )
        ),
        assertion_fn=lambda res: (
            self.assertIsInstance(res.result, local_types.FindFileResult),
            self.assertEqual(res.result.output, "/tmp/a.txt"),
        ),
    )

  def test_extract_search_directory(self):
    self._run_extraction_test(
        tool_name="search_directory",
        action_kwargs=dict(
            search_directory=localharness_pb2.ActionSearchDirectory(
                directory_path="/tmp",
                num_results=5,
            )
        ),
        assertion_fn=lambda res: (
            self.assertIsInstance(
                res.result, local_types.SearchDirectoryResult
            ),
            self.assertEqual(res.result.num_results, 5),
        ),
    )

  def test_extract_edit_file(self):
    self._run_extraction_test(
        tool_name="edit_file",
        action_kwargs=dict(
            text="Edited file",
            edit_file=localharness_pb2.ActionEditFile(
                file_path="/tmp/a.py",
                diff_block=[
                    localharness_pb2.ActionEditFile.DiffBlock(
                        start_line=1, end_line=2
                    )
                ],
            ),
        ),
        assertion_fn=lambda res: (
            self.assertIsInstance(res.result, local_types.EditFileResult),
            self.assertIn("Edited file", res.result.summary),
        ),
    )

  def test_extract_generate_image(self):
    self._run_extraction_test(
        tool_name="generate_image",
        action_kwargs=dict(
            generate_image=localharness_pb2.ActionGenerateImage(
                prompt="cat",
                image_name="cat_img",
            )
        ),
        assertion_fn=lambda res: (
            self.assertIsInstance(res.result, local_types.GenerateImageResult),
            self.assertEqual(res.result.image_name, "cat_img"),
        ),
    )

  def test_extract_search_web(self):
    self._run_extraction_test(
        tool_name="search_web",
        action_kwargs=dict(
            search_web=localharness_pb2.ActionSearchWeb(
                query="news",
                summary="news results",
            )
        ),
        assertion_fn=lambda res: (
            self.assertIsInstance(res.result, local_types.SearchWebResult),
            self.assertEqual(res.result.summary, "news results"),
        ),
    )

  def test_extract_read_url_content(self):
    self._run_extraction_test(
        tool_name="read_url_content",
        action_kwargs=dict(
            read_url_content=localharness_pb2.ActionReadUrlContent(
                url="https://example.com",
                title="Example Domain",
                summary="example domain summary",
                content_path="/tmp/content.md",
            )
        ),
        assertion_fn=lambda res: (
            self.assertIsInstance(res.result, local_types.ReadUrlContentResult),
            self.assertEqual(res.result.title, "Example Domain"),
            self.assertEqual(res.result.summary, "example domain summary"),
            self.assertEqual(res.result.content_path, "/tmp/content.md"),
        ),
    )

  def test_fallback_view_file(self):
    self._run_extraction_test(
        tool_name="view_file",
        action_kwargs=dict(
            text="Viewed file",
            view_file=localharness_pb2.ActionViewFile(file_path="/tmp/a.py"),
        ),
        assertion_fn=lambda res: self.assertIsInstance(res.result, str),
    )


if __name__ == "__main__":
  absltest.main()
