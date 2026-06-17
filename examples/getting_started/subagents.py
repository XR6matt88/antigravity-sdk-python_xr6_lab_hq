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

"""Example demonstrating subagents in Google Antigravity SDK.

This example shows two subagent scenarios:
  1. Dynamic Subagent: The agent dynamically spawns a clone of itself ("self")
     to delegate a heavy research task (listing/reading files), keeping its
     own context window clean.
  2. Custom Static Subagent: The agent is pre-configured with a custom, static
     subagent definition ('code_reviewer') that has its own system instructions
     and custom tools (e.g. get_reviewer_badge).

To run:
  python subagents.py

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. In Scenario 1, the agent dynamically spawns a subagent to research the
     examples directory.
  3. In Scenario 1, the agent produces a non-empty lesson plan.
  4. In Scenario 2, the 'code_reviewer' subagent audits target_code.py,
     producing warnings prefixed with '[AUDIT_WARNING]'.
  5. In Scenario 2, the subagent uses the 'get_reviewer_badge' tool to sign the
     report with 'Senior-L3-Auditor-Badge'.
  6. Subagent hook logs fire for both scenarios, showing start/done events.
"""

import asyncio
import logging
import pathlib
import sys
import tempfile
from typing import Any

from google.antigravity import Agent
from google.antigravity import LocalAgentConfig
from google.antigravity import types
from google.antigravity.hooks import hooks

_subagent_active = False


@hooks.pre_tool_call_decide
async def log_pre_tool(data: types.ToolCall) -> types.HookResult:
  """Logs all tool calls for visibility."""
  global _subagent_active

  if data.name == types.BuiltinTools.START_SUBAGENT.value:
    _subagent_active = True
    print("\n  --- 🤖 [Hook] Spawning Subagent ---")
    print(f"  Arguments: {data.args}\n")
  else:
    indent = "    " if _subagent_active else "  "
    print(f"{indent}- [Start]: {data.name} (ID: {data.id})", flush=True)
  return types.HookResult(allow=True)


@hooks.post_tool_call
async def log_post_tool(data: Any) -> None:
  """Logs tool results."""
  global _subagent_active

  if data.name == types.BuiltinTools.START_SUBAGENT.value:
    _subagent_active = False
    print("\n  --- 🤖 [Hook] Subagent Finished ---")
    print(f"  Result: {data.result}\n")
  else:
    indent = "    " if _subagent_active else "  "
    print(f"{indent}- [Done]: {data.name} (ID: {data.id}) ✅", flush=True)


def get_reviewer_badge() -> str:
  """Returns the reviewer's official certification badge name."""
  return "Senior-L3-Auditor-Badge"


async def run_dynamic_subagent() -> None:
  """Runs a dynamic subagent research scenario."""
  print("\n=== Scenario 1: Dynamic Subagent (Self Clone) ===")
  # Enable subagents in the config and add hooks for visibility.
  config = LocalAgentConfig(
      capabilities=types.CapabilitiesConfig(
          enable_subagents=True,
      ),
      hooks=[log_pre_tool, log_post_tool],
  )

  async with Agent(config) as my_agent:
    prompt = (
        "Use a subagent to research the Google Antigravity SDK examples in"
        " the parent directory. Delegate the task of listing and reading the"
        " files to the subagent, and then generate a lesson plan for me to"
        " learn more based on its findings."
    )
    print(f"  User: {prompt}")
    response = await my_agent.chat(prompt)
    response_text = await response.text()
    print(f"\n  Agent:\n{response_text}")


async def run_custom_static_subagent() -> None:
  """Runs a custom static subagent code-review audit scenario."""
  print("\n=== Scenario 2: Custom Static Subagent ===")
  with tempfile.TemporaryDirectory() as tmpdir:
    workspace_path = pathlib.Path(tmpdir) / "workspace"
    workspace_path.mkdir(parents=True, exist_ok=True)

    # Write a target file for the reviewer subagent to check.
    target_file = workspace_path / "target_code.py"
    target_file.write_text(
        "def hello():\n"
        "  print('hello')\n"
        "\n"
        "def add(a, b):\n"
        "  \"\"\"Adds two numbers.\"\"\"\n"
        "  return a + b\n",
        encoding="utf-8",
    )

    reviewer_subagent = types.SubagentConfig(
        name="code_reviewer",
        description="Audits source code files and reports missing docstrings.",
        system_instructions=(
            "You are a code reviewer. Read python files in the workspace and "
            "check if all function declarations have docstrings. For each "
            "function that is missing a docstring, output a warning prefixed "
            "with '[AUDIT_WARNING]'. "
            "CRITICAL: Every warning you output MUST start with "
            "'[AUDIT_WARNING]'. Use the 'get_reviewer_badge' tool to sign "
            "your final audit report with your official badge name. "
            "Output your report directly in your final response. Do not use "
            "the send_message tool to deliver it."
        ),
        tools=[get_reviewer_badge],
    )

    config = LocalAgentConfig(
        subagents=[reviewer_subagent],
        workspaces=[str(workspace_path)],
        tools=[get_reviewer_badge],
        hooks=[log_pre_tool, log_post_tool],
    )

    async with Agent(config) as my_agent:
      prompt = (
          f"Ask the 'code_reviewer' subagent to review {target_file.name} "
          "and sign the report with their reviewer badge name. "
          "Show me the exact warnings it produced verbatim, including any "
          "prefixes and the badge signature."
      )
      print(f"  User: {prompt}")

      response = await my_agent.chat(prompt)
      response_text = await response.text()
      print(f"\n  Agent:\n{response_text}")


async def main() -> None:
  # Configure logging
  root = logging.getLogger()
  root.setLevel(logging.INFO)
  for h in root.handlers[:]:
    root.removeHandler(h)
  ch = logging.StreamHandler(sys.stderr)
  ch.setLevel(logging.INFO)
  ch.setFormatter(
      logging.Formatter("%(levelname)s:%(name)s:%(message)s")
  )
  root.addHandler(ch)

  await run_dynamic_subagent()
  await run_custom_static_subagent()


if __name__ == "__main__":
  asyncio.run(main())
