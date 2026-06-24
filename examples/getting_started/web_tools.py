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

"""Example demonstrating web tools using Google Antigravity SDK.

To run:
  python web_tools.py
"""

import asyncio

from google.antigravity import Agent
from google.antigravity import CapabilitiesConfig
from google.antigravity import LocalAgentConfig
from google.antigravity import types


async def run_web_search_example() -> None:
  """Runs the web search tool example."""
  print("=== 1. Web Search Example ===\n")
  config = LocalAgentConfig(
      capabilities=CapabilitiesConfig(
          enabled_tools=[
              types.BuiltinTools.SEARCH_WEB,
          ]
      ),
  )

  async with Agent(config) as my_agent:
    prompt = (
        "What is the current weather and temperature in New York City right"
        " now? Please provide the source."
    )
    print(f"User: {prompt}\n")

    print("Agent is thinking and searching...")
    response = await my_agent.chat(prompt)

    # Await the full aggregated text response.
    response_text = await response.text()
    print(f"\nAgent Response:\n{response_text}")


async def run_url_fetching_example() -> None:
  """Runs the URL fetching tool example."""
  print("\n=== 2. URL Fetching (read_url_content) Example ===\n")
  config = LocalAgentConfig(
      capabilities=CapabilitiesConfig(
          enabled_tools=[
              types.BuiltinTools.READ_URL_CONTENT,
              # Note: For massive web pages or lengthy documentation,
              # read_url_content caches the downloaded payload to disk.
              # Enabling VIEW_FILE allows the agent to inspect those files.
              types.BuiltinTools.VIEW_FILE,
          ]
      ),
  )

  async with Agent(config) as my_agent:
    target_url = "https://en.wikipedia.org/wiki/Google"
    prompt = (
        f"Please read the full page content from {target_url} and tell me the"
        " exact date that Google acquired DeepMind Technologies."
    )
    print(f"User: {prompt}\n")

    print("Agent is fetching and reading URL content...")
    response = await my_agent.chat(prompt)

    response_text = await response.text()
    print(f"\nAgent Response:\n{response_text}")


async def main() -> None:
  """Runs the web tools examples.

  Demonstrates enabling and running SEARCH_WEB and READ_URL_CONTENT tools.
  """
  await run_web_search_example()
  await run_url_fetching_example()


if __name__ == "__main__":
  asyncio.run(main())
