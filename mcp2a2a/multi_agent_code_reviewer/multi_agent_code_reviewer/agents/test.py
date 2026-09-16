"""Test Agent backed exclusively by the run_tests MCP tool."""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.messages import AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama

from multi_agent_code_reviewer.paths import PROJECT_ROOT, QUALITY_TEST_DEMO

TEST_MCP_MODULE = "multi_agent_code_reviewer.mcp_servers.test"
DEMO_PROJECT = QUALITY_TEST_DEMO

TEST_SYSTEM_PROMPT = """你是 Test Agent，专门负责执行并解释 Python 项目的测试结果。

工作规则：
1. 当用户提供项目路径并要求执行测试时，必须调用 run_tests 工具，并把该项目路径原样传给 project_path 参数。
2. 只能依据 run_tests 返回的 status、summary、tests、stdout 和 stderr 进行分析，不得虚构未执行的测试或不存在的失败原因。
3. 必须先报告测试总数以及通过、失败、错误、跳过数量，再逐项说明失败或错误测试的位置、现象和工具返回的原因。
4. 测试全部通过时，只能说明本次 Pytest 执行通过，不能对未覆盖行为作保证。
5. 如果没有收集到测试或执行异常，必须明确说明对应状态和可见错误信息。
6. 不执行代码修改，不调用 run_tests 之外的工具。
"""


def create_test_mcp_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "test": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", TEST_MCP_MODULE],
                "cwd": str(PROJECT_ROOT),
            }
        }
    )


def get_structured_tool_result(tool_message: ToolMessage) -> dict[str, Any]:
    artifact = tool_message.artifact
    if not isinstance(artifact, dict):
        raise TypeError("run_tests ToolMessage did not contain an MCP artifact")
    structured = artifact.get("structured_content")
    if not isinstance(structured, dict):
        raise TypeError("run_tests MCP artifact had no structured_content")
    return structured


async def run_test_review(project_path: Path) -> dict[str, Any]:
    mcp_client = create_test_mcp_client()
    tools = await mcp_client.get_tools()
    test_tools = [tool for tool in tools if tool.name == "run_tests"]
    if len(test_tools) != 1:
        raise RuntimeError(
            f"Expected exactly one run_tests MCP tool, found {len(test_tools)}"
        )

    model_name = os.getenv("TEST_AGENT_MODEL", "qwen2.5:latest")
    model = ChatOllama(
        model=model_name,
        temperature=0,
        validate_model_on_init=True,
    )
    agent = create_agent(
        model=model,
        tools=test_tools,
        system_prompt=TEST_SYSTEM_PROMPT,
        name="test_agent",
    )
    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"请执行这个 Python 项目的测试并解释结果：{project_path}",
                }
            ]
        }
    )

    messages = result["messages"]
    tool_calls = [
        tool_call
        for message in messages
        if isinstance(message, AIMessage)
        for tool_call in message.tool_calls
        if tool_call.get("name") == "run_tests"
    ]
    tool_messages = [
        message
        for message in messages
        if isinstance(message, ToolMessage) and message.name == "run_tests"
    ]
    if not tool_calls or not tool_messages:
        raise RuntimeError("Test Agent did not call run_tests")

    structured_result = get_structured_tool_result(tool_messages[-1])
    final_message = messages[-1]
    if not isinstance(final_message, AIMessage):
        raise TypeError("Test Agent did not produce a final AI response")

    return {
        "model": model_name,
        "loaded_tools": [tool.name for tool in tools],
        "run_tests_calls": tool_calls,
        "tool_result": structured_result,
        "final_answer": final_message.content,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Test Agent.")
    parser.add_argument(
        "project_path",
        nargs="?",
        type=Path,
        default=DEMO_PROJECT,
        help="Local project directory. Defaults to quality_test_demo.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    project = arguments.project_path.expanduser().resolve()
    if not project.is_dir():
        raise SystemExit(f"Project directory does not exist: {project}")

    review = asyncio.run(run_test_review(project))
    print("=== Loaded LangChain tools ===")
    print(review["loaded_tools"])
    print("\n=== Agent run_tests tool calls ===")
    print(review["run_tests_calls"])
    print("\n=== MCP structured tool result ===")
    print(review["tool_result"])
    print("\n=== Test Agent final answer ===")
    print(review["final_answer"])
