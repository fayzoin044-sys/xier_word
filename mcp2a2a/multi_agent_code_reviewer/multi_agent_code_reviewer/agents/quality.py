"""Quality Agent backed exclusively by the check_quality MCP tool."""

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

QUALITY_MCP_MODULE = "multi_agent_code_reviewer.mcp_servers.quality"
DEMO_PROJECT = QUALITY_TEST_DEMO

QUALITY_SYSTEM_PROMPT = """你是 Quality Agent，专门负责 Python 代码质量审查。

工作规则：
1. 当用户提供项目路径并要求代码质量审查时，必须调用 check_quality 工具，并把该项目路径原样传给 project_path 参数。
2. 只能依据 check_quality 返回的 findings 进行分析，不得自行推测、补充或声称存在工具未发现的质量问题。
3. 对每一条 finding 分别说明规则编号、所在文件与行号、问题原因、影响和修复建议。
4. 每个结论都必须明确对应 findings 中的一条记录，不得把 Ruff 返回的事实改写成其他问题。
5. 说明影响时必须遵循 Ruff finding 的实际含义。除非工具明确报告安全风险，否则不得把代码质量问题扩展为安全风险；F401 unused import 只说明代码整洁性、可读性和可维护性问题。
6. 如果 findings 为空，只能说明本次 Ruff 检查未返回问题，不能宣称代码质量绝对完美。
7. 不执行代码修改，不调用 check_quality 之外的工具。
"""


def create_quality_mcp_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "quality": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", QUALITY_MCP_MODULE],
                "cwd": str(PROJECT_ROOT),
            }
        }
    )


def get_structured_tool_result(tool_message: ToolMessage) -> dict[str, Any]:
    artifact = tool_message.artifact
    if not isinstance(artifact, dict):
        raise TypeError("check_quality ToolMessage did not contain an MCP artifact")
    structured = artifact.get("structured_content")
    if not isinstance(structured, dict):
        raise TypeError("check_quality MCP artifact had no structured_content")
    return structured


async def run_quality_review(project_path: Path) -> dict[str, Any]:
    mcp_client = create_quality_mcp_client()
    tools = await mcp_client.get_tools()
    quality_tools = [tool for tool in tools if tool.name == "check_quality"]
    if len(quality_tools) != 1:
        raise RuntimeError(
            f"Expected exactly one check_quality MCP tool, found {len(quality_tools)}"
        )

    model_name = os.getenv("QUALITY_AGENT_MODEL", "qwen2.5:latest")
    model = ChatOllama(
        model=model_name,
        temperature=0,
        validate_model_on_init=True,
    )
    agent = create_agent(
        model=model,
        tools=quality_tools,
        system_prompt=QUALITY_SYSTEM_PROMPT,
        name="quality_agent",
    )
    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"请对这个项目进行 Python 代码质量审查：{project_path}",
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
        if tool_call.get("name") == "check_quality"
    ]
    tool_messages = [
        message
        for message in messages
        if isinstance(message, ToolMessage) and message.name == "check_quality"
    ]
    if not tool_calls or not tool_messages:
        raise RuntimeError("Quality Agent did not call check_quality")

    structured_result = get_structured_tool_result(tool_messages[-1])
    final_message = messages[-1]
    if not isinstance(final_message, AIMessage):
        raise TypeError("Quality Agent did not produce a final AI response")

    return {
        "model": model_name,
        "loaded_tools": [tool.name for tool in tools],
        "check_quality_calls": tool_calls,
        "tool_result": structured_result,
        "final_answer": final_message.content,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Quality Agent.")
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

    review = asyncio.run(run_quality_review(project))
    print("=== Loaded LangChain tools ===")
    print(review["loaded_tools"])
    print("\n=== Agent check_quality tool calls ===")
    print(review["check_quality_calls"])
    print("\n=== MCP structured tool result ===")
    print(review["tool_result"])
    print("\n=== Quality Agent final answer ===")
    print(review["final_answer"])
