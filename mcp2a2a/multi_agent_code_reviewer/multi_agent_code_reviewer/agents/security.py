"""Security Agent backed exclusively by the existing scan_security MCP tool."""

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

from multi_agent_code_reviewer.paths import PROJECT_ROOT, VULNERABLE_DEMO

SECURITY_MCP_MODULE = "multi_agent_code_reviewer.mcp_servers.security"
DEMO_PROJECT = VULNERABLE_DEMO

SECURITY_SYSTEM_PROMPT = """你是 Security Agent，专门负责代码安全审查。

工作规则：
1. 当用户提供项目路径并要求安全审查时，必须调用 scan_security 工具，并把该项目路径原样传给 project_path 参数。
2. 只能依据 scan_security 返回的 findings 进行分析。不得自行推测、补充或声称存在工具未发现的漏洞。
3. 对每一条 finding 分别说明：
   - 漏洞类型；
   - 风险；
   - 产生原因；
   - 修复建议。
4. 每个安全结论都必须明确对应 findings 中的一条记录，不得把工具返回的事实改写成其他漏洞。
5. 修复建议必须针对当前 finding 所在的实际代码和调用方式，不得使用与当前项目无关的固定命令或特定操作系统示例；只有 finding 或被审查代码明确涉及某个平台时，才可给出平台专属建议。
6. 不得为了演示而编造或替换成 finding 未返回的命令、第三方库、框架或代码示例。如果 finding 没有提供足够的源代码细节，只说明与该 finding 直接对应的修复原则和需要调整的调用参数。
7. 如果 findings 为空，只能说明本次扫描未返回安全发现，不能宣称代码绝对安全。
8. 不执行代码修改，不调用 scan_security 之外的工具。
"""


def create_security_mcp_client() -> MultiServerMCPClient:
    """Configure the existing Security MCP Server as a stdio tool source."""
    return MultiServerMCPClient(
        {
            "security": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", SECURITY_MCP_MODULE],
                "cwd": str(PROJECT_ROOT),
            }
        }
    )


def get_structured_tool_result(tool_message: ToolMessage) -> dict[str, Any]:
    artifact = tool_message.artifact
    if not isinstance(artifact, dict):
        raise TypeError("scan_security ToolMessage did not contain an MCP artifact")

    structured = artifact.get("structured_content")
    if not isinstance(structured, dict):
        raise TypeError("scan_security MCP artifact had no structured_content")

    return structured


async def run_security_review(project_path: Path) -> dict[str, Any]:
    mcp_client = create_security_mcp_client()
    tools = await mcp_client.get_tools()
    scan_tools = [tool for tool in tools if tool.name == "scan_security"]
    if len(scan_tools) != 1:
        raise RuntimeError(
            f"Expected exactly one scan_security MCP tool, found {len(scan_tools)}"
        )

    model_name = os.getenv("SECURITY_AGENT_MODEL", "qwen2.5:latest")
    model = ChatOllama(
        model=model_name,
        temperature=0,
        validate_model_on_init=True,
    )
    agent = create_agent(
        model=model,
        tools=scan_tools,
        system_prompt=SECURITY_SYSTEM_PROMPT,
        name="security_agent",
    )

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"请对这个项目进行安全审查：{project_path}",
                }
            ]
        }
    )

    messages = result["messages"]
    scan_calls = [
        tool_call
        for message in messages
        if isinstance(message, AIMessage)
        for tool_call in message.tool_calls
        if tool_call.get("name") == "scan_security"
    ]
    scan_messages = [
        message
        for message in messages
        if isinstance(message, ToolMessage) and message.name == "scan_security"
    ]
    if not scan_calls or not scan_messages:
        raise RuntimeError("Security Agent did not call scan_security")

    structured_result = get_structured_tool_result(scan_messages[-1])
    final_message = messages[-1]
    if not isinstance(final_message, AIMessage):
        raise TypeError("Security Agent did not produce a final AI response")

    return {
        "model": model_name,
        "loaded_tools": [tool.name for tool in tools],
        "scan_security_calls": scan_calls,
        "tool_result": structured_result,
        "final_answer": final_message.content,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Security Agent.")
    parser.add_argument(
        "project_path",
        nargs="?",
        type=Path,
        default=DEMO_PROJECT,
        help="Local project directory. Defaults to vulnerable_demo.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    project = arguments.project_path.expanduser().resolve()
    if not project.is_dir():
        raise SystemExit(f"Project directory does not exist: {project}")

    review = asyncio.run(run_security_review(project))
    print("=== Loaded LangChain tools ===")
    print(review["loaded_tools"])
    print("\n=== Agent scan_security tool calls ===")
    print(review["scan_security_calls"])
    print("\n=== MCP structured tool result ===")
    print(review["tool_result"])
    print("\n=== Security Agent final answer ===")
    print(review["final_answer"])
