"""Command-line orchestrator for the remote weather and RAG A2A agents."""

from __future__ import annotations

import asyncio
from typing import Any

from a2a.helpers import get_artifact_text, get_message_text
from a2a.types import Task, TaskState
from langchain.agents import create_agent
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai import ChatOpenAI

try:
    from a2a_mcp_demo.common.a2a_client import A2AClient
    from a2a_mcp_demo.config import (
        API_BASE_URL,
        API_KEY,
        MODEL_NAME,
        RAG_AGENT_URL,
        WEATHER_AGENT_URL,
    )
except ModuleNotFoundError:
    from common.a2a_client import A2AClient
    from config import (
        API_BASE_URL,
        API_KEY,
        MODEL_NAME,
        RAG_AGENT_URL,
        WEATHER_AGENT_URL,
    )


SYSTEM_PROMPT = """你是一个总调度助手，负责协调天气 Agent 和知识库 RAG Agent。

请根据用户问题自行决定调用方式：
- 涉及城市天气时，调用天气 Agent。
- 涉及知识库、内部资料、文档或产品信息时，调用 RAG Agent。
- 一个问题同时涉及天气和知识库时，可以同时调用两个 Agent，再综合结果回答。
- 寒暄或不需要专业 Agent 的问题可以直接回答。

必须忠实使用远程 Agent 返回的内容，不得编造远程 Agent 没有提供的信息。
如果远程任务失败、没有 Artifact 或返回“查不到”，应明确告诉用户。
"""


def task_result_to_text(task: Task) -> str:
    """Translate a remote A2A Task into text suitable for a LangChain tool."""
    state_name = TaskState.Name(task.status.state)
    if task.status.state != TaskState.TASK_STATE_COMPLETED:
        detail = ""
        if task.status.HasField("message"):
            detail = get_message_text(task.status.message).strip()
        return f"远程任务未成功完成，状态：{state_name}。{detail}".strip()

    artifact_texts = []
    for artifact in task.artifacts:
        artifact_text = get_artifact_text(artifact).strip()
        if artifact_text:
            artifact_texts.append(artifact_text)
    if not artifact_texts:
        return "远程任务已完成，但没有返回 Artifact 内容。"
    return "\n".join(artifact_texts)


class Orchestrator:
    """Route user requests to remote specialist agents over A2A."""

    def __init__(self) -> None:
        if not API_KEY.strip():
            raise ValueError("请先在 config.py 中填写 API_KEY")

        self._weather_client = A2AClient(WEATHER_AGENT_URL)
        self._rag_client = A2AClient(RAG_AGENT_URL)
        self._tools = self._create_tools()

        model = ChatOpenAI(
            model=MODEL_NAME,
            api_key=API_KEY,
            base_url=API_BASE_URL,
            use_responses_api=False,
        )
        self._agent = create_agent(
            model=model,
            tools=self._tools,
            system_prompt=SYSTEM_PROMPT,
        )

    def _create_tools(self) -> list[BaseTool]:
        async def call_weather_agent(query: str) -> str:
            """调用远程天气 Agent 查询城市天气。"""
            task = await self._weather_client.send_message(query)
            return task_result_to_text(task)

        async def call_rag_agent(query: str) -> str:
            """调用远程 RAG Agent 检索知识库。"""
            task = await self._rag_client.send_message(query)
            return task_result_to_text(task)

        return [
            StructuredTool.from_function(
                coroutine=call_weather_agent,
                name="call_weather_agent",
                description="向远程天气 Agent 查询指定城市的天气信息。",
            ),
            StructuredTool.from_function(
                coroutine=call_rag_agent,
                name="call_rag_agent",
                description="向远程 RAG Agent 查询知识库、内部资料或文档信息。",
            ),
        ]

    async def run(self, user_message: str) -> str:
        """Run one orchestration request."""
        if not user_message.strip():
            raise ValueError("user_message 不能为空")

        result = await self._agent.ainvoke(
            {"messages": [{"role": "user", "content": user_message}]}
        )
        content: Any = result["messages"][-1].content
        if isinstance(content, str):
            return content
        return str(content)


async def run_cli() -> None:
    """Run a continuous command-line chat session."""
    orchestrator = Orchestrator()
    print("A2A 调度助手已启动。输入 退出、exit 或 quit 可结束。")

    while True:
        try:
            user_message = await asyncio.to_thread(input, "\n用户：")
        except (EOFError, KeyboardInterrupt):
            print("\n助手：已退出。")
            return

        if user_message.strip().lower() in {"退出", "exit", "quit"}:
            print("助手：已退出。")
            return
        if not user_message.strip():
            continue

        try:
            answer = await orchestrator.run(user_message)
            print(f"助手：{answer}")
        except Exception as exc:
            print(f"助手：请求处理失败：{exc}")


if __name__ == "__main__":
    asyncio.run(run_cli())
