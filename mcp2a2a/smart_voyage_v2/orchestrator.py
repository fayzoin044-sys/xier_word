"""LangChain orchestrator that routes CLI requests to remote A2A agents."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from smart_voyage_v2.common.a2a_client import A2AClient, get_task_text
from smart_voyage_v2.config import (
    API_BASE_URL,
    API_KEY,
    ATTRACTION_A2A_URL,
    MODEL_NAME,
    ORDER_A2A_URL,
    TICKET_A2A_URL,
    WEATHER_A2A_URL,
)


ORCHESTRATOR_SYSTEM_PROMPT = """
你是 SmartVoyage 的旅行总控助手。
你不直接编造天气、车票、景点或订单数据，而是根据用户目标选择远程专业 Agent。
天气问题调用 WeatherAgent，火车票查询调用 TicketAgent，景点问题调用 AttractionAgent，
用户明确要求创建订单时调用 OrderAgent。
一个请求涉及多个领域时，可以调用多个专业 Agent，并将结果整合成清晰的中文回答。
查询车票不应调用 OrderAgent；只有用户明确表示预订、下单时才使用 OrderAgent。
如果专业 Agent 要求补充信息，应直接向用户询问缺少的信息。
""".strip()


@dataclass(frozen=True)
class RemoteAgentSpec:
    tool_name: str
    base_url: str


REMOTE_AGENTS = (
    RemoteAgentSpec("ask_weather_agent", WEATHER_A2A_URL),
    RemoteAgentSpec("ask_ticket_agent", TICKET_A2A_URL),
    RemoteAgentSpec("ask_attraction_agent", ATTRACTION_A2A_URL),
    RemoteAgentSpec("ask_order_agent", ORDER_A2A_URL),
)


class RemoteAgentQuery(BaseModel):
    query: str = Field(description="发送给远程专业 Agent 的完整用户问题。")


async def create_remote_agent_tool(spec: RemoteAgentSpec) -> StructuredTool:
    """Resolve one AgentCard and expose that remote agent as a LangChain tool."""
    client = A2AClient(spec.base_url)
    card = await client.get_agent_card()
    skill_descriptions = "；".join(
        f"{skill.name}：{skill.description}" for skill in card.skills
    )
    description = card.description
    if skill_descriptions:
        description = f"{description} 能力：{skill_descriptions}"

    async def call_remote_agent(query: str) -> str:
        task = await client.send_message(query)
        return get_task_text(task)

    return StructuredTool.from_function(
        coroutine=call_remote_agent,
        name=spec.tool_name,
        description=description,
        args_schema=RemoteAgentQuery,
    )


class TravelOrchestrator:
    """Choose and coordinate remote specialist agents through A2A tools."""

    def __init__(self, model: BaseChatModel | None = None) -> None:
        self._model = model
        self._agent_graph: Any | None = None
        self._messages: list[BaseMessage] = []

    def _get_model(self) -> BaseChatModel:
        if self._model is not None:
            return self._model
        if not API_KEY:
            raise RuntimeError("请先设置环境变量 DEEPSEEK_API_KEY。")

        self._model = ChatOpenAI(
            model=MODEL_NAME,
            api_key=API_KEY,
            base_url=API_BASE_URL,
            temperature=0,
        )
        return self._model

    async def _get_agent_graph(self) -> Any:
        if self._agent_graph is None:
            model = self._get_model()
            tools = await asyncio.gather(
                *(create_remote_agent_tool(spec) for spec in REMOTE_AGENTS)
            )
            self._agent_graph = create_agent(
                model=model,
                tools=list(tools),
                system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
                name="travel_orchestrator",
            )
        return self._agent_graph

    async def invoke(self, query: str) -> str:
        query = query.strip()
        if not query:
            return "请输入旅行相关问题。"

        agent_graph = await self._get_agent_graph()
        self._messages.append(HumanMessage(content=query))
        state = await agent_graph.ainvoke({"messages": self._messages})
        self._messages = list(state["messages"])

        content = self._messages[-1].content
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)

    def reset(self) -> None:
        """Clear command-line conversation history."""
        self._messages.clear()


async def run_cli() -> None:
    orchestrator = TravelOrchestrator()
    print("SmartVoyage 已启动。输入 exit、quit 或 退出可结束会话。")

    while True:
        try:
            query = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n会话已结束。")
            return

        if query.lower() in {"exit", "quit"} or query == "退出":
            print("会话已结束。")
            return
        if query.lower() == "reset" or query == "重置":
            orchestrator.reset()
            print("对话上下文已清空。")
            continue

        try:
            response = await orchestrator.invoke(query)
        except Exception as exc:
            print(f"总控调用失败：{exc}")
            continue
        print(f"SmartVoyage：{response}")


def main() -> None:
    asyncio.run(run_cli())


if __name__ == "__main__":
    main()
