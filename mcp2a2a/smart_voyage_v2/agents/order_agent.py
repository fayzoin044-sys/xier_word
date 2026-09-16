"""Order agent that collaborates with TicketAgent over A2A."""

from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from smart_voyage_v2.common.a2a_client import A2AClient, get_task_text
from smart_voyage_v2.common.mcp_tools import load_mcp_tools
from smart_voyage_v2.common.models import AgentResult
from smart_voyage_v2.config import (
    API_BASE_URL,
    API_KEY,
    MODEL_NAME,
    ORDER_MCP_URL,
    TICKET_A2A_URL,
)


ORDER_SYSTEM_PROMPT = """
你是 SmartVoyage 的火车票订单助手。
你负责协助用户确认车次并创建模拟订单。
查询或确认车次时，调用 query_ticket_agent，让 TicketAgent 通过 A2A 完成查询；
不要绕过 TicketAgent 自己编造车次、票价或余票。
只有当用户明确要求订票，并且已经提供车次、乘客姓名和出行日期时，
才调用 create_ticket_order 创建模拟订单。
缺少必要信息时应先询问用户，不得猜测乘客姓名或出行日期。
创建的是演示订单，不会真实支付或出票；回答时必须向用户说明这一点。
""".strip()


class TicketAgentQuery(BaseModel):
    """Input accepted by the TicketAgent A2A tool."""

    query: str = Field(description="需要发送给 TicketAgent 的完整车票查询问题。")


class OrderAgent:
    """Use an LLM to choose between remote A2A and local MCP tools."""

    def __init__(self, model: BaseChatModel | None = None) -> None:
        self._model = model
        self._agent_graph: Any | None = None
        self._ticket_client = A2AClient(TICKET_A2A_URL)

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

    async def _query_ticket_agent(self, query: str) -> str:
        task = await self._ticket_client.send_message(query)
        return get_task_text(task)

    async def _get_agent_graph(self) -> Any:
        if self._agent_graph is None:
            order_tools = await load_mcp_tools(ORDER_MCP_URL)
            if not order_tools:
                raise RuntimeError("Order MCP 没有提供可用工具。")

            ticket_a2a_tool = StructuredTool.from_function(
                coroutine=self._query_ticket_agent,
                name="query_ticket_agent",
                description=(
                    "通过 A2A 调用 TicketAgent，查询或确认火车车次、票价和余票。"
                ),
                args_schema=TicketAgentQuery,
            )
            self._agent_graph = create_agent(
                model=self._get_model(),
                tools=[ticket_a2a_tool, *order_tools],
                system_prompt=ORDER_SYSTEM_PROMPT,
                name="order_agent",
            )
        return self._agent_graph

    async def invoke(self, query: str) -> AgentResult:
        query = query.strip()
        if not query:
            return AgentResult(
                text="请提供需要查询或预订的车票信息。",
                requires_input=True,
            )

        agent_graph = await self._get_agent_graph()
        state = await agent_graph.ainvoke(
            {"messages": [HumanMessage(content=query)]}
        )
        content = state["messages"][-1].content
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)

        return AgentResult(text=content)
