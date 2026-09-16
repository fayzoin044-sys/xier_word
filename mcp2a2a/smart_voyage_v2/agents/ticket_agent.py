"""LangChain ticket agent backed by MCP Streamable HTTP tools."""

from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from smart_voyage_v2.common.mcp_tools import load_mcp_tools
from smart_voyage_v2.common.models import AgentResult
from smart_voyage_v2.config import (
    API_BASE_URL,
    API_KEY,
    MODEL_NAME,
    TICKET_MCP_URL,
)


TICKET_SYSTEM_PROMPT = """
你是 SmartVoyage 的火车票查询助手。
你只负责查询和解释火车票信息，不负责创建订单。
当用户给出出发地、目的地和出行日期时，根据问题自行判断并调用车票查询工具；
不得编造工具没有返回的车次、价格或余票。
如果缺少出发地、目的地或日期，请明确询问缺少的信息。
回答使用中文，并优先列出车次、出发到达时间、席别、价格和余票。
""".strip()


class TicketAgent:
    """Business agent used by SpecialistAgentExecutor."""

    def __init__(self, model: BaseChatModel | None = None) -> None:
        self._model = model
        self._agent_graph: Any | None = None

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
            tools = await load_mcp_tools(TICKET_MCP_URL)
            if not tools:
                raise RuntimeError("Ticket MCP 没有提供可用工具。")

            self._agent_graph = create_agent(
                model=self._get_model(),
                tools=tools,
                system_prompt=TICKET_SYSTEM_PROMPT,
                name="ticket_agent",
            )
        return self._agent_graph

    async def invoke(self, query: str) -> AgentResult:
        """Let the model decide whether to call the discovered ticket tool."""
        query = query.strip()
        if not query:
            return AgentResult(
                text="请提供出发地、目的地和出行日期。",
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
