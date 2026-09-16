"""LangChain weather agent backed by MCP Streamable HTTP tools."""

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
    WEATHER_MCP_URL,
)


WEATHER_SYSTEM_PROMPT = """
你是 SmartVoyage 的天气专业助手。
你只负责回答天气相关问题。
当用户询问具体城市天气时，根据问题自行判断并调用可用的天气工具；
不得编造工具没有返回的天气数据。
如果用户没有说明城市，请简短询问城市名称。
回答使用中文，并清楚说明城市、天气、温度、湿度和风力。
""".strip()


class WeatherAgent:
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
            tools = await load_mcp_tools(WEATHER_MCP_URL)
            if not tools:
                raise RuntimeError("Weather MCP 没有提供可用工具。")

            self._agent_graph = create_agent(
                model=self._get_model(),
                tools=tools,
                system_prompt=WEATHER_SYSTEM_PROMPT,
                name="weather_agent",
            )
        return self._agent_graph

    async def invoke(self, query: str) -> AgentResult:
        """Let the model decide whether to call the discovered weather tool."""
        query = query.strip()
        if not query:
            return AgentResult(
                text="请告诉我需要查询哪个城市的天气。",
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
