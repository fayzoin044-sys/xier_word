"""LangChain RAG agent backed by the stdio RAG MCP server."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from langchain.agents import create_agent
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai import ChatOpenAI
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import Tool

try:
    from a2a_mcp_demo.config import API_BASE_URL, API_KEY, MODEL_NAME
except ModuleNotFoundError:
    # Support importing this module when a2a_mcp_demo itself is the working directory.
    from config import API_BASE_URL, API_KEY, MODEL_NAME


SYSTEM_PROMPT = """你是一个知识库问答助手。
根据用户的问题自行判断是否需要调用知识库检索工具。当问题涉及事实、业务资料、
文档内容或其他需要知识库依据的信息时，应先调用检索工具。

回答必须严格依据工具返回的检索结果，不得补充或编造检索结果中没有的信息。
如果工具返回“查不到”、空内容或没有相关资料，应明确告诉用户知识库中暂未检索到
相关信息。对于寒暄、能力介绍等不需要知识库的问题，可以直接回答。
"""

RAG_MCP_PATH = Path(__file__).resolve().parents[1] / "mcp_servers" / "rag_mcp.py"


class RAGAgent:
    """Discover RAG MCP tools and expose them to a LangChain agent."""

    def __init__(self) -> None:
        self._server_parameters = StdioServerParameters(
            command=sys.executable,
            args=[str(RAG_MCP_PATH)],
        )

    @staticmethod
    def _convert_mcp_tool(session: ClientSession, mcp_tool: Tool) -> BaseTool:
        """Convert one MCP tool definition into a LangChain StructuredTool."""

        async def call_mcp_tool(**arguments: Any) -> Any:
            result = await session.call_tool(mcp_tool.name, arguments)

            if result.is_error:
                error_text = "\n".join(
                    item.text
                    for item in result.content
                    if hasattr(item, "text")
                )
                raise RuntimeError(error_text or f"MCP 工具 {mcp_tool.name} 调用失败")

            if result.structured_content is not None:
                return result.structured_content

            text_content = [
                item.text for item in result.content if hasattr(item, "text")
            ]
            return "\n".join(text_content)

        return StructuredTool.from_function(
            coroutine=call_mcp_tool,
            name=mcp_tool.name,
            description=mcp_tool.description or f"调用 MCP 工具 {mcp_tool.name}",
            args_schema=mcp_tool.input_schema,
            infer_schema=False,
        )

    @asynccontextmanager
    async def connect_tools(self) -> AsyncIterator[list[BaseTool]]:
        """Connect to the stdio MCP server and yield its LangChain tools."""
        async with stdio_client(self._server_parameters) as (
            read_stream,
            write_stream,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                mcp_tools = (await session.list_tools()).tools
                langchain_tools = [
                    self._convert_mcp_tool(session, tool) for tool in mcp_tools
                ]
                yield langchain_tools

    @staticmethod
    def _create_model() -> ChatOpenAI:
        if not API_KEY.strip():
            raise ValueError("请先在 config.py 中填写 API_KEY")

        return ChatOpenAI(
            model=MODEL_NAME,
            api_key=API_KEY,
            base_url=API_BASE_URL,
            use_responses_api=False,
        )

    async def run(self, user_message: str) -> str:
        """Run one request and let the model decide whether to search the RAG tool."""
        if not user_message.strip():
            raise ValueError("user_message 不能为空")

        model = self._create_model()

        async with self.connect_tools() as tools:
            agent = create_agent(
                model=model,
                tools=tools,
                system_prompt=SYSTEM_PROMPT,
            )
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": user_message}]}
            )

        content = result["messages"][-1].content
        if isinstance(content, str):
            return content
        return str(content)
