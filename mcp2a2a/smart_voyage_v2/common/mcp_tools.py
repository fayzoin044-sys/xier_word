"""Adapt MCP 2.0 Streamable HTTP tools into LangChain tools."""
#TODO 这是后面补的，把MCP→LangChain 工具
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent, Tool


async def call_mcp_tool(
    mcp_url: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    async with streamable_http_client(mcp_url) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)

    if not isinstance(result, CallToolResult):
        return result.model_dump_json(by_alias=True)

    if result.structured_content is not None:
        content = json.dumps(result.structured_content, ensure_ascii=False)
    else:
        text_parts = [
            item.text for item in result.content if isinstance(item, TextContent)
        ]
        content = "\n".join(text_parts)

    if result.is_error:
        raise ToolException(content or f"MCP 工具 {tool_name} 调用失败。")
    return content


def _to_langchain_tool(mcp_url: str, tool: Tool) -> BaseTool:
    async def call_tool(**arguments: Any) -> str:
        return await call_mcp_tool(mcp_url, tool.name, arguments)

    return StructuredTool(
        name=tool.name,
        description=tool.description or f"MCP tool: {tool.name}",
        args_schema=tool.input_schema,
        coroutine=call_tool,
    )


async def load_mcp_tools(mcp_url: str) -> list[BaseTool]:
    """Discover all tools at an MCP endpoint and wrap them for LangChain."""
    async with streamable_http_client(mcp_url) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            result = await session.list_tools()

    return [_to_langchain_tool(mcp_url, tool) for tool in result.tools]
