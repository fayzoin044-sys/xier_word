"""Mock RAG MCP server using the stdio transport."""

from mcp.server.mcpserver import MCPServer


server = MCPServer(
    name="rag-mcp",
    description="模拟知识库检索，用于本地 A2A/MCP 联调。",
)


@server.tool()
def search_knowledge(query: str) -> str:
    """查询模拟知识库。"""
    _ = query
    return "查不到"


if __name__ == "__main__":
    server.run(transport="stdio")
