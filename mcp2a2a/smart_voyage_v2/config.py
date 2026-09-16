"""Central configuration for SmartVoyage v2."""

from __future__ import annotations

"""
模型连接信息
+ 四个 Agent 的 A2A 地址
+ 四个 MCP 的 HTTP 地址
81xx = A2A 智能体服务
82xx = MCP 工具服务
PUBLIC_HOST和AGENT_CARD_PATH 表示
"""

import os

#TODO　1 模型配置，os.geten表示优先读取环境变量
# Model configuration. Secrets come from environment variables rather than source code.
MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
API_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
#TODO 2 A2A服务配置，

# Network configuration.
BIND_HOST = os.getenv("SMART_VOYAGE_BIND_HOST", "127.0.0.1")#服务只在本机监听。只监听本机地址
PUBLIC_HOST = os.getenv("SMART_VOYAGE_PUBLIC_HOST", "localhost")#我们启用外部服务的地址，就是给客户端的地址
A2A_RPC_PATH = "/a2a/jsonrpc/"#其他客户端应该通过哪个主机名访问这些服务。目前是本机
AGENT_CARD_PATH = "/.well-known/agent-card.json"#去这个 Agent 的哪个路径读取名片
MCP_PATH = "/mcp"#接口的路径，http://localhost:8201/mcp

# A2A agent ports.
WEATHER_AGENT_PORT = int(os.getenv("WEATHER_AGENT_PORT", "8101"))
TICKET_AGENT_PORT = int(os.getenv("TICKET_AGENT_PORT", "8102"))
ATTRACTION_AGENT_PORT = int(os.getenv("ATTRACTION_AGENT_PORT", "8103"))
ORDER_AGENT_PORT = int(os.getenv("ORDER_AGENT_PORT", "8104"))

# Streamable HTTP MCP ports.
WEATHER_MCP_PORT = int(os.getenv("WEATHER_MCP_PORT", "8201"))
TICKET_MCP_PORT = int(os.getenv("TICKET_MCP_PORT", "8202"))
ATTRACTION_MCP_PORT = int(os.getenv("ATTRACTION_MCP_PORT", "8203"))
ORDER_MCP_PORT = int(os.getenv("ORDER_MCP_PORT", "8204"))


def _http_url(port: int, path: str = "") -> str:
    return f"http://{PUBLIC_HOST}:{port}{path}"


# Public A2A base URLs. Clients use these to resolve Agent Cards.
WEATHER_A2A_URL = _http_url(WEATHER_AGENT_PORT)
TICKET_A2A_URL = _http_url(TICKET_AGENT_PORT)
ATTRACTION_A2A_URL = _http_url(ATTRACTION_AGENT_PORT)
ORDER_A2A_URL = _http_url(ORDER_AGENT_PORT)

# MCP Streamable HTTP endpoints used internally by specialist agents.
WEATHER_MCP_URL = _http_url(WEATHER_MCP_PORT, MCP_PATH)
TICKET_MCP_URL = _http_url(TICKET_MCP_PORT, MCP_PATH)
ATTRACTION_MCP_URL = _http_url(ATTRACTION_MCP_PORT, MCP_PATH)
ORDER_MCP_URL = _http_url(ORDER_MCP_PORT, MCP_PATH)
