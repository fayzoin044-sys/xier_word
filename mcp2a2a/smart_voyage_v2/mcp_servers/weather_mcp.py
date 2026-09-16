"""Hard-coded weather tools exposed through MCP Streamable HTTP."""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from smart_voyage_v2.config import (
    BIND_HOST,
    MCP_PATH,
    WEATHER_MCP_PORT,
)


server = MCPServer(
    name="weather-mcp",
    description="提供模拟天气数据，用于 SmartVoyage A2A/MCP 流程演示。",
)


WEATHER_DATA: dict[str, dict[str, Any]] = {
    "北京": {
        "weather": "晴",
        "temperature_c": 26,
        "humidity_percent": 38,
        "wind": "西北风 2 级",
    },
    "上海": {
        "weather": "多云",
        "temperature_c": 29,
        "humidity_percent": 67,
        "wind": "东南风 3 级",
    },
    "广州": {
        "weather": "小雨",
        "temperature_c": 31,
        "humidity_percent": 81,
        "wind": "南风 2 级",
    },
}


@server.tool()
def get_weather(city: str) -> dict[str, Any]:
    """查询指定城市的模拟天气；当前支持北京、上海和广州。"""
    normalized_city = city.strip()
    weather = WEATHER_DATA.get(normalized_city)

    if weather is None:
        return {
            "found": False,
            "city": normalized_city,
            "message": "未找到该城市的天气数据。",
            "supported_cities": list(WEATHER_DATA),
        }

    return {
        "found": True,
        "city": normalized_city,
        **weather,
    }


if __name__ == "__main__":
    server.run(
        transport="streamable-http",
        host=BIND_HOST,
        port=WEATHER_MCP_PORT,
        streamable_http_path=MCP_PATH,
    )
