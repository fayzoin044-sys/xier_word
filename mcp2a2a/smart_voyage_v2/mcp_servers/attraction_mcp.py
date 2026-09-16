"""Hard-coded attraction tools exposed through MCP Streamable HTTP."""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from smart_voyage_v2.config import ATTRACTION_MCP_PORT, BIND_HOST, MCP_PATH


server = MCPServer(
    name="attraction-mcp",
    description="提供模拟景点数据，用于 SmartVoyage A2A/MCP 流程演示。",
)


ATTRACTION_DATA: list[dict[str, Any]] = [
    {
        "name": "故宫博物院",
        "city": "北京",
        "category": "历史文化",
        "opening_hours": "08:30-17:00",
        "ticket_price": 60.0,
        "recommended_duration_hours": 4,
        "rating": 4.9,
    },
    {
        "name": "外滩",
        "city": "上海",
        "category": "城市风光",
        "opening_hours": "全天开放",
        "ticket_price": 0.0,
        "recommended_duration_hours": 2,
        "rating": 4.8,
    },
    {
        "name": "广州塔",
        "city": "广州",
        "category": "城市地标",
        "opening_hours": "09:30-22:30",
        "ticket_price": 150.0,
        "recommended_duration_hours": 3,
        "rating": 4.7,
    },
]


@server.tool()
def search_attractions(city: str, keyword: str = "") -> dict[str, Any]:
    """按城市查询模拟景点，可使用关键词筛选名称或景点类型。"""
    normalized_city = city.strip()
    normalized_keyword = keyword.strip()

    attractions = [
        attraction
        for attraction in ATTRACTION_DATA
        if attraction["city"] == normalized_city
        and (
            not normalized_keyword
            or normalized_keyword in attraction["name"]
            or normalized_keyword in attraction["category"]
        )
    ]

    return {
        "found": bool(attractions),
        "city": normalized_city,
        "keyword": normalized_keyword,
        "count": len(attractions),
        "attractions": attractions,
        "message": "查询成功。" if attractions else "未找到符合条件的模拟景点。",
    }


if __name__ == "__main__":
    server.run(
        transport="streamable-http",
        host=BIND_HOST,
        port=ATTRACTION_MCP_PORT,
        streamable_http_path=MCP_PATH,
    )
