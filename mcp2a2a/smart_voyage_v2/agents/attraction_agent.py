"""Deterministic attraction agent without an LLM."""

from __future__ import annotations

import json
from typing import Any

from smart_voyage_v2.common.mcp_tools import call_mcp_tool
from smart_voyage_v2.common.models import AgentResult
from smart_voyage_v2.config import ATTRACTION_MCP_URL


SUPPORTED_CITIES = ("北京", "上海", "广州")
ATTRACTION_CITY_HINTS = {
    "故宫": "北京",
    "外滩": "上海",
    "广州塔": "广州",
}
ATTRACTION_KEYWORDS = (
    "故宫",
    "历史文化",
    "外滩",
    "城市风光",
    "广州塔",
    "城市地标",
)


def _extract_city(query: str) -> str | None:
    for city in SUPPORTED_CITIES:
        if city in query:
            return city
    for hint, city in ATTRACTION_CITY_HINTS.items():
        if hint in query:
            return city
    return None


def _extract_keyword(query: str) -> str:
    for keyword in ATTRACTION_KEYWORDS:
        if keyword in query:
            return keyword
    return ""


def _format_attractions(payload: dict[str, Any]) -> str:
    attractions = payload.get("attractions", [])
    if not attractions:
        return str(payload.get("message", "未找到符合条件的景点。"))

    lines = []
    for item in attractions:
        lines.append(
            f"{item['name']}（{item['category']}）："
            f"开放时间 {item['opening_hours']}，"
            f"票价 {item['ticket_price']} 元，"
            f"建议游览 {item['recommended_duration_hours']} 小时，"
            f"评分 {item['rating']}。"
        )
    return "\n".join(lines)


class AttractionAgent:
    """Use fixed parsing rules and call Attraction MCP directly."""

    async def invoke(self, query: str) -> AgentResult:
        query = query.strip()
        if not query:
            return AgentResult(
                text="请告诉我需要查询哪个城市的景点。",
                requires_input=True,
            )

        city = _extract_city(query)
        if city is None:
            return AgentResult(
                text="目前支持北京、上海和广州，请补充需要查询的城市。",
                requires_input=True,
            )

        raw_result = await call_mcp_tool(
            ATTRACTION_MCP_URL,
            "search_attractions",
            {
                "city": city,
                "keyword": _extract_keyword(query),
            },
        )
        payload = json.loads(raw_result)
        return AgentResult(
            text=_format_attractions(payload),
            data=payload,
        )
