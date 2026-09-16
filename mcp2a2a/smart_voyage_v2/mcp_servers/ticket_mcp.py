"""Hard-coded train-ticket tools exposed through MCP Streamable HTTP."""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from smart_voyage_v2.config import BIND_HOST, MCP_PATH, TICKET_MCP_PORT


server = MCPServer(
    name="ticket-mcp",
    description="提供模拟火车票数据，用于 SmartVoyage A2A/MCP 流程演示。",
)


TICKET_DATA: list[dict[str, Any]] = [
    {
        "train_no": "G101",
        "departure": "北京",
        "destination": "上海",
        "departure_time": "06:43",
        "arrival_time": "12:40",
        "seat_type": "二等座",
        "price": 553.0,
        "available_seats": 32,
    },
    {
        "train_no": "G79",
        "departure": "北京",
        "destination": "广州",
        "departure_time": "10:00",
        "arrival_time": "17:38",
        "seat_type": "二等座",
        "price": 862.0,
        "available_seats": 18,
    },
    {
        "train_no": "G7351",
        "departure": "上海",
        "destination": "杭州",
        "departure_time": "08:30",
        "arrival_time": "09:15",
        "seat_type": "二等座",
        "price": 73.0,
        "available_seats": 40,
    },
]


@server.tool()
def search_train_tickets(
    departure: str,
    destination: str,
    travel_date: str,
) -> dict[str, Any]:
    """按出发地、目的地和日期查询模拟火车票，日期格式为 YYYY-MM-DD。"""
    normalized_departure = departure.strip()
    normalized_destination = destination.strip()
    normalized_date = travel_date.strip()

    tickets = [
        ticket
        for ticket in TICKET_DATA
        if ticket["departure"] == normalized_departure
        and ticket["destination"] == normalized_destination
    ]

    return {
        "found": bool(tickets),
        "departure": normalized_departure,
        "destination": normalized_destination,
        "travel_date": normalized_date,
        "count": len(tickets),
        "tickets": tickets,
        "message": "查询成功。" if tickets else "未找到符合条件的模拟车次。",
    }


if __name__ == "__main__":
    server.run(
        transport="streamable-http",
        host=BIND_HOST,
        port=TICKET_MCP_PORT,
        streamable_http_path=MCP_PATH,
    )
