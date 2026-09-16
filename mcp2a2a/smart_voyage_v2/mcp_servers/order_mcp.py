"""Hard-coded order tools exposed through MCP Streamable HTTP."""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from smart_voyage_v2.config import BIND_HOST, MCP_PATH, ORDER_MCP_PORT


server = MCPServer(
    name="order-mcp",
    description="提供模拟订票接口，用于 SmartVoyage A2A/MCP 流程演示。",
)


TRAIN_ORDER_DATA: dict[str, dict[str, Any]] = {
    "G101": {
        "departure": "北京",
        "destination": "上海",
        "seat_type": "二等座",
        "price": 553.0,
    },
    "G79": {
        "departure": "北京",
        "destination": "广州",
        "seat_type": "二等座",
        "price": 862.0,
    },
    "G7351": {
        "departure": "上海",
        "destination": "杭州",
        "seat_type": "二等座",
        "price": 73.0,
    },
}


@server.tool()
def create_ticket_order(
    train_no: str,
    passenger_name: str,
    travel_date: str,
) -> dict[str, Any]:
    """创建一条不落库、不扣票的模拟火车票订单。"""
    normalized_train_no = train_no.strip().upper()
    normalized_name = passenger_name.strip()
    normalized_date = travel_date.strip()
    train = TRAIN_ORDER_DATA.get(normalized_train_no)

    if train is None:
        return {
            "created": False,
            "train_no": normalized_train_no,
            "message": "模拟订单系统不支持该车次。",
        }

    if not normalized_name or not normalized_date:
        return {
            "created": False,
            "train_no": normalized_train_no,
            "message": "乘客姓名和出行日期不能为空。",
        }

    return {
        "created": True,
        "order_no": f"SV-DEMO-{normalized_train_no}-{normalized_date}",
        "status": "待支付",
        "passenger_name": normalized_name,
        "travel_date": normalized_date,
        "train_no": normalized_train_no,
        **train,
        "message": "模拟订单创建成功，本次操作不会真实出票。",
    }


if __name__ == "__main__":
    server.run(
        transport="streamable-http",
        host=BIND_HOST,
        port=ORDER_MCP_PORT,
        streamable_http_path=MCP_PATH,
    )
