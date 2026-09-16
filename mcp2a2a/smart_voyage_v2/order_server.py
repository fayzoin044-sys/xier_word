"""Publish OrderAgent as an A2A HTTP service on port 8104."""

from a2a.types import AgentSkill

from smart_voyage_v2.agents.order_agent import OrderAgent
from smart_voyage_v2.common.executor import SpecialistAgentExecutor
from smart_voyage_v2.common.server import (
    create_a2a_app,
    create_agent_card,
    run_a2a_server,
)
from smart_voyage_v2.config import (
    BIND_HOST,
    ORDER_A2A_URL,
    ORDER_AGENT_PORT,
)


ORDER_SKILL = AgentSkill(
    id="create_ticket_order",
    name="模拟火车票订购",
    description="通过 A2A 确认车次后创建不出票的模拟订单。",
    tags=["order", "订单", "订票", "A2A collaboration"],
    examples=["帮张三预订2026-08-20北京到上海的火车票。"],
)

order_agent = OrderAgent()
order_executor = SpecialistAgentExecutor(
    order_agent,
    artifact_name="order_result",
)
order_card = create_agent_card(
    name="OrderAgent",
    description="SmartVoyage 火车票订单专业 Agent。",
    base_url=ORDER_A2A_URL,
    skills=[ORDER_SKILL],
)
app = create_a2a_app(
    agent_card=order_card,
    executor=order_executor,
)


def main() -> None:
    run_a2a_server(app, host=BIND_HOST, port=ORDER_AGENT_PORT)


if __name__ == "__main__":
    main()
