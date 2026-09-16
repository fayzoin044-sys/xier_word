"""Publish TicketAgent as an A2A HTTP service on port 8102."""

from a2a.types import AgentSkill

from smart_voyage_v2.agents.ticket_agent import TicketAgent
from smart_voyage_v2.common.executor import SpecialistAgentExecutor
from smart_voyage_v2.common.server import (
    create_a2a_app,
    create_agent_card,
    run_a2a_server,
)
from smart_voyage_v2.config import (
    BIND_HOST,
    TICKET_A2A_URL,
    TICKET_AGENT_PORT,
)


TICKET_SKILL = AgentSkill(
    id="search_train_tickets",
    name="火车票查询",
    description="按出发地、目的地和日期查询模拟火车票。",
    tags=["ticket", "火车票", "车次", "余票"],
    examples=["查询2026-08-20北京到上海的火车票。"],
)

ticket_agent = TicketAgent()
ticket_executor = SpecialistAgentExecutor(
    ticket_agent,
    artifact_name="ticket_result",
)
ticket_card = create_agent_card(
    name="TicketAgent",
    description="SmartVoyage 火车票查询专业 Agent。",
    base_url=TICKET_A2A_URL,
    skills=[TICKET_SKILL],
)
app = create_a2a_app(
    agent_card=ticket_card,
    executor=ticket_executor,
)


def main() -> None:
    run_a2a_server(app, host=BIND_HOST, port=TICKET_AGENT_PORT)


if __name__ == "__main__":
    main()
