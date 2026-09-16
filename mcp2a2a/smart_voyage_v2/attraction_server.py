"""Publish deterministic AttractionAgent as an A2A service on port 8103."""

from a2a.types import AgentSkill

from smart_voyage_v2.agents.attraction_agent import AttractionAgent
from smart_voyage_v2.common.executor import SpecialistAgentExecutor
from smart_voyage_v2.common.server import (
    create_a2a_app,
    create_agent_card,
    run_a2a_server,
)
from smart_voyage_v2.config import (
    ATTRACTION_A2A_URL,
    ATTRACTION_AGENT_PORT,
    BIND_HOST,
)


ATTRACTION_SKILL = AgentSkill(
    id="search_attractions",
    name="景点查询",
    description="按城市和关键词查询模拟景点信息。",
    tags=["attraction", "景点", "旅游推荐"],
    examples=["推荐北京的历史文化景点。"],
)

attraction_agent = AttractionAgent()
attraction_executor = SpecialistAgentExecutor(
    attraction_agent,
    artifact_name="attraction_result",
)
attraction_card = create_agent_card(
    name="AttractionAgent",
    description="SmartVoyage 无大模型的固定规则景点查询 Agent。",
    base_url=ATTRACTION_A2A_URL,
    skills=[ATTRACTION_SKILL],
)
app = create_a2a_app(
    agent_card=attraction_card,
    executor=attraction_executor,
)


def main() -> None:
    run_a2a_server(app, host=BIND_HOST, port=ATTRACTION_AGENT_PORT)


if __name__ == "__main__":
    main()
