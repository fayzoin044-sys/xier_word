"""Publish WeatherAgent as an A2A HTTP service on port 8101."""

from a2a.types import AgentSkill

from smart_voyage_v2.agents.weather_agent import WeatherAgent
from smart_voyage_v2.common.executor import SpecialistAgentExecutor
from smart_voyage_v2.common.server import (
    create_a2a_app,
    create_agent_card,
    run_a2a_server,
)
from smart_voyage_v2.config import (
    BIND_HOST,
    WEATHER_A2A_URL,
    WEATHER_AGENT_PORT,
)


WEATHER_SKILL = AgentSkill(
    id="query_weather",
    name="天气查询",
    description="查询指定城市的模拟天气信息。",
    tags=["weather", "天气", "气温"],
    examples=["北京今天天气怎么样？"],
)

weather_agent = WeatherAgent()
weather_executor = SpecialistAgentExecutor(
    weather_agent,
    artifact_name="weather_result",
)
weather_card = create_agent_card(
    name="WeatherAgent",
    description="SmartVoyage 天气查询专业 Agent。",
    base_url=WEATHER_A2A_URL,
    skills=[WEATHER_SKILL],
)
app = create_a2a_app(
    agent_card=weather_card,
    executor=weather_executor,
)


def main() -> None:
    run_a2a_server(app, host=BIND_HOST, port=WEATHER_AGENT_PORT)


if __name__ == "__main__":
    main()
