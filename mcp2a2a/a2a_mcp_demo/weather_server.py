"""A2A HTTP server entry point for the weather agent."""

from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill

try:
    from a2a_mcp_demo.agents.weather_agent import WeatherAgent
    from a2a_mcp_demo.common.executor import BusinessAgentExecutor
    from a2a_mcp_demo.common.server import (
        DEFAULT_RPC_PATH,
        create_a2a_app,
        run_a2a_server,
    )
    from a2a_mcp_demo.config import WEATHER_AGENT_URL
except ModuleNotFoundError:
    from agents.weather_agent import WeatherAgent
    from common.executor import BusinessAgentExecutor
    from common.server import DEFAULT_RPC_PATH, create_a2a_app, run_a2a_server
    from config import WEATHER_AGENT_URL


HOST = "localhost"
PORT = 8001


def create_weather_agent_card() -> AgentCard:
    """Describe the weather agent and its A2A endpoint."""
    skill = AgentSkill(
        id="weather_query",
        name="天气查询",
        description="查询指定城市的天气信息。",
        tags=["weather", "天气"],
        input_modes=["text/plain"],
        output_modes=["text/plain"],
        examples=["北京今天天气怎么样？", "查询上海的天气"],
    )
    return AgentCard(
        name="Weather Agent",
        description="根据用户问题查询并回答城市天气信息。",
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                url=f"{WEATHER_AGENT_URL.rstrip('/')}{DEFAULT_RPC_PATH}",
            )
        ],
        version="0.1.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


def create_weather_app():
    """Wire WeatherAgent into the shared executor and A2A server."""
    agent = WeatherAgent()
    executor = BusinessAgentExecutor(agent, artifact_name="weather_result")
    return create_a2a_app(create_weather_agent_card(), executor)


app = create_weather_app()


if __name__ == "__main__":
    run_a2a_server(app, host=HOST, port=PORT)
