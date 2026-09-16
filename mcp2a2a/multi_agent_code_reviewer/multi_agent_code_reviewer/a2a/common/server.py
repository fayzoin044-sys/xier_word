"""Shared A2A AgentCard, application, and Uvicorn server helpers."""

import uvicorn
from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from a2a.utils.constants import PROTOCOL_VERSION_CURRENT, TransportProtocol
from starlette.applications import Starlette

A2A_RPC_PATH = "/a2a/jsonrpc/"
AGENT_CARD_PATH = "/.well-known/agent-card.json"


def create_agent_card(
    *,
    name: str,
    description: str,
    base_url: str,
    skills: list[AgentSkill],
    version: str = "1.0.0",
) -> AgentCard:
    """Create the public AgentCard for one code-review specialist."""
    rpc_url = f"{base_url.rstrip('/')}{A2A_RPC_PATH}"
    return AgentCard(
        name=name,
        description=description,
        supported_interfaces=[
            AgentInterface(
                url=rpc_url,
                protocol_binding=TransportProtocol.JSONRPC.value,
                protocol_version=PROTOCOL_VERSION_CURRENT,
            )
        ],
        version=version,
        capabilities=AgentCapabilities(
            streaming=False,
            push_notifications=False,
        ),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain", "application/json"],
        skills=skills,
    )


def create_a2a_app(
    *,
    agent_card: AgentCard,
    executor: AgentExecutor,
) -> Starlette:
    """Mount the AgentCard and JSON-RPC handler on a Starlette app."""
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    routes = [
        *create_agent_card_routes(agent_card, card_url=AGENT_CARD_PATH),
        *create_jsonrpc_routes(
            request_handler,
            rpc_url=A2A_RPC_PATH,
            enable_v0_3_compat=False,
        ),
    ]
    return Starlette(routes=routes)


def run_a2a_server(app: Starlette, *, host: str, port: int) -> None:
    """Run one independent A2A service with Uvicorn."""
    uvicorn.run(app, host=host, port=port)
