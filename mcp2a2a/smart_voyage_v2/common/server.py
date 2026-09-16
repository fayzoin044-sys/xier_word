"""Shared helpers for constructing and running specialist A2A servers."""

from __future__ import annotations

import uvicorn
from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from a2a.utils.constants import PROTOCOL_VERSION_CURRENT, TransportProtocol
from starlette.applications import Starlette

from smart_voyage_v2.config import A2A_RPC_PATH, AGENT_CARD_PATH


def create_agent_card(
    *,
    name: str,
    description: str,
    base_url: str,
    skills: list[AgentSkill],
    version: str = "1.0.0",
) -> AgentCard:
    """Create the public capability card for one specialist agent."""
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
    """Mount an executor and its Agent Card on a Starlette application."""
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    routes = [
        *create_agent_card_routes(
            agent_card,
            card_url=AGENT_CARD_PATH,
        ),
        *create_jsonrpc_routes(
            request_handler,
            rpc_url=A2A_RPC_PATH,
            enable_v0_3_compat=False,
        ),
    ]
    return Starlette(routes=routes)


def run_a2a_server(
    app: Starlette,
    *,
    host: str,
    port: int,
) -> None:
    """Run one A2A Starlette application with Uvicorn."""
    uvicorn.run(app, host=host, port=port)
