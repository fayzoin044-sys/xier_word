"""Shared A2A server construction helpers."""

from __future__ import annotations

import uvicorn
from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard
from starlette.applications import Starlette


DEFAULT_RPC_PATH = "/a2a/jsonrpc/"


def create_a2a_app(
    agent_card: AgentCard,
    executor: AgentExecutor,
    *,
    rpc_path: str = DEFAULT_RPC_PATH,
) -> Starlette:
    """Mount an AgentExecutor and its Agent Card on a Starlette application."""
    if not rpc_path.startswith("/"):
        raise ValueError("rpc_path 必须以 / 开头")

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    routes = [
        *create_agent_card_routes(agent_card),
        *create_jsonrpc_routes(
            request_handler,
            rpc_url=rpc_path,
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
    """Run a constructed A2A Starlette application with Uvicorn."""
    uvicorn.run(app, host=host, port=port)
