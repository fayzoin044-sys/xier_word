"""Resolve a remote Agent Card and send non-streaming A2A messages."""

from __future__ import annotations

import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import get_text_parts, new_text_message
from a2a.types import AgentCard, Role, SendMessageRequest, Task

from smart_voyage_v2.config import AGENT_CARD_PATH


class A2AResponseError(RuntimeError):
    """Raised when a remote agent does not return an A2A Task."""


def get_task_text(task: Task) -> str:
    """Extract human-readable text from a remote Task's artifacts or status."""
    texts: list[str] = []
    for artifact in task.artifacts:
        texts.extend(get_text_parts(artifact.parts))

    if texts:
        return "\n".join(texts)

    if task.status.HasField("message"):
        texts.extend(get_text_parts(task.status.message.parts))
    if texts:
        return "\n".join(texts)

    return "远程 Agent 没有返回文本结果。"


class A2AClient:
    """Small client used by the orchestrator and collaborating agents."""

    def __init__(self, target_url: str) -> None:
        target_url = target_url.strip()
        if not target_url:
            raise ValueError("target_url 不能为空。")
        self._target_url = target_url.rstrip("/")

    async def get_agent_card(self) -> AgentCard:
        """Read the remote agent's public card from its well-known URL."""
        async with httpx.AsyncClient() as http_client:
            resolver = A2ACardResolver(
                httpx_client=http_client,
                base_url=self._target_url,
                agent_card_path=AGENT_CARD_PATH,
            )
            return await resolver.get_agent_card()

    async def send_message(self, user_message: str) -> Task:
        """Send one text Message and return the remote agent's final Task."""
        user_message = user_message.strip()
        if not user_message:
            raise ValueError("user_message 不能为空。")

        agent_card = await self.get_agent_card()
        client = await create_client(
            agent_card,
            client_config=ClientConfig(
                streaming=False,
                polling=False,
                supported_protocol_bindings=["JSONRPC"],
                accepted_output_modes=["text/plain", "application/json"],
            ),
        )

        request = SendMessageRequest(
            message=new_text_message(
                user_message,
                role=Role.ROLE_USER,
            )
        )

        task: Task | None = None
        try:
            async for response in client.send_message(request):
                if response.HasField("task"):
                    task = response.task
        finally:
            await client.close()

        if task is None:
            raise A2AResponseError("远程 Agent 没有返回 A2A Task。")
        return task
