"""Small A2A client wrapper that returns the remote Task object."""

from __future__ import annotations

from a2a.client import ClientConfig, create_client
from a2a.helpers import new_text_message
from a2a.types import Role, SendMessageRequest, Task


class A2AResponseError(RuntimeError):
    """Raised when the remote agent does not return an A2A Task."""


class A2AClient:
    """Resolve an Agent Card and send non-streaming A2A messages."""

    def __init__(self, target_url: str) -> None:
        target_url = target_url.strip()
        if not target_url:
            raise ValueError("target_url 不能为空")
        self._target_url = target_url.rstrip("/")

    async def send_message(self, user_message: str) -> Task:
        """Send one user message and return the complete remote Task."""
        if not user_message.strip():
            raise ValueError("user_message 不能为空")

        client = await create_client(
            self._target_url,
            client_config=ClientConfig(
                streaming=False,
                polling=False,
                supported_protocol_bindings=["JSONRPC"],
                accepted_output_modes=["text/plain"],
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
            raise A2AResponseError("远程 Agent 没有返回 A2A Task")
        return task
