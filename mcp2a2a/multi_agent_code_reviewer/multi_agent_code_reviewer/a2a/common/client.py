"""Resolve AgentCards and send non-streaming A2A messages."""

import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import get_data_parts, get_text_parts, new_text_message
from a2a.types import AgentCard, Role, SendMessageRequest, Task

from multi_agent_code_reviewer.a2a.common.server import AGENT_CARD_PATH

INTEGER_DATA_FIELDS = frozenset(
    {
        "end_column",
        "end_line",
        "errors",
        "exit_code",
        "failed",
        "findings_count",
        "line",
        "passed",
        "skipped",
        "start_column",
        "start_line",
        "total",
    }
)


class A2AResponseError(RuntimeError):
    """Raised when a remote Agent does not return an A2A Task."""


def get_task_text(task: Task) -> str:
    texts: list[str] = []
    for artifact in task.artifacts:
        texts.extend(get_text_parts(artifact.parts))
    if texts:
        return "\n".join(texts)

    if task.status.HasField("message"):
        texts.extend(get_text_parts(task.status.message.parts))
    return "\n".join(texts) if texts else "远程 Agent 没有返回文本结果。"


def normalize_a2a_data(value: object, *, field_name: str | None = None) -> object:
    """Restore schema-defined integers after protobuf Value deserialization."""
    if isinstance(value, dict):
        return {
            key: normalize_a2a_data(item, field_name=key)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [normalize_a2a_data(item, field_name=field_name) for item in value]
    if (
        field_name in INTEGER_DATA_FIELDS
        and isinstance(value, float)
        and value.is_integer()
    ):
        return int(value)
    return value


def get_task_data(task: Task) -> list[object]:
    """Extract machine-readable data with integer field types restored."""
    data: list[object] = []
    for artifact in task.artifacts:
        data.extend(normalize_a2a_data(item) for item in get_data_parts(artifact.parts))
    return data


class A2AClient:
    """Small reusable client copied from the validated SmartVoyage design."""

    def __init__(self, target_url: str) -> None:
        target_url = target_url.strip()
        if not target_url:
            raise ValueError("target_url 不能为空。")
        self._target_url = target_url.rstrip("/")

    async def get_agent_card(self) -> AgentCard:
        async with httpx.AsyncClient(timeout=30, trust_env=False) as http_client:
            resolver = A2ACardResolver(
                httpx_client=http_client,
                base_url=self._target_url,
                agent_card_path=AGENT_CARD_PATH,
            )
            return await resolver.get_agent_card()

    async def send_message(self, user_message: str) -> Task:
        user_message = user_message.strip()
        if not user_message:
            raise ValueError("user_message 不能为空。")

        agent_card = await self.get_agent_card()
        http_client = httpx.AsyncClient(timeout=300, trust_env=False)
        client = await create_client(
            agent_card,
            client_config=ClientConfig(
                streaming=False,
                polling=False,
                httpx_client=http_client,
                supported_protocol_bindings=["JSONRPC"],
                accepted_output_modes=["text/plain", "application/json"],
            ),
        )
        request = SendMessageRequest(
            message=new_text_message(user_message, role=Role.ROLE_USER)
        )

        task: Task | None = None
        try:
            async for response in client.send_message(request):
                if response.HasField("task"):
                    task = response.task
        finally:
            await client.close()
            await http_client.aclose()

        if task is None:
            raise A2AResponseError("远程 Agent 没有返回 A2A Task。")
        return task
