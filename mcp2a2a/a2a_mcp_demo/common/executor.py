"""Translate A2A tasks to business-agent calls and back to A2A events."""

from __future__ import annotations

import logging
from typing import Protocol

from a2a.helpers import new_task_from_user_message, new_text_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState


logger = logging.getLogger(__name__)


class BusinessAgent(Protocol):
    """Interface implemented by WeatherAgent and RAGAgent."""

    async def run(self, user_message: str) -> str:
        """Process one user message and return the final text response."""
        ...


class BusinessAgentExecutor(AgentExecutor):
    """Adapt an async business agent to the A2A Task lifecycle."""

    def __init__(
        self,
        agent: BusinessAgent,
        *,
        artifact_name: str = "agent_result",
    ) -> None:
        self._agent = agent
        self._artifact_name = artifact_name

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute a business agent and publish its result as an A2A artifact."""
        task = context.current_task
        if task is None:
            if context.message is None:
                raise ValueError("A2A 请求中缺少 message")
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task.id,
            context_id=task.context_id,
        )

        user_message = context.get_user_input().strip()
        if not user_message:
            await updater.failed(
                new_text_message(
                    "请求中没有可处理的文本内容",
                    context_id=task.context_id,
                    task_id=task.id,
                )
            )
            return

        await updater.update_status(
            TaskState.TASK_STATE_WORKING,
            new_text_message(
                "正在处理请求",
                context_id=task.context_id,
                task_id=task.id,
            ),
        )

        try:
            result = await self._agent.run(user_message)
        except Exception as exc:
            logger.exception("Business agent execution failed")
            await updater.failed(
                new_text_message(
                    f"业务 Agent 执行失败：{exc}",
                    context_id=task.context_id,
                    task_id=task.id,
                )
            )
            return

        await updater.add_artifact(
            parts=[new_text_part(str(result), media_type="text/plain")],
            name=self._artifact_name,
            last_chunk=True,
        )
        await updater.complete(
            new_text_message(
                "请求处理完成",
                context_id=task.context_id,
                task_id=task.id,
            )
        )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Publish a canceled terminal state for the current A2A task."""
        task_id = context.task_id
        context_id = context.context_id
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task_id,
            context_id=context_id,
        )
        await updater.cancel(
            new_text_message(
                "任务已取消",
                context_id=context_id,
                task_id=task_id,
            )
        )
