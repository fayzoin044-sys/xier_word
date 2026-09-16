"""Translate A2A requests to business-Agent calls and A2A task events."""

import logging
from typing import Protocol

from a2a.helpers import (
    new_data_part,
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater

from multi_agent_code_reviewer.a2a.common.models import AgentResult

logger = logging.getLogger(__name__)


class BusinessAgent(Protocol):
    async def invoke(self, query: str) -> AgentResult:
        """Run one query and return a transport-independent result."""
        ...


class SpecialistAgentExecutor(AgentExecutor):
    """Expose one existing specialist Agent through the A2A task lifecycle."""

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
        task = context.current_task
        if task is None:
            if context.message is None:
                raise ValueError("A2A request does not contain a message.")
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task.id,
            context_id=task.context_id,
        )
        query = context.get_user_input().strip()
        if not query:
            await updater.requires_input(
                new_text_message(
                    "请提供本机项目目录的绝对路径。",
                    context_id=task.context_id,
                    task_id=task.id,
                )
            )
            return

        await updater.start_work(
            new_text_message(
                "正在执行代码审查任务。",
                context_id=task.context_id,
                task_id=task.id,
            )
        )
        try:
            result = await self._agent.invoke(query)
        except Exception as exc:
            logger.exception("Specialist Agent execution failed")
            await updater.failed(
                new_text_message(
                    f"Agent 执行失败：{exc}",
                    context_id=task.context_id,
                    task_id=task.id,
                )
            )
            return

        if result.requires_input:
            await updater.requires_input(
                new_text_message(
                    result.text,
                    context_id=task.context_id,
                    task_id=task.id,
                )
            )
            return

        parts = [new_text_part(result.text, media_type="text/plain")]
        if result.data:
            parts.append(new_data_part(result.data, media_type="application/json"))

        await updater.add_artifact(
            parts=parts,
            name=self._artifact_name,
            last_chunk=True,
        )
        await updater.complete(
            new_text_message(
                "代码审查任务已完成。",
                context_id=task.context_id,
                task_id=task.id,
            )
        )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        if context.task_id is None or context.context_id is None:
            raise ValueError("Cannot cancel an A2A task without task identifiers.")
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id,
            context_id=context.context_id,
        )
        await updater.cancel(
            new_text_message(
                "任务已取消。",
                context_id=context.context_id,
                task_id=context.task_id,
            )
        )
