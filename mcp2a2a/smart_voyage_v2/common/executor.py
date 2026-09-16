"""
收到 A2A Request
→ 创建/取得 Task
→ 从 Message 提取用户问题
→ 标记 WORKING
→ 调用业务 Agent.invoke(query)
→ 得到 AgentResult
→ 写入 Artifact
→ 标记 COMPLETED

就是：
1. 收到 A2A 请求
2. 提取用户问题
3. 调用业务 Agent
4. 把结果放进 Artifact
5. 标记任务完成

再详细一些
1. 收到 A2A 请求

2. 获取现有 Task
   没有 Task 就根据本次 Message 创建一个

3. 从本次请求的 Message 中提取用户问题

4. 通过 TaskUpdater 标记 WORKING

5. 调用业务 Agent
   Agent 返回统一的 AgentResult

6. Executor 把 AgentResult 转成 Artifact

7. 通过 TaskUpdater 标记 COMPLETED

8. A2A Server 将最终 Task 返回客户端

"""

from __future__ import annotations

"""Translate A2A requests to business-agent calls and back to A2A events."""

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

from smart_voyage_v2.common.models import AgentResult


logger = logging.getLogger(__name__)


class BusinessAgent(Protocol):#TODO 这里是我们定义的规范化的类，agent输出规范化
    """The small interface required by the A2A executor."""

    async def invoke(self, query: str) -> AgentResult:
        """Run one user query and return a transport-independent result."""
        ...

#TODO AgentExecutor这个类表示，A2A SDK 规定的执行器基类。自定义 Executor 必须实现execute()和cancel()
class SpecialistAgentExecutor(AgentExecutor):
    """Expose one specialist business agent through the A2A task lifecycle."""

    def __init__(
        self,
        agent: BusinessAgent,#TODO 传入我们的智能体和我们智能体的名字
        *,
        artifact_name: str = "agent_result",
    ) -> None:
        self._agent = agent
        self._artifact_name = artifact_name
#TODO 这个函数内代码需要看懂
    async def execute(
        self,
        context: RequestContext,#TODO 代表本次请求的上下文
        event_queue: EventQueue,#TODO Executor 不直接向客户端返回结果，而是把 Task 和更新事件放进事件队列
    ) -> None:
        task = context.current_task#TODO 取task
        if task is None:#TODO  如果没有就根据当前messages创建task
            if context.message is None:
                raise ValueError("A2A request does not contain a message.")
            task = new_task_from_user_message(context.message)#TODO 这个代表创建task
            await event_queue.enqueue_event(task)#TODO 不直接向客户端返回结果，先把task和事件更新放到事件队列
#TODO 这里表示更新task,后面的可以这样去看
# await updater.start_work()  # WORKING
# await updater.add_artifact(...)  # 添加结果
# await updater.complete()  # COMPLETED
# await updater.requires_input(...)  # INPUT_REQUIRED
# await updater.failed(...)  # FAILED
# await updater.cancel(...)  # CANCELED
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task.id,
            context_id=task.context_id,
        )

        query = context.get_user_input().strip()
        if not query:
            await updater.requires_input(
                new_text_message(
                    "请提供需要处理的问题。",
                    context_id=task.context_id,
                    task_id=task.id,
                )
            )
            return

        await updater.start_work(
            new_text_message(
                "正在处理请求。",
                context_id=task.context_id,
                task_id=task.id,
            )
        )

        try:
            result = await self._agent.invoke(query)#TODO 这里调用agent
        except Exception as exc:
            logger.exception("Specialist agent execution failed")
            await updater.failed(
                new_text_message(
                    f"业务 Agent 执行失败：{exc}",
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
                "请求处理完成。",
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
