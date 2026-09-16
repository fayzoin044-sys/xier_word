"""Thin adapters from existing review functions to the common AgentResult."""

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from multi_agent_code_reviewer.a2a.common.models import AgentResult

ReviewRunner = Callable[[Path], Awaitable[dict[str, Any]]]


class ExistingReviewAgentAdapter:
    """Expose an existing run_*_review function without changing its logic."""

    def __init__(
        self,
        *,
        agent_name: str,
        runner: ReviewRunner,
        tool_calls_key: str,
    ) -> None:
        self._agent_name = agent_name
        self._runner = runner
        self._tool_calls_key = tool_calls_key

    async def invoke(self, query: str) -> AgentResult:
        raw_path = query.strip().strip('"')
        if not raw_path:
            return AgentResult(
                text="请直接提供本机项目目录的绝对路径。",
                requires_input=True,
            )

        project = Path(raw_path).expanduser().resolve()
        if not project.is_dir():
            return AgentResult(
                text=f"项目目录不存在，请重新提供有效的绝对路径：{project}",
                requires_input=True,
            )

        review = await self._runner(project)
        final_answer = review["final_answer"]
        if not isinstance(final_answer, str):
            final_answer = json.dumps(final_answer, ensure_ascii=False)

        return AgentResult(
            text=final_answer,
            data={
                "agent": self._agent_name,
                "model": review["model"],
                "loaded_tools": review["loaded_tools"],
                "tool_calls": review[self._tool_calls_key],
                "tool_result": review["tool_result"],
            },
        )
