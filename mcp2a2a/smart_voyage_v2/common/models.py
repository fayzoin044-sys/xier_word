"""Business-layer result types shared by all specialist agents."""

from __future__ import annotations

"""
不同业务 Agent 的输出
        ↓ 统一格式
AgentResult(到这里)
        ↓ Executor 转换
A2A Task / Artifact
"""
from typing import Any

from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    """A transport-independent result returned by a specialist agent.

    The business agent only describes its result here. The A2A executor is
    responsible for translating it into an A2A Message, Artifact and Task
    state later.
    """

    text: str = Field(description="Human-readable response for the caller.")
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured business data returned with the text.",
    )
    requires_input: bool = Field(
        default=False,
        description="Whether the caller must provide more information.",
    )
