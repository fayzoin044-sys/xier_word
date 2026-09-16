"""Internal result models shared by business agents and the A2A executor."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AgentResultState(str, Enum):
    """Business-level states that will later map to A2A TaskState values."""

    COMPLETED = "completed"
    INPUT_REQUIRED = "input_required"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Result returned by a business agent before A2A protocol conversion."""

    state: AgentResultState
    content: str
    artifact_name: str | None = None

    @classmethod
    def completed(cls, content: str, *, artifact_name: str) -> AgentResult:
        """Create a successful result whose content becomes an A2A Artifact."""
        return cls(
            state=AgentResultState.COMPLETED,
            content=content,
            artifact_name=artifact_name,
        )

    @classmethod
    def input_required(cls, content: str) -> AgentResult:
        """Ask the caller to provide missing information."""
        return cls(
            state=AgentResultState.INPUT_REQUIRED,
            content=content,
        )

    @classmethod
    def failed(cls, content: str) -> AgentResult:
        """Create a failed business result."""
        return cls(
            state=AgentResultState.FAILED,
            content=content,
        )
