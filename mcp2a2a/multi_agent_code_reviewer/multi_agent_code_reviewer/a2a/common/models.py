"""Transport-independent result shared by all A2A specialist services."""

from typing import Any

from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    """Result returned by a business Agent before A2A protocol conversion."""

    text: str = Field(description="Human-readable response for the caller.")
    data: dict[str, Any] = Field(default_factory=dict)
    requires_input: bool = False
