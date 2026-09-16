"""Call each code-review A2A service once and print its final result."""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from a2a.types import TaskState

from multi_agent_code_reviewer.a2a.common.client import (
    A2AClient,
    get_task_data,
    get_task_text,
)
from multi_agent_code_reviewer.config import (
    QUALITY_A2A_URL,
    SECURITY_A2A_URL,
    TEST_A2A_URL,
)
from multi_agent_code_reviewer.paths import QUALITY_TEST_DEMO, VULNERABLE_DEMO


@dataclass(frozen=True)
class SmokeTarget:
    label: str
    url: str
    project_path: Path


TARGETS = [
    SmokeTarget(
        label="security",
        url=SECURITY_A2A_URL,
        project_path=VULNERABLE_DEMO,
    ),
    SmokeTarget(
        label="quality",
        url=QUALITY_A2A_URL,
        project_path=QUALITY_TEST_DEMO,
    ),
    SmokeTarget(
        label="test",
        url=TEST_A2A_URL,
        project_path=QUALITY_TEST_DEMO,
    ),
]


async def call_target(target: SmokeTarget) -> None:
    client = A2AClient(target.url)
    card = await client.get_agent_card()
    task = await client.send_message(str(target.project_path.resolve()))
    text = get_task_text(task)
    data = get_task_data(task)
    if not text.strip() or not data:
        raise RuntimeError(f"{target.label} A2A service returned an empty result")

    print(f"=== {target.label.upper()} A2A ===")
    print(f"URL: {target.url}")
    print(f"AgentCard: {card.name} | {card.description}")
    for skill in card.skills:
        print(f"Skill: {skill.id} | {skill.name} | {skill.description}")
    print(f"Task state: {TaskState.Name(task.status.state)}")
    print("Text result:")
    print(text)
    print("Structured data:")
    print(json.dumps(data, ensure_ascii=False, indent=2))


async def main() -> None:
    for target in TARGETS:
        await call_target(target)


if __name__ == "__main__":
    asyncio.run(main())
