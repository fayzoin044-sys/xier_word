"""Collect real specialist reports and call the Fix Agent A2A service."""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from a2a.types import TaskState

from multi_agent_code_reviewer.a2a.common.client import (
    A2AClient,
    get_task_data,
    get_task_text,
)
from multi_agent_code_reviewer.config import (
    FIX_A2A_URL,
    QUALITY_A2A_URL,
    SECURITY_A2A_URL,
    TEST_A2A_URL,
)
from multi_agent_code_reviewer.paths import QUALITY_TEST_DEMO


async def get_review(url: str, project: Path) -> dict[str, Any]:
    task = await A2AClient(url).send_message(str(project))
    if task.status.state != TaskState.TASK_STATE_COMPLETED:
        raise RuntimeError(f"Specialist A2A task failed: {TaskState.Name(task.status.state)}")
    data_parts = get_task_data(task)
    if len(data_parts) != 1 or not isinstance(data_parts[0], dict):
        raise TypeError("Specialist A2A service did not return one data object")
    return {"text": get_task_text(task), "data": data_parts[0]}


async def call_fix_service(project: Path) -> None:
    project = project.resolve()
    security_result, quality_result, test_result = await asyncio.gather(
        get_review(SECURITY_A2A_URL, project),
        get_review(QUALITY_A2A_URL, project),
        get_review(TEST_A2A_URL, project),
    )
    payload = {
        "project_path": str(project),
        "security_result": security_result,
        "quality_result": quality_result,
        "test_result": test_result,
    }

    client = A2AClient(FIX_A2A_URL)
    card = await client.get_agent_card()
    task = await client.send_message(json.dumps(payload, ensure_ascii=False))
    if task.status.state != TaskState.TASK_STATE_COMPLETED:
        raise RuntimeError(f"Fix A2A task failed: {TaskState.Name(task.status.state)}")

    data_parts = get_task_data(task)
    if len(data_parts) != 1 or not isinstance(data_parts[0], dict):
        raise TypeError("Fix A2A service did not return one data object")
    data = data_parts[0]
    if data.get("agent") != "FixAgent" or not data.get("patch"):
        raise RuntimeError("Fix A2A service returned an incomplete result")
    if any("test" in path.lower() for path in data.get("changed_files", [])):
        raise RuntimeError("Fix A2A service attempted to modify a test file")

    print(f"AgentCard: {card.name} | {card.description}")
    for skill in card.skills:
        print(f"Skill: {skill.id} | {skill.name} | {skill.description}")
    print(f"Task state: {TaskState.Name(task.status.state)}")
    print("Text result:")
    print(get_task_text(task))
    print("Structured data:")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call the Fix Agent A2A service.")
    parser.add_argument(
        "project_path",
        nargs="?",
        type=Path,
        default=QUALITY_TEST_DEMO,
    )
    return parser.parse_args()


async def main(project_path: Path) -> None:
    await call_fix_service(project_path)


if __name__ == "__main__":
    asyncio.run(main(parse_args().project_path))
