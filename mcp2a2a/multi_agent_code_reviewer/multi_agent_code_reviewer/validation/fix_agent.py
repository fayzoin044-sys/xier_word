"""Real two-project validation for the Patch-only Fix Agent."""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from a2a.types import TaskState

from multi_agent_code_reviewer.a2a.common.client import (
    A2AClient,
    get_task_data,
    get_task_text,
)
from multi_agent_code_reviewer.agents.fix import run_fix_agent
from multi_agent_code_reviewer.config import (
    QUALITY_A2A_URL,
    SECURITY_A2A_URL,
    TEST_A2A_URL,
)
from multi_agent_code_reviewer.paths import QUALITY_TEST_DEMO, VULNERABLE_DEMO


@dataclass(frozen=True)
class ValidationProject:
    label: str
    path: Path
    required_patch_fragments: tuple[str, ...]


VALIDATION_PROJECTS = [
    ValidationProject(
        label="vulnerable_demo",
        path=VULNERABLE_DEMO,
        required_patch_fragments=(
            '-    subprocess.run(f"nslookup {host}", shell=True, check=False)',
            '+    subprocess.run(["nslookup", host], check=False)',
            '-    query = "SELECT id, username FROM users WHERE username = \'" + username + "\'"',
            "+    cursor.execute(query,",
        ),
    ),
    ValidationProject(
        label="quality_test_demo",
        path=QUALITY_TEST_DEMO,
        required_patch_fragments=(
            "-import os",
            "-    return left + right",
            "+    return left - right",
        ),
    ),
]


async def get_a2a_result(url: str, project_path: Path) -> dict[str, Any]:
    task = await A2AClient(url).send_message(str(project_path.resolve()))
    if task.status.state != TaskState.TASK_STATE_COMPLETED:
        raise RuntimeError(
            f"A2A Agent returned {TaskState.Name(task.status.state)}"
        )
    data_parts = get_task_data(task)
    if len(data_parts) != 1 or not isinstance(data_parts[0], dict):
        raise TypeError("A2A Agent did not return exactly one structured data object")
    return {"text": get_task_text(task), "data": data_parts[0]}


async def validate_project(target: ValidationProject) -> None:
    print(f"=== Collecting real A2A reports: {target.label} ===")
    security_result, quality_result, test_result = await asyncio.gather(
        get_a2a_result(SECURITY_A2A_URL, target.path),
        get_a2a_result(QUALITY_A2A_URL, target.path),
        get_a2a_result(TEST_A2A_URL, target.path),
    )
    fix_result = await run_fix_agent(
        project_path=target.path,
        security_result=security_result,
        quality_result=quality_result,
        test_result=test_result,
    )
    if not fix_result.patch:
        raise RuntimeError(f"{target.label} did not produce a Patch")
    missing_fragments = [
        fragment
        for fragment in target.required_patch_fragments
        if fragment not in fix_result.patch
    ]
    if missing_fragments:
        raise RuntimeError(
            f"{target.label} Patch did not cover all expected report issues: "
            f"{missing_fragments}"
        )

    print(f"\n=== Fix Agent result: {target.label} ===")
    print(
        json.dumps(
            {
                "model": fix_result.model,
                "loaded_tools": fix_result.loaded_tools,
                "read_files": fix_result.read_files,
                "changed_files": fix_result.changed_files,
                "summary": fix_result.summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("\n--- Unified diff Patch ---")
    print(fix_result.patch)


async def main() -> None:
    for target in VALIDATION_PROJECTS:
        await validate_project(target)


if __name__ == "__main__":
    asyncio.run(main())
