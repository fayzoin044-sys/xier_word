"""Real Reject and Approve validation for the resumable LangGraph workflow."""

import asyncio
import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from multi_agent_code_reviewer.paths import QUALITY_TEST_DEMO
from multi_agent_code_reviewer.workflows.review import (
    ReviewState,
    build_review_workflow,
    create_initial_state,
    get_tool_result,
    snapshot_project_files,
)

SOURCE_DEMO = QUALITY_TEST_DEMO


def copy_demo(temporary_root: Path) -> Path:
    """Create one disposable project without tool-generated caches."""
    workspace = temporary_root / "quality_test_demo"
    shutil.copytree(
        SOURCE_DEMO,
        workspace,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".semgrep"),
    )
    return workspace


def get_interrupt_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Require one Patch approval interrupt and return its public value."""
    requests = result.get("__interrupt__")
    if not isinstance(requests, (list, tuple)) or len(requests) != 1:
        raise RuntimeError(f"Expected exactly one HITL interrupt, received: {requests}")
    payload = requests[0].value
    if not isinstance(payload, dict) or payload.get("type") != "patch_approval":
        raise TypeError(f"Unexpected HITL payload: {payload}")
    for field_name in ("summary", "changed_files", "patch"):
        if not payload.get(field_name):
            raise RuntimeError(f"HITL payload is missing {field_name}")
    return payload


def assert_initial_failures(state: ReviewState) -> None:
    """Confirm the copied demo starts with the intended Ruff and Pytest failures."""
    quality = get_tool_result(state["quality_result"], branch_name="quality_result")
    tests = get_tool_result(state["test_result"], branch_name="test_result")
    if quality.get("findings_count") != 1:
        raise RuntimeError(f"Expected one initial Ruff finding: {quality}")
    summary = tests.get("summary")
    if not isinstance(summary, dict) or summary.get("passed") != 1:
        raise RuntimeError(f"Expected one initially passing test: {tests}")
    if summary.get("failed") != 1 or tests.get("exit_code") != 1:
        raise RuntimeError(f"Expected one initially failing test: {tests}")


def assert_verified_pass(state: ReviewState) -> None:
    """Confirm the post-apply A2A verification is fully clean."""
    security = get_tool_result(
        state["security_result"],
        branch_name="security_result",
    )
    quality = get_tool_result(state["quality_result"], branch_name="quality_result")
    tests = get_tool_result(state["test_result"], branch_name="test_result")
    summary = tests.get("summary")
    if security.get("findings_count") != 0:
        raise RuntimeError(f"Security verification failed: {security}")
    if quality.get("findings_count") != 0:
        raise RuntimeError(f"Quality verification failed: {quality}")
    if tests.get("exit_code") != 0 or not isinstance(summary, dict):
        raise RuntimeError(f"Test verification failed: {tests}")
    if summary.get("passed") != 2 or summary.get("failed") != 0:
        raise RuntimeError(f"Expected two passing tests after apply: {tests}")


async def start_until_hitl(
    workspace: Path,
    *,
    thread_id: str,
) -> tuple[Any, dict[str, Any], ReviewState]:
    """Run one fresh graph until its first Patch approval interrupt."""
    workflow = build_review_workflow(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": thread_id}}
    result = await workflow.ainvoke(create_initial_state(workspace), config=config)
    get_interrupt_payload(result)
    snapshot = await workflow.aget_state(config)
    state = snapshot.values
    if state.get("approval") != "PENDING" or state.get("final_status") != "RUNNING":
        raise RuntimeError(f"Unexpected State at HITL: {state}")
    if "hitl_node" not in snapshot.next:
        raise RuntimeError(f"Checkpoint is not paused at hitl_node: {snapshot.next}")
    assert_initial_failures(state)
    return workflow, config, state


async def validate_reject() -> dict[str, Any]:
    """Reject the Patch and prove that no copied project file changed."""
    temporary_path: Path | None = None
    with tempfile.TemporaryDirectory(prefix="review-reject-") as temporary:
        temporary_path = Path(temporary)
        workspace = copy_demo(temporary_path)
        before = snapshot_project_files(workspace)
        thread_id = f"reject-{uuid.uuid4().hex}"
        workflow, config, paused_state = await start_until_hitl(
            workspace,
            thread_id=thread_id,
        )
        snapshot_at_interrupt = snapshot_project_files(workspace)
        if snapshot_at_interrupt != before:
            raise RuntimeError("Reject workspace changed before human approval")

        final_state = await workflow.ainvoke(Command(resume="REJECT"), config=config)
        if "__interrupt__" in final_state:
            raise RuntimeError("Reject flow unexpectedly paused again")
        if final_state.get("approval") != "REJECT":
            raise RuntimeError(f"Reject approval was not recorded: {final_state}")
        if final_state.get("final_status") != "REJECTED":
            raise RuntimeError(f"Reject final_status is wrong: {final_state}")
        if snapshot_project_files(workspace) != before:
            raise RuntimeError("Reject flow modified copied project files")

        result = {
            "thread_id": thread_id,
            "paused": True,
            "interrupt_summary": paused_state["patch_result"]["data"]["summary"],
            "approval": final_state["approval"],
            "final_status": final_state["final_status"],
            "sha256_unchanged": True,
            "retry_count": final_state["retry_count"],
        }
    if temporary_path is None or temporary_path.exists():
        raise RuntimeError("Reject temporary workspace was not deleted")
    result["temporary_workspace_deleted"] = True
    return result


async def validate_approve() -> dict[str, Any]:
    """Approve, apply, and verify the Patch in a disposable project copy."""
    temporary_path: Path | None = None
    with tempfile.TemporaryDirectory(prefix="review-approve-") as temporary:
        temporary_path = Path(temporary)
        workspace = copy_demo(temporary_path)
        before = snapshot_project_files(workspace)
        thread_id = f"approve-{uuid.uuid4().hex}"
        workflow, config, paused_state = await start_until_hitl(
            workspace,
            thread_id=thread_id,
        )

        approval_count = 1
        final_state = await workflow.ainvoke(Command(resume="APPROVE"), config=config)
        while "__interrupt__" in final_state:
            payload = get_interrupt_payload(final_state)
            approval_count += 1
            print("\n=== Retry HITL payload ===")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            final_state = await workflow.ainvoke(
                Command(resume="APPROVE"),
                config=config,
            )

        if final_state.get("decision") != "PASS":
            raise RuntimeError(f"Approve flow decision is not PASS: {final_state}")
        if final_state.get("final_status") != "PASS":
            raise RuntimeError(f"Approve flow final_status is not PASS: {final_state}")
        apply_result = final_state.get("apply_result")
        if not isinstance(apply_result, dict) or apply_result.get("success") is not True:
            raise RuntimeError(f"Patch was not applied successfully: {apply_result}")
        if apply_result.get("changed_files") != ["calculator.py"]:
            raise RuntimeError(f"Unexpected applied files: {apply_result}")
        if snapshot_project_files(workspace) == before:
            raise RuntimeError("Approve flow did not change the temporary workspace")

        calculator = (workspace / "calculator.py").read_text(encoding="utf-8")
        if "import os" in calculator or "return left - right" not in calculator:
            raise RuntimeError("Applied calculator.py content is not repaired")
        assert_verified_pass(final_state)

        result = {
            "thread_id": thread_id,
            "initial_quality_findings": get_tool_result(
                paused_state["quality_result"],
                branch_name="quality_result",
            )["findings_count"],
            "initial_tests": {
                **get_tool_result(
                    paused_state["test_result"],
                    branch_name="test_result",
                )["summary"],
                "exit_code": get_tool_result(
                    paused_state["test_result"],
                    branch_name="test_result",
                )["exit_code"],
            },
            "approval": final_state["approval"],
            "apply_result": apply_result,
            "verified_security_findings": 0,
            "verified_quality_findings": 0,
            "verified_tests": {"passed": 2, "failed": 0, "exit_code": 0},
            "decision": final_state["decision"],
            "final_status": final_state["final_status"],
            "retry_count": final_state["retry_count"],
            "retry_route_observed": approval_count > 1,
        }
    if temporary_path is None or temporary_path.exists():
        raise RuntimeError("Approve temporary workspace was not deleted")
    result["temporary_workspace_deleted"] = True
    return result


async def main() -> None:
    original_before = snapshot_project_files(SOURCE_DEMO)
    reject_result = await validate_reject()
    approve_result = await validate_approve()
    if snapshot_project_files(SOURCE_DEMO) != original_before:
        raise RuntimeError("Original quality_test_demo was modified")

    print("\n=== Reject validation ===")
    print(json.dumps(reject_result, ensure_ascii=False, indent=2))
    print("\n=== Approve validation ===")
    print(json.dumps(approve_result, ensure_ascii=False, indent=2))
    print("\nOriginal quality_test_demo SHA-256: unchanged")


if __name__ == "__main__":
    asyncio.run(main())
