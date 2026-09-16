"""First LangGraph workflow for parallel A2A code review orchestration."""

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Literal, TypedDict

from a2a.types import TaskState
from langchain.agents import create_agent
from langchain.messages import AIMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

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

DEMO_PROJECT = QUALITY_TEST_DEMO

Decision = Literal["PASS", "NEEDS_FIX"]
Approval = Literal["PENDING", "APPROVE", "REJECT"]
FinalStatus = Literal[
    "RUNNING",
    "PASS",
    "REJECTED",
    "APPLY_FAILED",
    "FAILED_AFTER_MAX_RETRIES",
]
OrchestratorRoute = Literal["PASS", "FIX", "RETRY"]


class ReviewAgentResult(TypedDict):
    """State-safe subset of an A2A response."""

    text: str
    data: dict[str, Any]


class ApplyResult(TypedDict):
    """Deterministic result of applying one approved Patch."""

    success: bool
    error: str
    changed_files: list[str]


class ReviewState(TypedDict, total=False):
    project_path: str
    security_result: ReviewAgentResult
    quality_result: ReviewAgentResult
    test_result: ReviewAgentResult
    patch_result: ReviewAgentResult
    decision: Decision
    summary: str
    approval: Approval
    retry_count: int
    max_retries: int
    apply_result: ApplyResult
    final_status: FinalStatus


ORCHESTRATOR_SYSTEM_PROMPT = """你是 multi_agent_code_reviewer 的 Orchestrator Agent。

你的唯一职责是根据输入中的 Security、Quality、Test 三路结果生成中文汇总和 key issues。

规则：
1. decision 已由程序规则确定，你不得重新判断、修改或质疑 decision。
2. 只能依据三路结果中的 text 和 data 进行汇总，不得补充任何结果中没有的问题。
3. 清楚说明 Security finding 数量、Quality finding 数量和测试通过/失败情况。
4. key issues 必须逐项对应三路工具实际返回的问题；没有问题的分支要明确说明未返回问题。
5. 输出一段可直接交付用户的中文纯文本 summary，不输出 JSON，不调用工具，不修改代码。
"""

_orchestrator_agent: Any | None = None


def get_orchestrator_agent() -> Any:
    """Lazily create the summary-only Ollama Agent."""
    global _orchestrator_agent
    if _orchestrator_agent is None:
        model_name = os.getenv("ORCHESTRATOR_MODEL", "qwen2.5:latest")
        model = ChatOllama(
            model=model_name,
            temperature=0,
            validate_model_on_init=True,
        )
        _orchestrator_agent = create_agent(
            model=model,
            tools=[],
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
            name="review_orchestrator_agent",
        )
    return _orchestrator_agent


async def call_a2a_agent(
    *,
    node_name: str,
    target_url: str,
    project_path: str,
) -> ReviewAgentResult:
    """Call one remote A2A Agent and retain only text plus structured data."""
    print(f"[{node_name}] START -> {target_url} | project={project_path}")
    task = await A2AClient(target_url).send_message(project_path)
    if task.status.state != TaskState.TASK_STATE_COMPLETED:
        state_name = TaskState.Name(task.status.state)
        raise RuntimeError(f"{node_name} returned non-completed Task: {state_name}")

    data_parts = get_task_data(task)
    if len(data_parts) != 1 or not isinstance(data_parts[0], dict):
        raise TypeError(
            f"{node_name} must return exactly one structured A2A data object"
        )

    result = ReviewAgentResult(
        text=get_task_text(task),
        data=data_parts[0],
    )
    print(f"[{node_name}] END -> {TaskState.Name(task.status.state)}")
    return result


async def security_node(state: ReviewState) -> dict[str, ReviewAgentResult]:
    """Call Security Agent through A2A and update only security_result."""
    result = await call_a2a_agent(
        node_name="security_node",
        target_url=SECURITY_A2A_URL,
        project_path=state["project_path"],
    )
    return {"security_result": result}


async def quality_node(state: ReviewState) -> dict[str, ReviewAgentResult]:
    """Call Quality Agent through A2A and update only quality_result."""
    result = await call_a2a_agent(
        node_name="quality_node",
        target_url=QUALITY_A2A_URL,
        project_path=state["project_path"],
    )
    return {"quality_result": result}


async def test_node(state: ReviewState) -> dict[str, ReviewAgentResult]:
    """Call Test Agent through A2A and update only test_result."""
    result = await call_a2a_agent(
        node_name="test_node",
        target_url=TEST_A2A_URL,
        project_path=state["project_path"],
    )
    return {"test_result": result}


async def fix_node(state: ReviewState) -> dict[str, Any]:
    """Call Fix Agent through A2A and prepare a fresh approval cycle."""
    payload = {
        "project_path": state["project_path"],
        "security_result": state["security_result"],
        "quality_result": state["quality_result"],
        "test_result": state["test_result"],
    }
    print(f"[fix_node] START -> {FIX_A2A_URL} | decision={state['decision']}")
    task = await A2AClient(FIX_A2A_URL).send_message(
        json.dumps(payload, ensure_ascii=False)
    )
    if task.status.state != TaskState.TASK_STATE_COMPLETED:
        state_name = TaskState.Name(task.status.state)
        raise RuntimeError(f"fix_node returned non-completed Task: {state_name}")

    data_parts = get_task_data(task)
    if len(data_parts) != 1 or not isinstance(data_parts[0], dict):
        raise TypeError("fix_node must return exactly one structured A2A data object")
    result = ReviewAgentResult(
        text=get_task_text(task),
        data=data_parts[0],
    )
    print(f"[fix_node] END -> {TaskState.Name(task.status.state)}")
    return {
        "patch_result": result,
        "approval": "PENDING",
        "apply_result": {
            "success": False,
            "error": "",
            "changed_files": [],
        },
        "final_status": "RUNNING",
    }


def get_patch_data(state: ReviewState) -> dict[str, Any]:
    """Extract the machine-readable Fix A2A result from State."""
    patch_result = state.get("patch_result")
    if not isinstance(patch_result, dict):
        raise TypeError("patch_result is missing from ReviewState")
    data = patch_result.get("data")
    if not isinstance(data, dict):
        raise TypeError("patch_result has no structured A2A data object")
    return data


def hitl_node(state: ReviewState) -> dict[str, str]:
    """Pause for explicit approval of the current Patch."""
    patch_data = get_patch_data(state)
    approval_value = interrupt(
        {
            "type": "patch_approval",
            "summary": patch_data.get("summary", ""),
            "changed_files": patch_data.get("changed_files", []),
            "patch": patch_data.get("patch", ""),
            "retry_count": state["retry_count"],
            "max_retries": state["max_retries"],
            "allowed_responses": ["APPROVE", "REJECT"],
        }
    )
    if isinstance(approval_value, dict):
        approval_value = approval_value.get("approval")
    if not isinstance(approval_value, str):
        raise TypeError("HITL resume value must be APPROVE or REJECT")

    approval = approval_value.strip().upper()
    if approval == "REJECT":
        print("[hitl_node] REJECT -> no project files will be modified")
        return {
            "approval": "REJECT",
            "final_status": "REJECTED",
        }
    if approval == "APPROVE":
        print("[hitl_node] APPROVE -> continue to deterministic Patch apply")
        return {
            "approval": "APPROVE",
            "final_status": "RUNNING",
        }
    raise ValueError("HITL resume value must be APPROVE or REJECT")


def parse_patch_files(patch: str) -> list[str]:
    """Extract matching existing-file headers from one unified diff."""
    old_paths = re.findall(r"(?m)^--- a/(.+)$", patch)
    new_paths = re.findall(r"(?m)^\+\+\+ b/(.+)$", patch)
    if not old_paths or old_paths != new_paths or "@@" not in patch:
        raise ValueError("Patch must modify existing files with matching a/ and b/ headers")
    if len(old_paths) != len(set(old_paths)):
        raise ValueError("Patch contains duplicate file headers")
    for relative_path in old_paths:
        path = PurePosixPath(relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Patch path escapes project: {relative_path}")
    return old_paths


def snapshot_project_files(project: Path) -> dict[str, str]:
    """Hash project files while ignoring tool-generated caches."""
    ignored_parts = {".pytest_cache", "__pycache__", ".semgrep"}
    snapshot: dict[str, str] = {}
    for path in sorted(project.rglob("*")):
        if not path.is_file() or set(path.parts) & ignored_parts:
            continue
        relative_path = path.relative_to(project).as_posix()
        snapshot[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def run_git_apply(
    *,
    project: Path,
    patch: str,
    check_only: bool,
) -> subprocess.CompletedProcess[str]:
    """Run git apply with the Patch supplied through stdin."""
    command = ["git", "apply"]
    if check_only:
        command.append("--check")
    command.extend(["--recount", "--ignore-space-change", "-"])
    return subprocess.run(
        command,
        cwd=str(project),
        input=patch,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        check=False,
        timeout=30,
    )


def apply_patch_node(state: ReviewState) -> dict[str, Any]:
    """Apply only an explicitly approved Patch and record exact file changes."""
    if state.get("approval") != "APPROVE":
        raise ValueError("apply_patch_node requires approval=APPROVE")

    project = Path(state["project_path"]).expanduser().resolve()
    try:
        if not project.is_dir():
            raise ValueError(f"project_path is not a directory: {project}")
        patch_data = get_patch_data(state)
        patch = patch_data.get("patch")
        declared_files = patch_data.get("changed_files")
        if not isinstance(patch, str) or not patch.strip():
            raise ValueError("Approved patch_result.patch is empty")
        if not isinstance(declared_files, list) or not all(
            isinstance(path, str) for path in declared_files
        ):
            raise TypeError("patch_result.changed_files must be a string list")

        patch_files = parse_patch_files(patch)
        if patch_files != declared_files:
            raise ValueError(
                "Patch headers do not match patch_result.changed_files: "
                f"headers={patch_files}, declared={declared_files}"
            )
        for relative_path in patch_files:
            target = (project / Path(*PurePosixPath(relative_path).parts)).resolve()
            try:
                target.relative_to(project)
            except ValueError as error:
                raise ValueError(
                    f"Patch target escapes project: {relative_path}"
                ) from error
            if not target.is_file():
                raise ValueError(f"Patch target is not an existing file: {relative_path}")

        before_snapshot = snapshot_project_files(project)
        check_result = run_git_apply(project=project, patch=patch, check_only=True)
        if check_result.returncode != 0:
            raise RuntimeError(
                "git apply --check failed: "
                + (check_result.stderr.strip() or check_result.stdout.strip())
            )

        apply_result = run_git_apply(project=project, patch=patch, check_only=False)
        if apply_result.returncode != 0:
            raise RuntimeError(
                "git apply failed: "
                + (apply_result.stderr.strip() or apply_result.stdout.strip())
            )

        after_snapshot = snapshot_project_files(project)
        actual_changed_files = sorted(
            path
            for path in set(before_snapshot) | set(after_snapshot)
            if before_snapshot.get(path) != after_snapshot.get(path)
        )
        if actual_changed_files != sorted(patch_files):
            raise RuntimeError(
                "Applied Patch changed unexpected files: "
                f"expected={sorted(patch_files)}, actual={actual_changed_files}"
            )

        print(f"[apply_patch_node] SUCCESS -> changed={actual_changed_files}")
        return {
            "apply_result": {
                "success": True,
                "error": "",
                "changed_files": actual_changed_files,
            },
            "final_status": "RUNNING",
        }
    except (
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        TypeError,
        ValueError,
    ) as error:
        print(f"[apply_patch_node] FAILED -> {error}")
        return {
            "apply_result": {
                "success": False,
                "error": str(error),
                "changed_files": [],
            },
            "final_status": "APPLY_FAILED",
        }


def verify_start_node(state: ReviewState) -> dict[str, str]:
    """Fan out to the unchanged specialist nodes after a successful apply."""
    print(
        "[verify_start_node] START -> rerun Security / Quality / Test | "
        f"retry_count={state['retry_count']}"
    )
    return {"final_status": "RUNNING"}


def get_tool_result(result: ReviewAgentResult, *, branch_name: str) -> dict[str, Any]:
    """Extract one branch's machine-readable MCP tool result."""
    tool_result = result["data"].get("tool_result")
    if not isinstance(tool_result, dict):
        raise TypeError(f"{branch_name} result has no structured tool_result")
    return tool_result


def require_int(value: object, *, field_name: str) -> int:
    """Reject missing or non-integer policy inputs."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    return value


def calculate_decision(
    security_result: ReviewAgentResult,
    quality_result: ReviewAgentResult,
    test_result: ReviewAgentResult,
) -> Decision:
    """Apply the deterministic first-version review policy."""
    security_tool_result = get_tool_result(
        security_result,
        branch_name="security_result",
    )
    quality_tool_result = get_tool_result(
        quality_result,
        branch_name="quality_result",
    )
    test_tool_result = get_tool_result(test_result, branch_name="test_result")

    security_findings = require_int(
        security_tool_result.get("findings_count"),
        field_name="security_result.findings_count",
    )
    quality_findings = require_int(
        quality_tool_result.get("findings_count"),
        field_name="quality_result.findings_count",
    )
    test_exit_code = require_int(
        test_tool_result.get("exit_code"),
        field_name="test_result.exit_code",
    )

    if security_findings > 0 or quality_findings > 0 or test_exit_code != 0:
        return "NEEDS_FIX"
    return "PASS"


async def orchestrator_node(state: ReviewState) -> dict[str, str]:
    """Calculate decision in code and let Ollama write only the summary."""
    print("[orchestrator_node] START -> all three parallel results received")
    security_result = state["security_result"]
    quality_result = state["quality_result"]
    test_result = state["test_result"]
    decision = calculate_decision(
        security_result,
        quality_result,
        test_result,
    )
    print(f"[orchestrator_node] deterministic decision={decision}")

    summary_input = {
        "fixed_decision": decision,
        "security_result": security_result,
        "quality_result": quality_result,
        "test_result": test_result,
    }
    agent_result = await get_orchestrator_agent().ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "下面是三路代码审查结果和程序已经确定的 decision。"
                        "请只生成 summary 和 key issues，不要重新决策。\n"
                        + json.dumps(summary_input, ensure_ascii=False, indent=2)
                    ),
                }
            ]
        }
    )
    final_message = agent_result["messages"][-1]
    if not isinstance(final_message, AIMessage):
        raise TypeError("Orchestrator Agent did not produce a final AI response")
    summary = final_message.content
    if not isinstance(summary, str) or not summary.strip():
        raise TypeError("Orchestrator Agent summary must be a non-empty string")

    print("[orchestrator_node] END -> summary generated")
    return {"decision": decision, "summary": summary}


def route_after_orchestrator(state: ReviewState) -> OrchestratorRoute:
    """Route initial findings to Fix and verification findings to Retry."""
    decision = state["decision"]
    if decision not in ("PASS", "NEEDS_FIX"):
        raise ValueError(f"Unsupported workflow decision: {decision}")
    if decision == "PASS":
        return "PASS"

    apply_result = state.get("apply_result", {})
    if state.get("approval") == "APPROVE" and apply_result.get("success") is True:
        return "RETRY"
    return "FIX"


def pass_node(state: ReviewState) -> dict[str, str]:
    """Mark either an initial or post-apply clean review as successful."""
    print(f"[pass_node] PASS -> retry_count={state['retry_count']}")
    return {"final_status": "PASS"}


def retry_node(state: ReviewState) -> dict[str, Any]:
    """Increment the retry counter or stop at the configured limit."""
    retry_count = require_int(state.get("retry_count"), field_name="retry_count")
    max_retries = require_int(state.get("max_retries"), field_name="max_retries")
    if retry_count < max_retries:
        next_retry = retry_count + 1
        print(f"[retry_node] RETRY -> {next_retry}/{max_retries}")
        return {
            "retry_count": next_retry,
            "approval": "PENDING",
            "final_status": "RUNNING",
        }

    print(f"[retry_node] FAILED -> retry limit reached ({max_retries})")
    return {"final_status": "FAILED_AFTER_MAX_RETRIES"}


def route_after_hitl(state: ReviewState) -> Literal["APPROVE", "REJECT"]:
    """Route the explicit human decision."""
    approval = state["approval"]
    if approval not in ("APPROVE", "REJECT"):
        raise ValueError(f"Unsupported approval: {approval}")
    return approval


def route_after_apply(state: ReviewState) -> Literal["VERIFY", "APPLY_FAILED"]:
    """Verify only successfully applied Patches."""
    apply_result = state["apply_result"]
    return "VERIFY" if apply_result["success"] else "APPLY_FAILED"


def route_after_retry(state: ReviewState) -> Literal["FIX", "FAILED"]:
    """Generate a new Patch only while retry capacity remains."""
    if state["final_status"] == "FAILED_AFTER_MAX_RETRIES":
        return "FAILED"
    return "FIX"


def build_review_workflow(*, checkpointer: Any | None = None) -> Any:
    """Build review, Patch approval, apply, verify, and bounded retry."""
    graph = StateGraph(ReviewState)
    graph.add_node("security_node", security_node)
    graph.add_node("quality_node", quality_node)
    graph.add_node("test_node", test_node)
    graph.add_node("orchestrator_node", orchestrator_node)
    graph.add_node("fix_node", fix_node)
    graph.add_node("hitl_node", hitl_node)
    graph.add_node("apply_patch_node", apply_patch_node)
    graph.add_node("verify_start_node", verify_start_node)
    graph.add_node("pass_node", pass_node)
    graph.add_node("retry_node", retry_node)

    graph.add_edge(START, "security_node")
    graph.add_edge(START, "quality_node")
    graph.add_edge(START, "test_node")
    graph.add_edge(
        ["security_node", "quality_node", "test_node"],
        "orchestrator_node",
    )
    graph.add_conditional_edges(
        "orchestrator_node",
        route_after_orchestrator,
        {
            "PASS": "pass_node",
            "FIX": "fix_node",
            "RETRY": "retry_node",
        },
    )
    graph.add_edge("pass_node", END)
    graph.add_edge("fix_node", "hitl_node")
    graph.add_conditional_edges(
        "hitl_node",
        route_after_hitl,
        {
            "APPROVE": "apply_patch_node",
            "REJECT": END,
        },
    )
    graph.add_conditional_edges(
        "apply_patch_node",
        route_after_apply,
        {
            "VERIFY": "verify_start_node",
            "APPLY_FAILED": END,
        },
    )
    graph.add_edge("verify_start_node", "security_node")
    graph.add_edge("verify_start_node", "quality_node")
    graph.add_edge("verify_start_node", "test_node")
    graph.add_conditional_edges(
        "retry_node",
        route_after_retry,
        {
            "FIX": "fix_node",
            "FAILED": END,
        },
    )
    return graph.compile(checkpointer=checkpointer)


workflow_checkpointer = InMemorySaver()
review_workflow = build_review_workflow(checkpointer=workflow_checkpointer)


def create_initial_state(project: Path, *, max_retries: int = 2) -> ReviewState:
    """Create a complete first-version State for one local review run."""
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    return {
        "project_path": str(project.expanduser().resolve()),
        "approval": "PENDING",
        "retry_count": 0,
        "max_retries": max_retries,
        "apply_result": {
            "success": False,
            "error": "",
            "changed_files": [],
        },
        "final_status": "RUNNING",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LangGraph review workflow.")
    parser.add_argument(
        "project_path",
        nargs="?",
        type=Path,
        default=DEMO_PROJECT,
        help="Local project directory. Defaults to quality_test_demo.",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="LangGraph checkpoint thread ID. Defaults to a generated UUID.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum number of re-fix attempts after the initial Patch.",
    )
    return parser.parse_args()


def print_interrupt_request(workflow_result: dict[str, Any]) -> None:
    """Display every pending HITL request returned by LangGraph."""
    requests = workflow_result.get("__interrupt__", ())
    for request in requests:
        print("\n=== HITL: Patch approval required ===")
        print(json.dumps(request.value, ensure_ascii=False, indent=2))


async def run_interactive_workflow(
    initial_state: ReviewState,
    *,
    thread_id: str,
) -> ReviewState:
    """Run one in-memory checkpoint thread and resume each HITL interactively."""
    config = {"configurable": {"thread_id": thread_id}}
    result = await review_workflow.ainvoke(initial_state, config=config)
    while "__interrupt__" in result:
        print_interrupt_request(result)
        approval = input("请输入 APPROVE 或 REJECT：").strip().upper()
        result = await review_workflow.ainvoke(
            Command(resume=approval),
            config=config,
        )
    return result


async def main() -> None:
    arguments = parse_args()
    project = arguments.project_path.expanduser().resolve()
    if not project.is_dir():
        raise SystemExit(f"Project directory does not exist: {project}")

    thread_id = arguments.thread_id or uuid.uuid4().hex
    initial_state = create_initial_state(
        project,
        max_retries=arguments.max_retries,
    )
    print("=== LangGraph workflow input ===")
    print(json.dumps(initial_state, ensure_ascii=False, indent=2))
    print(f"thread_id={thread_id}")
    final_state = await run_interactive_workflow(
        initial_state,
        thread_id=thread_id,
    )

    print("\n=== Parallel node results ===")
    for field_name in ("security_result", "quality_result", "test_result"):
        print(f"\n--- {field_name} ---")
        print(json.dumps(final_state[field_name], ensure_ascii=False, indent=2))

    if "patch_result" in final_state:
        print("\n--- patch_result ---")
        print(json.dumps(final_state["patch_result"], ensure_ascii=False, indent=2))

    print("\n=== Final State ===")
    print(json.dumps(final_state, ensure_ascii=False, indent=2))
    print(f"\n=== Final decision: {final_state['decision']} ===")
    print(f"=== Final status: {final_state['final_status']} ===")


if __name__ == "__main__":
    asyncio.run(main())
