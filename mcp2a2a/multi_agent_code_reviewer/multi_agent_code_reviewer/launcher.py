"""Local all-in-one launcher for PyCharm and command-line development."""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx

from multi_agent_code_reviewer.config import (
    FIX_A2A_URL,
    QUALITY_A2A_URL,
    SECURITY_A2A_URL,
    TEST_A2A_URL,
)
from multi_agent_code_reviewer.paths import PROJECT_ROOT, QUALITY_TEST_DEMO
from multi_agent_code_reviewer.workflows.review import (
    ReviewState,
    create_initial_state,
    run_interactive_workflow,
)


@dataclass(frozen=True)
class ServiceSpec:
    """One local A2A service managed by the launcher."""

    name: str
    agent_name: str
    module: str
    base_url: str


SERVICES = (
    ServiceSpec(
        name="Security",
        agent_name="SecurityAgent",
        module="multi_agent_code_reviewer.a2a.servers.security",
        base_url=SECURITY_A2A_URL,
    ),
    ServiceSpec(
        name="Quality",
        agent_name="QualityAgent",
        module="multi_agent_code_reviewer.a2a.servers.quality",
        base_url=QUALITY_A2A_URL,
    ),
    ServiceSpec(
        name="Test",
        agent_name="TestAgent",
        module="multi_agent_code_reviewer.a2a.servers.test",
        base_url=TEST_A2A_URL,
    ),
    ServiceSpec(
        name="Fix",
        agent_name="FixAgent",
        module="multi_agent_code_reviewer.a2a.servers.fix",
        base_url=FIX_A2A_URL,
    ),
)


def is_service_ready(service: ServiceSpec) -> bool:
    """Check both HTTP readiness and AgentCard identity."""
    card_url = f"{service.base_url}/.well-known/agent-card.json"
    try:
        response = httpx.get(card_url, timeout=1, trust_env=False)
        response.raise_for_status()
        card = response.json()
    except (httpx.HTTPError, json.JSONDecodeError):
        return False
    return isinstance(card, dict) and card.get("name") == service.agent_name


def wait_for_service(
    service: ServiceSpec,
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: float = 30,
) -> None:
    """Wait until one spawned A2A service publishes its expected card."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"{service.name} A2A service exited during startup: {return_code}"
            )
        if is_service_ready(service):
            print(f"[launcher] {service.name} ready -> {service.base_url}")
            return
        time.sleep(0.2)
    raise TimeoutError(
        f"Timed out waiting for {service.name} A2A service: {service.base_url}"
    )


def stop_process(process: subprocess.Popen[bytes]) -> None:
    """Stop one launcher-owned child process with a bounded fallback."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


@contextmanager
def local_a2a_services() -> Iterator[None]:
    """Start missing services, reuse existing ones, and clean up owned children."""
    managed_processes: list[subprocess.Popen[bytes]] = []
    try:
        for service in SERVICES:
            if is_service_ready(service):
                print(f"[launcher] {service.name} already running -> reuse")
                continue

            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                [sys.executable, "-m", service.module],
                cwd=str(PROJECT_ROOT),
                env=environment,
                stdin=subprocess.DEVNULL,
                shell=False,
            )
            managed_processes.append(process)
            wait_for_service(service, process)
        yield
    finally:
        if managed_processes:
            print("[launcher] stopping launcher-owned A2A services")
        for process in reversed(managed_processes):
            stop_process(process)


@contextmanager
def review_target(project_path: Path | None) -> Iterator[Path]:
    """Use an explicit project directly or a disposable copy of the demo."""
    if project_path is not None:
        project = project_path.expanduser().resolve()
        if not project.is_dir():
            raise ValueError(f"Project directory does not exist: {project}")
        yield project
        return

    with tempfile.TemporaryDirectory(prefix="code-review-main-") as temporary:
        project = Path(temporary) / "quality_test_demo"
        shutil.copytree(
            QUALITY_TEST_DEMO,
            project,
            ignore=shutil.ignore_patterns(
                "__pycache__",
                ".pytest_cache",
                ".ruff_cache",
                ".semgrep",
            ),
        )
        print(f"[launcher] using disposable demo copy -> {project}")
        yield project
        print("[launcher] disposable demo copy deleted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start local A2A services and run the interactive workflow."
    )
    parser.add_argument(
        "project_path",
        nargs="?",
        type=Path,
        help="Project to review. Omit to use a disposable quality_test_demo copy.",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="LangGraph thread ID. Defaults to a generated UUID.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum re-fix attempts after the initial Patch.",
    )
    return parser.parse_args()


async def run_review(
    project_path: Path | None,
    *,
    thread_id: str | None = None,
    max_retries: int = 2,
) -> ReviewState:
    """Run one local review while managing its dependent A2A services."""
    resolved_thread_id = thread_id or f"local-{uuid.uuid4().hex}"
    with review_target(project_path) as project, local_a2a_services():
        initial_state = create_initial_state(
            project,
            max_retries=max_retries,
        )
        print(f"[launcher] workflow thread_id={resolved_thread_id}")
        final_state = await run_interactive_workflow(
            initial_state,
            thread_id=resolved_thread_id,
        )
        print("\n=== Workflow completed ===")
        print(
            json.dumps(
                {
                    "project_path": final_state["project_path"],
                    "decision": final_state["decision"],
                    "approval": final_state["approval"],
                    "retry_count": final_state["retry_count"],
                    "apply_result": final_state["apply_result"],
                    "final_status": final_state["final_status"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return final_state


async def run() -> ReviewState:
    """Parse command-line arguments and run the all-in-one application."""
    arguments = parse_args()
    return await run_review(
        arguments.project_path,
        thread_id=arguments.thread_id,
        max_retries=arguments.max_retries,
    )


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n[launcher] interrupted by user")


if __name__ == "__main__":
    main()
