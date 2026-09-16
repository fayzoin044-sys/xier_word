"""Thin MCP adapter over Pytest installed in the current environment."""

import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from importlib.metadata import version
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

mcp = FastMCP(
    "test-mcp-server",
    instructions="Delegate local Python test execution to Pytest.",
)


class TestCaseResult(BaseModel):
    name: str
    classname: str
    file: str | None
    line: int | None
    duration_seconds: float
    status: str
    details: str | None


class TestSummary(BaseModel):
    total: int
    passed: int
    failed: int
    errors: int
    skipped: int


class PytestResult(BaseModel):
    project_path: str
    pytest_version: str
    command: list[str]
    exit_code: int
    status: str
    summary: TestSummary
    tests: list[TestCaseResult]
    stdout: str
    stderr: str


def parse_test_case(testcase: ET.Element) -> TestCaseResult:
    status = "passed"
    details = None
    for child_status in ("failure", "error", "skipped"):
        child = testcase.find(child_status)
        if child is not None:
            status = "failed" if child_status == "failure" else child_status
            message = child.get("message", "").strip()
            body = (child.text or "").strip()
            details = "\n".join(part for part in (message, body) if part) or None
            break

    raw_line = testcase.get("line")
    return TestCaseResult(
        name=testcase.get("name", ""),
        classname=testcase.get("classname", ""),
        file=testcase.get("file"),
        line=int(raw_line) + 1 if raw_line is not None else None,
        duration_seconds=float(testcase.get("time", "0")),
        status=status,
        details=details,
    )


def parse_junit_report(report_path: Path) -> tuple[TestSummary, list[TestCaseResult]]:
    if not report_path.is_file():
        return TestSummary(total=0, passed=0, failed=0, errors=0, skipped=0), []

    root = ET.parse(report_path).getroot()
    tests = [parse_test_case(testcase) for testcase in root.iter("testcase")]
    failed = sum(test.status == "failed" for test in tests)
    errors = sum(test.status == "error" for test in tests)
    skipped = sum(test.status == "skipped" for test in tests)
    passed = sum(test.status == "passed" for test in tests)
    return (
        TestSummary(
            total=len(tests),
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
        ),
        tests,
    )


@mcp.tool(
    name="run_tests",
    description="Run Pytest for a local Python project and return structured results.",
    structured_output=True,
)
def run_tests(project_path: str) -> PytestResult:
    """Execute Pytest and read its native JUnit XML report."""
    project = Path(project_path).expanduser().resolve()
    if not project.is_dir():
        raise ValueError(f"project_path must be an existing directory: {project}")

    with tempfile.TemporaryDirectory(prefix="test-mcp-") as temporary_directory:
        report_path = Path(temporary_directory) / "pytest-report.xml"
        command = [
            sys.executable,
            "-m",
            "pytest",
            str(project),
            "-q",
            "-o",
            "junit_family=legacy",
            f"--junitxml={report_path}",
        ]
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            check=False,
            timeout=300,
            cwd=str(project),
        )
        summary, tests = parse_junit_report(report_path)

    statuses = {
        0: "passed",
        1: "failed",
        2: "interrupted",
        3: "internal_error",
        4: "usage_error",
        5: "no_tests_collected",
    }
    return PytestResult(
        project_path=str(project),
        pytest_version=version("pytest"),
        command=command,
        exit_code=completed.returncode,
        status=statuses.get(completed.returncode, "error"),
        summary=summary,
        tests=tests,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


if __name__ == "__main__":
    try:
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        pass
