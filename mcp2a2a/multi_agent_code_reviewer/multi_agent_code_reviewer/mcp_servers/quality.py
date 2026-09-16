"""Thin MCP adapter over the Ruff CLI installed in the current environment."""

import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel


def default_ruff_executable() -> Path:
    if os.name == "nt":
        return Path(sys.prefix) / "Scripts" / "ruff.exe"
    return Path(sys.prefix) / "bin" / "ruff"


mcp = FastMCP(
    "quality-mcp-server",
    instructions="Delegate local Python code-quality checks to Ruff.",
)


class QualityFinding(BaseModel):
    rule_id: str
    file: str
    start_line: int
    start_column: int
    end_line: int
    end_column: int
    message: str
    fix_available: bool


class QualityCheckResult(BaseModel):
    project_path: str
    ruff_version: str
    command: list[str]
    findings_count: int
    findings: list[QualityFinding]


def get_ruff_executable() -> Path:
    configured = os.environ.get("RUFF_EXECUTABLE")
    executable = (
        Path(configured).expanduser() if configured else default_ruff_executable()
    ).resolve()

    if not executable.is_file():
        raise RuntimeError(
            "Ruff executable not found. Set RUFF_EXECUTABLE to its absolute path. "
            f"Checked: {executable}"
        )

    return executable


@mcp.tool(
    name="check_quality",
    description="Check a local Python project with Ruff and return structured findings.",
    structured_output=True,
)
def check_quality(project_path: str) -> QualityCheckResult:
    """Run Ruff and map its JSON diagnostics to a small stable result."""
    project = Path(project_path).expanduser().resolve()
    if not project.is_dir():
        raise ValueError(f"project_path must be an existing directory: {project}")

    ruff_executable = get_ruff_executable()
    command = [
        str(ruff_executable),
        "check",
        "--output-format",
        "json",
        "--no-cache",
        str(project),
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
        timeout=120,
    )

    try:
        raw_findings = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Ruff did not return valid JSON "
            f"(exit code {completed.returncode}). stderr: {completed.stderr.strip()}"
        ) from error

    # Ruff returns 1 when it found violations; 2 means a configuration/runtime error.
    if completed.returncode not in (0, 1):
        raise RuntimeError(
            "Ruff check failed "
            f"(exit code {completed.returncode}): {completed.stderr.strip()}"
        )

    findings = [
        QualityFinding(
            rule_id=item.get("code") or "",
            file=item.get("filename") or "",
            start_line=item.get("location", {}).get("row", 0),
            start_column=item.get("location", {}).get("column", 0),
            end_line=item.get("end_location", {}).get("row", 0),
            end_column=item.get("end_location", {}).get("column", 0),
            message=item.get("message") or "",
            fix_available=item.get("fix") is not None,
        )
        for item in raw_findings
    ]

    return QualityCheckResult(
        project_path=str(project),
        ruff_version=version("ruff"),
        command=command,
        findings_count=len(findings),
        findings=findings,
    )


if __name__ == "__main__":
    try:
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        pass
