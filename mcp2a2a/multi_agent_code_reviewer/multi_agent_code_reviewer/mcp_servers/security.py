"""Thin MCP adapter over the locally installed Semgrep CE CLI."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel


def default_semgrep_executable() -> Path:
    if os.name == "nt":
        return Path(sys.prefix) / "Scripts" / "semgrep.exe"
    return Path(sys.prefix) / "bin" / "semgrep"


mcp = FastMCP(
    "security-mcp-server",
    instructions="Delegate local project security scans to Semgrep CE.",
)


class SecurityFinding(BaseModel):
    rule_id: str
    file: str
    start_line: int
    end_line: int
    severity: str
    message: str


class SecurityScanResult(BaseModel):
    project_path: str
    semgrep_version: str | None
    command: list[str]
    findings_count: int
    findings: list[SecurityFinding]
    semgrep_errors: list[dict[str, Any]]


def get_semgrep_executable() -> Path:
    configured = os.environ.get("SEMGREP_EXECUTABLE")
    executable = (
        Path(configured).expanduser() if configured else default_semgrep_executable()
    ).resolve()

    if not executable.is_file():
        raise RuntimeError(
            "Semgrep executable not found. Set SEMGREP_EXECUTABLE to its absolute "
            f"path. Checked: {executable}"
        )

    return executable


@mcp.tool(
    name="scan_security",
    description="Scan a local project directory with Semgrep CE.",
    structured_output=True,
)
def scan_security(project_path: str) -> SecurityScanResult:
    """Run Semgrep CE and map its JSON findings to a small stable result."""
    project = Path(project_path).expanduser().resolve()
    if not project.is_dir():
        raise ValueError(f"project_path must be an existing directory: {project}")

    semgrep_executable = get_semgrep_executable()
    command = [
        str(semgrep_executable),
        "scan",
        "--config",
        "auto",
        "--json",
        "--quiet",
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
        timeout=300,
    )

    try:
        raw_result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Semgrep did not return valid JSON "
            f"(exit code {completed.returncode}). stderr: {completed.stderr.strip()}"
        ) from error

    if completed.returncode != 0:
        raise RuntimeError(
            "Semgrep scan failed "
            f"(exit code {completed.returncode}): {raw_result.get('errors', [])}"
        )

    findings = []
    for result in raw_result.get("results", []):
        extra = result.get("extra", {})
        findings.append(
            SecurityFinding(
                rule_id=result.get("check_id", ""),
                file=result.get("path", ""),
                start_line=result.get("start", {}).get("line", 0),
                end_line=result.get("end", {}).get("line", 0),
                severity=extra.get("severity", "UNKNOWN"),
                message=extra.get("message", ""),
            )
        )

    return SecurityScanResult(
        project_path=str(project),
        semgrep_version=raw_result.get("version"),
        command=command,
        findings_count=len(findings),
        findings=findings,
        semgrep_errors=raw_result.get("errors", []),
    )


if __name__ == "__main__":
    try:
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        pass
