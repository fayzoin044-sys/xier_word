"""Intentionally vulnerable implementation for Security evaluation."""

import subprocess


def run_user_command(program: str, argument: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [program, argument],
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )


def greeting(name: str) -> str:
    return f"Hello, {name}!"
