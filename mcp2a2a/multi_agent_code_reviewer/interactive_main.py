"""PyCharm entry point that prompts for an absolute project directory."""

import asyncio
from pathlib import Path

from multi_agent_code_reviewer.launcher import run_review


def prompt_project_path() -> Path:
    """Read and validate one absolute project directory from the console."""
    raw_path = input("请输入要审查的项目目录绝对路径：").strip().strip('"')
    if not raw_path:
        raise SystemExit("未输入项目目录。")

    project = Path(raw_path).expanduser()
    if not project.is_absolute():
        raise SystemExit(f"必须输入绝对路径：{project}")
    project = project.resolve()
    if not project.is_dir():
        raise SystemExit(f"项目目录不存在：{project}")
    return project


def main() -> None:
    project = prompt_project_path()
    try:
        asyncio.run(run_review(project))
    except KeyboardInterrupt:
        print("\n审查已由用户中止。")


if __name__ == "__main__":
    main()
