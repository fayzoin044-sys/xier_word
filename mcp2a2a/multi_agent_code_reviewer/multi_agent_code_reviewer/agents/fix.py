"""Patch-only Fix Agent backed by the official read-only Filesystem MCP tools."""

import argparse
import ast
import asyncio
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

FILESYSTEM_MCP_PACKAGE = os.getenv(
    "FILESYSTEM_MCP_PACKAGE",
    "@modelcontextprotocol/server-filesystem@2026.7.10",
)
READ_ONLY_FILESYSTEM_TOOLS = frozenset(
    {
        "read_text_file",
    }
)
TEST_DIRECTORY_NAMES = frozenset({"test", "tests"})
MAX_TEST_RELATED_SOURCE_FILES = 20

FIX_SYSTEM_PROMPT = """你是 Fix Agent，专门根据专业审查结果和已经通过 Filesystem MCP 读取的源码上下文提出最小代码修复。

输入包括 project_path、report_issue_checklist、filesystem_test_context、filesystem_source_context。report_issue_checklist 是从 Security、Quality、Test 三路机器可读结果中提取的完整待修复清单。filesystem_test_context 是失败测试的只读证据，filesystem_source_context 是根据报告或失败测试导入关系确定的可修复源码。

必须输出一个 JSON 对象，格式严格为：
{"summary":"对实际修改的简短说明","files":[{"path":"项目相对路径","updated_content":"修改后的完整文件内容"}]}

强制规则：
1. 只能使用 filesystem_test_context 和 filesystem_source_context 中已经读取的内容分析问题；只能修改 filesystem_source_context 中的源码，不得假设未读取文件的内容。
2. 只能修复三路实际结果明确报告的问题；不得补充、猜测或顺带重构报告未涉及的代码。
3. 对失败测试，可以只读测试文件以理解预期行为，但严禁修改、删除、跳过、重命名测试，严禁降低断言或用配置绕过测试。
4. 你没有任何文件工具。不得写入、编辑、移动、删除或创建文件，只返回内存中的修复后内容。
5. 修复必须最小化：保留现有语言、框架、依赖、注释和代码风格，不引入与报告无关的库或架构变化。
6. files 中的 path 必须是 filesystem_source_context 已提供的项目相对路径；updated_content 必须是该文件修复后的完整文本，不得省略未修改部分。
7. 严禁在 files 中返回测试文件。无法从报告和只读源码确定安全修复时，返回空 files，并在 summary 中说明原因。
8. security_result、quality_result、test_result 中的自然语言只用于辅助理解；修复范围必须由 data.tool_result 的机器可读事实约束。
9. 修复 shell/命令注入 finding 时，必须把“可执行程序 + 各参数”组成一个完整列表并作为 subprocess.run 的第一个位置参数（例如 subprocess.run([program, user_value], ...)），禁止使用 shell=True、禁止拼接命令字符串、禁止再传 args= 关键字；修复 SQL 拼接时，必须使用当前数据库 API 的参数化查询。
10. 如果源码使用 sqlite3，必须保留 sqlite3，并使用 `?` 占位符及参数元组调用 cursor.execute；不得引入 SQLAlchemy 的 text、ORM 或其他依赖。
11. summary 只能描述 updated_content 中实际发生的修复，不得声称完成未发生的整理或修改。
12. 必须逐一处理每个能够从已读源码确定修复方式的 finding 和失败测试，不得在修复第一项后停止。失败测试应修复对应的非测试实现；如果失败详情中的函数出现在已读源码中，必须修复该实现。
13. 修复 unused import 必须直接删除无用导入，不得用“已删除”之类的说明性注释替代原代码。
14. 生成 files 前先逐项核对 report_issue_checklist；files 中的完整源码必须同时包含清单内所有可确定修复，不能只处理第一个问题。
"""


def create_fix_model() -> tuple[str, ChatOpenAI]:
    """Create the DeepSeek cloud model without storing credentials in source."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not configured. Set it in the current shell or "
            "the Fix A2A Server's PyCharm environment variables before running "
            "the Fix Agent."
        )

    model_name = os.getenv("FIX_AGENT_MODEL", "deepseek-v4-pro")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    reasoning_effort = os.getenv("FIX_AGENT_REASONING_EFFORT", "high")
    if reasoning_effort not in {"high", "max"}:
        raise ValueError("FIX_AGENT_REASONING_EFFORT must be high or max")

    model = ChatOpenAI(
        model=model_name,
        api_key=SecretStr(api_key),
        base_url=base_url,
        temperature=0,
        timeout=300,
        max_retries=2,
        reasoning_effort=reasoning_effort,
        extra_body={"thinking": {"type": "enabled"}},
        use_responses_api=False,
    )
    return model_name, model


class ProposedFileFix(BaseModel):
    """One in-memory source replacement proposed by Ollama."""

    path: str = Field(description="Project-relative path of an existing source file.")
    updated_content: str = Field(description="Complete repaired file content.")


class FixPlanOutput(BaseModel):
    """Internal structured repair plan converted deterministically to a Patch."""

    summary: str = Field(description="Concise explanation of the proposed fixes.")
    files: list[ProposedFileFix] = Field(
        description="Only existing non-test source files that require reported fixes."
    )


class FixAgentRunResult(BaseModel):
    """Observable result used by validation and future orchestration."""

    model: str
    loaded_tools: list[str]
    filesystem_tool_calls: list[dict[str, Any]]
    read_files: list[str]
    summary: str
    patch: str
    changed_files: list[str]


def get_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    data = result.get("data")
    tool_result = data.get("tool_result") if isinstance(data, dict) else None
    return tool_result if isinstance(tool_result, dict) else {}


def build_issue_checklist(
    *,
    security_result: dict[str, Any],
    quality_result: dict[str, Any],
    test_result: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Keep only actionable machine facts so a small local model sees every issue."""
    security_findings = get_tool_result(security_result).get("findings", [])
    quality_findings = get_tool_result(quality_result).get("findings", [])
    tests = get_tool_result(test_result).get("tests", [])

    return {
        "security_findings": [
            finding
            for finding in security_findings
            if isinstance(finding, dict)
        ]
        if isinstance(security_findings, list)
        else [],
        "quality_findings": [
            finding
            for finding in quality_findings
            if isinstance(finding, dict)
        ]
        if isinstance(quality_findings, list)
        else [],
        "failed_tests": [
            test
            for test in tests
            if isinstance(test, dict)
            and test.get("status") in {"failed", "error"}
        ]
        if isinstance(tests, list)
        else [],
    }


def collect_report_files(
    *,
    project_path: Path,
    security_result: dict[str, Any],
    quality_result: dict[str, Any],
    test_result: dict[str, Any],
) -> list[Path]:
    """Select only files explicitly named by machine-readable reports."""
    project = project_path.resolve()
    candidates: set[Path] = set()
    for result in (security_result, quality_result):
        findings = get_tool_result(result).get("findings", [])
        if not isinstance(findings, list):
            continue
        for finding in findings:
            file_value = finding.get("file") if isinstance(finding, dict) else None
            if isinstance(file_value, str):
                candidates.add(Path(file_value))

    tests = get_tool_result(test_result).get("tests", [])
    if isinstance(tests, list):
        for test in tests:
            if not isinstance(test, dict) or test.get("status") not in {
                "failed",
                "error",
            }:
                continue
            file_value = test.get("file")
            if isinstance(file_value, str):
                candidates.add(Path(file_value))

    selected: list[Path] = []
    for candidate in candidates:
        resolved = (
            candidate.expanduser().resolve()
            if candidate.is_absolute()
            else (project / candidate).resolve()
        )
        try:
            resolved.relative_to(project)
        except ValueError as error:
            raise ValueError(f"Report file escapes project: {candidate}") from error
        if not resolved.is_file():
            raise ValueError(f"Report references a missing file: {resolved}")
        selected.append(resolved)
    return sorted(set(selected))


def _resolve_module_file(
    module_name: str,
    *,
    project_path: Path,
    importing_file: Path,
    relative_level: int = 0,
) -> Path | None:
    """Resolve one Python import to an existing project-local source file."""
    project = project_path.resolve()
    module_parts = [part for part in module_name.split(".") if part]

    search_bases: list[Path]
    if relative_level:
        base = importing_file.parent
        for _ in range(max(relative_level - 1, 0)):
            base = base.parent
        search_bases = [base]
    else:
        search_bases = [project, project / "src"]

    for base in search_bases:
        module_path = base.joinpath(*module_parts) if module_parts else base
        candidates = [module_path.with_suffix(".py"), module_path / "__init__.py"]
        for candidate in candidates:
            resolved = candidate.resolve()
            try:
                resolved.relative_to(project)
            except ValueError:
                continue
            if resolved.is_file() and not is_test_path(
                resolved.relative_to(project).as_posix()
            ):
                return resolved
    return None


def resolve_test_import_files(
    *,
    project_path: Path,
    test_file: Path,
    test_content: str,
) -> list[Path]:
    """Resolve direct project-local imports used by one failed Python test file."""
    if test_file.suffix.lower() != ".py":
        return []
    try:
        tree = ast.parse(test_content, filename=str(test_file))
    except SyntaxError:
        return []

    resolved_files: set[Path] = set()
    for node in ast.walk(tree):
        modules: list[tuple[str, int]] = []
        if isinstance(node, ast.Import):
            modules.extend((alias.name, 0) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append((node.module or "", node.level))
        else:
            continue

        for module_name, relative_level in modules:
            resolved = _resolve_module_file(
                module_name,
                project_path=project_path,
                importing_file=test_file,
                relative_level=relative_level,
            )
            if resolved is not None:
                resolved_files.add(resolved)

    return sorted(resolved_files)[:MAX_TEST_RELATED_SOURCE_FILES]


def extract_tool_text(value: object) -> str:
    """Flatten LangChain MCP text content into one source string."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        texts = [
            item.get("text", "")
            for item in value
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(texts)
    raise TypeError(f"Unexpected Filesystem MCP content: {type(value).__name__}")


def build_unified_diff(
    plan: FixPlanOutput,
    *,
    source_by_path: dict[str, str],
) -> str:
    """Convert Ollama's in-memory file proposals into a valid unified diff."""
    chunks: list[str] = []
    seen_paths: set[str] = set()
    for proposed in plan.files:
        relative_path = PurePosixPath(proposed.path).as_posix()
        if relative_path in seen_paths:
            raise ValueError(f"Fix plan contains duplicate file: {relative_path}")
        seen_paths.add(relative_path)
        if relative_path not in source_by_path:
            raise ValueError(f"Fix plan references an unread file: {relative_path}")
        if is_test_path(relative_path):
            raise ValueError(f"Fix plan must not modify test files: {relative_path}")

        original = source_by_path[relative_path].replace("\r\n", "\n")
        updated = proposed.updated_content.replace("\r\n", "\n")
        if original.endswith("\n") and not updated.endswith("\n"):
            updated += "\n"
        if original == updated:
            continue
        if relative_path.endswith(".py"):
            validate_python_content(updated, relative_path=relative_path)
        chunks.extend(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=f"a/{relative_path}",
                tofile=f"b/{relative_path}",
            )
        )
    return "".join(chunks)


def validate_python_content(content: str, *, relative_path: str) -> None:
    """Reject syntax errors and newly undefined Python names without file writes."""
    try:
        compile(content, relative_path, "exec")
    except SyntaxError as error:
        raise ValueError(
            f"Proposed Python content has invalid syntax: {relative_path}: {error}"
        ) from error

    ruff_executable = (
        Path(sys.prefix) / ("Scripts/ruff.exe" if os.name == "nt" else "bin/ruff")
    ).resolve()
    if not ruff_executable.is_file():
        raise RuntimeError(f"Ruff executable not found: {ruff_executable}")
    completed = subprocess.run(
        [
            str(ruff_executable),
            "check",
            "--select",
            "F821",
            "--output-format",
            "json",
            "--stdin-filename",
            relative_path,
            "-",
        ],
        input=content,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        check=False,
        timeout=30,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError(
            f"Ruff validation failed for {relative_path}: {completed.stderr.strip()}"
        )
    findings = json.loads(completed.stdout)
    if findings:
        raise ValueError(
            f"Proposed Python content contains undefined names: {findings}"
        )


def get_npx_command() -> str:
    configured = os.getenv("FILESYSTEM_MCP_COMMAND")
    command = configured or shutil.which("npx.cmd") or shutil.which("npx")
    if not command:
        raise RuntimeError(
            "npx was not found. Install Node.js or set FILESYSTEM_MCP_COMMAND."
        )
    return str(Path(command).resolve())


def create_filesystem_mcp_client(project_path: Path) -> MultiServerMCPClient:
    """Restrict the official Filesystem MCP Server to one project directory."""
    return MultiServerMCPClient(
        {
            "filesystem": {
                "transport": "stdio",
                "command": get_npx_command(),
                "args": [FILESYSTEM_MCP_PACKAGE, str(project_path)],
                "cwd": str(project_path),
            }
        }
    )


def extract_read_files(
    tool_calls: list[dict[str, Any]],
    *,
    project_path: Path,
) -> list[str]:
    """Collect file paths passed to the read-only MCP tools."""
    project = project_path.resolve()
    read_files: set[str] = set()
    for tool_call in tool_calls:
        args = tool_call.get("args", {})
        candidates: list[str] = []
        path = args.get("path") if isinstance(args, dict) else None
        paths = args.get("paths") if isinstance(args, dict) else None
        if isinstance(path, str):
            candidates.append(path)
        if isinstance(paths, list):
            candidates.extend(item for item in paths if isinstance(item, str))

        for candidate in candidates:
            resolved = Path(candidate).expanduser().resolve()
            try:
                relative = resolved.relative_to(project)
            except ValueError:
                continue
            if resolved.is_file():
                read_files.add(relative.as_posix())
    return sorted(read_files)


def parse_changed_files(patch: str) -> list[str]:
    """Extract and validate matching a/ and b/ unified-diff file headers."""
    old_paths = re.findall(r"(?m)^--- a/(.+)$", patch)
    new_paths = re.findall(r"(?m)^\+\+\+ b/(.+)$", patch)
    if not old_paths or len(old_paths) != len(new_paths):
        raise ValueError(
            "Patch must contain matching --- a/ and +++ b/ headers. "
            f"Received: {patch!r}"
        )
    if old_paths != new_paths:
        raise ValueError("First version Patch may only modify existing files")
    if "@@" not in patch:
        raise ValueError("Patch must contain at least one @@ hunk")
    return old_paths


def is_test_path(relative_path: str) -> bool:
    path = PurePosixPath(relative_path)
    lowered_parts = {part.lower() for part in path.parts[:-1]}
    filename = path.name.lower()
    return bool(
        lowered_parts & TEST_DIRECTORY_NAMES
        or filename.startswith("test_")
        or filename.endswith("_test.py")
    )


def validate_patch(
    patch: str,
    *,
    project_path: Path,
    read_files: list[str],
) -> list[str]:
    """Reject test edits, unread files, invalid paths, and malformed patches."""
    if not patch.strip():
        return []
    if "```" in patch:
        raise ValueError("Patch must not contain Markdown code fences")

    project = project_path.resolve()
    changed_files = parse_changed_files(patch)
    read_file_set = set(read_files)
    for relative_path in changed_files:
        pure_path = PurePosixPath(relative_path)
        if pure_path.is_absolute() or ".." in pure_path.parts:
            raise ValueError(f"Patch path must stay inside project: {relative_path}")
        if is_test_path(relative_path):
            raise ValueError(f"Patch must not modify test files: {relative_path}")

        target = (project / Path(*pure_path.parts)).resolve()
        try:
            target.relative_to(project)
        except ValueError as error:
            raise ValueError(f"Patch path escapes project: {relative_path}") from error
        if not target.is_file():
            raise ValueError(f"Patch may only modify existing files: {relative_path}")
        if relative_path not in read_file_set:
            raise ValueError(
                f"Patch modifies a file not read through Filesystem MCP: {relative_path}"
            )

    completed = subprocess.run(
        [
            "git",
            "apply",
            "--check",
            "--recount",
            "--ignore-space-change",
            "-",
        ],
        input=patch,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        check=False,
        timeout=30,
        cwd=str(project),
    )
    if completed.returncode != 0:
        raise ValueError(
            "Generated unified diff failed git apply --check: "
            + completed.stderr.strip()
            + f". Patch: {patch!r}"
        )
    return changed_files


def snapshot_project(project_path: Path) -> dict[str, str]:
    """Hash project files to prove the Patch-only Agent made no writes."""
    ignored_parts = {".pytest_cache", "__pycache__", ".semgrep"}
    snapshot: dict[str, str] = {}
    for path in sorted(project_path.rglob("*")):
        if not path.is_file() or set(path.parts) & ignored_parts:
            continue
        relative = path.relative_to(project_path).as_posix()
        snapshot[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


async def run_fix_agent(
    *,
    project_path: Path,
    security_result: dict[str, Any],
    quality_result: dict[str, Any],
    test_result: dict[str, Any],
) -> FixAgentRunResult:
    """Generate and validate a Patch without modifying the project."""
    project = project_path.expanduser().resolve()
    if not project.is_dir():
        raise ValueError(f"project_path must be an existing directory: {project}")

    model_name, model = create_fix_model()
    before_snapshot = snapshot_project(project)
    mcp_client = create_filesystem_mcp_client(project)
    discovered_tools = await mcp_client.get_tools()
    tools = [
        tool for tool in discovered_tools if tool.name in READ_ONLY_FILESYSTEM_TOOLS
    ]
    loaded_tool_names = [tool.name for tool in tools]
    if set(loaded_tool_names) != READ_ONLY_FILESYSTEM_TOOLS:
        raise RuntimeError(
            "Filesystem MCP did not expose the expected read-only tools: "
            f"{loaded_tool_names}"
        )

    issue_checklist = build_issue_checklist(
        security_result=security_result,
        quality_result=quality_result,
        test_result=test_result,
    )
    report_files = collect_report_files(
        project_path=project,
        security_result=security_result,
        quality_result=quality_result,
        test_result=test_result,
    )
    if not report_files:
        raise ValueError("The three reports do not reference any files to repair")

    read_text_tool = tools[0]
    filesystem_tool_calls: list[dict[str, Any]] = []
    source_by_path: dict[str, str] = {}
    filesystem_test_context: list[dict[str, str]] = []
    filesystem_source_context: list[dict[str, str]] = []
    for path in report_files:
        tool_call = {"name": "read_text_file", "args": {"path": str(path)}}
        content = await read_text_tool.ainvoke(tool_call["args"])
        text = extract_tool_text(content)
        relative_path = path.relative_to(project).as_posix()
        filesystem_tool_calls.append(tool_call)
        source_by_path[relative_path] = text
        if is_test_path(relative_path):
            filesystem_test_context.append(
                {"path": relative_path, "content": text}
            )
        else:
            filesystem_source_context.append(
                {"path": relative_path, "content": text}
            )

    # A Pytest failure usually names the test file rather than the faulty source.
    # Resolve only direct project-local imports from failed tests, then read those
    # implementation files through the same read-only Filesystem MCP boundary.
    related_source_files: set[Path] = set()
    for test_context in filesystem_test_context:
        test_path = (project / test_context["path"]).resolve()
        related_source_files.update(
            resolve_test_import_files(
                project_path=project,
                test_file=test_path,
                test_content=test_context["content"],
            )
        )

    for path in sorted(related_source_files)[:MAX_TEST_RELATED_SOURCE_FILES]:
        relative_path = path.relative_to(project).as_posix()
        if relative_path in source_by_path:
            continue
        tool_call = {"name": "read_text_file", "args": {"path": str(path)}}
        content = await read_text_tool.ainvoke(tool_call["args"])
        text = extract_tool_text(content)
        filesystem_tool_calls.append(tool_call)
        source_by_path[relative_path] = text
        filesystem_source_context.append(
            {"path": relative_path, "content": text}
        )

    patch_generator = model.with_structured_output(
        FixPlanOutput,
        method="json_mode",
    )
    structured_response = await patch_generator.ainvoke(
        [
            {"role": "system", "content": FIX_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "project_path": str(project),
                        "report_issue_checklist": issue_checklist,
                        "filesystem_test_context": filesystem_test_context,
                        "filesystem_source_context": filesystem_source_context,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]
    )
    if not isinstance(structured_response, FixPlanOutput):
        raise TypeError("Fix Agent Patch generator returned an invalid schema")

    patch = build_unified_diff(
        structured_response,
        source_by_path=source_by_path,
    ).replace("\r\n", "\n").rstrip("\r\n")
    if patch:
        patch += "\n"
    read_files = sorted(source_by_path)
    changed_files = validate_patch(
        patch,
        project_path=project,
        read_files=read_files,
    )
    after_snapshot = snapshot_project(project)
    if before_snapshot != after_snapshot:
        raise RuntimeError("Fix Agent modified project files; Patch-only mode was violated")

    return FixAgentRunResult(
        model=model_name,
        loaded_tools=loaded_tool_names,
        filesystem_tool_calls=filesystem_tool_calls,
        read_files=read_files,
        summary=structured_response.summary,
        patch=patch,
        changed_files=changed_files,
    )


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Patch with Fix Agent.")
    parser.add_argument("project_path", type=Path)
    parser.add_argument("security_result", type=Path)
    parser.add_argument("quality_result", type=Path)
    parser.add_argument("test_result", type=Path)
    return parser.parse_args()


async def main() -> None:
    arguments = parse_args()
    result = await run_fix_agent(
        project_path=arguments.project_path,
        security_result=load_json_object(arguments.security_result),
        quality_result=load_json_object(arguments.quality_result),
        test_result=load_json_object(arguments.test_result),
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
