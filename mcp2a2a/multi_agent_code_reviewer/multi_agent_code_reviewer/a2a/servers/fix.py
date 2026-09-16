"""Publish the Patch-only Fix Agent as an independent A2A service."""

import json
from pathlib import Path
from typing import Any

from a2a.types import AgentSkill
from pydantic import BaseModel, ValidationError

from multi_agent_code_reviewer.a2a.common.executor import SpecialistAgentExecutor
from multi_agent_code_reviewer.a2a.common.models import AgentResult
from multi_agent_code_reviewer.a2a.common.server import (
    create_a2a_app,
    create_agent_card,
    run_a2a_server,
)
from multi_agent_code_reviewer.agents.fix import run_fix_agent
from multi_agent_code_reviewer.config import BIND_HOST, FIX_A2A_PORT, FIX_A2A_URL


class FixA2AInput(BaseModel):
    """Machine-readable input carried by one A2A text message."""

    project_path: str
    security_result: dict[str, Any]
    quality_result: dict[str, Any]
    test_result: dict[str, Any]


class FixAgentA2AAdapter:
    """Translate the A2A JSON message to the existing Fix Agent call."""

    async def invoke(self, query: str) -> AgentResult:
        try:
            payload = FixA2AInput.model_validate_json(query)
        except (ValidationError, ValueError) as error:
            return AgentResult(
                text=(
                    "请提供 JSON 对象，字段为 project_path、security_result、"
                    f"quality_result、test_result。输入错误：{error}"
                ),
                requires_input=True,
            )

        project = Path(payload.project_path).expanduser().resolve()
        if not project.is_dir():
            return AgentResult(
                text=f"项目目录不存在，请重新提供有效的绝对路径：{project}",
                requires_input=True,
            )

        result = await run_fix_agent(
            project_path=project,
            security_result=payload.security_result,
            quality_result=payload.quality_result,
            test_result=payload.test_result,
        )
        text = result.summary
        if result.patch:
            text = f"{text}\n\n{result.patch}"
        return AgentResult(
            text=text,
            data={
                "agent": "FixAgent",
                **result.model_dump(),
            },
        )


FIX_SKILL = AgentSkill(
    id="generate_code_fix_patch",
    name="生成代码修复 Patch",
    description=(
        "接收项目路径及 Security、Quality、Test 三路结构化结果，"
        "通过只读 Filesystem MCP 获取相关源码，并返回 unified diff Patch；"
        "不会直接修改项目文件或测试。"
    ),
    tags=["fix", "patch", "filesystem-mcp", "code-review", "代码修复"],
    examples=[
        json.dumps(
            {
                "project_path": r"D:\path\to\python-project",
                "security_result": {"text": "...", "data": {}},
                "quality_result": {"text": "...", "data": {}},
                "test_result": {"text": "...", "data": {}},
            },
            ensure_ascii=False,
        )
    ],
)

fix_a2a_agent = FixAgentA2AAdapter()
fix_executor = SpecialistAgentExecutor(
    fix_a2a_agent,
    artifact_name="fix_patch_result",
)
fix_card = create_agent_card(
    name="FixAgent",
    description=(
        "根据三个专业审查 Agent 的实际结果，使用只读 Filesystem MCP "
        "生成最小 unified diff Patch 的专业 Agent。"
    ),
    base_url=FIX_A2A_URL,
    skills=[FIX_SKILL],
)
app = create_a2a_app(agent_card=fix_card, executor=fix_executor)


def main() -> None:
    run_a2a_server(app, host=BIND_HOST, port=FIX_A2A_PORT)


if __name__ == "__main__":
    main()
