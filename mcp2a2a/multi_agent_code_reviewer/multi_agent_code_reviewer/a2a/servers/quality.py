"""Publish the existing Quality Agent as an independent A2A service."""

from a2a.types import AgentSkill

from multi_agent_code_reviewer.a2a.common.agent_adapters import (
    ExistingReviewAgentAdapter,
)
from multi_agent_code_reviewer.a2a.common.executor import SpecialistAgentExecutor
from multi_agent_code_reviewer.a2a.common.server import (
    create_a2a_app,
    create_agent_card,
    run_a2a_server,
)
from multi_agent_code_reviewer.agents.quality import run_quality_review
from multi_agent_code_reviewer.config import (
    BIND_HOST,
    QUALITY_A2A_PORT,
    QUALITY_A2A_URL,
)

QUALITY_SKILL = AgentSkill(
    id="check_python_quality",
    name="Python 代码质量审查",
    description=(
        "接收本机项目目录的绝对路径，通过 Quality Agent 调用 Ruff MCP，"
        "返回真实代码质量问题及逐项解释。"
    ),
    tags=["quality", "ruff", "python", "code-review", "代码质量"],
    examples=[r"D:\path\to\python-project"],
)

quality_a2a_agent = ExistingReviewAgentAdapter(
    agent_name="QualityAgent",
    runner=run_quality_review,
    tool_calls_key="check_quality_calls",
)
quality_executor = SpecialistAgentExecutor(
    quality_a2a_agent,
    artifact_name="quality_review_result",
)
quality_card = create_agent_card(
    name="QualityAgent",
    description="使用 Ruff MCP 对本机 Python 项目执行代码质量审查的专业 Agent。",
    base_url=QUALITY_A2A_URL,
    skills=[QUALITY_SKILL],
)
app = create_a2a_app(agent_card=quality_card, executor=quality_executor)


def main() -> None:
    run_a2a_server(app, host=BIND_HOST, port=QUALITY_A2A_PORT)


if __name__ == "__main__":
    main()
