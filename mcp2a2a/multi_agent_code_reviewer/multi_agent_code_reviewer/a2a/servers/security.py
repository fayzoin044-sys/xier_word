"""Publish the existing Security Agent as an independent A2A service."""

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
from multi_agent_code_reviewer.agents.security import run_security_review
from multi_agent_code_reviewer.config import (
    BIND_HOST,
    SECURITY_A2A_PORT,
    SECURITY_A2A_URL,
)

SECURITY_SKILL = AgentSkill(
    id="scan_project_security",
    name="项目安全审查",
    description=(
        "接收本机项目目录的绝对路径，通过 Security Agent 调用 Semgrep MCP，"
        "返回真实安全扫描结果及逐项解释。"
    ),
    tags=["security", "semgrep", "code-review", "安全审查"],
    examples=[r"D:\path\to\python-project"],
)

security_a2a_agent = ExistingReviewAgentAdapter(
    agent_name="SecurityAgent",
    runner=run_security_review,
    tool_calls_key="scan_security_calls",
)
security_executor = SpecialistAgentExecutor(
    security_a2a_agent,
    artifact_name="security_review_result",
)
security_card = create_agent_card(
    name="SecurityAgent",
    description="使用 Semgrep MCP 对本机代码项目执行安全审查的专业 Agent。",
    base_url=SECURITY_A2A_URL,
    skills=[SECURITY_SKILL],
)
app = create_a2a_app(agent_card=security_card, executor=security_executor)


def main() -> None:
    run_a2a_server(app, host=BIND_HOST, port=SECURITY_A2A_PORT)


if __name__ == "__main__":
    main()
