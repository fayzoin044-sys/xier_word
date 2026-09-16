"""Publish the existing Test Agent as an independent A2A service."""

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
from multi_agent_code_reviewer.agents.test import run_test_review
from multi_agent_code_reviewer.config import BIND_HOST, TEST_A2A_PORT, TEST_A2A_URL

TEST_SKILL = AgentSkill(
    id="run_python_tests",
    name="Python 测试执行",
    description=(
        "接收本机项目目录的绝对路径，通过 Test Agent 调用 Pytest MCP，"
        "返回真实测试汇总、失败详情及解释。"
    ),
    tags=["testing", "pytest", "python", "code-review", "测试"],
    examples=[r"D:\path\to\python-project"],
)

test_a2a_agent = ExistingReviewAgentAdapter(
    agent_name="TestAgent",
    runner=run_test_review,
    tool_calls_key="run_tests_calls",
)
test_executor = SpecialistAgentExecutor(
    test_a2a_agent,
    artifact_name="test_review_result",
)
test_card = create_agent_card(
    name="TestAgent",
    description="使用 Pytest MCP 对本机 Python 项目执行测试并解释结果的专业 Agent。",
    base_url=TEST_A2A_URL,
    skills=[TEST_SKILL],
)
app = create_a2a_app(agent_card=test_card, executor=test_executor)


def main() -> None:
    run_a2a_server(app, host=BIND_HOST, port=TEST_A2A_PORT)


if __name__ == "__main__":
    main()
