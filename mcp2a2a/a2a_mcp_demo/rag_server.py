"""A2A HTTP server entry point for the RAG agent."""

from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill

try:
    from a2a_mcp_demo.agents.rag_agent import RAGAgent
    from a2a_mcp_demo.common.executor import BusinessAgentExecutor
    from a2a_mcp_demo.common.server import (
        DEFAULT_RPC_PATH,
        create_a2a_app,
        run_a2a_server,
    )
    from a2a_mcp_demo.config import RAG_AGENT_URL
except ModuleNotFoundError:
    from agents.rag_agent import RAGAgent
    from common.executor import BusinessAgentExecutor
    from common.server import DEFAULT_RPC_PATH, create_a2a_app, run_a2a_server
    from config import RAG_AGENT_URL


HOST = "localhost"
PORT = 8002


def create_rag_agent_card() -> AgentCard:
    """Describe the RAG agent and its A2A endpoint."""
    skill = AgentSkill(
        id="knowledge_search",
        name="知识库检索",
        description="检索知识库并根据检索结果回答问题。",
        tags=["rag", "knowledge", "知识库"],
        input_modes=["text/plain"],
        output_modes=["text/plain"],
        examples=["查询知识库中的产品说明", "知识库里有相关资料吗？"],
    )
    return AgentCard(
        name="RAG Agent",
        description="根据用户问题检索知识库并生成有依据的回答。",
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                url=f"{RAG_AGENT_URL.rstrip('/')}{DEFAULT_RPC_PATH}",
            )
        ],
        version="0.1.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


def create_rag_app():
    """Wire RAGAgent into the shared executor and A2A server."""
    agent = RAGAgent()
    executor = BusinessAgentExecutor(agent, artifact_name="rag_result")
    return create_a2a_app(create_rag_agent_card(), executor)


app = create_rag_app()


if __name__ == "__main__":
    run_a2a_server(app, host=HOST, port=PORT)
