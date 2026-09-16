"""RAG 问答系统的核心调度逻辑。

前面的模块已经分别完成了查询分类、策略选择、向量检索和提示词管理，
本模块负责把它们按照业务顺序连接起来：

1. 通用知识：不查询 Milvus，直接调用大模型回答；
2. 专业咨询：选择检索策略，查询知识库，再根据父块上下文生成答案。

本模块不负责文档入库、不训练 BERT，也不重复实现向量检索和父块重排。
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document
from openai import OpenAI


# 兼容两种运行方式：
# 1. 由整个项目导入 rag_qa.core.rag_system；
# 2. 在 PyCharm 中直接运行当前文件。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base.config import config
from base.logger import logger
from rag_qa.core.prompts import RAGPrompts
from rag_qa.core.query_classifier import QueryClassifier
from rag_qa.core.strategy_selector import StrategySelector
from rag_qa.core.vector_store import VectorStore


class RAGSystem:
    """整合查询分类、策略选择、知识库检索和答案生成。"""

    GENERAL_CATEGORY = "通用知识"
    PROFESSIONAL_CATEGORY = "专业咨询"

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        client: Any | None = None,
        query_classifier: QueryClassifier | None = None,
        strategy_selector: StrategySelector | None = None,
    ) -> None:
        """初始化 RAG 系统所需组件。

        Args:
            vector_store: 向量存储实例。正常运行不传时会创建 ``VectorStore``；
                离线测试可以传入模拟对象，避免连接真实 Milvus。
            client: OpenAI 兼容的大模型客户端。正常运行不传时读取项目配置；
                离线测试可以传入模拟客户端，避免调用真实 DeepSeek API。
            query_classifier: 可选的查询分类器，默认使用已微调的本地 BERT。
            strategy_selector: 可选的策略选择器，默认与本模块共用同一个客户端。

        注意：不传 ``vector_store`` 会立即连接 Milvus 并加载 BGE 模型，
        因此服务尚未启动时应在测试中传入模拟向量库。
        """

        self.model = config.LLM_MODEL

        # 显式判断 None，避免模拟对象自定义布尔值时被错误替换。
        self.vector_store = (
            vector_store if vector_store is not None else VectorStore()
        )
        self.query_classifier = (
            query_classifier
            if query_classifier is not None
            else QueryClassifier()
        )

        if client is None:
            if not config.LLM_API_KEY.strip():
                raise ValueError(
                    "大模型 API Key 为空，请在环境变量 DEEPSEEK_API_KEY "
                    "或 config.ini 的 llm.api_key 中配置"
                )

            self.client = OpenAI(
                api_key=config.LLM_API_KEY,
                base_url=config.LLM_BASE_URL,
                timeout=60.0,
                max_retries=1,
            )
        else:
            self.client = client

        # 默认与 RAGSystem 共用一个客户端，减少重复初始化；测试时也可以
        # 单独传入模拟策略选择器，让不同分支能够离线验证。
        self.strategy_selector = (
            strategy_selector
            if strategy_selector is not None
            else StrategySelector(client=self.client)
        )

        self.rag_prompt = RAGPrompts.rag_prompt()
        self.hyde_prompt = RAGPrompts.hyde_prompt()
        self.subquery_prompt = RAGPrompts.subquery_prompt()
        self.backtracking_prompt = RAGPrompts.backtracking_prompt()
        logger.info("RAGSystem 初始化完成")

    # =========================
    # 第一部分：统一的大模型 API 调用
    # =========================
    def _call_llm(
        self,
        prompt: str,
        *,
        system_message: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        """调用 DeepSeek API，并返回非空文本。

        ``prompt`` 是具体任务内容，``system_message`` 用于限定模型角色。
        查询改写和最终回答需要的长度不同，因此由调用方传入输出上限。
        """

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("发送给大模型的 prompt 不能为空")
        if max_tokens <= 0:
            raise ValueError("max_tokens 必须大于0")

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt.strip()},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            # 当前所有任务都有明确格式或知识库约束，不需要长推理。
            # 关闭 DeepSeek V4 思考模式可减少延迟和 token 消耗，并避免
            # 推理内容占满输出额度后最终 content 为空。
            extra_body={"thinking": {"type": "disabled"}},
        )

        if not completion.choices:
            raise RuntimeError("大模型 API 没有返回候选结果")

        content = completion.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("大模型 API 返回内容为空")

        return content.strip()

    # =========================
    # 第二部分：四种检索策略的真正执行
    # =========================
    def _direct_search(
        self,
        query: str,
        source_filter: str | None,
    ) -> list[Document]:
        """使用用户原问题直接执行混合检索。"""

        logger.info("执行直接检索策略")
        return self.vector_store.hybrid_search_with_rerank(
            query=query,
            source_filter=source_filter,
        )

    def _retrieve_with_hyde(
        self,
        query: str,
        source_filter: str | None,
    ) -> list[Document]:
        """生成一段假设答案，并用假设答案进行向量检索。

        HyDE 生成的内容只用于提高检索匹配效果，绝不会直接作为最终答案。
        如果假设答案生成失败，则回退为使用原问题直接检索。
        """

        logger.info("执行假设问题检索（HyDE）策略")
        prompt = self.hyde_prompt.format(query=query)

        try:
            hypothetical_answer = self._call_llm(
                prompt,
                system_message=(
                    "你只负责生成用于知识库检索的简短假设答案，"
                    "不要把它当作最终用户答案。"
                ),
                max_tokens=512,
            )
            logger.info("HyDE 假设答案生成完成")
        except Exception as exc:
            logger.error(
                "HyDE 假设答案生成失败（%s），回退为直接检索",
                type(exc).__name__,
            )
            return self._direct_search(query, source_filter)

        return self.vector_store.hybrid_search_with_rerank(
            query=hypothetical_answer,
            source_filter=source_filter,
        )

    def _retrieve_with_subqueries(
        self,
        query: str,
        source_filter: str | None,
    ) -> list[Document]:
        """把复杂问题拆成多个子查询，分别检索后合并父块。"""

        logger.info("执行子查询检索策略")
        prompt = self.subquery_prompt.format(query=query)

        try:
            raw_subqueries = self._call_llm(
                prompt,
                system_message=(
                    "你只负责把复杂查询拆成可独立检索的子查询，"
                    "每行输出一个，不要回答问题。"
                ),
                max_tokens=512,
            )
            subqueries = self._parse_subqueries(raw_subqueries)
        except Exception as exc:
            logger.error(
                "子查询生成失败（%s），回退为直接检索",
                type(exc).__name__,
            )
            return self._direct_search(query, source_filter)

        if not subqueries:
            logger.warning("没有生成有效子查询，回退为直接检索")
            return self._direct_search(query, source_filter)

        logger.info("有效子查询数量：%d", len(subqueries))
        result_groups: list[list[Document]] = []
        for subquery in subqueries:
            documents = self.vector_store.hybrid_search_with_rerank(
                query=subquery,
                source_filter=source_filter,
            )
            result_groups.append(documents)

        # 不同子查询可能命中同一个父块。这里按 parent_id 去重，并采用
        # 轮询合并，让多个子查询的高排名结果都有机会进入最终上下文。
        return self._merge_subquery_results(result_groups)

    def _retrieve_with_backtracking(
        self,
        query: str,
        source_filter: str | None,
    ) -> list[Document]:
        """把具体问题改写为更基础的问题，再执行向量检索。"""

        logger.info("执行回溯问题检索策略")
        prompt = self.backtracking_prompt.format(query=query)

        try:
            simplified_query = self._call_llm(
                prompt,
                system_message=(
                    "你只负责把具体查询改写成一个更基础的问题，"
                    "不要回答问题。"
                ),
                max_tokens=256,
            )
            logger.info("回溯问题生成完成")
        except Exception as exc:
            logger.error(
                "回溯问题生成失败（%s），回退为直接检索",
                type(exc).__name__,
            )
            return self._direct_search(query, source_filter)

        return self.vector_store.hybrid_search_with_rerank(
            query=simplified_query,
            source_filter=source_filter,
        )

    @staticmethod
    def _parse_subqueries(raw_subqueries: str) -> list[str]:
        """清理大模型返回的子查询，每次最多保留4个。"""

        unique_subqueries: list[str] = []
        seen: set[str] = set()

        for line in raw_subqueries.splitlines():
            # 兼容模型偶尔添加的“-”“1.”“2、”等前缀。
            cleaned = re.sub(
                r"^\s*(?:[-*•]|\d+[.、)])\s*",
                "",
                line,
            ).strip()
            if not cleaned or cleaned in seen:
                continue

            unique_subqueries.append(cleaned)
            seen.add(cleaned)
            if len(unique_subqueries) >= 4:
                break

        return unique_subqueries

    @staticmethod
    def _document_key(document: Document) -> str:
        """生成父块去重键，优先使用 parent_id，其次使用文档ID或正文。"""

        parent_id = document.metadata.get("parent_id")
        document_id = document.metadata.get("id")
        return str(parent_id or document_id or document.page_content)

    def _merge_subquery_results(
        self,
        result_groups: Iterable[list[Document]],
    ) -> list[Document]:
        """轮询合并多组检索结果、按父块去重并限制最终数量。"""

        groups = [group for group in result_groups if group]
        if not groups:
            return []

        merged_documents: list[Document] = []
        seen_keys: set[str] = set()
        maximum_group_size = max(len(group) for group in groups)
        result_limit = max(1, int(config.CANDIDATE_M))

        for rank_index in range(maximum_group_size):
            for group in groups:
                if rank_index >= len(group):
                    continue

                document = group[rank_index]
                document_key = self._document_key(document)
                if document_key in seen_keys:
                    continue

                merged_documents.append(document)
                seen_keys.add(document_key)
                if len(merged_documents) >= result_limit:
                    logger.info(
                        "子查询结果合并完成：最终父块=%d",
                        len(merged_documents),
                    )
                    return merged_documents

        logger.info(
            "子查询结果合并完成：最终父块=%d",
            len(merged_documents),
        )
        return merged_documents

    def retrieve_and_merge(
        self,
        query: str,
        source_filter: str | None = None,
        strategy: str | None = None,
    ) -> list[Document]:
        """根据指定策略检索父块；未指定策略时自动选择。

        这是专业咨询分支的统一检索入口，最终始终返回父块
        ``list[Document]``。
        """

        query = self._validate_query(query)
        source_filter = self._validate_source_filter(source_filter)

        if strategy is None:
            strategy = self.strategy_selector.select_strategy(query)

        if strategy == StrategySelector.HYDE:
            return self._retrieve_with_hyde(query, source_filter)
        if strategy == StrategySelector.SUBQUERY:
            return self._retrieve_with_subqueries(query, source_filter)
        if strategy == StrategySelector.BACKTRACKING:
            return self._retrieve_with_backtracking(query, source_filter)

        if strategy != StrategySelector.DIRECT:
            logger.warning("无法识别策略 %r，回退为直接检索", strategy)
        return self._direct_search(query, source_filter)

    # =========================
    # 第三部分：查询分类与最终答案生成
    # =========================
    def _generate_general_answer(self, query: str) -> str:
        """通用知识不查询 Milvus，直接让大模型回答。"""

        return self._call_llm(
            query,
            system_message=(
                "你是一个面向 IT 教育场景的中文问答助手。"
                "请根据可靠的通用知识简洁回答。"
                "如果问题涉及本机构的学费、课程政策、教师、就业承诺等"
                "专属信息，不要猜测，应说明需要查询机构知识库。"
            ),
            max_tokens=1024,
            temperature=0.1,
        )

    def _generate_rag_answer(
        self,
        query: str,
        context_documents: list[Document],
    ) -> str:
        """把父块拼成上下文，并依据 RAG 提示词生成最终答案。"""

        context_parts = [
            f"[资料{index}]\n{document.page_content.strip()}"
            for index, document in enumerate(context_documents, start=1)
            if document.page_content.strip()
        ]
        context = "\n\n".join(context_parts)

        if not context:
            return self._no_context_answer()

        prompt = self.rag_prompt.format(
            context=context,
            question=query,
            phone=config.CUSTOMER_SERVICE_PHONE,
        )
        return self._call_llm(
            prompt,
            system_message=(
                "你是基于知识库回答问题的中文助手。"
                "必须遵守用户提示中给出的知识库回答规则。"
            ),
            max_tokens=1536,
            temperature=0.1,
        )

    def generate_answer(
        self,
        query: str,
        source_filter: str | None = None,
    ) -> str:
        """处理一条问题并返回最终答案字符串。

        主流程：
            1. BERT 判断通用知识或专业咨询；
            2. 通用知识直接调用大模型；
            3. 专业咨询选择策略、检索父块，再生成 RAG 答案。
        """

        start_time = time.perf_counter()
        query = self._validate_query(query)
        source_filter = self._validate_source_filter(source_filter)
        logger.info("RAG 开始处理查询：%r", query)

        try:
            category = self.query_classifier.predict_category(query)
        except Exception as exc:
            logger.error("查询分类失败（%s）", type(exc).__name__)
            return self._service_error_answer("查询分类")

        logger.info("RAG 查询类别：%s", category)

        if category == self.GENERAL_CATEGORY:
            try:
                answer = self._generate_general_answer(query)
            except Exception as exc:
                logger.error(
                    "通用知识回答生成失败（%s）",
                    type(exc).__name__,
                )
                answer = self._service_error_answer("通用知识回答")

            logger.info(
                "通用知识处理完成，耗时 %.2f 秒",
                time.perf_counter() - start_time,
            )
            return answer

        if category != self.PROFESSIONAL_CATEGORY:
            logger.error("查询分类器返回未知类别：%r", category)
            return self._service_error_answer("查询分类")

        # 只有专业咨询才选择策略并查询 Milvus。
        try:
            strategy = self.strategy_selector.select_strategy(query)
            logger.info("专业咨询采用策略：%s", strategy)
            context_documents = self.retrieve_and_merge(
                query=query,
                source_filter=source_filter,
                strategy=strategy,
            )
        except Exception as exc:
            logger.error("知识库检索失败（%s）", type(exc).__name__)
            return self._service_error_answer("知识库检索")

        logger.info("专业咨询检索到父块数量：%d", len(context_documents))
        if not context_documents:
            answer = self._no_context_answer()
        else:
            try:
                answer = self._generate_rag_answer(
                    query,
                    context_documents,
                )
            except Exception as exc:
                logger.error(
                    "RAG 最终答案生成失败（%s）",
                    type(exc).__name__,
                )
                answer = self._service_error_answer("最终答案生成")

        logger.info(
            "专业咨询处理完成，耗时 %.2f 秒",
            time.perf_counter() - start_time,
        )
        return answer

    # =========================
    # 第四部分：输入校验、兜底回答与资源释放
    # =========================
    @staticmethod
    def _validate_query(query: str) -> str:
        """校验并清理用户问题。"""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")
        return query.strip()

    @staticmethod
    def _validate_source_filter(source_filter: str | None) -> str | None:
        """校验知识库来源过滤条件。"""

        if source_filter is None:
            return None
        if not isinstance(source_filter, str):
            raise ValueError("source_filter 必须是字符串或 None")

        source_filter = source_filter.strip()
        if not source_filter:
            return None
        if source_filter not in config.VALID_SOURCES:
            raise ValueError(
                f"无效的 source_filter：{source_filter}；"
                f"允许值：{config.VALID_SOURCES}"
            )
        return source_filter

    @staticmethod
    def _no_context_answer() -> str:
        """专业咨询没有检索到可靠资料时返回固定提示，禁止编造。"""

        return (
            "抱歉，当前知识库中没有找到足够的信息来回答这个问题。"
            f"建议联系人工客服进一步确认，电话：{config.CUSTOMER_SERVICE_PHONE}。"
        )

    @staticmethod
    def _service_error_answer(stage: str) -> str:
        """外部服务或模型异常时返回统一、可理解的用户提示。"""

        return (
            f"抱歉，{stage}服务暂时不可用，请稍后再试。"
            f"如需帮助，请联系人工客服：{config.CUSTOMER_SERVICE_PHONE}。"
        )

    def close(self) -> None:
        """释放向量存储持有的 Milvus 连接。重复调用应保持安全。"""

        close_method = getattr(self.vector_store, "close", None)
        if callable(close_method):
            close_method()
        logger.info("RAGSystem 资源已释放")


def main() -> None:
    """PyCharm 真实环境测试入口，会连接 Milvus 并调用 DeepSeek API。"""

    rag_system: RAGSystem | None = None
    try:
        rag_system = RAGSystem()
        query = input("请输入需要测试的RAG问题：").strip()
        source_filter = input(
            "请输入知识库分类（可直接回车跳过）："
        ).strip()
        answer = rag_system.generate_answer(
            query=query,
            source_filter=source_filter or None,
        )
        print("RAG回答：", answer)
    finally:
        if rag_system is not None:
            rag_system.close()


if __name__ == "__main__":
    main()
