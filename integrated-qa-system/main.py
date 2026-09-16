"""FAQ 与 RAG 的统一问答入口。

本模块只负责系统调度：先让 FAQ 尝试回答，FAQ 无法可靠回答时，
再把用户原问题转交给 RAGSystem。这里不重复实现检索、分类或大模型调用。
"""

from __future__ import annotations

from base.logger import logger
from mysql_qa.main import MySQLQASystem
from rag_qa.core.rag_system import RAGSystem


class IntegratedQASystem:
    """统一调度 FAQ 系统和 RAG 系统。"""

    def __init__(self) -> None:
        # FAQ 启动时只初始化一次，后续所有问题复用 MySQL、Redis 和 BM25。
        self.faq_system = MySQLQASystem()

        # RAG 加载本地模型并连接 Milvus，开销较大，因此采用延迟初始化：
        # 只有 FAQ 第一次无法回答时才创建，创建后也会一直复用。
        self.rag_system: RAGSystem | None = None
        self._closed = False

        logger.info("统一问答系统初始化完成")

    def _get_rag_system(self) -> RAGSystem:
        """获取可复用的 RAGSystem，第一次调用时才真正初始化。"""

        if self.rag_system is None:
            logger.info("首次需要 RAG，开始初始化 RAGSystem")
            self.rag_system = RAGSystem()

        return self.rag_system

    def query(self, user_query: str) -> str:
        """接收一个用户问题，并返回 FAQ 或 RAG 生成的最终答案。"""

        if self._closed:
            raise RuntimeError("统一问答系统已经关闭")

        if not isinstance(user_query, str) or not user_query.strip():
            raise ValueError("user_query 必须是非空字符串")

        user_query = user_query.strip()
        logger.info("统一问答系统收到用户问题：%s", user_query)

        # 第一层：优先使用速度更快、不消耗大模型额度的 FAQ 系统。
        faq_answer, need_rag = self.faq_system.query(user_query)

        if not need_rag:
            logger.info("本次问题由 FAQ 系统回答")
            return faq_answer

        # 第二层：FAQ 没有可靠答案时，将用户原问题完整交给 RAG。
        logger.info("FAQ 未找到可靠答案，开始转交 RAGSystem")
        rag_system = self._get_rag_system()
        answer = rag_system.generate_answer(query=user_query)
        logger.info("本次问题由 RAG 系统回答")
        return answer

    def close(self) -> None:
        """关闭已经创建的 RAG、Redis 和 MySQL 资源。"""

        if self._closed:
            return

        # 先标记关闭，保证重复调用 close() 时不会重复释放资源。
        self._closed = True

        if self.rag_system is not None:
            try:
                self.rag_system.close()
            except Exception as exc:
                logger.error("RAGSystem 关闭失败（%s）", type(exc).__name__)

        try:
            self.faq_system.close()
        except Exception as exc:
            logger.error("FAQ 系统关闭失败（%s）", type(exc).__name__)

        logger.info("统一问答系统资源已释放")


def main() -> None:
    """PyCharm 交互式运行入口。"""

    qa_system: IntegratedQASystem | None = None

    try:
        qa_system = IntegratedQASystem()

        print("\n欢迎使用智能问答系统")
        print("输入问题开始查询，输入 exit 退出。\n")

        while True:
            user_query = input("请输入问题：").strip()

            if user_query.lower() == "exit":
                print("已退出智能问答系统")
                break

            if not user_query:
                print("问题不能为空，请重新输入。\n")
                continue

            try:
                answer = qa_system.query(user_query)
                print("\n回答：")
                print(answer)
                print("-" * 50)
            except Exception as exc:
                # 只记录异常类型，避免把 API Key 等敏感配置输出到控制台或日志。
                logger.error("当前问题处理失败（%s）", type(exc).__name__)
                print("\n抱歉，系统暂时无法处理这个问题，请稍后再试。")
                print("-" * 50)

    except (KeyboardInterrupt, EOFError):
        print("\n检测到用户中断，系统退出")
    except Exception as exc:
        logger.error("统一问答系统启动失败（%s）", type(exc).__name__)
        print("系统启动失败，请检查 MySQL、Redis 和项目配置。")
    finally:
        if qa_system is not None:
            qa_system.close()


if __name__ == "__main__":
    main()
