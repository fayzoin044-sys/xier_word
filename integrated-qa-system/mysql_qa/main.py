# mysql_qa/main.py

import sys
from pathlib import Path

# 保证从 mysql_qa/main.py 直接运行时，也能找到项目根目录下的 base、mysql_qa 包
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from base.logger import logger
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.retrieval.bm25_search import BM25Search


class MySQLQASystem:
    """
    FAQ 问答系统总调度类。

    负责把：kk
    MySQLClient
    RedisClient
    BM25Search

    串成完整查询流程。
    """

    def __init__(self):
        # 连接 MySQL
        self.mysql_client = MySQLClient()

        # 连接 Redis
        self.redis_client = RedisClient()

        # 初始化 BM25 检索器
        # 内部会加载标准问题、分词、初始化 BM25
        self.bm25_search = BM25Search(
            redis_client=self.redis_client,
            mysql_client=self.mysql_client
        )

        logger.info("FAQ 问答系统初始化完成")

    def query(self, user_query):
        """
        接收用户问题，返回 FAQ 查询结果。

        返回：
            answer：答案内容
            need_rag：是否需要进入 RAG
        """

        logger.info(f"收到用户问题：{user_query}")

        answer, need_rag = self.bm25_search.search(
            query=user_query,
            threshold=0.85
        )

        if answer is not None and not need_rag:
            logger.info("FAQ 系统成功返回答案")
            return answer, False

        logger.info("FAQ 系统未找到可靠答案，需要转交 RAG")

        # 当前章节还没有真正接入 RAG，
        # 所以这里只返回占位提示。
        return "FAQ 未找到可靠答案，后续应调用 RAG 系统", True

    def close(self):
        """关闭 MySQL 和 Redis 连接。"""

        if self.redis_client is not None:
            self.redis_client.close()

        if self.mysql_client is not None:
            self.mysql_client.close()

        logger.info("FAQ 问答系统资源已释放")


def main():
    qa_system = MySQLQASystem()

    try:
        print("\n欢迎使用 FAQ 问答系统")
        print("输入问题开始查询，输入 exit 退出。\n")

        while True:
            user_query = input("请输入问题：").strip()

            if user_query.lower() == "exit":
                print("已退出 FAQ 问答系统")
                break

            if not user_query:
                print("问题不能为空，请重新输入。\n")
                continue

            answer, need_rag = qa_system.query(user_query)

            print("\n回答：")
            print(answer)

            print("\n是否需要调用 RAG：", need_rag)
            print("-" * 50)

    except KeyboardInterrupt:
        print("\n手动中断，系统退出")

    except Exception as e:
        logger.error(f"系统运行异常：{e}")
        print(f"系统运行异常：{e}")

    finally:
        qa_system.close()


if __name__ == "__main__":
    main()
