# mysql_qa/retrieval/bm25_search.py

import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base.logger import logger
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.utils.preprocess import preprocess_text


class BM25Search:
    """负责 FAQ 问题加载、BM25 检索和答案缓存。"""

    def __init__(self, redis_client, mysql_client):
        # 接收外部已经创建好的 Redis 和 MySQL 客户端
        self.redis_client = redis_client
        self.mysql_client = mysql_client

        # BM25 模型
        self.bm25 = None

        # 原始标准问题列表
        # 例如：
        # ["如何切割字符串", "如何读取文件"]
        self.original_questions = []

        # 分词后的标准问题列表
        # 例如：
        # [["如何", "切割", "字符串"], ["如何", "读取", "文件"]]
        self.tokenized_questions = []

        # 创建对象时自动加载问题并初始化 BM25
        self._load_data()

    def _load_data(self):
        """
        加载标准问题和分词结果。

        优先从 Redis 读取；
        Redis 没有时，再从 MySQL 读取并写入 Redis。
        """

        original_key = "qa_original_questions"
        tokenized_key = "qa_tokenized_questions"

        # 1. 优先从 Redis 读取缓存
        original_questions = self.redis_client.get_data(original_key)
        tokenized_questions = self.redis_client.get_data(tokenized_key)

        # 2. Redis 中缺少任意一份数据时，从 MySQL 重新加载
        if not original_questions or not tokenized_questions:
            logger.info("Redis 中没有完整的问题缓存，开始从 MySQL 加载")

            # 从 MySQL 获取标准问题
            original_questions = self.mysql_client.fetch_questions()

            if not original_questions:
                logger.warning("MySQL 中没有可用的标准问题")
                return

            # 对所有标准问题进行分词
            tokenized_questions = [
                preprocess_text(question)
                for question in original_questions
            ]

            # 将原始问题和分词结果写入 Redis
            self.redis_client.set_data(
                original_key,
                original_questions
            )

            self.redis_client.set_data(
                tokenized_key,
                tokenized_questions
            )

            logger.info("标准问题和分词结果已写入 Redis")

        else:
            logger.info("已从 Redis 读取标准问题缓存")

        # 3. 保存到当前 BM25Search 对象
        self.original_questions = original_questions
        self.tokenized_questions = tokenized_questions

        # 防止两份列表数量不一致
        if len(self.original_questions) != len(self.tokenized_questions):
            logger.error("原始问题与分词问题数量不一致")
            return

        # 4. 使用分词后的标准问题初始化 BM25
        self.bm25 = BM25Okapi(self.tokenized_questions)#创建检索器

        logger.info(
            f"BM25 初始化完成，共加载 "
            f"{len(self.original_questions)} 个标准问题"
        )

    def _softmax(self, scores):
        """
        将 BM25 原始分数转换为总和为 1 的分数。

        输入：
            [4.8, 0.2, 0.0]

        输出：
            [0.98, 0.01, 0.01]
        """

        scores = np.asarray(scores, dtype=np.float64)

        # 减去最大值，防止指数计算时数值溢出
        exp_scores = np.exp(scores - np.max(scores))

        return exp_scores / exp_scores.sum()

    def search(self, query, threshold=0.85):
        """
        查询 FAQ。

        返回：
            (answer, need_rag)

        FAQ 找到可靠答案：
            ("答案内容", False)

        FAQ 无法可靠回答：
            (None, True)
        """

        # 1. 检查输入是否合法
        if not isinstance(query, str) or not query.strip():
            logger.warning("用户查询为空或类型不正确")
            return None, True

        query = query.strip()#
        # 2. 最先检查答案缓存
        cached_answer = self.redis_client.get_answer(query)

        if cached_answer is not None:
            logger.info("直接返回 Redis 答案缓存")
            return cached_answer, False

        # BM25 未成功初始化
        if self.bm25 is None:
            logger.error("BM25 尚未成功初始化")
            return None, True

        try:
            # 3. 对用户问题分词
            query_tokens = preprocess_text(query)

            if not query_tokens:
                logger.warning("用户问题分词结果为空")
                return None, True

            # 4. 计算用户问题与所有标准问题的 BM25 分数
            scores = self.bm25.get_scores(query_tokens)

            if len(scores) == 0:
                logger.warning("BM25 没有返回有效分数")
                return None, True

            # 5. 按教材使用 Softmax 转换分数
            softmax_scores = self._softmax(scores)

            # 6. 找到最高分所在索引
            best_idx = int(np.argmax(softmax_scores))

            # 最高分
            best_score = float(softmax_scores[best_idx])

            # 通过相同索引找到原始标准问题
            best_question = self.original_questions[best_idx]

            logger.info(
                f"最佳标准问题：{best_question}，"
                f"Softmax 分数：{best_score:.4f}"
            )

            # 7. 判断最高分是否达到阈值
            if best_score >= threshold:
                # 根据标准问题从 MySQL 查询答案
                answer = self.mysql_client.fetch_answer(
                    best_question
                )

                if answer is not None:
                    # 将“用户原始问题 → 答案”写入 Redis
                    self.redis_client.set_answer(
                        query,
                        answer
                    )

                    logger.info("FAQ 找到可靠答案")

                    return answer, False

            # 8. 分数未达到阈值，或者 MySQL 没有答案
            logger.info("FAQ 未找到可靠答案，需要转交 RAG")

            return None, True

        except Exception as e:
            logger.error(f"BM25 检索失败：{e}")
            return None, True


if __name__ == "__main__":
    mysql_client = MySQLClient()
    redis_client = RedisClient()

    try:
        bm25_search = BM25Search(
            redis_client=redis_client,
            mysql_client=mysql_client
        )

        query = input("请输入测试问题：").strip()

        answer, need_rag = bm25_search.search(
            query,
            threshold=0.85
        )

        print("FAQ 答案：", answer)
        print("是否需要调用 RAG：", need_rag)

    finally:
        redis_client.close()
        mysql_client.close()
