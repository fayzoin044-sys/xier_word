# mysql_qa/cache/redis_client.py

import json
import sys
from pathlib import Path

import redis

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base.config import config
from base.logger import logger


class RedisClient:
    """负责 FAQ 系统中的 Redis 缓存操作。"""

    def __init__(self):
        self.client = None

        try:
            # 创建 Redis 客户端
            self.client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                password=config.REDIS_PASSWORD,
                db=config.REDIS_DB,
                decode_responses=True
            )

            # 真正向 Redis 发送请求，验证当前连接有效
            self.client.ping()

            logger.info("Redis 连接成功")

        except redis.RedisError as e:
            logger.error(f"Redis 连接失败：{e}")
            raise

    def set_data(self, key, value):
        """
        保存通用缓存。

        可保存字符串、列表、字典等能够转换为 JSON 的数据。
        """

        try:
            # Python 对象转换为 JSON 字符串
            json_value = json.dumps(
                value,
                ensure_ascii=False#防止中文乱码
            )

            self.client.set(key, json_value)#存入数据库

            logger.info(f"Redis 数据写入成功：{key}")

            return True

        except (redis.RedisError, TypeError) as e:
            logger.error(f"Redis 数据写入失败：{e}")
            return False

    def get_data(self, key):
        """读取通用缓存，并恢复成原来的 Python 数据。"""

        try:
            data = self.client.get(key)

            if data is None:
                return None

            # JSON 字符串恢复为 Python 对象
            return json.loads(data)

        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.error(f"Redis 数据读取失败：{e}")
            return None

    def set_answer(self, query, answer):
        """
        缓存用户问题对应的答案。

        例如：
        answer:字符串怎么切割
        """

        return self.set_data(
            key=f"answer:{query}",
            value=answer
        )

    def get_answer(self, query):
        """查询某个用户问题是否已有答案缓存。"""

        answer = self.get_data(
            key=f"answer:{query}"
        )

        if answer is not None:
            logger.info(f"Redis 答案缓存命中：{query}")

        return answer

    def delete(self, key):
        """删除指定缓存。"""

        try:
            deleted_count = self.client.delete(key)

            if deleted_count > 0:
                logger.info(f"Redis 缓存删除成功：{key}")

            return deleted_count

        except redis.RedisError as e:
            logger.error(f"Redis 缓存删除失败：{e}")
            return 0

    def exists(self, key):
        """判断缓存键是否存在。"""

        try:
            return bool(self.client.exists(key))

        except redis.RedisError as e:
            logger.error(f"Redis 缓存检查失败：{e}")
            return False

    def close(self):
        """关闭 Redis 客户端。"""

        if self.client is not None:
            self.client.close()

        logger.info("Redis 连接已关闭")


if __name__ == "__main__":
    redis_client = RedisClient()

    try:
        # 测试通用缓存
        redis_client.set_data(
            key="test:faq",
            value={
                "question": "字符串怎么切割",
                "answer": "使用 split 方法"
            }
        )

        result = redis_client.get_data("test:faq")
        print("通用缓存读取结果：", result)

        # 测试答案缓存
        redis_client.set_answer(
            query="字符串怎么切割",
            answer="使用 split 方法"
        )

        answer = redis_client.get_answer(
            "字符串怎么切割"
        )

        print("答案缓存读取结果：", answer)

    finally:
        # 删除测试数据，避免污染正式缓存
        redis_client.delete("test:faq")
        redis_client.delete("answer:字符串怎么切割")

        redis_client.close()
