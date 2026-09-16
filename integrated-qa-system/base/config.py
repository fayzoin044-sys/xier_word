# base/config.py

import configparser
import json
import os
from pathlib import Path


class Config:
    """
    负责读取项目根目录下的 config.ini，
    并将配置内容转换为 Python 属性。
    """

    def __init__(self, config_file: str | Path | None = None):
        # 当前文件位置：
        # integrated_qa_system/base/config.py
        #
        # parent：
        # integrated_qa_system/base
        #
        # parent.parent：
        # integrated_qa_system
        self.project_root = Path(__file__).resolve().parent.parent

        # 如果没有手动指定配置文件，
        # 默认读取项目根目录下的 config.ini
        if config_file is None:
            config_path = self.project_root / "config.ini"
            if not config_path.exists():
                config_path = self.project_root / "config.example.ini"
        else:
            config_path = Path(config_file)

            # 如果传入的是相对路径，
            # 就让它相对于项目根目录
            if not config_path.is_absolute():
                config_path = self.project_root / config_path

        # 检查配置文件是否存在
        if not config_path.exists():
            raise FileNotFoundError(
                f"配置文件不存在：{config_path}"
            )

        self.config_path = config_path

        # 创建配置解析器
        self.config = configparser.ConfigParser()

        # 读取配置文件
        read_files = self.config.read(
            config_path,
            encoding="utf-8"
        )

        # 如果没有成功读取文件，则主动报错
        if not read_files:
            raise RuntimeError(
                f"配置文件读取失败：{config_path}"
            )

        # Environment overrides keep credentials out of tracked files.
        for section, fields in {
            "mysql": ("host", "port", "user", "password", "database"),
            "redis": ("host", "port", "password", "db"),
            "milvus": ("host", "port", "database_name", "collection_name"),
        }.items():
            for field in fields:
                value = os.getenv(f"{section}_{field}".upper())
                if value is not None:
                    if not self.config.has_section(section):
                        self.config.add_section(section)
                    self.config.set(section, field, value)

        # =========================
        # MySQL 配置
        # =========================
        self.MYSQL_HOST = self.config.get(
            "mysql",
            "host",
            fallback="localhost"
        )

        self.MYSQL_PORT = self.config.getint(
            "mysql",
            "port",
            fallback=3306
        )

        self.MYSQL_USER = self.config.get(
            "mysql",
            "user",
            fallback="root"
        )

        self.MYSQL_PASSWORD = self.config.get(
            "mysql",
            "password",
            fallback=""
        )

        self.MYSQL_DATABASE = self.config.get(
            "mysql",
            "database",
            fallback="subjects_kg"
        )

        # =========================
        # Redis 配置
        # =========================
        self.REDIS_HOST = self.config.get(
            "redis",
            "host",
            fallback="localhost"
        )

        self.REDIS_PORT = self.config.getint(
            "redis",
            "port",
            fallback=6379
        )

        self.REDIS_PASSWORD = self.config.get(
            "redis",
            "password",
            fallback=""
        )

        self.REDIS_DB = self.config.getint(
            "redis",
            "db",
            fallback=0
        )

        # =========================
        # RAG 配置  新增
        # =========================

        # Milvus 配置
        self.MILVUS_HOST = self.config.get(
            "milvus",
            "host",
            fallback="localhost"
        )

        self.MILVUS_PORT = self.config.getint(
            "milvus",
            "port",
            fallback=19530
        )

        self.MILVUS_DATABASE_NAME = self.config.get(
            "milvus",
            "database_name",
            fallback="itcast"
        )

        self.MILVUS_COLLECTION_NAME = self.config.get(
            "milvus",
            "collection_name",
            fallback="edurag_final"
        )

        self.MILVUS_URI = (
            f"http://{self.MILVUS_HOST}:{self.MILVUS_PORT}"
        )

        # 大模型配置
        self.LLM_MODEL = self.config.get(
            "llm",
            "model",
            fallback="deepseek-v4-pro"
        )

        # 优先从环境变量读取密钥，避免把密钥直接写进配置文件
        self.LLM_API_KEY = os.getenv(
            "DEEPSEEK_API_KEY",
            self.config.get(
                "llm",
                "api_key",
                fallback=""
            )
        )

        self.LLM_BASE_URL = self.config.get(
            "llm",
            "base_url",
            fallback="https://api.deepseek.com"
        )

        # 检索配置
        # 父块大小：父块包含较完整的上下文，最终用于交给大模型回答
        self.PARENT_CHUNK_SIZE = self.config.getint(
            "retrieval",
            "parent_chunk_size",
            fallback=1200
        )

        # 子块大小：子块用于向量化和检索，便于更精确地匹配问题
        self.CHILD_CHUNK_SIZE = self.config.getint(
            "retrieval",
            "child_chunk_size",
            fallback=300
        )

        # 块重叠大小：让相邻块保留部分相同内容，避免语义被切断
        self.CHUNK_OVERLAP = self.config.getint(
            "retrieval",
            "chunk_overlap",
            fallback=50
        )

        # 初始检索数量：从 Milvus 召回相似度最高的前 K 个子块
        self.RETRIEVAL_K = self.config.getint(
            "retrieval",
            "retrieval_k",
            fallback=5
        )

        # 最终候选数量：筛选或重排后保留给大模型的上下文数量
        self.CANDIDATE_M = self.config.getint(
            "retrieval",
            "candidate_m",
            fallback=2
        )

        # 应用配置
        valid_sources_text = self.config.get(
            "app",
            "valid_sources",
            fallback='["ai", "java", "test", "ops", "bigdata"]'
        )

        try:
            self.VALID_SOURCES = json.loads(valid_sources_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "config.ini 中 app.valid_sources 必须是 JSON 列表"
            ) from exc

        self.CUSTOMER_SERVICE_PHONE = self.config.get(
            "app",
            "customer_service_phone",
            fallback="12345678"
        )

        # rag部分新增

        # =========================
        # 日志配置
        # =========================
        self.LOG_FILE = self.config.get(
            "logger",
            "log_file",
            fallback="logs/app.log"
        )


# 创建一个全局配置对象
# 后续模块可以直接导入 config 使用
config = Config()


if __name__ == "__main__":
    # 配置模块测试
    print("配置文件路径：", config.config_path)

    print("MySQL 地址：", config.MYSQL_HOST)
    print("MySQL 端口：", config.MYSQL_PORT)
    print("MySQL 用户：", config.MYSQL_USER)
    print("MySQL 数据库：", config.MYSQL_DATABASE)

    print("Redis 地址：", config.REDIS_HOST)
    print("Redis 端口：", config.REDIS_PORT)
    print("Redis 数据库编号：", config.REDIS_DB)

    print("日志路径：", config.LOG_FILE)

    print("MySQL 端口类型：", type(config.MYSQL_PORT))
    print("Redis 端口类型：", type(config.REDIS_PORT))
    print("Redis DB 类型：", type(config.REDIS_DB))
