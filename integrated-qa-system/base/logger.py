# base/logger.py

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base.config import config


def setup_logging(log_file: str | Path | None = None) -> logging.Logger:
    """
    创建项目统一使用的日志器。

    日志同时输出到：
    1. PyCharm 控制台
    2. logs/app.log 文件
    """

    # 没有手动传入路径时，读取 config.ini 中的日志路径
    if log_file is None:
        log_file = config.LOG_FILE

    # 转换为 Path 对象
    log_path = Path(log_file)

    # 如果配置的是相对路径 logs/app.log，
    # 就将它拼接到项目根目录
    if not log_path.is_absolute():
        log_path = config.project_root / log_path

    # 自动创建日志目录
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 获取名为 EduRAG 的日志器
    logger = logging.getLogger("EduRAG")

    # 记录 INFO 及以上级别的日志
    logger.setLevel(logging.INFO)

    # 防止日志继续传给根日志器，避免重复输出
    logger.propagate = False

    # 防止重复导入模块时重复添加处理器
    if logger.handlers:
        return logger

    # 设置日志显示格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 文件处理器：将日志写入 app.log
    file_handler = logging.FileHandler(
        filename=log_path,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # 控制台处理器：将日志显示在 PyCharm 控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 把两个处理器交给 logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# 创建项目统一使用的日志器对象
logger = setup_logging()
