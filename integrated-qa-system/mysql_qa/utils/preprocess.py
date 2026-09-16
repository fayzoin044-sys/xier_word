import sys
from pathlib import Path

import jieba

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base.logger import logger


def preprocess_text(text):
    """
    将原始文本转换为 BM25 可以使用的分词列表。

    输入：
        "字符串怎么切割"

    输出：
        ["字符串", "怎么", "切割"]
    """

    # 输入必须是非空字符串
    if not isinstance(text, str) or not text.strip():
        logger.warning("文本为空或类型不正确")
        return []

    try:
        # 去除前后空格、转为小写、使用 jieba 分词
        tokens = jieba.lcut(text.strip().lower())

        # 去除分词结果中的空白内容
        return [
            token.strip()
            for token in tokens
            if token.strip()
        ]

    except Exception as e:
        logger.error(f"文本预处理失败：{e}")
        return []


if __name__ == "__main__":
    text = "字符串怎么切割"

    result = preprocess_text(text)

    print("原始文本：", text)
    print("分词结果：", result)
#TODO 就是jeaba 分词，输入字符串输出一个分词后的列表
