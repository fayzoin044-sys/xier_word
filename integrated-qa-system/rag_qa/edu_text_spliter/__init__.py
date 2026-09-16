"""EduRAG 文本切分工具的公共入口。"""

# 当前 RAG 文档处理流程只使用中文递归切分器。
# AliTextSplitter 依赖较重的 ModelScope，若以后确实需要语义切分，
# 再从 edu_model_text_spliter 中单独导入，避免普通切分被可选依赖阻塞。
from .edu_chinese_recursive_text_splitter import ChineseRecursiveTextSplitter

__all__ = ["ChineseRecursiveTextSplitter"]
