"""RAG 文档加载与父子分块。

这个模块只负责把原始文件处理成适合后续向量化的子块 ``Document``，
不负责连接或写入 Milvus。
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime
from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter


# 兼容两种运行方式：
# 1. 在项目中导入 rag_qa.core.document_processor
# 2. 直接运行本文件进行测试
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base.config import config
from base.logger import logger
from rag_qa.edu_document_loaders import (
    OCRDOCLoader,
    OCRIMGLoader,
    OCRPDFLoader,
    OCRPPTLoader,
)
from rag_qa.edu_text_spliter import ChineseRecursiveTextSplitter


# 文件扩展名与加载器的对应关系。
# 加载器的共同输出都是 list[Document]，因此后续切分逻辑不需要关心原文件格式。
DOCUMENT_LOADERS = {
    ".txt": TextLoader,
    ".pdf": OCRPDFLoader,
    ".docx": OCRDOCLoader,
    ".pptx": OCRPPTLoader,
    ".jpg": OCRIMGLoader,
    ".jpeg": OCRIMGLoader,
    ".png": OCRIMGLoader,
    ".md": UnstructuredMarkdownLoader,
}


def _source_from_directory(directory: Path) -> str:
    """根据知识目录名生成分类，例如 ``ai_data`` 转换为 ``ai``。"""

    return directory.name.removesuffix("_data")


def _stable_id(prefix: str, *parts: object) -> str:
    """根据文件和块信息生成稳定 ID，避免多次处理不同目录时发生撞号。"""

    raw_value = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(raw_value.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}_{digest}"


def _create_loader(file_path: Path):
    """根据扩展名创建对应的文档加载器。"""

    extension = file_path.suffix.lower()
    loader_class = DOCUMENT_LOADERS[extension]

    # 普通文本需要明确指定 UTF-8，避免不同操作系统使用不同默认编码。
    if extension == ".txt":
        return loader_class(str(file_path), encoding="utf-8")

    return loader_class(str(file_path))


def load_documents_from_directory(
    directory_path: str | Path,
) -> list[Document]:
    """递归读取目录中的支持文件，返回尚未父子切分的原始文档。

    Args:
        directory_path: 知识文件所在目录，例如 ``rag_qa/data/ai_data``。

    Returns:
        加载成功的原始 ``Document`` 列表。每个文档都包含 ``source``、
        ``file_path`` 和 ``timestamp`` 元数据。

    Raises:
        FileNotFoundError: 目录不存在。
        NotADirectoryError: 传入路径不是目录。
    """

    directory = Path(directory_path).expanduser()
    if not directory.is_absolute():
        directory = (Path.cwd() / directory).resolve()
    else:
        directory = directory.resolve()

    if not directory.exists():
        raise FileNotFoundError(f"文档目录不存在：{directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"传入路径不是目录：{directory}")

    source = _source_from_directory(directory)
    documents: list[Document] = []

    # sorted() 让每次处理顺序一致，便于测试和排查问题。
    for file_path in sorted(path for path in directory.rglob("*") if path.is_file()):
        extension = file_path.suffix.lower()
        if extension not in DOCUMENT_LOADERS:
            logger.warning("跳过不支持的文件类型：%s", file_path)
            continue

        try:
            loader = _create_loader(file_path)
            loaded_documents = loader.load()
            timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

            valid_documents = 0
            for document in loaded_documents:
                if not document.page_content.strip():
                    logger.warning("文件未提取到有效文字，已跳过：%s", file_path)
                    continue

                # 加载器可能已经提供 metadata；这里保留其他字段，只统一项目需要的字段。
                document.metadata["source"] = source
                document.metadata["file_path"] = str(file_path)
                document.metadata["timestamp"] = timestamp
                documents.append(document)
                valid_documents += 1

            logger.info(
                "成功加载文件：%s，得到 %d 个原始 Document",
                file_path,
                valid_documents,
            )
        except Exception:
            # 单个文件失败不能影响同一目录中的其他知识文件。
            logger.exception("加载文件失败：%s", file_path)

    logger.info("目录加载完成：%s，共得到 %d 个原始 Document", directory, len(documents))
    return documents


def _validate_split_parameters(
    parent_chunk_size: int,
    child_chunk_size: int,
    chunk_overlap: int,
) -> None:
    """提前检查切块参数，提供比第三方库更容易理解的错误信息。"""

    if parent_chunk_size <= 0 or child_chunk_size <= 0:
        raise ValueError("父块大小和子块大小必须大于 0")
    if child_chunk_size > parent_chunk_size:
        raise ValueError("子块大小不能大于父块大小")
    if chunk_overlap < 0:
        raise ValueError("块重叠大小不能小于 0")
    if chunk_overlap >= child_chunk_size:
        raise ValueError("块重叠大小必须小于子块大小")


def process_documents(
    directory_path: str | Path,
    parent_chunk_size: int = config.PARENT_CHUNK_SIZE,
    child_chunk_size: int = config.CHILD_CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[Document]:
    """加载目录中的文档，并依次切成父块和子块。

    最终返回的是子块列表：子块正文保存在 ``page_content`` 中；对应父块的
    ID 和完整内容保存在 ``metadata['parent_id']`` 与
    ``metadata['parent_content']`` 中，供后续向量检索和上下文还原使用。
    """

    _validate_split_parameters(
        parent_chunk_size=parent_chunk_size,
        child_chunk_size=child_chunk_size,
        chunk_overlap=chunk_overlap,
    )
    documents = load_documents_from_directory(directory_path)

    parent_splitter = ChineseRecursiveTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=chunk_overlap,
    )
    child_splitter = ChineseRecursiveTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=chunk_overlap,
    )
    markdown_parent_splitter = MarkdownTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=chunk_overlap,
    )
    markdown_child_splitter = MarkdownTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=chunk_overlap,
    )

    child_chunks: list[Document] = []
    parent_count = 0

    for document_index, document in enumerate(documents):
        file_path = document.metadata.get("file_path", "")
        is_markdown = Path(file_path).suffix.lower() == ".md"
        selected_parent_splitter = (
            markdown_parent_splitter if is_markdown else parent_splitter
        )
        selected_child_splitter = (
            markdown_child_splitter if is_markdown else child_splitter
        )

        logger.info(
            "开始切分文档：%s，切分器：%s",
            file_path,
            "Markdown" if is_markdown else "ChineseRecursive",
        )

        parent_documents = selected_parent_splitter.split_documents([document])
        parent_count += len(parent_documents)

        for parent_index, parent_document in enumerate(parent_documents):
            parent_id = _stable_id(
                "parent",
                file_path,
                document_index,
                parent_index,
                parent_document.page_content,
            )
            parent_content = parent_document.page_content

            parent_document.metadata["parent_id"] = parent_id
            parent_document.metadata["parent_content"] = parent_content

            sub_chunks = selected_child_splitter.split_documents([parent_document])
            for child_index, child_document in enumerate(sub_chunks):
                child_document.metadata["parent_id"] = parent_id
                child_document.metadata["parent_content"] = parent_content
                child_document.metadata["id"] = _stable_id(
                    "child",
                    parent_id,
                    child_index,
                    child_document.page_content,
                )
                child_chunks.append(child_document)

    logger.info(
        "文档处理完成：原始 Document %d 个，父块 %d 个，子块 %d 个",
        len(documents),
        parent_count,
        len(child_chunks),
    )
    return child_chunks


if __name__ == "__main__":
    # 直接运行本文件时，默认处理项目中的 ai_data，方便观察实际输出。
    demo_directory = Path(__file__).resolve().parents[1] / "data" / "ai_data"
    demo_chunks = process_documents(demo_directory)

    print(f"\n共生成 {len(demo_chunks)} 个子块 Document")
    for index, chunk in enumerate(demo_chunks[:3], start=1):
        print(f"\n===== 子块 {index} =====")
        print("Document 类型：", type(chunk))
        print("page_content：")
        print(chunk.page_content)
        print("metadata：")
        print(chunk.metadata)
