"""Milvus 向量存储、混合检索与父块重排序。

本模块负责两件事：
1. 将 ``document_processor.py`` 生成的子块向量化后写入 Milvus；
2. 根据用户问题检索子块，再恢复、去重并重排对应的父块。

注意：本模块不调用 DeepSeek，也不负责生成最终回答。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import AnnSearchRequest, DataType, MilvusClient, WeightedRanker
from sentence_transformers import CrossEncoder


# 兼容两种运行方式：
# 1. 在项目中导入 rag_qa.core.vector_store；
# 2. 直接运行本文件附近的测试代码。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base.config import config
from base.logger import logger


# 当前项目已经保存了老师提供的本地模型，因此默认直接读取本地目录，
# 不再使用 Hugging Face 模型名称，也不会在首次运行时重复联网下载。
RAG_QA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDING_MODEL = str(RAG_QA_ROOT / "models" / "bge-m3")
DEFAULT_RERANKER_MODEL = str(
    RAG_QA_ROOT / "models" / "bge-reranker-large"
)


class VectorStore:
    """封装 BGE-M3、Milvus 混合检索和 BGE 重排序。"""

    # Collection 中显式定义的字段。加载已有 Collection 时会检查这些字段，
    # 避免连接到了同名但结构不兼容的旧集合后，在写入阶段才出现难以理解的错误。
    REQUIRED_FIELDS = {
        "id",
        "text",
        "dense_vector",
        "sparse_vector",
        "parent_id",
        "parent_content",
        "source",
        "file_path",
        "timestamp",
    }

    def __init__(
        self,
        collection_name: str = config.MILVUS_COLLECTION_NAME,
        uri: str = config.MILVUS_URI,
        database: str = config.MILVUS_DATABASE_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        device: str = "cpu",
    ) -> None:
        """初始化模型，确保 Milvus 数据库和 Collection 可用。

        Args:
            collection_name: Milvus Collection 名称。
            uri: Milvus 服务地址，例如 ``http://127.0.0.1:19530``。
            database: Milvus 数据库名称。
            embedding_model: BGE-M3 模型名称或本地目录。
            reranker_model: BGE Reranker 模型名称或本地目录。
            device: 模型运行设备。当前默认使用 CPU，GPU 可传入 ``cuda``。

        注意：创建 ``VectorStore`` 会连接 Milvus 并从项目的 ``models``
        目录加载两个本地模型，所以初始化仍可能需要一定时间，但不会重复下载。
        """

        self.collection_name = collection_name
        self.uri = uri
        self.database = database
        self.device = device
        self.client: MilvusClient | None = None

        try:
            # 先确认 Milvus 可连接。这样当 Milvus 未启动时，不会先花时间下载大模型。
            self._connect_and_select_database()

            logger.info("开始加载 BGE-M3 向量模型：%s", embedding_model)
            self.embedding_function = BGEM3EmbeddingFunction(
                model_name=embedding_model,
                use_fp16=False,
                device=device,
            )
            self.dense_dim = int(self.embedding_function.dim["dense"])

            logger.info("开始加载 BGE 重排序模型：%s", reranker_model)
            self.reranker = CrossEncoder(
                reranker_model,
                device=device,
            )

            self._create_or_load_collection()
            logger.info(
                "VectorStore 初始化完成：database=%s，collection=%s，dense_dim=%d",
                self.database,
                self.collection_name,
                self.dense_dim,
            )
        except Exception:
            # 初始化任意环节失败都关闭已建立的客户端，避免连接泄漏。
            self.close()
            logger.exception("VectorStore 初始化失败")
            raise

    # =========================
    # 第一部分：数据库连接
    # =========================
    def _connect_and_select_database(self) -> None:
        """连接 Milvus，必要时创建数据库，并切换到目标数据库。"""

        # 不能直接使用 db_name=self.database 连接，因为数据库第一次运行时可能不存在。
        self.client = MilvusClient(uri=self.uri)
        databases = self.client.list_databases()

        if self.database not in databases:
            self.client.create_database(db_name=self.database)
            logger.info("已创建 Milvus 数据库：%s", self.database)
        else:
            logger.info("Milvus 数据库已存在：%s", self.database)

        self.client.use_database(self.database)
        logger.info("已切换到 Milvus 数据库：%s", self.database)

    # =========================
    # 第二部分：Collection 和索引
    # =========================
    def _create_or_load_collection(self) -> None:
        """创建或加载保存子块的 Collection。"""

        client = self._require_client()

        if client.has_collection(self.collection_name):
            self._validate_existing_collection()
            logger.info("Milvus Collection 已存在：%s", self.collection_name)
        else:
            # auto_id=False：子块 ID 由 document_processor.py 生成并作为主键。
            # enable_dynamic_field=False：字段全部显式定义，数据结构更容易检查和维护。
            schema = client.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=100,
            )
            schema.add_field(
                field_name="text",
                datatype=DataType.VARCHAR,
                max_length=65535,
            )
            schema.add_field(
                field_name="dense_vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=self.dense_dim,
            )
            schema.add_field(
                field_name="sparse_vector",
                datatype=DataType.SPARSE_FLOAT_VECTOR,
            )
            schema.add_field(
                field_name="parent_id",
                datatype=DataType.VARCHAR,
                max_length=100,
            )
            schema.add_field(
                field_name="parent_content",
                datatype=DataType.VARCHAR,
                max_length=65535,
            )
            schema.add_field(
                field_name="source",
                datatype=DataType.VARCHAR,
                max_length=50,
            )
            schema.add_field(
                field_name="file_path",
                datatype=DataType.VARCHAR,
                max_length=2048,
            )
            schema.add_field(
                field_name="timestamp",
                datatype=DataType.VARCHAR,
                max_length=64,
            )

            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128},
            )
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2},
            )

            client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
            )
            logger.info("已创建 Milvus Collection：%s", self.collection_name)

        # Collection 必须加载到查询节点后才能执行向量检索。
        client.load_collection(self.collection_name)
        logger.info("已加载 Milvus Collection：%s", self.collection_name)

    def _validate_existing_collection(self) -> None:
        """检查同名 Collection 是否包含当前代码需要的字段。"""

        client = self._require_client()
        description = client.describe_collection(self.collection_name)
        current_fields = {
            field.get("name")
            for field in description.get("fields", [])
            if field.get("name")
        }
        missing_fields = self.REQUIRED_FIELDS - current_fields

        if missing_fields:
            missing_text = ", ".join(sorted(missing_fields))
            raise RuntimeError(
                f"Collection {self.collection_name} 缺少字段：{missing_text}。"
                "请确认是否连接到了旧版同名 Collection。"
            )

    # =========================
    # 第三部分：子块向量化并写入 Milvus
    # =========================
    def add_documents(
        self,
        documents: Iterable[Document],
        batch_size: int = 32,
    ) -> int:
        """将子块 Document 分批向量化并 upsert 到 Milvus。

        Args:
            documents: ``process_documents()`` 返回的子块。
            batch_size: 每批处理的子块数量，避免一次加载过多文本占用内存。

        Returns:
            成功提交给 Milvus 的子块数量。
        """

        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")

        document_list = list(documents)
        if not document_list:
            logger.warning("没有需要写入 Milvus 的子块 Document")
            return 0

        client = self._require_client()
        total_upserted = 0

        for start in range(0, len(document_list), batch_size):
            batch = document_list[start : start + batch_size]
            texts = [document.page_content.strip() for document in batch]

            if any(not text for text in texts):
                raise ValueError("待写入的子块中存在空 page_content")

            # 一次调用同时产生稠密向量和稀疏向量。
            embeddings = self.embedding_function.encode_documents(texts)
            entities: list[dict[str, Any]] = []

            for index, document in enumerate(batch):
                metadata = document.metadata
                child_id = self._required_metadata(metadata, "id")
                parent_id = self._required_metadata(metadata, "parent_id")
                parent_content = self._required_metadata(metadata, "parent_content")

                entities.append(
                    {
                        "id": str(child_id),
                        "text": texts[index],
                        "dense_vector": self._dense_vector_to_list(
                            embeddings["dense"][index]
                        ),
                        "sparse_vector": self._sparse_row_to_dict(
                            embeddings["sparse"], index
                        ),
                        "parent_id": str(parent_id),
                        "parent_content": str(parent_content),
                        "source": str(metadata.get("source", "unknown")),
                        "file_path": str(metadata.get("file_path", "unknown")),
                        "timestamp": str(metadata.get("timestamp", "unknown")),
                    }
                )

            # upsert 根据主键决定插入或更新，重复处理同一资料不会无限增加重复记录。
            client.upsert(
                collection_name=self.collection_name,
                data=entities,
            )
            total_upserted += len(entities)
            logger.info(
                "已写入 Milvus：本批 %d 个子块，累计 %d/%d",
                len(entities),
                total_upserted,
                len(document_list),
            )

        # 所有批次完成后统一 flush，确保已提交数据持久化并可用于后续查询。
        client.flush(self.collection_name)
        logger.info("子块入库完成，共 upsert %d 条记录", total_upserted)
        return total_upserted

    @staticmethod
    def _required_metadata(metadata: dict[str, Any], key: str) -> Any:
        """读取必需元数据，缺失时给出明确错误，而不是写入残缺记录。"""

        value = metadata.get(key)
        if value is None or value == "":
            raise ValueError(f"子块 metadata 缺少必需字段：{key}")
        return value

    @staticmethod
    def _dense_vector_to_list(vector: Any) -> list[float]:
        """把 NumPy 等向量类型转换为 Milvus 可序列化的 Python 列表。"""

        values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        return [float(value) for value in values]

    @staticmethod
    def _sparse_row_to_dict(sparse_vectors: Any, row_index: int) -> dict[int, float]:
        """将一行 SciPy 稀疏向量转换为 Milvus 接受的字典。

        当前 SciPy 返回的是 ``csr_array``，它没有教材代码使用的 ``getrow()``；
        因此先按索引取行，再统一转换成 COO 格式读取列号和权重。
        """

        row = sparse_vectors[row_index]
        if isinstance(row, dict):
            return {int(index): float(value) for index, value in row.items()}

        if hasattr(row, "tocoo"):
            row = row.tocoo()

        indices = getattr(row, "col", None)
        if indices is None:
            indices = getattr(row, "indices", None)
        values = getattr(row, "data", None)

        if indices is None or values is None:
            raise TypeError("无法识别 BGE-M3 返回的稀疏向量格式")

        return {
            int(index): float(value)
            for index, value in zip(indices, values)
        }

    # =========================
    # 第四部分：混合检索、父块去重与重排序
    # =========================
    def hybrid_search_with_rerank(
        self,
        query: str,
        k: int = config.RETRIEVAL_K,
        source_filter: str | None = None,
    ) -> list[Document]:
        """执行稠密+稀疏混合检索，最终返回重排后的父块。"""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")
        if k <= 0:
            raise ValueError("k 必须大于 0")

        query = query.strip()
        filter_expression = self._build_source_filter(source_filter)
        query_embeddings = self.embedding_function.encode_queries([query])
        dense_query_vector = self._dense_vector_to_list(
            query_embeddings["dense"][0]
        )
        sparse_query_vector = self._sparse_row_to_dict(
            query_embeddings["sparse"], 0
        )

        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=k,
            expr=filter_expression,
        )
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=k,
            expr=filter_expression,
        )

        # 权重必须与请求顺序一致：dense_request=1.0，sparse_request=0.7。
        ranker = WeightedRanker(1.0, 0.7)
        client = self._require_client()
        search_batches = client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=k,
            output_fields=[
                "id",
                "text",
                "parent_id",
                "parent_content",
                "source",
                "file_path",
                "timestamp",
            ],
        )

        hits = search_batches[0] if search_batches else []
        child_documents = [self._doc_from_hit(hit) for hit in hits]
        parent_documents = self._get_unique_parent_docs(child_documents)

        logger.info(
            "混合检索完成：查询=%r，命中子块=%d，去重父块=%d",
            query,
            len(child_documents),
            len(parent_documents),
        )

        if not parent_documents:
            return []

        # 只有一个父块时无需调用重排序模型，直接返回即可。
        if len(parent_documents) == 1:
            return parent_documents[: config.CANDIDATE_M]

        pairs = [[query, document.page_content] for document in parent_documents]
        raw_scores = self.reranker.predict(
            pairs,
            show_progress_bar=False,
        )
        scored_documents: list[tuple[float, Document]] = []

        for raw_score, document in zip(raw_scores, parent_documents):
            score = self._score_to_float(raw_score)
            document.metadata["rerank_score"] = score
            scored_documents.append((score, document))

        # 只按数值分数排序，避免分数相同时 Python 尝试比较 Document 对象。
        scored_documents.sort(key=lambda item: item[0], reverse=True)
        final_documents = [
            document
            for _, document in scored_documents[: config.CANDIDATE_M]
        ]
        logger.info(
            "父块重排序完成：候选=%d，最终返回=%d",
            len(parent_documents),
            len(final_documents),
        )
        return final_documents

    @staticmethod
    def _build_source_filter(source_filter: str | None) -> str | None:
        """校验知识分类，并生成 Milvus 标量过滤表达式。"""

        if source_filter is None or not source_filter.strip():
            return None

        source_filter = source_filter.strip()
        if source_filter not in config.VALID_SOURCES:
            raise ValueError(
                f"无效的 source_filter：{source_filter}；"
                f"允许值：{config.VALID_SOURCES}"
            )

        # 虽然配置中的分类名是可信值，仍进行转义，避免将来修改配置时破坏表达式。
        escaped_source = source_filter.replace("\\", "\\\\").replace('"', '\\"')
        return f'source == "{escaped_source}"'

    @staticmethod
    def _doc_from_hit(hit: dict[str, Any]) -> Document:
        """把 Milvus 命中的一条子块记录恢复成 LangChain Document。"""

        entity = hit.get("entity", hit)
        retrieval_score = hit.get("distance", hit.get("score"))
        child_id = hit.get("id", entity.get("id"))

        return Document(
            page_content=str(entity.get("text", "")),
            metadata={
                "id": child_id,
                "parent_id": entity.get("parent_id"),
                "parent_content": entity.get("parent_content"),
                "source": entity.get("source", "unknown"),
                "file_path": entity.get("file_path", "unknown"),
                "timestamp": entity.get("timestamp", "unknown"),
                "retrieval_score": retrieval_score,
            },
        )

    @staticmethod
    def _get_unique_parent_docs(
        child_documents: Iterable[Document],
    ) -> list[Document]:
        """按照 parent_id 恢复并去重父块，同时保持原检索排名顺序。"""

        seen_parent_ids: set[str] = set()
        parent_documents: list[Document] = []

        for child_document in child_documents:
            parent_id = child_document.metadata.get("parent_id")
            parent_content = child_document.metadata.get("parent_content")

            if not parent_id or not parent_content:
                logger.warning(
                    "检索结果缺少 parent_id 或 parent_content，已跳过：%s",
                    child_document.metadata.get("id"),
                )
                continue

            parent_id = str(parent_id)
            if parent_id in seen_parent_ids:
                continue

            parent_metadata = dict(child_document.metadata)
            parent_metadata.pop("parent_content", None)
            parent_metadata["parent_id"] = parent_id
            parent_documents.append(
                Document(
                    page_content=str(parent_content),
                    metadata=parent_metadata,
                )
            )
            seen_parent_ids.add(parent_id)

        return parent_documents

    @staticmethod
    def _score_to_float(score: Any) -> float:
        """把 NumPy/Torch 标量统一转换为 Python float。"""

        if hasattr(score, "item"):
            score = score.item()
        return float(score)

    # =========================
    # 第五部分：资源管理
    # =========================
    def _require_client(self) -> MilvusClient:
        """返回已初始化客户端；未初始化时主动报错。"""

        if self.client is None:
            raise RuntimeError("Milvus 客户端尚未初始化或已经关闭")
        return self.client

    def close(self) -> None:
        """关闭 Milvus 客户端。重复调用也不会报错。"""

        if self.client is not None:
            self.client.close()
            self.client = None
            logger.info("Milvus 客户端已关闭")

    def __enter__(self) -> "VectorStore":
        """支持 ``with VectorStore() as store`` 的资源管理写法。"""

        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


# =========================
# 第六部分：当前模块的独立测试入口
# =========================
def main() -> None:
    """通过命令行独立测试文档入库或向量检索。

    这个入口只用于当前模块的学习和验证。完整 RAG 系统完成后，正常业务流程
    会由 ``rag_qa/main.py`` 和 ``rag_system.py`` 调用 ``VectorStore``。
    """

    import argparse

    parser = argparse.ArgumentParser(
        description="测试 RAG 的 Milvus 文档入库与混合检索",
    )
    parser.add_argument(
        "--mode",
        choices=["store", "query"],
        required=True,
        help="store=处理文档并入库，query=输入问题并检索父块",
    )
    parser.add_argument(
        "--data-dir",
        default=str(RAG_QA_ROOT / "data" / "ai_data"),
        help="store 模式使用的知识文件目录",
    )
    parser.add_argument(
        "--query",
        help="query 模式使用的用户问题；不传时会在控制台提示输入",
    )
    parser.add_argument(
        "--source",
        choices=config.VALID_SOURCES,
        help="query 模式可选的知识分类过滤条件",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="store 模式每批向量化和写入的子块数量",
    )
    args = parser.parse_args()

    # with 代码块结束时会自动调用 VectorStore.close()，关闭 Milvus 客户端。
    with VectorStore() as vector_store:
        if args.mode == "store":
            # 文档处理属于 document_processor.py；这里只接收其输出并负责向量入库。
            from rag_qa.core.document_processor import process_documents

            child_documents = process_documents(args.data_dir)
            stored_count = vector_store.add_documents(
                child_documents,
                batch_size=args.batch_size,
            )
            print("\n===== 文档入库测试完成 =====")
            print(f"知识目录：{Path(args.data_dir).resolve()}")
            print(f"生成子块：{len(child_documents)} 个")
            print(f"写入 Milvus：{stored_count} 条")
            return

        query = args.query or input("请输入要检索的问题：").strip()
        parent_documents = vector_store.hybrid_search_with_rerank(
            query=query,
            source_filter=args.source,
        )

        print("\n===== 混合检索测试完成 =====")
        print(f"问题：{query}")
        print(f"返回父块：{len(parent_documents)} 个")

        for index, document in enumerate(parent_documents, start=1):
            print(f"\n----- 父块 {index} -----")
            print("parent_id：", document.metadata.get("parent_id"))
            print("source：", document.metadata.get("source"))
            print("file_path：", document.metadata.get("file_path"))
            print("retrieval_score：", document.metadata.get("retrieval_score"))
            print("rerank_score：", document.metadata.get("rerank_score"))
            print("page_content：")
            print(document.page_content)


if __name__ == "__main__":
    #测试部分
    # main()


    # 直接测试文档入库
    from rag_qa.core.document_processor import process_documents

    # 需要导入的知识文件目录
    data_dir = RAG_QA_ROOT / "data" / "ai_data"

    # 读取文档并生成父子分块
    child_documents = process_documents(data_dir)

    # 连接 Milvus 并写入向量数据
    with VectorStore() as vector_store:
        stored_count = vector_store.add_documents(
            child_documents,
            batch_size=32,
        )

    print("\n===== 文档入库完成 =====")
    print(f"知识目录：{data_dir.resolve()}")
    print(f"生成子块：{len(child_documents)} 个")
    print(f"写入 Milvus：{stored_count} 条")
