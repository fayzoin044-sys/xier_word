"""不依赖虚拟机的 RAG 检索与重排评测。

保留生产链路中的 PDF 解析、父子分块、BGE-M3 dense/sparse 表示、
WeightedRanker(1.0, 0.7) 融合和 BGE Reranker；仅将 Milvus ANN 查询替换为
小语料上的进程内精确内积。脚本不会连接 MySQL、Redis、Milvus，也不会调用 LLM API。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


EVALUATION_ROOT = Path(__file__).resolve().parent
INTEGRATED_ROOT = EVALUATION_ROOT.parents[1]
if str(INTEGRATED_ROOT) not in sys.path:
    sys.path.insert(0, str(INTEGRATED_ROOT))

from base.config import config
from rag_qa.core.document_processor import process_documents
from rag_qa.core.vector_store import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANKER_MODEL


DEFAULT_GOLDEN = EVALUATION_ROOT / "data" / "rag_retrieval_golden.jsonl"
DEFAULT_DOCUMENTS = INTEGRATED_ROOT / "rag_qa" / "data" / "ai_data"
DEFAULT_CACHE = EVALUATION_ROOT / "cache"
DEFAULT_RESULTS = EVALUATION_ROOT / "results"


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    question: str
    case_type: str
    reference_parent_indexes: tuple[int, ...]
    reference_answer: str


@dataclass
class RetrievalResult:
    parent_ids: list[str]
    parent_indexes: list[int]
    scores: list[float]
    latency_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--retrieval-k", type=int, default=config.RETRIEVAL_K)
    parser.add_argument("--candidate-m", type=int, default=config.CANDIDATE_M)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--sparse-weight", type=float, default=0.7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--skip-reranker",
        action="store_true",
        help="只跑 dense/sparse/hybrid，便于快速验证数据与向量缓存。",
    )
    return parser.parse_args()


def load_golden(path: Path) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    seen_ids: set[str] = set()
    with path.resolve().open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            case = GoldenCase(
                case_id=str(value["case_id"]),
                question=str(value["question"]).strip(),
                case_type=str(value["case_type"]),
                reference_parent_indexes=tuple(
                    int(item) for item in value["reference_parent_indexes"]
                ),
                reference_answer=str(value["reference_answer"]).strip(),
            )
            if not case.question:
                raise ValueError(f"第 {line_number} 行 question 为空")
            if case.case_id in seen_ids:
                raise ValueError(f"case_id 重复：{case.case_id}")
            cases.append(case)
            seen_ids.add(case.case_id)
    if not cases:
        raise ValueError(f"测试集为空：{path}")
    return cases


def unique_parents(child_chunks: list[Any]) -> tuple[list[str], list[str], dict[str, int]]:
    parent_ids: list[str] = []
    parent_contents: list[str] = []
    parent_index_by_id: dict[str, int] = {}
    for chunk in child_chunks:
        parent_id = str(chunk.metadata["parent_id"])
        if parent_id in parent_index_by_id:
            continue
        parent_index_by_id[parent_id] = len(parent_ids)
        parent_ids.append(parent_id)
        parent_contents.append(str(chunk.metadata["parent_content"]))
    return parent_ids, parent_contents, parent_index_by_id


def corpus_fingerprint(child_chunks: list[Any]) -> str:
    digest = hashlib.sha256()
    digest.update(str(config.PARENT_CHUNK_SIZE).encode())
    digest.update(str(config.CHILD_CHUNK_SIZE).encode())
    digest.update(str(config.CHUNK_OVERLAP).encode())
    for chunk in child_chunks:
        digest.update(str(chunk.metadata["id"]).encode("utf-8"))
        digest.update(chunk.page_content.encode("utf-8"))
    return digest.hexdigest()


def load_or_build_document_embeddings(
    child_chunks: list[Any], cache_dir: Path, device: str, rebuild: bool
) -> tuple[Any, np.ndarray, sparse.csr_array, bool]:
    from milvus_model.hybrid import BGEM3EmbeddingFunction

    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    dense_path = cache_dir / "child_dense.npy"
    sparse_path = cache_dir / "child_sparse.npz"
    expected = {
        "fingerprint": corpus_fingerprint(child_chunks),
        "embedding_model": str(Path(DEFAULT_EMBEDDING_MODEL).resolve()),
        "child_count": len(child_chunks),
    }
    embedding_function = BGEM3EmbeddingFunction(
        model_name=DEFAULT_EMBEDDING_MODEL,
        use_fp16=False,
        device=device,
    )
    if not rebuild and metadata_path.exists() and dense_path.exists() and sparse_path.exists():
        actual = json.loads(metadata_path.read_text(encoding="utf-8"))
        if actual == expected:
            return (
                embedding_function,
                np.load(dense_path),
                sparse.load_npz(sparse_path).tocsr(),
                True,
            )

    texts = [chunk.page_content for chunk in child_chunks]
    encoded = embedding_function.encode_documents(texts)
    dense_vectors = np.asarray(encoded["dense"], dtype=np.float32)
    sparse_vectors = sparse.csr_array(encoded["sparse"], dtype=np.float32)
    np.save(dense_path, dense_vectors)
    sparse.save_npz(sparse_path, sparse_vectors)
    metadata_path.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return embedding_function, dense_vectors, sparse_vectors, False


def top_indices(scores: np.ndarray, k: int) -> list[int]:
    safe_k = min(k, len(scores))
    return sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))[:safe_k]


def normalize_ip(score: float) -> float:
    """复现 Milvus WeightedRanker 对 IP 分数的 arctan 归一化。"""

    return 0.5 + math.atan(score) / math.pi


def parent_rank_from_children(
    ranked_child_indexes: list[int],
    ranked_scores: list[float],
    child_chunks: list[Any],
    parent_index_by_id: dict[str, int],
    base_latency_ms: float,
) -> RetrievalResult:
    parent_ids: list[str] = []
    parent_indexes: list[int] = []
    parent_scores: list[float] = []
    seen: set[str] = set()
    for child_index, score in zip(ranked_child_indexes, ranked_scores):
        parent_id = str(child_chunks[child_index].metadata["parent_id"])
        if parent_id in seen:
            continue
        seen.add(parent_id)
        parent_ids.append(parent_id)
        parent_indexes.append(parent_index_by_id[parent_id])
        parent_scores.append(float(score))
    return RetrievalResult(parent_ids, parent_indexes, parent_scores, base_latency_ms)


def score_query(
    question: str,
    embedding_function: Any,
    dense_vectors: np.ndarray,
    sparse_vectors: sparse.csr_array,
    child_chunks: list[Any],
    parent_index_by_id: dict[str, int],
    retrieval_k: int,
    dense_weight: float,
    sparse_weight: float,
) -> dict[str, RetrievalResult]:
    start = time.perf_counter()
    encoded = embedding_function.encode_queries([question])
    query_dense = np.asarray(encoded["dense"][0], dtype=np.float32)
    query_sparse = sparse.csr_array(encoded["sparse"], dtype=np.float32)
    dense_scores = dense_vectors @ query_dense
    sparse_scores = np.asarray((sparse_vectors @ query_sparse.T).todense()).reshape(-1)
    model_latency_ms = (time.perf_counter() - start) * 1000

    dense_children = top_indices(dense_scores, retrieval_k)
    sparse_children = top_indices(sparse_scores, retrieval_k)
    dense_result = parent_rank_from_children(
        dense_children,
        [float(dense_scores[index]) for index in dense_children],
        child_chunks,
        parent_index_by_id,
        model_latency_ms,
    )
    sparse_result = parent_rank_from_children(
        sparse_children,
        [float(sparse_scores[index]) for index in sparse_children],
        child_chunks,
        parent_index_by_id,
        model_latency_ms,
    )

    dense_set = set(dense_children)
    sparse_set = set(sparse_children)
    fused_scores: dict[int, float] = {}
    for index in dense_set | sparse_set:
        fused_scores[index] = (
            dense_weight * normalize_ip(float(dense_scores[index]))
            if index in dense_set
            else 0.0
        ) + (
            sparse_weight * normalize_ip(float(sparse_scores[index]))
            if index in sparse_set
            else 0.0
        )
    hybrid_children = sorted(
        fused_scores, key=lambda index: (-fused_scores[index], index)
    )[:retrieval_k]
    hybrid_result = parent_rank_from_children(
        hybrid_children,
        [fused_scores[index] for index in hybrid_children],
        child_chunks,
        parent_index_by_id,
        model_latency_ms,
    )
    return {"dense": dense_result, "sparse": sparse_result, "hybrid": hybrid_result}


def rerank_hybrid(
    question: str,
    hybrid: RetrievalResult,
    parent_contents: list[str],
    reranker: Any,
    candidate_m: int,
) -> RetrievalResult:
    start = time.perf_counter()
    pairs = [[question, parent_contents[index]] for index in hybrid.parent_indexes]
    raw_scores = reranker.predict(pairs, show_progress_bar=False) if pairs else []
    reranked = sorted(
        zip(raw_scores, hybrid.parent_ids, hybrid.parent_indexes),
        key=lambda item: float(np.asarray(item[0]).reshape(-1)[0]),
        reverse=True,
    )[:candidate_m]
    return RetrievalResult(
        [item[1] for item in reranked],
        [item[2] for item in reranked],
        [float(np.asarray(item[0]).reshape(-1)[0]) for item in reranked],
        hybrid.latency_ms + (time.perf_counter() - start) * 1000,
    )


def case_metrics(reference: tuple[int, ...], retrieved: list[int]) -> dict[str, float]:
    if not reference:
        return {}
    reference_set = set(reference)
    matches = reference_set.intersection(retrieved)
    first_rank = next(
        (rank for rank, index in enumerate(retrieved, start=1) if index in reference_set), None
    )
    return {
        "hit": float(bool(matches)),
        "hit_at_1": float(bool(retrieved and retrieved[0] in reference_set)),
        "mrr": 0.0 if first_rank is None else 1.0 / first_rank,
        "id_precision": len(matches) / len(retrieved) if retrieved else 0.0,
        "id_recall": len(matches) / len(reference_set),
    }


def aggregate(rows: list[dict[str, Any]], strategies: list[str]) -> dict[str, Any]:
    positives = [row for row in rows if row["reference_parent_indexes"]]
    negatives = [row for row in rows if not row["reference_parent_indexes"]]
    output: dict[str, Any] = {
        "positive_cases": len(positives),
        "hard_negative_cases": len(negatives),
        "note": "生产检索无拒答阈值，hard negative 只记录上下文暴露。",
        "strategies": {},
    }
    for strategy in strategies:
        values = [row["metrics"][strategy] for row in positives]
        metrics = {
            name: round(sum(value[name] for value in values) / len(values), 6)
            for name in ("hit", "hit_at_1", "mrr", "id_precision", "id_recall")
        }
        metrics["avg_latency_ms"] = round(
            sum(row["results"][strategy]["latency_ms"] for row in rows) / len(rows), 3
        )
        metrics["hard_negative_context_exposure_rate"] = (
            round(
                sum(bool(row["results"][strategy]["parent_indexes"]) for row in negatives)
                / len(negatives),
                6,
            )
            if negatives
            else None
        )
        output["strategies"][strategy] = metrics
    return output


def write_outputs(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    parent_ids: list[str],
    parent_contents: list[str],
    results_dir: Path,
    settings: dict[str, Any],
) -> tuple[Path, Path, Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    details_path = results_dir / f"rag_details_{stamp}.csv"
    summary_path = results_dir / f"rag_summary_{stamp}.json"
    report_path = results_dir / f"rag_report_{stamp}.md"
    manifest_path = results_dir / f"parent_manifest_{stamp}.json"
    strategy_names = list(summary["strategies"])

    with details_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = ["case_id", "case_type", "question", "reference_parent_indexes"]
        for name in strategy_names:
            fieldnames.extend(
                [
                    f"{name}_parent_indexes",
                    f"{name}_scores",
                    f"{name}_hit",
                    f"{name}_mrr",
                    f"{name}_latency_ms",
                ]
            )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat: dict[str, Any] = {
                "case_id": row["case_id"],
                "case_type": row["case_type"],
                "question": row["question"],
                "reference_parent_indexes": json.dumps(row["reference_parent_indexes"]),
            }
            for name in strategy_names:
                result = row["results"][name]
                metrics = row["metrics"].get(name, {})
                flat[f"{name}_parent_indexes"] = json.dumps(result["parent_indexes"])
                flat[f"{name}_scores"] = json.dumps(result["scores"])
                flat[f"{name}_hit"] = metrics.get("hit", "")
                flat[f"{name}_mrr"] = metrics.get("mrr", "")
                flat[f"{name}_latency_ms"] = result["latency_ms"]
            writer.writerow(flat)

    payload = {"settings": settings, "summary": summary, "cases": rows}
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            [
                {"parent_index": index, "parent_id": parent_id, "content": content}
                for index, (parent_id, content) in enumerate(zip(parent_ids, parent_contents))
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# RAG 离线检索评测报告",
        "",
        "> 未连接 MySQL、Redis、Milvus，也未调用 DeepSeek。Milvus ANN 被 35 个子块上的精确内积替代，其余核心模型与参数沿用项目配置。",
        "",
        "## 配置",
        "",
        f"- 正例：{summary['positive_cases']}；hard negative：{summary['hard_negative_cases']}",
        f"- 父块 / 子块：{settings['parent_count']} / {settings['child_count']}",
        f"- chunk：parent={settings['parent_chunk_size']}，child={settings['child_chunk_size']}，overlap={settings['chunk_overlap']}",
        f"- retrieval_k={settings['retrieval_k']}，candidate_m={settings['candidate_m']}",
        f"- 混合权重：dense={settings['dense_weight']}，sparse={settings['sparse_weight']}；IP arctan 归一化",
        f"- 文档向量缓存命中：{settings['cache_hit']}",
        "",
        "## 正例结果",
        "",
        "| 策略 | Hit@1 | Hit@K/M | MRR | ID Precision | ID Recall | 平均耗时(ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in summary["strategies"].items():
        lines.append(
            f"| {name} | {metrics['hit_at_1']:.2%} | {metrics['hit']:.2%} | "
            f"{metrics['mrr']:.4f} | {metrics['id_precision']:.2%} | "
            f"{metrics['id_recall']:.2%} | {metrics['avg_latency_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Hard negative",
            "",
            "当前生产 RAG 检索没有相关性阈值，知识库非空就会返回上下文。这里报告“不相关上下文暴露率”，不伪造拒答能力。",
            "",
            "| 策略 | 不相关上下文暴露率 |",
            "|---|---:|",
        ]
    )
    for name, metrics in summary["strategies"].items():
        lines.append(f"| {name} | {metrics['hard_negative_context_exposure_rate']:.2%} |")
    lines.extend(["", "## 最终策略未命中正例", ""])
    final_strategy = strategy_names[-1]
    misses = []
    for row in rows:
        metrics = row["metrics"].get(final_strategy)
        if metrics and not metrics["hit"]:
            misses.append(
                f"- {row['case_id']}：{row['question']}；"
                f"gold={row['reference_parent_indexes']}；"
                f"got={row['results'][final_strategy]['parent_indexes']}"
            )
    lines.extend(misses or ["- 无"])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path, summary_path, details_path, manifest_path


def main() -> int:
    args = parse_args()
    if args.retrieval_k <= 0 or args.candidate_m <= 0:
        raise ValueError("retrieval-k 和 candidate-m 必须大于 0")
    cases = load_golden(args.golden)
    print("[1/5] 解析 PDF 并执行真实父子分块……", flush=True)
    child_chunks = process_documents(args.documents.resolve())
    if not child_chunks:
        raise RuntimeError("没有得到任何子块")
    parent_ids, parent_contents, parent_index_by_id = unique_parents(child_chunks)
    for case in cases:
        for index in case.reference_parent_indexes:
            if index < 0 or index >= len(parent_ids):
                raise ValueError(f"{case.case_id} 引用了不存在的 parent index：{index}")

    print("[2/5] 加载 BGE-M3，并读取或构建本地向量缓存……", flush=True)
    embedding_function, dense_vectors, sparse_vectors, cache_hit = (
        load_or_build_document_embeddings(
            child_chunks, args.cache_dir.resolve(), args.device, args.rebuild_cache
        )
    )
    reranker = None
    if not args.skip_reranker:
        from sentence_transformers import CrossEncoder

        print("[3/5] 加载本地 BGE Reranker……", flush=True)
        reranker = CrossEncoder(DEFAULT_RERANKER_MODEL, device=args.device)
    else:
        print("[3/5] 已按参数跳过 BGE Reranker。", flush=True)

    print(f"[4/5] 评测 {len(cases)} 条问题……", flush=True)
    rows: list[dict[str, Any]] = []
    strategies = ["dense", "sparse", "hybrid"]
    if reranker is not None:
        strategies.append("hybrid_rerank")
    for position, case in enumerate(cases, start=1):
        results = score_query(
            case.question,
            embedding_function,
            dense_vectors,
            sparse_vectors,
            child_chunks,
            parent_index_by_id,
            args.retrieval_k,
            args.dense_weight,
            args.sparse_weight,
        )
        if reranker is not None:
            results["hybrid_rerank"] = rerank_hybrid(
                case.question,
                results["hybrid"],
                parent_contents,
                reranker,
                args.candidate_m,
            )
        row_results = {
            name: {
                "parent_ids": result.parent_ids,
                "parent_indexes": result.parent_indexes,
                "scores": [round(score, 6) for score in result.scores],
                "latency_ms": round(result.latency_ms, 3),
            }
            for name, result in results.items()
        }
        rows.append(
            {
                "case_id": case.case_id,
                "case_type": case.case_type,
                "question": case.question,
                "reference_parent_indexes": list(case.reference_parent_indexes),
                "reference_parent_ids": [
                    parent_ids[index] for index in case.reference_parent_indexes
                ],
                "reference_answer": case.reference_answer,
                "results": row_results,
                "metrics": {
                    name: case_metrics(case.reference_parent_indexes, result.parent_indexes)
                    for name, result in results.items()
                    if case.reference_parent_indexes
                },
            }
        )
        print(f"  {position:02d}/{len(cases)} {case.case_id}", flush=True)

    summary = aggregate(rows, strategies)
    settings = {
        "mode": "offline_exact_ip_no_database",
        "documents": str(args.documents.resolve()),
        "golden": str(args.golden.resolve()),
        "parent_count": len(parent_ids),
        "child_count": len(child_chunks),
        "parent_chunk_size": config.PARENT_CHUNK_SIZE,
        "child_chunk_size": config.CHILD_CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "retrieval_k": args.retrieval_k,
        "candidate_m": args.candidate_m,
        "dense_weight": args.dense_weight,
        "sparse_weight": args.sparse_weight,
        "embedding_model": str(Path(DEFAULT_EMBEDDING_MODEL).resolve()),
        "reranker_model": (
            None if reranker is None else str(Path(DEFAULT_RERANKER_MODEL).resolve())
        ),
        "device": args.device,
        "cache_hit": cache_hit,
        "external_services_called": [],
    }
    print("[5/5] 写入报告……", flush=True)
    paths = write_outputs(
        rows, summary, parent_ids, parent_contents, args.results_dir.resolve(), settings
    )
    print(json.dumps({"report": str(paths[0]), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
