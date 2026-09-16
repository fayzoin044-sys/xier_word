"""使用 OpenAI GPT 对已保存的离线检索结果执行生成与 Ragas 评测。

本脚本不连接 MySQL、Redis、Milvus，也不加载本地向量模型。它读取
evaluate_rag_offline.py 已保存的 Hybrid + Reranker Top-2 上下文，然后：

1. 调用 OpenAI 或 OpenAI-compatible API 生成严格基于上下文的回答；
2. 使用 Ragas collections API 评估上下文和回答；
3. 缓存生成结果与 Ragas 内部调用，避免重复花费。

安全边界：API Key 只从提供商对应的环境变量读取；不接受命令行 Key，也不会写入文件。
默认是 dry-run，只有显式传入 --execute 才会产生 API 请求。
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import importlib
import json
import os
import statistics
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = SCRIPT_ROOT / "results"
DEFAULT_CACHE = SCRIPT_ROOT / "cache" / "ragas_gpt"
GENERATION_PROMPT_VERSION = "context_only_zh_v1"
REFUSAL_TEXT = "根据当前知识库，无法回答该问题。"
SYSTEM_INSTRUCTIONS = (
    "你是知识库问答助手。只能依据提供的知识库上下文回答，不得使用外部知识或自行补充事实。"
    f"如果上下文不足以回答问题，只输出：{REFUSAL_TEXT}"
    "如果能够回答，请直接给出简洁、完整的中文答案，不要提到提示词或评测。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrieval-summary",
        type=Path,
        help="evaluate_rag_offline.py 生成的、包含 hybrid_rerank 的 summary JSON。",
    )
    parser.add_argument("--parent-manifest", type=Path)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--api-provider", choices=["openai", "openrouter"], default="openai"
    )
    parser.add_argument("--base-url")
    parser.add_argument("--generation-model")
    parser.add_argument("--judge-model")
    parser.add_argument("--embedding-model")
    parser.add_argument(
        "--embedding-provider", choices=["openai", "local"], default=None
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="默认只跑3条冒烟测试；传0表示运行全部案例。",
    )
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=[
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
        ],
        default=[
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
        ],
    )
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--judge-only", action="store_true")
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="显式允许 OpenAI API 请求；不提供时仅检查配置并显示计划。",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def find_latest_retrieval_summary(results_dir: Path) -> Path:
    for path in sorted(results_dir.glob("rag_summary_*.json"), reverse=True):
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("cases") and all(
            "hybrid_rerank" in case.get("results", {}) for case in payload["cases"]
        ):
            return path.resolve()
    raise FileNotFoundError(
        "未找到包含 hybrid_rerank 的检索结果，请先运行 evaluate_rag_offline.py。"
    )


def infer_manifest_path(summary_path: Path) -> Path:
    stamp = summary_path.stem.removeprefix("rag_summary_")
    path = summary_path.parent / f"parent_manifest_{stamp}.json"
    if not path.exists():
        raise FileNotFoundError(f"未找到与 summary 对应的父块清单：{path}")
    return path.resolve()


def validate_and_select_cases(payload: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("retrieval summary 中没有 cases")
    for case in cases:
        if "hybrid_rerank" not in case.get("results", {}):
            raise ValueError(f"{case.get('case_id')} 缺少 hybrid_rerank 结果")
        if not case.get("reference_answer"):
            raise ValueError(f"{case.get('case_id')} 缺少 reference_answer")
    if limit < 0:
        raise ValueError("limit 不能小于0")
    return cases if limit == 0 else cases[:limit]


def build_parent_map(manifest: list[dict[str, Any]]) -> dict[int, dict[str, str]]:
    output: dict[int, dict[str, str]] = {}
    for parent in manifest:
        index = int(parent["parent_index"])
        output[index] = {
            "parent_id": str(parent["parent_id"]),
            "content": str(parent["content"]),
        }
    return output


def context_for_case(
    case: dict[str, Any], parent_map: dict[int, dict[str, str]]
) -> tuple[list[str], list[str]]:
    indexes = case["results"]["hybrid_rerank"]["parent_indexes"]
    contexts: list[str] = []
    context_ids: list[str] = []
    for raw_index in indexes:
        index = int(raw_index)
        if index not in parent_map:
            raise ValueError(f"{case['case_id']} 引用了不存在的 parent index：{index}")
        contexts.append(parent_map[index]["content"])
        context_ids.append(parent_map[index]["parent_id"])
    return contexts, context_ids


def cache_key(provider: str, model: str, question: str, contexts: list[str]) -> str:
    value = json.dumps(
        {
            "prompt_version": GENERATION_PROMPT_VERSION,
            "provider": provider,
            "model": model,
            "question": question,
            "contexts": contexts,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_generation_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"生成缓存格式错误：{path}")
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def response_usage(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


async def generate_answer(
    client: Any,
    provider: str,
    model: str,
    question: str,
    contexts: list[str],
    max_output_tokens: int,
) -> tuple[str, dict[str, int | None]]:
    numbered_context = "\n\n".join(
        f"[上下文 {index}]\n{context}" for index, context in enumerate(contexts, start=1)
    )
    user_input = f"知识库上下文：\n{numbered_context}\n\n用户问题：{question}"
    if provider == "openrouter":
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": user_input},
            ],
            max_tokens=max_output_tokens,
            temperature=0.0,
        )
        answer = str(response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        usage_value = {
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
    else:
        request: dict[str, Any] = {
            "model": model,
            "instructions": SYSTEM_INSTRUCTIONS,
            "input": user_input,
            "max_output_tokens": max_output_tokens,
            "store": False,
        }
        if model.startswith("gpt-5"):
            request["reasoning"] = {"effort": "none"}
        response = await client.responses.create(**request)
        answer = str(response.output_text).strip()
        usage_value = response_usage(response)
    if not answer:
        raise RuntimeError(f"{provider} 返回了空答案")
    return answer, usage_value


def resolve_provider_settings(args: argparse.Namespace) -> None:
    """补齐不同 API 提供商的默认模型、地址和 Embedding 方式。"""

    if args.api_provider == "openrouter":
        args.base_url = args.base_url or "https://openrouter.ai/api/v1"
        args.generation_model = args.generation_model or "openai/gpt-4o-mini"
        args.judge_model = args.judge_model or args.generation_model
        args.embedding_provider = args.embedding_provider or "local"
        args.embedding_model = args.embedding_model or "bge-m3-local"
        args.api_key_env = "OPENROUTER_API_KEY"
    else:
        args.generation_model = args.generation_model or "gpt-5.6-luna"
        args.judge_model = args.judge_model or args.generation_model
        args.embedding_provider = args.embedding_provider or "openai"
        args.embedding_model = args.embedding_model or "text-embedding-3-small"
        args.api_key_env = "OPENAI_API_KEY"


def create_local_bge_embeddings(base_class: Any, cache: Any) -> Any:
    """创建供 Ragas Answer Relevancy/Correctness 使用的本地 BGE-M3 Embedding。"""

    import numpy as np
    from milvus_model.hybrid import BGEM3EmbeddingFunction

    integrated_root = SCRIPT_ROOT.parents[1]
    model_path = integrated_root / "rag_qa" / "models" / "bge-m3"

    class LocalBGEM3Embedding(base_class):
        def __init__(self) -> None:
            super().__init__(cache=cache)
            self.function = BGEM3EmbeddingFunction(
                model_name=str(model_path.resolve()), use_fp16=False, device="cpu"
            )

        def embed_text(self, text: str, **kwargs: Any) -> list[float]:
            vector = self.function.encode_documents([text])["dense"][0]
            return np.asarray(vector, dtype=np.float32).tolist()

        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            return await asyncio.to_thread(self.embed_text, text, **kwargs)

        def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            vectors = self.function.encode_documents(texts)["dense"]
            return np.asarray(vectors, dtype=np.float32).tolist()

        async def aembed_texts(
            self, texts: list[str], **kwargs: Any
        ) -> list[list[float]]:
            return await asyncio.to_thread(self.embed_texts, texts, **kwargs)

    return LocalBGEM3Embedding()


def install_ragas_vertex_compatibility() -> bool:
    """绕过 Ragas 0.4.3 对已移除 VertexAI 路径的无条件导入。"""

    module_name = "langchain_community.chat_models.vertexai"
    try:
        importlib.import_module(module_name)
        return False
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
    module = types.ModuleType(module_name)
    module.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules[module_name] = module
    return True


def metric_value(result: Any) -> float:
    value = getattr(result, "value", result)
    return float(value)


async def score_case(
    case: dict[str, Any],
    answer: str,
    contexts: list[str],
    metrics: dict[str, Any],
) -> tuple[dict[str, float | None], dict[str, str]]:
    scores: dict[str, float | None] = {}
    errors: dict[str, str] = {}
    positive = bool(case["reference_parent_indexes"])
    for name, metric in metrics.items():
        if name == "context_recall" and not positive:
            scores[name] = None
            continue
        kwargs: dict[str, Any]
        if name == "context_precision":
            kwargs = {
                "user_input": case["question"],
                "reference": case["reference_answer"],
                "retrieved_contexts": contexts,
            }
        elif name == "context_recall":
            kwargs = {
                "user_input": case["question"],
                "reference": case["reference_answer"],
                "retrieved_contexts": contexts,
            }
        elif name == "faithfulness":
            kwargs = {
                "user_input": case["question"],
                "response": answer,
                "retrieved_contexts": contexts,
            }
        elif name == "answer_relevancy":
            kwargs = {"user_input": case["question"], "response": answer}
        elif name == "answer_correctness":
            kwargs = {
                "user_input": case["question"],
                "response": answer,
                "reference": case["reference_answer"],
            }
        else:
            raise ValueError(f"未知指标：{name}")
        try:
            scores[name] = metric_value(await metric.ascore(**kwargs))
        except Exception as error:  # 单个评审失败时保留其他指标与错误信息。
            scores[name] = None
            errors[name] = f"{type(error).__name__}: {error}"
    return scores, errors


def aggregate_scores(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, Any]:
    positives = [row for row in rows if row["is_positive"]]
    negatives = [row for row in rows if not row["is_positive"]]

    def group_summary(group: list[dict[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {"cases": len(group), "metrics": {}}
        for name in metric_names:
            values = [row["scores"].get(name) for row in group]
            valid = [float(value) for value in values if value is not None]
            output["metrics"][name] = {
                "mean": round(statistics.fmean(valid), 6) if valid else None,
                "valid": len(valid),
                "failed_or_skipped": len(values) - len(valid),
            }
        output["abstention_rate"] = (
            round(sum(row["abstained"] for row in group) / len(group), 6)
            if group
            else None
        )
        return output

    return {
        "positive": group_summary(positives),
        "hard_negative": group_summary(negatives),
    }


def write_report(
    rows: list[dict[str, Any]], summary: dict[str, Any], settings: dict[str, Any], results_dir: Path
) -> tuple[Path, Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = results_dir / f"ragas_report_{stamp}.md"
    json_path = results_dir / f"ragas_summary_{stamp}.json"
    csv_path = results_dir / f"ragas_details_{stamp}.csv"
    write_json_atomic(json_path, {"settings": settings, "summary": summary, "cases": rows})

    metric_names = settings["metrics"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "case_id",
            "case_type",
            "question",
            "response",
            "reference_answer",
            "is_positive",
            "abstained",
            *metric_names,
            "errors",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "case_type": row["case_type"],
                    "question": row["question"],
                    "response": row["response"],
                    "reference_answer": row["reference_answer"],
                    "is_positive": row["is_positive"],
                    "abstained": row["abstained"],
                    **{name: row["scores"].get(name) for name in metric_names},
                    "errors": json.dumps(row["errors"], ensure_ascii=False),
                }
            )

    positive_abstention = summary["positive"]["abstention_rate"]
    negative_abstention = summary["hard_negative"]["abstention_rate"]
    positive_abstention_text = (
        "-" if positive_abstention is None else f"{positive_abstention:.2%}"
    )
    negative_abstention_text = (
        "-（本次未选择 hard negative）"
        if negative_abstention is None
        else f"{negative_abstention:.2%}"
    )

    lines = [
        "# GPT + Ragas 生成质量评测",
        "",
        f"- 案例数：{len(rows)}",
        f"- 生成模型：`{settings['generation_model']}`",
        f"- 评审模型：`{settings['judge_model']}`",
        f"- Embedding：`{settings['embedding_model']}`",
        "- 上下文：离线 Hybrid + BGE Reranker Top-2",
        "",
        "## 知识库内问题",
        "",
        "| 指标 | 平均分 | 有效样本 | 失败/跳过 |",
        "|---|---:|---:|---:|",
    ]
    for name, value in summary["positive"]["metrics"].items():
        mean = "-" if value["mean"] is None else f"{value['mean']:.4f}"
        lines.append(f"| {name} | {mean} | {value['valid']} | {value['failed_or_skipped']} |")
    lines.extend(
        [
            "",
            f"知识库内问题误拒答率：{positive_abstention_text}",
            "",
            "## Hard negative",
            "",
            f"正确拒答率：{negative_abstention_text}",
            "",
            "Hard negative 的 Context Recall 不具有稳定业务含义，因此脚本跳过该项；其余指标单独汇总，不与知识库内问题混合平均。",
            "",
            "| 指标 | 平均分 | 有效样本 | 失败/跳过 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, value in summary["hard_negative"]["metrics"].items():
        mean = "-" if value["mean"] is None else f"{value['mean']:.4f}"
        lines.append(f"| {name} | {mean} | {value['valid']} | {value['failed_or_skipped']} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path, json_path, csv_path


async def run(args: argparse.Namespace) -> int:
    resolve_provider_settings(args)
    retrieval_summary = (
        args.retrieval_summary.resolve()
        if args.retrieval_summary
        else find_latest_retrieval_summary(args.results_dir.resolve())
    )
    manifest_path = (
        args.parent_manifest.resolve()
        if args.parent_manifest
        else infer_manifest_path(retrieval_summary)
    )
    retrieval_payload = load_json(retrieval_summary)
    cases = validate_and_select_cases(retrieval_payload, args.limit)
    parent_map = build_parent_map(load_json(manifest_path))
    prepared = []
    for case in cases:
        contexts, context_ids = context_for_case(case, parent_map)
        prepared.append((case, contexts, context_ids))

    print("GPT + Ragas 运行计划")
    print(f"  retrieval summary: {retrieval_summary}")
    print(f"  parent manifest:   {manifest_path}")
    print(f"  cases:             {len(prepared)}")
    print(f"  API provider:      {args.api_provider}")
    print(f"  base URL:          {args.base_url or 'OpenAI default'}")
    print(f"  generation model:  {args.generation_model}")
    print(f"  judge model:       {args.judge_model}")
    print(f"  embedding:         {args.embedding_provider}/{args.embedding_model}")
    print(f"  metrics:           {', '.join(args.metrics)}")
    print(f"  external service:  {args.api_provider} API")
    print("  API key persisted: no")
    if not args.execute:
        print(
            f"\nDRY-RUN 完成：未发送任何 API 请求。设置 {args.api_key_env} 后添加 --execute 才会运行。"
        )
        return 0

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"未检测到 {args.api_key_env}。请在当前 PowerShell 会话设置环境变量，不要把 Key 写入代码。"
        )
    if args.generation_only and args.judge_only:
        raise ValueError("generation-only 和 judge-only 不能同时使用")

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=2,
    )
    cache_dir = args.cache_dir.resolve()
    generation_cache_path = cache_dir / "generation_cache.json"
    generation_cache = load_generation_cache(generation_cache_path)
    generated_rows: list[dict[str, Any]] = []

    for position, (case, contexts, context_ids) in enumerate(prepared, start=1):
        key = cache_key(
            args.api_provider, args.generation_model, case["question"], contexts
        )
        cached = generation_cache.get(key)
        if args.judge_only and cached is None:
            raise RuntimeError(f"{case['case_id']} 没有生成缓存，无法使用 --judge-only")
        if cached is not None and not args.force_regenerate:
            answer = str(cached["response"])
            usage = cached.get("usage", {})
            cache_hit = True
        else:
            print(f"[生成 {position}/{len(prepared)}] {case['case_id']}", flush=True)
            answer, usage = await generate_answer(
                client,
                args.api_provider,
                args.generation_model,
                case["question"],
                contexts,
                args.max_output_tokens,
            )
            generation_cache[key] = {
                "case_id": case["case_id"],
                "model": args.generation_model,
                "response": answer,
                "usage": usage,
            }
            write_json_atomic(generation_cache_path, generation_cache)
            cache_hit = False
        generated_rows.append(
            {
                "case": case,
                "contexts": contexts,
                "context_ids": context_ids,
                "response": answer,
                "usage": usage,
                "generation_cache_hit": cache_hit,
            }
        )

    if args.generation_only:
        print(f"生成完成，缓存：{generation_cache_path}")
        return 0

    compatibility_used = install_ragas_vertex_compatibility()
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    from ragas.cache import DiskCacheBackend
    from ragas.embeddings.base import BaseRagasEmbedding, embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    ragas_cache = DiskCacheBackend(str(cache_dir / "ragas_internal"))
    judge_llm = llm_factory(
        args.judge_model,
        client=client,
        cache=ragas_cache,
        temperature=0.01,
        top_p=0.1,
        max_tokens=1024,
    )
    if args.embedding_provider == "local":
        embeddings = create_local_bge_embeddings(BaseRagasEmbedding, ragas_cache)
    else:
        embeddings = embedding_factory(
            "openai",
            model=args.embedding_model,
            client=client,
            interface="modern",
            cache=ragas_cache,
        )
    available_metrics = {
        "context_precision": ContextPrecision(llm=judge_llm),
        "context_recall": ContextRecall(llm=judge_llm),
        "faithfulness": Faithfulness(llm=judge_llm),
        "answer_relevancy": AnswerRelevancy(
            llm=judge_llm, embeddings=embeddings, strictness=3
        ),
        "answer_correctness": AnswerCorrectness(
            llm=judge_llm, embeddings=embeddings
        ),
    }
    selected_metrics = {name: available_metrics[name] for name in args.metrics}

    rows: list[dict[str, Any]] = []
    for position, generated in enumerate(generated_rows, start=1):
        case = generated["case"]
        print(f"[评审 {position}/{len(generated_rows)}] {case['case_id']}", flush=True)
        scores, errors = await score_case(
            case,
            generated["response"],
            generated["contexts"],
            selected_metrics,
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "case_type": case["case_type"],
                "question": case["question"],
                "response": generated["response"],
                "reference_answer": case["reference_answer"],
                "retrieved_context_ids": generated["context_ids"],
                "is_positive": bool(case["reference_parent_indexes"]),
                "abstained": generated["response"].strip() == REFUSAL_TEXT,
                "generation_cache_hit": generated["generation_cache_hit"],
                "generation_usage": generated["usage"],
                "scores": scores,
                "errors": errors,
            }
        )

    summary = aggregate_scores(rows, args.metrics)
    settings = {
        "retrieval_summary": str(retrieval_summary),
        "parent_manifest": str(manifest_path),
        "api_provider": args.api_provider,
        "base_url": args.base_url,
        "generation_model": args.generation_model,
        "judge_model": args.judge_model,
        "embedding_provider": args.embedding_provider,
        "embedding_model": args.embedding_model,
        "metrics": args.metrics,
        "case_limit": args.limit,
        "prompt_version": GENERATION_PROMPT_VERSION,
        "ragas_version": "0.4.3",
        "ragas_vertex_compatibility_used": compatibility_used,
        "ragas_telemetry_disabled": True,
        "api_key_persisted": False,
        "external_services_called": [f"{args.api_provider} API"],
    }
    paths = write_report(rows, summary, settings, args.results_dir.resolve())
    print(json.dumps({"report": str(paths[0]), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.max_output_tokens <= 0:
        raise ValueError("max-output-tokens 必须大于0")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
