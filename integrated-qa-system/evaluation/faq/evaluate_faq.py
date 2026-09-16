"""离线评测 FAQ/BM25 的 Top-1 匹配与阈值路由。

脚本复用生产代码中的分词、BM25 和 Softmax 实现，但使用内存数据源与空缓存，
不会连接或修改正式 MySQL、Redis。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SYSTEM_ROOT = Path(__file__).resolve().parents[2]
if str(SYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEM_ROOT))

from mysql_qa.retrieval.bm25_search import BM25Search
from mysql_qa.utils.preprocess import preprocess_text
from base.logger import logger


FAQ_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = FAQ_DIR / "data" / "faq_golden.jsonl"
DEFAULT_RESULTS_DIR = FAQ_DIR / "results"
DEFAULT_FAQ_CSV = SYSTEM_ROOT / "mysql_qa" / "data" / "JP学科知识问答.csv"
VALID_CASE_TYPES = {"exact", "paraphrase", "hard_negative"}
VALID_ROUTES = {"faq", "rag"}


class NoCacheClient:
    """满足 BM25Search 缓存接口，但不读写任何外部状态。"""

    def get_data(self, key: str) -> None:
        del key
        return None

    def set_data(self, key: str, value: Any) -> bool:
        del key, value
        return True

    def get_answer(self, query: str) -> None:
        del query
        return None

    def set_answer(self, query: str, answer: str) -> bool:
        del query, answer
        return True


class InMemoryFAQRepository:
    """使用 CSV 内容模拟生产 MySQLClient 的只读查询接口。"""

    def __init__(self, rows: list[dict[str, str]]) -> None:
        self._questions = [row["question"] for row in rows]
        self._answers: dict[str, str] = {}
        for row in rows:
            self._answers.setdefault(row["question"], row["answer"])

    def fetch_questions(self) -> list[str]:
        return list(self._questions)

    def fetch_answer(self, question: str) -> str | None:
        return self._answers.get(question)


def _clean_cell(value: str | None) -> str:
    return "" if value is None else str(value).strip()


def load_faq_rows(csv_path: Path) -> list[dict[str, str]]:
    """读取生产 FAQ CSV，并保留有效问题在文件中的原始顺序。"""

    if not csv_path.exists():
        raise FileNotFoundError(f"FAQ CSV 不存在：{csv_path}")

    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "gbk"):
        try:
            with csv_path.open("r", encoding=encoding, newline="") as file:
                reader = csv.DictReader(file)
                required = {"学科名称", "问题", "答案"}
                missing = required - set(reader.fieldnames or [])
                if missing:
                    raise ValueError(f"FAQ CSV 缺少字段：{sorted(missing)}")

                rows: list[dict[str, str]] = []
                for row in reader:
                    question = _clean_cell(row.get("问题"))
                    answer = _clean_cell(row.get("答案"))
                    if not question or not answer:
                        continue
                    rows.append(
                        {
                            "subject": _clean_cell(row.get("学科名称")),
                            "question": question,
                            "answer": answer,
                        }
                    )
                if not rows:
                    raise ValueError("FAQ CSV 中没有有效问答")
                return rows
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeError(f"FAQ CSV 编码无法识别：{csv_path}") from last_error


def load_cases(dataset_path: Path) -> list[dict[str, Any]]:
    """读取并进行基础结构校验。"""

    if not dataset_path.exists():
        raise FileNotFoundError(f"黄金测试集不存在：{dataset_path}")

    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"测试集第 {line_number} 行不是合法 JSON"
                ) from exc

            case_id = _clean_cell(case.get("id"))
            case_type = _clean_cell(case.get("case_type"))
            query = _clean_cell(case.get("query"))
            expected_route = _clean_cell(case.get("expected_route"))
            expected_question = case.get("expected_standard_question")
            if expected_question is not None:
                expected_question = _clean_cell(expected_question)

            if not case_id or case_id in seen_ids:
                raise ValueError(f"测试集第 {line_number} 行 ID 为空或重复：{case_id!r}")
            if case_type not in VALID_CASE_TYPES:
                raise ValueError(
                    f"测试集第 {line_number} 行 case_type 无效：{case_type!r}"
                )
            if not query:
                raise ValueError(f"测试集第 {line_number} 行 query 不能为空")
            if expected_route not in VALID_ROUTES:
                raise ValueError(
                    f"测试集第 {line_number} 行 expected_route 无效："
                    f"{expected_route!r}"
                )
            if expected_route == "faq" and not expected_question:
                raise ValueError(
                    f"测试集第 {line_number} 行预期 FAQ，但没有标准问题"
                )
            if expected_route == "rag" and expected_question:
                raise ValueError(
                    f"测试集第 {line_number} 行预期 RAG，不应设置标准问题"
                )

            normalized = dict(case)
            normalized.update(
                {
                    "id": case_id,
                    "case_type": case_type,
                    "query": query,
                    "expected_route": expected_route,
                    "expected_standard_question": expected_question,
                    "notes": _clean_cell(case.get("notes")),
                }
            )
            cases.append(normalized)
            seen_ids.add(case_id)

    if not cases:
        raise ValueError("黄金测试集不能为空")
    return cases


def validate_references(
    cases: Iterable[dict[str, Any]],
    faq_rows: list[dict[str, str]],
) -> None:
    """确认所有正例引用的标准问题确实存在于生产 FAQ CSV。"""

    question_counts = Counter(row["question"] for row in faq_rows)
    errors: list[str] = []
    for case in cases:
        expected = case["expected_standard_question"]
        if not expected:
            continue
        count = question_counts[expected]
        if count == 0:
            errors.append(f"{case['id']}: 标准问题不存在：{expected!r}")
        elif count > 1:
            errors.append(f"{case['id']}: 标准问题在 CSV 中重复 {count} 次：{expected!r}")

    if errors:
        raise ValueError("黄金测试集引用校验失败：\n" + "\n".join(errors))


def audit_faq_corpus(faq_rows: list[dict[str, str]]) -> dict[str, Any]:
    """检查会影响分词、Softmax 和答案唯一性的重复标准问题。"""

    groups: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for row_number, row in enumerate(faq_rows, start=1):
        normalized_question = row["question"].strip().lower()
        groups[normalized_question].append((row_number, row))

    duplicate_groups: list[dict[str, Any]] = []
    for normalized_question, entries in groups.items():
        if len(entries) <= 1:
            continue
        answers = {entry[1]["answer"] for entry in entries}
        duplicate_groups.append(
            {
                "normalized_question": normalized_question,
                "count": len(entries),
                "row_numbers": [entry[0] for entry in entries],
                "original_questions": [entry[1]["question"] for entry in entries],
                "answer_variant_count": len(answers),
                "has_conflicting_answers": len(answers) > 1,
            }
        )

    conflicting_groups = [
        group for group in duplicate_groups if group["has_conflicting_answers"]
    ]
    return {
        "row_count": len(faq_rows),
        "unique_normalized_question_count": len(groups),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_row_count": sum(group["count"] for group in duplicate_groups),
        "conflicting_answer_group_count": len(conflicting_groups),
        "duplicate_groups": duplicate_groups,
    }


def _top_prediction(
    searcher: BM25Search,
    query: str,
) -> tuple[str | None, float, float]:
    """返回 Top-1 标准问题、Softmax 分数和原始 BM25 分数。"""

    if searcher.bm25 is None:
        raise RuntimeError("BM25 尚未初始化")
    tokens = preprocess_text(query)
    if not tokens:
        return None, 0.0, 0.0

    raw_scores = np.asarray(searcher.bm25.get_scores(tokens), dtype=np.float64)
    if raw_scores.size == 0:
        return None, 0.0, 0.0
    probabilities = searcher._softmax(raw_scores)
    best_index = int(np.argmax(probabilities))
    return (
        searcher.original_questions[best_index],
        float(probabilities[best_index]),
        float(raw_scores[best_index]),
    )


def evaluate_threshold(
    searcher: BM25Search,
    repository: InMemoryFAQRepository,
    cases: list[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []

    for case in cases:
        predicted_question, best_score, raw_bm25_score = _top_prediction(
            searcher,
            case["query"],
        )
        answer, need_rag = searcher.search(case["query"], threshold=threshold)
        actual_route = "rag" if need_rag else "faq"
        expected_route = case["expected_route"]
        expected_question = case["expected_standard_question"]
        expected_answer = (
            repository.fetch_answer(expected_question) if expected_question else None
        )
        top1_correct = bool(
            expected_question and predicted_question == expected_question
        )
        answer_correct = bool(
            expected_answer is not None
            and answer is not None
            and answer.strip() == expected_answer.strip()
        )
        route_correct = actual_route == expected_route
        if expected_route == "faq":
            overall_correct = actual_route == "faq" and top1_correct and answer_correct
            error_type = ""
            if actual_route == "rag":
                error_type = "false_rejection"
            elif not top1_correct:
                error_type = "wrong_faq_match"
            elif not answer_correct:
                error_type = "wrong_answer"
        else:
            overall_correct = actual_route == "rag"
            error_type = "" if overall_correct else "false_acceptance"

        details.append(
            {
                "threshold": threshold,
                "id": case["id"],
                "case_type": case["case_type"],
                "query": case["query"],
                "expected_route": expected_route,
                "actual_route": actual_route,
                "expected_standard_question": expected_question or "",
                "predicted_standard_question": predicted_question or "",
                "softmax_top1_score": best_score,
                "raw_bm25_top1_score": raw_bm25_score,
                "route_correct": route_correct,
                "top1_correct": top1_correct,
                "answer_correct": answer_correct,
                "overall_correct": overall_correct,
                "error_type": error_type,
                "notes": case["notes"],
            }
        )

    return details


def evaluate_full_corpus(
    searcher: BM25Search,
    repository: InMemoryFAQRepository,
    faq_rows: list[dict[str, str]],
    threshold: float,
) -> dict[str, Any]:
    """用每条标准问题原句检查整个 FAQ 语料，不混入人工集总体分数。"""

    top1_correct_count = 0
    accepted_correct_count = 0
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(faq_rows, start=1):
        question = row["question"]
        predicted_question, best_score, raw_bm25_score = _top_prediction(
            searcher,
            question,
        )
        top1_correct = predicted_question == question
        answer = repository.fetch_answer(predicted_question) if predicted_question else None
        accepted_correct = bool(
            best_score >= threshold
            and top1_correct
            and answer is not None
            and answer.strip() == row["answer"].strip()
        )
        top1_correct_count += int(top1_correct)
        accepted_correct_count += int(accepted_correct)
        if not accepted_correct:
            failures.append(
                {
                    "row_number": index,
                    "question": question,
                    "predicted_standard_question": predicted_question or "",
                    "softmax_top1_score": best_score,
                    "raw_bm25_top1_score": raw_bm25_score,
                    "top1_correct": top1_correct,
                    "accepted_by_threshold": best_score >= threshold,
                }
            )

    total = len(faq_rows)
    return {
        "count": total,
        "top1_correct_count": top1_correct_count,
        "top1_accuracy": _safe_ratio(top1_correct_count, total),
        "accepted_correct_count": accepted_correct_count,
        "accepted_answer_accuracy": _safe_ratio(accepted_correct_count, total),
        "failure_count": len(failures),
        "failures": failures,
    }


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _finite_mean(values: Iterable[float]) -> float | None:
    finite_values = [value for value in values if math.isfinite(value)]
    return sum(finite_values) / len(finite_values) if finite_values else None


def summarize(details: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    positives = [item for item in details if item["expected_route"] == "faq"]
    negatives = [item for item in details if item["expected_route"] == "rag"]
    accepted = [item for item in details if item["actual_route"] == "faq"]
    correct_accepted = [
        item
        for item in accepted
        if item["expected_route"] == "faq"
        and item["top1_correct"]
        and item["answer_correct"]
    ]

    case_type_summaries: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in details:
        grouped[item["case_type"]].append(item)
    for case_type, items in sorted(grouped.items()):
        case_type_summaries[case_type] = {
            "count": len(items),
            "overall_correct_count": sum(item["overall_correct"] for item in items),
            "overall_accuracy": _safe_ratio(
                sum(item["overall_correct"] for item in items), len(items)
            ),
            "average_softmax_top1_score": _finite_mean(
                item["softmax_top1_score"] for item in items
            ),
        }

    false_rejections = [item for item in positives if item["actual_route"] == "rag"]
    wrong_faq_matches = [
        item
        for item in positives
        if item["actual_route"] == "faq" and not item["top1_correct"]
    ]
    false_acceptances = [item for item in negatives if item["actual_route"] == "faq"]
    rejected_negatives = [item for item in negatives if item["actual_route"] == "rag"]

    return {
        "threshold": threshold,
        "total_cases": len(details),
        "faq_expected_cases": len(positives),
        "rag_expected_cases": len(negatives),
        "overall_correct_count": sum(item["overall_correct"] for item in details),
        "overall_accuracy": _safe_ratio(
            sum(item["overall_correct"] for item in details), len(details)
        ),
        "route_accuracy": _safe_ratio(
            sum(item["route_correct"] for item in details), len(details)
        ),
        "top1_accuracy": _safe_ratio(
            sum(item["top1_correct"] for item in positives), len(positives)
        ),
        "accepted_answer_accuracy": _safe_ratio(
            len(correct_accepted), len(positives)
        ),
        "accepted_faq_precision": _safe_ratio(len(correct_accepted), len(accepted)),
        "false_rejection_count": len(false_rejections),
        "false_rejection_rate": _safe_ratio(len(false_rejections), len(positives)),
        "wrong_faq_match_count": len(wrong_faq_matches),
        "wrong_faq_match_rate": _safe_ratio(len(wrong_faq_matches), len(positives)),
        "hard_negative_rejection_count": len(rejected_negatives),
        "hard_negative_rejection_accuracy": _safe_ratio(
            len(rejected_negatives), len(negatives)
        ),
        "false_acceptance_count": len(false_acceptances),
        "false_acceptance_rate": _safe_ratio(
            len(false_acceptances), len(negatives)
        ),
        "case_types": case_type_summaries,
        "error_counts": dict(
            sorted(Counter(item["error_type"] for item in details if item["error_type"]).items())
        ),
    }


def _format_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def write_outputs(
    all_details: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    results_dir: Path,
    dataset_path: Path,
    csv_path: Path,
    corpus_audit: dict[str, Any],
) -> tuple[Path, Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = results_dir / f"faq_details_{timestamp}.csv"
    summary_path = results_dir / f"faq_summary_{timestamp}.json"
    report_path = results_dir / f"faq_report_{timestamp}.md"

    fieldnames = list(all_details[0].keys())
    with detail_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_details)

    summary_document = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": str(dataset_path.resolve()),
        "faq_csv": str(csv_path.resolve()),
        "corpus_audit": corpus_audit,
        "summaries": summaries,
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary_document, file, ensure_ascii=False, indent=2)

    report_lines = [
        "# FAQ/BM25 离线评测报告",
        "",
        f"- 测试集：`{dataset_path.name}`",
        f"- 测试样本：{summaries[0]['total_cases']} 条",
        f"- FAQ 语料：{csv_path.name}",
        "- 缓存：离线空缓存，不读写正式 Redis",
        "",
        "## 阈值对比",
        "",
        "| 阈值 | 黄金集总体 | 黄金集FAQ正确回答 | 硬负例拒绝 | 误接收 | 误拒绝 | 467条原句正确回答 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        report_lines.append(
            "| {threshold:.2f} | {overall} | {accepted} | "
            "{negative} | {false_accept} | {false_reject} | {corpus_exact} |".format(
                threshold=summary["threshold"],
                overall=_format_percent(summary["overall_accuracy"]),
                accepted=_format_percent(summary["accepted_answer_accuracy"]),
                negative=_format_percent(
                    summary["hard_negative_rejection_accuracy"]
                ),
                false_accept=_format_percent(summary["false_acceptance_rate"]),
                false_reject=_format_percent(summary["false_rejection_rate"]),
                corpus_exact=_format_percent(
                    summary["full_corpus_self_check"]["accepted_answer_accuracy"]
                ),
            )
        )

    report_lines.extend(["", "## 分类型结果", ""])
    for summary in summaries:
        report_lines.append(f"### threshold={summary['threshold']:.2f}")
        report_lines.append("")
        report_lines.append("| 类型 | 数量 | 正确数 | 正确率 | 平均Top-1 Softmax |")
        report_lines.append("|---|---:|---:|---:|---:|")
        for case_type, values in summary["case_types"].items():
            average_score = values["average_softmax_top1_score"]
            score_text = "N/A" if average_score is None else f"{average_score:.4f}"
            report_lines.append(
                f"| {case_type} | {values['count']} | "
                f"{values['overall_correct_count']} | "
                f"{_format_percent(values['overall_accuracy'])} | {score_text} |"
            )
        report_lines.append("")
        report_lines.append(
            "错误统计："
            + (
                "、".join(
                    f"{name}={count}"
                    for name, count in summary["error_counts"].items()
                )
                or "无"
            )
        )
        report_lines.append("")

    report_lines.extend(
        [
            "## FAQ 语料审计",
            "",
            f"- 有效数据行：{corpus_audit['row_count']}",
            "- 忽略大小写后的唯一标准问题："
            f"{corpus_audit['unique_normalized_question_count']}",
            f"- 重复问题组：{corpus_audit['duplicate_group_count']}",
            "- 存在不同答案的重复问题组："
            f"{corpus_audit['conflicting_answer_group_count']}",
            "",
            "| 归一化问题 | 行号 | 重复数 | 答案版本数 | 答案冲突 |",
            "|---|---|---:|---:|---|",
        ]
    )
    for group in corpus_audit["duplicate_groups"]:
        question = group["normalized_question"].replace("|", "\\|")
        row_numbers = ", ".join(str(value) for value in group["row_numbers"])
        report_lines.append(
            f"| {question} | {row_numbers} | {group['count']} | "
            f"{group['answer_variant_count']} | "
            f"{'是' if group['has_conflicting_answers'] else '否'} |"
        )
    report_lines.append("")

    report_lines.extend(
        [
            "## 解释",
            "",
            "- `overall_correct` 对 FAQ 正例要求：通过阈值、Top-1 标准问题正确且答案一致。",
            "- 对硬负例要求：未通过 FAQ 阈值并正确转交 RAG。",
            "- Softmax 分数依赖 FAQ 语料规模，本报告中的阈值只对当前 467 条语料有效。",
            "- 逐条错误及预测标准问题请查看同时间生成的 CSV。",
            "",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return detail_path, summary_path, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线评测 FAQ/BM25")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--faq-csv", type=Path, default=DEFAULT_FAQ_CSV)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.85],
        help="需要比较的一个或多个 FAQ Softmax 阈值",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只校验测试集结构与标准问题引用，不执行评测",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示生产 BM25Search 的逐条 INFO 日志",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.verbose:
        logger.setLevel(logging.WARNING)
    thresholds = sorted(set(args.thresholds))
    if any(not 0.0 <= threshold <= 1.0 for threshold in thresholds):
        raise ValueError("thresholds 必须位于 [0, 1]")

    faq_rows = load_faq_rows(args.faq_csv)
    cases = load_cases(args.dataset)
    validate_references(cases, faq_rows)
    corpus_audit = audit_faq_corpus(faq_rows)
    type_counts = Counter(case["case_type"] for case in cases)
    print(
        "黄金测试集校验通过："
        f"FAQ语料={len(faq_rows)}，测试样本={len(cases)}，"
        + "，".join(f"{name}={count}" for name, count in sorted(type_counts.items()))
    )
    if args.validate_only:
        return

    repository = InMemoryFAQRepository(faq_rows)
    searcher = BM25Search(
        redis_client=NoCacheClient(),
        mysql_client=repository,
    )

    all_details: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for threshold in thresholds:
        details = evaluate_threshold(searcher, repository, cases, threshold)
        summary = summarize(details, threshold)
        summary["full_corpus_self_check"] = evaluate_full_corpus(
            searcher=searcher,
            repository=repository,
            faq_rows=faq_rows,
            threshold=threshold,
        )
        all_details.extend(details)
        summaries.append(summary)
        print(
            f"threshold={threshold:.2f}："
            f"总体={_format_percent(summary['overall_accuracy'])}，"
            f"FAQ正确回答={_format_percent(summary['accepted_answer_accuracy'])}，"
            "硬负例拒绝="
            f"{_format_percent(summary['hard_negative_rejection_accuracy'])}"
        )

    detail_path, summary_path, report_path = write_outputs(
        all_details=all_details,
        summaries=summaries,
        results_dir=args.results_dir,
        dataset_path=args.dataset,
        csv_path=args.faq_csv,
        corpus_audit=corpus_audit,
    )
    print(f"逐条结果：{detail_path.resolve()}")
    print(f"汇总数据：{summary_path.resolve()}")
    print(f"评测报告：{report_path.resolve()}")


if __name__ == "__main__":
    main()
