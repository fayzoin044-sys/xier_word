

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = ROOT_DIR / "processed" / "train.jsonl"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "audit_reports"

FOCUS_STRATEGIES = {
    "策略9 反馈请求",
    "策略10 关系延续",
    "策略11 感谢与告别",
    "策略12 其它",
}


RULES: dict[str, list[tuple[str, float, str]]] = {
    "策略9 反馈请求": [
        (r"满意度|满意吗|是否满意", 3.0, "询问满意度"),
        (
            r"(请问|能否|是否方便|方便的话|期待|邀请|辛苦您).{0,16}(评价|(?<!信用)评分|打分)",
            3.0,
            "请求评价或打分",
        ),
        (r"(请|期待|邀请).{0,10}(您|你).{0,12}(反馈|评价).{0,8}(服务|体验|效果)", 3.0, "请求服务反馈"),
        (r"(操作|尝试).{0,10}后.{0,10}(请|欢迎|麻烦).{0,8}(告诉|告知|反馈)", 3.0, "请求操作后反馈"),
        (r"请问.{0,12}是否.{0,8}(解决|解答)", 2.5, "确认问题是否解决"),
    ],
    "策略10 关系延续": [
        (r"持续跟进|继续跟进|后续跟进", 3.0, "承诺后续跟进"),
        (r"有进展.{0,8}(通知|告知)|及时.{0,6}(通知|告知)", 3.0, "承诺通知进展"),
        (r"后续(将|会|由).{0,8}(联系|通知|沟通)|保持联系", 3.0, "保持后续联系"),
        (r"专员.{0,8}(稍后|明天|将在).{0,8}联系", 2.5, "安排专员联系"),
        (r"回访", 2.0, "安排回访"),
        (r"订阅|关注.{0,8}(公众号|服务号)", 2.0, "引导持续关注"),
        (r"随时.{0,8}(联系|咨询|致电)", 1.0, "邀请后续联系"),
    ],
    "策略11 感谢与告别": [
        (r"再见|结束本次|结束服务|通话结束", 3.5, "明确结束对话"),
        (r"生活愉快|工作顺利|一切顺利|祝您", 2.5, "结束祝福"),
        (r"不客气|这是我们应该做的", 2.0, "回应感谢"),
        (r"感谢.{0,8}(来电|咨询|支持|理解|配合|选择|信任)", 1.5, "表达感谢"),
    ],
}

CSV_FIELDS = [
    "line_number",
    "conversation_id",
    "source",
    "original_strategy",
    "reference_strategy",
    "suggested_strategy",
    "status",
    "rule_scores",
    "matched_reasons",
    "last_user_message",
    "recent_context",
    "reference_response",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审查策略9/10/11/12的语义标签")
    parser.add_argument("--input", type=Path,
                        default=DEFAULT_INPUT, help="输入JSONL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="审查报告目录",
    )
    return parser.parse_args()


def parse_reference_completion(sample: dict[str, Any]) -> tuple[str, str]:

    completion = sample.get("completion") or []
    if not completion:
        return "", ""

    content = completion[-1].get("content", "")
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return "", str(content)

    return str(parsed.get("strategy", "")), str(parsed.get("response", ""))


def get_last_user_message(prompt: list[dict[str, Any]]) -> str:
    for message in reversed(prompt):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def format_recent_context(prompt: list[dict[str, Any]], limit: int = 4) -> str:

    messages = [message for message in prompt if message.get(
        "role") != "system"]
    lines = []
    for message in messages[-limit:]:
        role = "用户" if message.get("role") == "user" else "客服"
        content = str(message.get("content", "")).replace(
            "\r", " ").replace("\n", " ")
        lines.append(f"{role}：{content}")
    return "\n".join(lines)


def score_response(response: str) -> tuple[dict[str, float], dict[str, list[str]]]:
    scores: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}

    for strategy, rules in RULES.items():
        strategy_score = 0.0
        strategy_reasons = []
        for pattern, weight, reason in rules:
            if re.search(pattern, response):
                strategy_score += weight
                strategy_reasons.append(reason)
        scores[strategy] = strategy_score
        reasons[strategy] = strategy_reasons

    return scores, reasons


def suggest_strategy(scores: dict[str, float]) -> tuple[str, str]:

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_strategy, best_score = ranked[0]
    second_score = ranked[1][1]

    if best_score < 2.5:
        return "", "unclassified"
    if second_score >= 2.5 and best_score - second_score < 1.0:
        return "", "ambiguous"
    return best_strategy, "candidate"


def make_audit_row(sample: dict[str, Any], line_number: int) -> dict[str, Any]:
    original_strategy = str(sample.get("target_strategy", ""))
    reference_strategy, reference_response = parse_reference_completion(sample)
    prompt = sample.get("prompt") or []
    scores, reasons = score_response(reference_response)
    suggested_strategy, suggestion_state = suggest_strategy(scores)

    if reference_strategy and reference_strategy != original_strategy:
        status = "field_mismatch"
    elif suggestion_state == "ambiguous":
        status = "ambiguous"
    elif suggestion_state == "unclassified":
        status = "needs_review"
    elif suggested_strategy == original_strategy:
        status = "aligned"
    else:
        status = "conflict"

    matched_reasons = []
    for strategy, strategy_reasons in reasons.items():
        if strategy_reasons:
            matched_reasons.append(f"{strategy}: {'、'.join(strategy_reasons)}")

    nonzero_scores = {
        strategy: score for strategy, score in scores.items() if score > 0
    }

    return {
        "line_number": line_number,
        "conversation_id": sample.get("conversation_id", ""),
        "source": sample.get("source", ""),
        "original_strategy": original_strategy,
        "reference_strategy": reference_strategy,
        "suggested_strategy": suggested_strategy,
        "status": status,
        "rule_scores": json.dumps(nonzero_scores, ensure_ascii=False),
        "matched_reasons": " | ".join(matched_reasons),
        "last_user_message": get_last_user_message(prompt),
        "recent_context": format_recent_context(prompt),
        "reference_response": reference_response,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    strategy_counts: Counter[str] = Counter()
    invalid_json_lines = 0
    total_lines = 0

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            total_lines += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                invalid_json_lines += 1
                continue

            strategy = str(sample.get("target_strategy", ""))
            if strategy not in FOCUS_STRATEGIES:
                continue

            row = make_audit_row(sample, line_number)
            all_rows.append(row)
            strategy_counts[strategy] += 1
            status_counts[row["status"]] += 1

    review_rows = [row for row in all_rows if row["status"] != "aligned"]
    conflict_rows = [
        row
        for row in all_rows
        if row["status"] in {"conflict", "field_mismatch", "ambiguous"}
    ]

    all_path = output_dir / "strategy_9_10_11_12.csv"
    review_path = output_dir / "needs_review.csv"
    conflict_path = output_dir / "label_conflicts.csv"
    summary_path = output_dir / "audit_summary.json"

    write_csv(all_path, all_rows)
    write_csv(review_path, review_rows)
    write_csv(conflict_path, conflict_rows)

    summary = {
        "input_file": input_path.name,
        "total_jsonl_lines": total_lines,
        "invalid_json_lines": invalid_json_lines,
        "focus_sample_count": len(all_rows),
        "strategy_counts": dict(sorted(strategy_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "needs_review_count": len(review_rows),
        "conflict_or_ambiguous_count": len(conflict_rows),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"读取总样本：{total_lines}")
    print(f"重点策略样本：{len(all_rows)}")
    print(f"需要复查：{len(review_rows)}")
    print(f"冲突或歧义：{len(conflict_rows)}")
    print(f"报告目录：{output_dir}")


if __name__ == "__main__":
    main()
