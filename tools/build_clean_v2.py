

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
SOURCE_DIR = ROOT_DIR / "processed"
CANDIDATE_DIR = ROOT_DIR / "processed_v2"
TRAIN_AUDIT_DIR = SCRIPT_DIR / "audit_reports"
DEV_AUDIT_DIR = SCRIPT_DIR / "audit_reports_dev"

SOURCE_TRAIN = SOURCE_DIR / "train.jsonl"
SOURCE_DEV = SOURCE_DIR / "dev.jsonl"
DEV_CANDIDATE = CANDIDATE_DIR / "dev_candidate.jsonl"
TEST_CANDIDATE = CANDIDATE_DIR / "test_candidate.jsonl"
TRAIN_AUDIT = TRAIN_AUDIT_DIR / "strategy_9_10_11_12.csv"
DEV_AUDIT = DEV_AUDIT_DIR / "strategy_9_10_11_12.csv"
AI_REVIEW = TRAIN_AUDIT_DIR / "AI语义初审.csv"

TRAIN_CLEAN_POOL = CANDIDATE_DIR / "train_clean_pool_v2.jsonl"
TRAIN_30K = CANDIDATE_DIR / "train_30k_v2.jsonl"
DEV_CLEAN = CANDIDATE_DIR / "dev_clean_v2.jsonl"
DEV_1000 = CANDIDATE_DIR / "dev_1000_v2.jsonl"
TEST_CLEAN = CANDIDATE_DIR / "test_clean_v2.jsonl"
SUMMARY_PATH = SCRIPT_DIR / "dataset_v2_summary.json"

STRATEGIES = [
    "策略1 礼貌问候",
    "策略2 确认身份",
    "策略3 重述或转述",
    "策略4 细化问题",
    "策略5 情感管理",
    "策略6 提供建议",
    "策略7 信息传达",
    "策略8 解决实施",
    "策略9 反馈请求",
    "策略10 关系延续",
    "策略11 感谢与告别",
    "策略12 其它",
]

FOCUS_STRATEGIES = set(STRATEGIES[8:])

V2_SYSTEM_PROMPT = """你是一个中文客服助手。请根据当前完整对话历史，判断客服下一步最主要的策略，并生成合适的客服回复。

strategy 必须从以下12种策略中选择：
策略1 礼貌问候：开始服务时进行礼貌问候。
策略2 确认身份：为查询或办理业务核实必要身份信息。
策略3 重述或转述：复述用户问题，确认理解是否准确。
策略4 细化问题：信息不足时追问具体情况。
策略5 情感管理：安抚情绪、表达理解或歉意。
策略6 提供建议：给出可执行建议或替代方案。
策略7 信息传达：说明事实、规则、状态或业务信息。
策略8 解决实施：执行、提交、设置或指导具体解决操作。
策略9 反馈请求：方案或操作完成后，请用户确认结果、反馈效果或评价服务。
策略10 关系延续：问题仍需后续处理，承诺跟进、通知进展、保持联系，或延续长期服务关系。
策略11 感谢与告别：问题已经结束，且用户明确感谢或结束对话时进行感谢告别。问题尚未解决、仍在等待结果时不能选择此策略。
策略12 其它：用户请求超出客服权限或业务范围，因隐私、安全、合规等原因无法提供，并引导至正规或专业渠道；仅在确实不属于前11类时使用。

只选择当前回复最主要的一个策略。不得编造处理状态、到账时间、政策条款、电话号码、回访安排或其他未在对话中提供的事实；信息不足时应说明需要核实。

输出必须是合法JSON，只包含strategy和response两个字段，不要输出其他内容。"""


TRAIN_S9_OVERRIDES = {
    1820: "策略7 信息传达",
    10378: "策略7 信息传达",
    16667: "策略7 信息传达",
    136748: "策略6 提供建议",
    4944: "策略8 解决实施",
    36203: "策略7 信息传达",
    63297: "策略6 提供建议",
    69677: "策略7 信息传达",
    93248: "策略8 解决实施",
    21932: "策略7 信息传达",
    33590: "策略10 关系延续",
    86066: "策略7 信息传达",
    36956: "策略7 信息传达",
    85501: "策略7 信息传达",
    100334: "策略8 解决实施",
    45734: "策略8 解决实施",
    101333: "策略8 解决实施",
    85354: "策略7 信息传达",
    87117: "策略7 信息传达",
    101328: "策略8 解决实施",
    122414: "策略8 解决实施",
}


STRATEGY12_BOUNDARY_PATTERN = re.compile(
    r"无法|暂不|不支持|未(开通|开展|开放|接入|推出|提供)|不能|不便|"
    r"超出.{0,6}权限|建议.{0,16}(联系|咨询|查询|通过|前往)|"
    r"属于.{0,8}范畴|涉及.{0,8}(内部|隐私|专业|风险|合规)"
)


def load_candidate_conversation_ids(path: Path) -> set[str]:
    ids = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            ids.add(str(json.loads(line)["conversation_id"]))
    return ids


def load_audit_status(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {
            int(row["line_number"]): row["status"]
            for row in csv.DictReader(file)
        }


def load_ai_review(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {
            int(row["line_number"]): row
            for row in csv.DictReader(file)
        }


def parse_completion(sample: dict[str, Any]) -> dict[str, Any]:
    content = sample["completion"][-1]["content"]
    parsed = json.loads(content)
    if set(parsed) != {"strategy", "response"}:
        raise ValueError(f"completion字段不符合Schema：{parsed}")
    return parsed


def update_sample(sample: dict[str, Any], final_strategy: str) -> dict[str, Any]:

    sample["target_strategy"] = final_strategy
    completion = parse_completion(sample)
    completion["strategy"] = final_strategy
    sample["completion"][-1]["content"] = json.dumps(
        completion, ensure_ascii=False, separators=(",", ":")
    )

    for message in sample.get("prompt", []):
        if message.get("role") == "system":
            message["content"] = V2_SYSTEM_PROMPT
            break
    return sample


def decide_train_source_sample(
    line_number: int,
    sample: dict[str, Any],
    status: str | None,
    ai_review: dict[int, dict[str, str]],
) -> tuple[str, str, str]:

    original = str(sample["target_strategy"])

    if line_number in ai_review:
        decision = ai_review[line_number]
        if decision["ai_action"] == "remove":
            return "remove", "", "train_ai_review_remove"
        return "keep", decision["ai_final_strategy"], "train_ai_review"

    if original == "策略9 反馈请求" and line_number in TRAIN_S9_OVERRIDES:
        return "keep", TRAIN_S9_OVERRIDES[line_number], "train_s9_manual_override"

    return decide_by_focus_policy(original, parse_completion(sample)["response"], status)


def decide_by_focus_policy(
    original: str, response: str, status: str | None
) -> tuple[str, str, str]:
    if original not in FOCUS_STRATEGIES:
        return "keep", original, "non_focus_keep"

    if original in {"策略9 反馈请求", "策略10 关系延续"}:
        if status in {"aligned", "needs_review"}:
            return "keep", original, "rare_strategy_keep"
        return "remove", "", "rare_strategy_conflict_remove"

    if original == "策略11 感谢与告别":
        if status == "aligned":
            return "keep", original, "clear_goodbye_keep"
        return "remove", "", "unclear_goodbye_remove"

    if original == "策略12 其它":
        if status == "needs_review" and STRATEGY12_BOUNDARY_PATTERN.search(response):
            return "keep", original, "clear_boundary_keep"
        return "remove", "", "unclear_other_remove"

    raise ValueError(f"未处理策略：{original}")


def write_clean_sample(
    output_file: Any,
    sample: dict[str, Any],
    final_strategy: str,
    split_counts: Counter[str],
    split_ids: set[str],
) -> None:
    cleaned = update_sample(sample, final_strategy)
    output_file.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
    split_counts[final_strategy] += 1
    split_ids.add(str(cleaned["conversation_id"]))


def scan_dataset(path: Path) -> tuple[Counter[str], set[str]]:
    counts: Counter[str] = Counter()
    conversation_ids = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            sample = json.loads(line)
            counts[str(sample["target_strategy"])] += 1
            conversation_ids.add(str(sample["conversation_id"]))
    return counts, conversation_ids


def rebalance_strategy12_holdout(
    train_path: Path,
    dev_path: Path,
    test_path: Path,
    target_per_holdout: int = 80,
    seed: int = 42,
) -> dict[str, Any]:

    train_s12_by_conversation: Counter[str] = Counter()
    with train_path.open("r", encoding="utf-8") as file:
        for line in file:
            sample = json.loads(line)
            if sample["target_strategy"] == "策略12 其它":
                train_s12_by_conversation[str(sample["conversation_id"])] += 1

    dev_counts, _ = scan_dataset(dev_path)
    test_counts, _ = scan_dataset(test_path)
    dev_need = max(0, target_per_holdout - dev_counts["策略12 其它"])
    test_need = max(0, target_per_holdout - test_counts["策略12 其它"])

    candidates = list(train_s12_by_conversation)
    random.Random(seed).shuffle(candidates)
    move_to_dev = set()
    move_to_test = set()
    dev_added = 0
    test_added = 0

    for conversation_id in candidates:
        count = train_s12_by_conversation[conversation_id]
        if dev_added < dev_need:
            move_to_dev.add(conversation_id)
            dev_added += count
        elif test_added < test_need:
            move_to_test.add(conversation_id)
            test_added += count
        else:
            break

    if dev_added < dev_need or test_added < test_need:
        raise RuntimeError("训练池中的清晰策略12不足以构建验证/测试集")

    train_temp = train_path.with_suffix(".tmp")
    dev_temp = dev_path.with_suffix(".tmp")
    test_temp = test_path.with_suffix(".tmp")

    with (
        train_temp.open("w", encoding="utf-8") as train_output,
        dev_temp.open("w", encoding="utf-8") as dev_output,
        test_temp.open("w", encoding="utf-8") as test_output,
    ):
        with dev_path.open("r", encoding="utf-8") as file:
            for line in file:
                dev_output.write(line)
        with test_path.open("r", encoding="utf-8") as file:
            for line in file:
                test_output.write(line)

        with train_path.open("r", encoding="utf-8") as file:
            for line in file:
                sample = json.loads(line)
                conversation_id = str(sample["conversation_id"])
                if conversation_id in move_to_dev:
                    dev_output.write(line)
                elif conversation_id in move_to_test:
                    test_output.write(line)
                else:
                    train_output.write(line)

    train_temp.replace(train_path)
    dev_temp.replace(dev_path)
    test_temp.replace(test_path)

    return {
        "target_strategy12_per_holdout": target_per_holdout,
        "moved_conversations_to_dev": len(move_to_dev),
        "moved_conversations_to_test": len(move_to_test),
        "strategy12_added_to_dev": dev_added,
        "strategy12_added_to_test": test_added,
    }


def reservoir_by_strategy(
    path: Path, quota: int, seed: int
) -> tuple[dict[str, list[str]], Counter[str]]:
    rng = random.Random(seed)
    reservoirs: dict[str, list[str]] = defaultdict(list)
    seen: Counter[str] = Counter()

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            strategy = str(json.loads(line)["target_strategy"])
            seen[strategy] += 1
            bucket = reservoirs[strategy]
            if len(bucket) < quota:
                bucket.append(line)
            else:
                candidate_index = rng.randrange(seen[strategy])
                if candidate_index < quota:
                    bucket[candidate_index] = line

    return reservoirs, seen


def build_balanced_train(path: Path, output: Path, seed: int = 42) -> dict[str, Any]:
    quota = 2500
    reservoirs, seen = reservoir_by_strategy(path, quota, seed)
    rng = random.Random(seed)
    selected_lines = []
    output_counts = Counter()
    duplicate_counts = Counter()

    for strategy in STRATEGIES:
        bucket = reservoirs[strategy]
        if not bucket:
            raise ValueError(f"训练池缺少{strategy}")
        chosen = []
        while len(chosen) < quota:
            cycle = list(bucket)
            rng.shuffle(cycle)
            remaining = quota - len(chosen)
            chosen.extend(cycle[:remaining])
        duplicate_counts[strategy] = max(0, quota - len(bucket))
        selected_lines.extend(chosen)
        output_counts[strategy] = len(chosen)

    rng.shuffle(selected_lines)
    with output.open("w", encoding="utf-8") as file:
        file.writelines(selected_lines)

    return {
        "source_counts": dict(seen),
        "output_counts": dict(output_counts),
        "oversampled_rows": dict(duplicate_counts),
        "total": len(selected_lines),
    }


def build_random_dev_sample(
    path: Path, output: Path, sample_size: int = 1000, seed: int = 42
) -> dict[str, Any]:
    rng = random.Random(seed)
    reservoir: list[str] = []
    seen = 0

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            seen += 1
            if len(reservoir) < sample_size:
                reservoir.append(line)
            else:
                candidate_index = rng.randrange(seen)
                if candidate_index < sample_size:
                    reservoir[candidate_index] = line

    rng.shuffle(reservoir)
    with output.open("w", encoding="utf-8") as file:
        file.writelines(reservoir)

    counts = Counter(
        json.loads(line)["target_strategy"] for line in reservoir
    )
    return {"source_total": seen, "sample_total": len(reservoir), "counts": dict(counts)}


def main() -> None:
    required_files = [
        SOURCE_TRAIN,
        SOURCE_DEV,
        DEV_CANDIDATE,
        TEST_CANDIDATE,
        TRAIN_AUDIT,
        DEV_AUDIT,
        AI_REVIEW,
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少构建输入：{missing}")

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    dev_ids = load_candidate_conversation_ids(DEV_CANDIDATE)
    test_ids = load_candidate_conversation_ids(TEST_CANDIDATE)
    if dev_ids & test_ids:
        raise RuntimeError("dev/test候选会话发生重叠")

    train_status = load_audit_status(TRAIN_AUDIT)
    dev_status = load_audit_status(DEV_AUDIT)
    ai_review = load_ai_review(AI_REVIEW)

    counts = {"train": Counter(), "dev": Counter(), "test": Counter()}
    conversation_ids = {"train": set(), "dev": set(), "test": set()}
    removal_reasons = Counter()
    change_counts = Counter()

    with (
        TRAIN_CLEAN_POOL.open("w", encoding="utf-8") as train_file,
        DEV_CLEAN.open("w", encoding="utf-8") as dev_file,
        TEST_CLEAN.open("w", encoding="utf-8") as test_file,
    ):
        with SOURCE_TRAIN.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                sample = json.loads(line)
                original = str(sample["target_strategy"])
                action, final_strategy, reason = decide_train_source_sample(
                    line_number,
                    sample,
                    train_status.get(line_number),
                    ai_review,
                )
                if action == "remove":
                    removal_reasons[reason] += 1
                    continue
                if final_strategy != original:
                    change_counts[f"{original} -> {final_strategy}"] += 1
                write_clean_sample(
                    train_file,
                    sample,
                    final_strategy,
                    counts["train"],
                    conversation_ids["train"],
                )

        with SOURCE_DEV.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                sample = json.loads(line)
                conversation_id = str(sample["conversation_id"])
                original = str(sample["target_strategy"])
                response = str(parse_completion(sample)["response"])
                action, final_strategy, reason = decide_by_focus_policy(
                    original,
                    response,
                    dev_status.get(line_number),
                )
                if action == "remove":
                    removal_reasons[f"dev_source_{reason}"] += 1
                    continue

                if conversation_id in dev_ids:
                    split_name = "dev"
                    output_file = dev_file
                elif conversation_id in test_ids:
                    split_name = "test"
                    output_file = test_file
                else:
                    split_name = "train"
                    output_file = train_file

                write_clean_sample(
                    output_file,
                    sample,
                    final_strategy,
                    counts[split_name],
                    conversation_ids[split_name],
                )

    holdout_rebalance = rebalance_strategy12_holdout(
        TRAIN_CLEAN_POOL,
        DEV_CLEAN,
        TEST_CLEAN,
    )

    counts = {}
    conversation_ids = {}
    for split_name, path in {
        "train": TRAIN_CLEAN_POOL,
        "dev": DEV_CLEAN,
        "test": TEST_CLEAN,
    }.items():
        split_counts, split_ids = scan_dataset(path)
        counts[split_name] = split_counts
        conversation_ids[split_name] = split_ids

    overlaps = {
        "train_dev": len(conversation_ids["train"] & conversation_ids["dev"]),
        "train_test": len(conversation_ids["train"] & conversation_ids["test"]),
        "dev_test": len(conversation_ids["dev"] & conversation_ids["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"清洗后存在会话泄漏：{overlaps}")

    train_30k_summary = build_balanced_train(TRAIN_CLEAN_POOL, TRAIN_30K)
    dev_1000_summary = build_random_dev_sample(DEV_CLEAN, DEV_1000)

    summary = {
        "outputs": {
            "train_clean_pool": str(TRAIN_CLEAN_POOL.relative_to(ROOT_DIR)),
            "train_30k": str(TRAIN_30K.relative_to(ROOT_DIR)),
            "dev_clean": str(DEV_CLEAN.relative_to(ROOT_DIR)),
            "dev_1000": str(DEV_1000.relative_to(ROOT_DIR)),
            "test_clean": str(TEST_CLEAN.relative_to(ROOT_DIR)),
        },
        "clean_counts": {
            split: dict(sorted(split_counts.items()))
            for split, split_counts in counts.items()
        },
        "clean_totals": {split: sum(split_counts.values()) for split, split_counts in counts.items()},
        "conversation_counts": {
            split: len(ids) for split, ids in conversation_ids.items()
        },
        "conversation_overlap": overlaps,
        "strategy12_holdout_rebalance": holdout_rebalance,
        "label_changes": dict(sorted(change_counts.items())),
        "removal_reasons": dict(sorted(removal_reasons.items())),
        "train_30k": train_30k_summary,
        "dev_1000": dev_1000_summary,
        "system_prompt_version": "v2_with_strategy_boundaries_and_no_fabrication",
        "original_files_modified": False,
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"清洗训练池：{summary['clean_totals']['train']} 条")
    print(f"清洗验证集：{summary['clean_totals']['dev']} 条")
    print(f"清洗测试集：{summary['clean_totals']['test']} 条")
    print(f"均衡训练集：{train_30k_summary['total']} 条")
    print(f"会话交集：{overlaps}")
    print(f"统计报告：{SUMMARY_PATH}")


if __name__ == "__main__":
    main()
