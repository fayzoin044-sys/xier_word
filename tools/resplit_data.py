

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import TextIO


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_TRAIN = ROOT_DIR / "processed" / "train.jsonl"
DEFAULT_DEV = ROOT_DIR / "processed" / "dev.jsonl"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "processed_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按完整会话重新划分数据")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_conversation_ids(path: Path) -> list[str]:
    conversation_ids = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            sample = json.loads(line)
            conversation_ids.add(str(sample["conversation_id"]))
    return sorted(conversation_ids)


def copy_original_train(
    source: Path,
    destination: TextIO,
    counts: Counter[str],
    conversation_ids: set[str],
) -> int:
    sample_count = 0
    with source.open("r", encoding="utf-8") as file:
        for line in file:
            sample = json.loads(line)
            destination.write(line)
            sample_count += 1
            counts[str(sample["target_strategy"])] += 1
            conversation_ids.add(str(sample["conversation_id"]))
    return sample_count


def main() -> None:
    args = parse_args()
    train_path = args.train.resolve()
    dev_path = args.dev.resolve()
    output_dir = args.output_dir.resolve()

    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError("找不到原始train.jsonl或dev.jsonl")
    if not 0 < args.train_ratio < 1 or not 0 < args.dev_ratio < 1:
        raise ValueError("划分比例必须在0到1之间")
    if args.train_ratio + args.dev_ratio >= 1:
        raise ValueError("train_ratio + dev_ratio必须小于1，以保留测试集")

    dev_conversation_ids = collect_conversation_ids(dev_path)
    rng = random.Random(args.seed)
    rng.shuffle(dev_conversation_ids)

    conversation_count = len(dev_conversation_ids)
    train_end = round(conversation_count * args.train_ratio)
    dev_end = train_end + round(conversation_count * args.dev_ratio)

    augment_train_ids = set(dev_conversation_ids[:train_end])
    new_dev_ids = set(dev_conversation_ids[train_end:dev_end])
    new_test_ids = set(dev_conversation_ids[dev_end:])

    if augment_train_ids & new_dev_ids or augment_train_ids & new_test_ids or new_dev_ids & new_test_ids:
        raise RuntimeError("会话划分发生重叠")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_output = output_dir / "train_pool.jsonl"
    dev_output = output_dir / "dev_candidate.jsonl"
    test_output = output_dir / "test_candidate.jsonl"

    split_counts = {
        "train": Counter(),
        "dev": Counter(),
        "test": Counter(),
    }
    split_sample_counts = Counter()
    split_conversation_ids = {
        "train": set(),
        "dev": set(),
        "test": set(),
    }

    with (
        train_output.open("w", encoding="utf-8") as train_file,
        dev_output.open("w", encoding="utf-8") as dev_file,
        test_output.open("w", encoding="utf-8") as test_file,
    ):
        split_sample_counts["train"] += copy_original_train(
            train_path,
            train_file,
            split_counts["train"],
            split_conversation_ids["train"],
        )

        with dev_path.open("r", encoding="utf-8") as file:
            for line in file:
                sample = json.loads(line)
                conversation_id = str(sample["conversation_id"])
                strategy = str(sample["target_strategy"])

                if conversation_id in augment_train_ids:
                    split_name = "train"
                    output_file = train_file
                elif conversation_id in new_dev_ids:
                    split_name = "dev"
                    output_file = dev_file
                elif conversation_id in new_test_ids:
                    split_name = "test"
                    output_file = test_file
                else:
                    raise RuntimeError(f"会话{conversation_id}没有分配集合")

                output_file.write(line)
                split_sample_counts[split_name] += 1
                split_counts[split_name][strategy] += 1
                split_conversation_ids[split_name].add(conversation_id)

    train_dev_overlap = split_conversation_ids["train"] & split_conversation_ids["dev"]
    train_test_overlap = split_conversation_ids["train"] & split_conversation_ids["test"]
    dev_test_overlap = split_conversation_ids["dev"] & split_conversation_ids["test"]
    if train_dev_overlap or train_test_overlap or dev_test_overlap:
        raise RuntimeError("输出集合存在conversation_id泄漏")

    summary = {
        "seed": args.seed,
        "source_train": train_path.name,
        "source_dev": dev_path.name,
        "ratios_for_source_dev": {
            "augment_train": args.train_ratio,
            "dev": args.dev_ratio,
            "test": 1 - args.train_ratio - args.dev_ratio,
        },
        "sample_counts": dict(split_sample_counts),
        "conversation_counts": {
            name: len(ids) for name, ids in split_conversation_ids.items()
        },
        "strategy_counts": {
            name: dict(sorted(counts.items())) for name, counts in split_counts.items()
        },
        "conversation_overlap": {
            "train_dev": len(train_dev_overlap),
            "train_test": len(train_test_overlap),
            "dev_test": len(dev_test_overlap),
        },
    }
    summary_path = SCRIPT_DIR / "resplit_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"训练候选：{split_sample_counts['train']} 条")
    print(f"验证候选：{split_sample_counts['dev']} 条")
    print(f"测试候选：{split_sample_counts['test']} 条")
    print("conversation_id交集：0")
    print(f"统计报告：{summary_path}")


if __name__ == "__main__":
    main()
