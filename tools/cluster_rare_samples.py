

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR / "audit_reports"
DEFAULT_INPUT = REPORT_DIR / "strategy_9_10_11_12.csv"

RARE_STRATEGIES = {
    "策略9 反馈请求",
    "策略10 关系延续",
    "策略12 其它",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="聚类策略9/10/12的待审查样本")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--threshold",
        type=int,
        default=24,
        help="SimHash最大汉明距离；越大，聚类越少、组内差异越大",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = re.sub(r"\[[^\]]+\]|\*+|X+|x+|\d+(?:\.\d+)?", "<槽位>", text)
    return re.sub(r"[^\u4e00-\u9fff<>]", "", text)


def simhash(text: str) -> int:
    normalized = normalize_text(text)
    grams = [
        normalized[index: index + 2]
        for index in range(max(1, len(normalized) - 1))
    ]
    vector = [0] * 64

    for gram in grams:
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(64):
            vector[bit] += 1 if (value >> bit) & 1 else -1

    fingerprint = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def greedy_cluster(
    rows: list[dict[str, str]], threshold: int
) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    representative_hashes: list[int] = []

    for row in rows:
        item: dict[str, Any] = dict(row)
        item["_simhash"] = simhash(row["reference_response"])

        distances = [
            hamming_distance(item["_simhash"], fingerprint)
            for fingerprint in representative_hashes
        ]
        if distances and min(distances) <= threshold:
            cluster_index = distances.index(min(distances))
            clusters[cluster_index].append(item)
        else:
            representative_hashes.append(item["_simhash"])
            clusters.append([item])

    return clusters


def choose_medoid(cluster: list[dict[str, Any]]) -> dict[str, Any]:

    if len(cluster) == 1:
        return cluster[0]

    distance_sums = []
    for candidate in cluster:
        total_distance = sum(
            hamming_distance(candidate["_simhash"], other["_simhash"])
            for other in cluster
        )
        distance_sums.append(total_distance)
    return cluster[distance_sums.index(min(distance_sums))]


def load_pending_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return [
        row
        for row in rows
        if row["status"] == "needs_review"
        and row["original_strategy"] in RARE_STRATEGIES
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"找不到审查文件：{input_path}")
    if not 0 <= args.threshold <= 64:
        raise ValueError("threshold必须在0到64之间")

    pending_rows = load_pending_rows(input_path)
    representative_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    cluster_counts: Counter[str] = Counter()

    for strategy in sorted(RARE_STRATEGIES):
        strategy_rows = [
            row for row in pending_rows if row["original_strategy"] == strategy
        ]
        clusters = greedy_cluster(strategy_rows, args.threshold)
        cluster_counts[strategy] = len(clusters)

        strategy_number = re.search(r"策略(\d+)", strategy).group(1)
        for index, cluster in enumerate(clusters, start=1):
            cluster_id = f"S{strategy_number}-{index:03d}"
            medoid = choose_medoid(cluster)
            max_distance = max(
                hamming_distance(medoid["_simhash"], item["_simhash"])
                for item in cluster
            )

            representative = dict(medoid)
            representative.update(
                {
                    "cluster_id": cluster_id,
                    "cluster_size": len(cluster),
                    "max_distance_from_representative": max_distance,
                    "ai_decision": "",
                    "ai_final_strategy": "",
                }
            )
            representative_rows.append(representative)

            for item in cluster:
                detail = dict(item)
                detail.update(
                    {
                        "cluster_id": cluster_id,
                        "cluster_size": len(cluster),
                        "distance_from_representative": hamming_distance(
                            medoid["_simhash"], item["_simhash"]
                        ),
                        "is_representative": item["line_number"]
                        == medoid["line_number"],
                    }
                )
                detail_rows.append(detail)

    representative_fields = [
        "cluster_id",
        "cluster_size",
        "max_distance_from_representative",
        "original_strategy",
        "line_number",
        "conversation_id",
        "source",
        "last_user_message",
        "recent_context",
        "reference_response",
        "ai_decision",
        "ai_final_strategy",
    ]
    detail_fields = [
        "cluster_id",
        "cluster_size",
        "distance_from_representative",
        "is_representative",
        "original_strategy",
        "line_number",
        "conversation_id",
        "source",
        "last_user_message",
        "recent_context",
        "reference_response",
    ]

    representatives_path = REPORT_DIR / "稀有类别代表样本.csv"
    details_path = REPORT_DIR / "稀有类别聚类明细.csv"
    summary_path = REPORT_DIR / "稀有类别聚类摘要.json"

    write_csv(representatives_path, representative_rows, representative_fields)
    write_csv(details_path, detail_rows, detail_fields)

    summary = {
        "input_file": str(input_path),
        "threshold": args.threshold,
        "pending_sample_count": len(pending_rows),
        "representative_count": len(representative_rows),
        "cluster_counts": dict(sorted(cluster_counts.items())),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"待审查样本：{len(pending_rows)}")
    print(f"代表样本：{len(representative_rows)}")
    print(f"报告目录：{REPORT_DIR}")


if __name__ == "__main__":
    main()
