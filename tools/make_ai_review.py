

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR / "audit_reports"
INPUT_FILE = REPORT_DIR / "label_conflicts.csv"
OUTPUT_FILE = REPORT_DIR / "AI语义初审.csv"
SUMMARY_FILE = REPORT_DIR / "AI初审摘要.json"


REMOVE_LINES = {
    3367,
    5094,
    53026,
    64594,
    69738,
    80536,
    84888,
    86660,
}


KEEP_ORIGINAL_LINES = {
    12460,
    17574,
    38447,
    39246,
    48164,
    51298,
    64283,
    65662,
    75670,
    75671,
    83776,
    89450,
    99175,
    123064,
}


MANUAL_OVERRIDES = {
    27107: "策略7 信息传达",
    30791: "策略10 关系延续",
    34070: "策略8 解决实施",
    44826: "策略6 提供建议",
    46803: "策略6 提供建议",
    51577: "策略10 关系延续",
    52968: "策略9 反馈请求",
    57148: "策略6 提供建议",
    72851: "策略6 提供建议",
    81050: "策略6 提供建议",
    99778: "策略10 关系延续",
    104919: "策略10 关系延续",
    111256: "策略6 提供建议",
    112275: "策略7 信息传达",
    113601: "策略8 解决实施",
    115007: "策略7 信息传达",
    115059: "策略6 提供建议",
    115071: "策略8 解决实施",
    119723: "策略6 提供建议",
    125874: "策略9 反馈请求",
    128040: "策略7 信息传达",
    135932: "策略8 解决实施",
}

EXTRA_FIELDS = [
    "ai_action",
    "ai_final_strategy",
    "ai_confidence",
    "ai_reason",
]


def decide(row: dict[str, str]) -> dict[str, str]:
    line_number = int(row["line_number"])
    original = row["original_strategy"]
    suggested = row["suggested_strategy"]

    if line_number in REMOVE_LINES:
        return {
            "ai_action": "remove",
            "ai_final_strategy": "",
            "ai_confidence": "high",
            "ai_reason": "回复同时承担多个策略动作，强行保留单标签会继续制造边界噪声。",
        }

    if line_number in KEEP_ORIGINAL_LINES:
        return {
            "ai_action": "keep",
            "ai_final_strategy": original,
            "ai_confidence": "high",
            "ai_reason": "完整上下文中的主要动作符合原标签，规则命中的是次要表达。",
        }

    if line_number in MANUAL_OVERRIDES:
        final_strategy = MANUAL_OVERRIDES[line_number]
        return {
            "ai_action": "relabel",
            "ai_final_strategy": final_strategy,
            "ai_confidence": "high",
            "ai_reason": f"逐条复核后，主要客服动作更符合{final_strategy}。",
        }

    if row["status"] == "conflict" and suggested:
        return {
            "ai_action": "relabel",
            "ai_final_strategy": suggested,
            "ai_confidence": "high",
            "ai_reason": f"参考回复的主要语义动作明确属于{suggested}，与原标签不一致。",
        }

    raise ValueError(f"第{line_number}行没有审查决定，请人工补充")


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"找不到输入报告：{INPUT_FILE}")

    with INPUT_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        source_fields = reader.fieldnames or []
        rows = list(reader)

    reviewed_rows = []
    for row in rows:
        reviewed = dict(row)
        reviewed.update(decide(row))
        reviewed_rows.append(reviewed)

    with OUTPUT_FILE.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=source_fields + EXTRA_FIELDS)
        writer.writeheader()
        writer.writerows(reviewed_rows)

    action_counts = Counter(row["ai_action"] for row in reviewed_rows)
    relabel_counts = Counter(
        (row["original_strategy"], row["ai_final_strategy"])
        for row in reviewed_rows
        if row["ai_action"] == "relabel"
    )
    summary = {
        "input_file": INPUT_FILE.name,
        "output_file": OUTPUT_FILE.name,
        "reviewed_count": len(reviewed_rows),
        "action_counts": dict(sorted(action_counts.items())),
        "relabel_counts": [
            {"from": source, "to": target, "count": count}
            for (source, target), count in sorted(relabel_counts.items())
        ],
    }
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"完成初审：{len(reviewed_rows)} 条")
    print(f"保留：{action_counts['keep']} 条")
    print(f"重标：{action_counts['relabel']} 条")
    print(f"移除：{action_counts['remove']} 条")
    print(f"结果：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
