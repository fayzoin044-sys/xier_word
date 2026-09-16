"""Deterministic issue extraction and evaluation metrics."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

ISSUE_CATEGORIES = ("security", "quality", "test")


def get_tool_result(branch_result: object) -> dict[str, Any]:
    """Return one A2A branch's machine-readable MCP result."""
    if not isinstance(branch_result, dict):
        return {}
    data = branch_result.get("data")
    if not isinstance(data, dict):
        return {}
    tool_result = data.get("tool_result")
    return tool_result if isinstance(tool_result, dict) else {}


def normalize_file(file_value: object, *, project_path: Path) -> str:
    """Normalize absolute or relative tool paths to a comparable POSIX path."""
    if not isinstance(file_value, str) or not file_value.strip():
        return ""
    raw = Path(file_value)
    if raw.is_absolute():
        try:
            return raw.resolve().relative_to(project_path.resolve()).as_posix()
        except ValueError:
            return raw.as_posix()
    return raw.as_posix().lstrip("./")


def extract_observed_issues(
    state: dict[str, Any],
    *,
    project_path: Path,
) -> list[dict[str, Any]]:
    """Flatten the initial Security, Quality, and Test results."""
    observed: list[dict[str, Any]] = []

    security = get_tool_result(state.get("security_result"))
    for finding in security.get("findings", []):
        if not isinstance(finding, dict):
            continue
        observed.append(
            {
                "category": "security",
                "rule_id": str(finding.get("rule_id", "")),
                "file": normalize_file(
                    finding.get("file"),
                    project_path=project_path,
                ),
                "line": finding.get("start_line"),
                "message": str(finding.get("message", "")),
            }
        )

    quality = get_tool_result(state.get("quality_result"))
    for finding in quality.get("findings", []):
        if not isinstance(finding, dict):
            continue
        observed.append(
            {
                "category": "quality",
                "rule_id": str(finding.get("rule_id", "")),
                "file": normalize_file(
                    finding.get("file"),
                    project_path=project_path,
                ),
                "line": finding.get("start_line"),
                "message": str(finding.get("message", "")),
            }
        )

    tests = get_tool_result(state.get("test_result"))
    for test in tests.get("tests", []):
        if not isinstance(test, dict) or test.get("status") not in {
            "failed",
            "error",
        }:
            continue
        observed.append(
            {
                "category": "test",
                "test_name": str(test.get("name", "")),
                "file": normalize_file(
                    test.get("file"),
                    project_path=project_path,
                ),
                "line": test.get("line"),
                "status": str(test.get("status", "")),
                "message": str(test.get("details", "")),
            }
        )
    return observed


def file_matches(expected_file: str, observed_file: str) -> bool:
    """Match a project-relative label against tool paths from any platform."""
    expected = expected_file.replace("\\", "/").lstrip("./")
    observed = observed_file.replace("\\", "/").lstrip("./")
    return observed == expected or observed.endswith(f"/{expected}")


def issue_matches(expected: dict[str, Any], observed: dict[str, Any]) -> bool:
    """Match one manually labelled issue to one actual tool finding."""
    if expected.get("category") != observed.get("category"):
        return False

    expected_file = expected.get("file")
    if isinstance(expected_file, str) and expected_file:
        observed_file = observed.get("file")
        if not isinstance(observed_file, str) or not file_matches(
            expected_file,
            observed_file,
        ):
            return False

    expected_rule = expected.get("rule_id")
    if (
        isinstance(expected_rule, str)
        and expected_rule
        and observed.get("rule_id") != expected_rule
    ):
        return False

    rule_fragment = expected.get("rule_id_contains")
    if (
        isinstance(rule_fragment, str)
        and rule_fragment
        and rule_fragment.lower() not in str(observed.get("rule_id", "")).lower()
    ):
        return False

    test_name = expected.get("test_name")
    if isinstance(test_name, str) and test_name:
        observed_name = str(observed.get("test_name", ""))
        if observed_name != test_name and not observed_name.endswith(test_name):
            return False

    expected_line = expected.get("line")
    if isinstance(expected_line, int):
        observed_line = observed.get("line")
        tolerance = expected.get("line_tolerance", 0)
        if not isinstance(observed_line, int) or not isinstance(tolerance, int):
            return False
        if abs(observed_line - expected_line) > tolerance:
            return False

    return True


def calculate_detection_counts(
    expected: list[dict[str, Any]],
    observed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Perform one-to-one matching and return raw Precision/Recall counts."""
    unmatched_observed = set(range(len(observed)))
    matched_pairs: list[dict[str, int]] = []
    missed_expected: list[dict[str, Any]] = []

    for expected_index, expected_issue in enumerate(expected):
        matched_index = next(
            (
                observed_index
                for observed_index in sorted(unmatched_observed)
                if issue_matches(expected_issue, observed[observed_index])
            ),
            None,
        )
        if matched_index is None:
            missed_expected.append(expected_issue)
            continue
        unmatched_observed.remove(matched_index)
        matched_pairs.append(
            {
                "expected_index": expected_index,
                "observed_index": matched_index,
            }
        )

    matched = len(matched_pairs)
    expected_count = len(expected)
    observed_count = len(observed)
    return {
        "expected": expected_count,
        "reported": observed_count,
        "matched": matched,
        "false_positives": observed_count - matched,
        "missed": expected_count - matched,
        "matched_pairs": matched_pairs,
        "missed_expected": missed_expected,
        "unexpected_observed": [observed[index] for index in sorted(unmatched_observed)],
    }


def ratio(numerator: int, denominator: int) -> float:
    """Treat an empty denominator as a perfect no-error result."""
    return numerator / denominator if denominator else 1.0


def add_metric_rates(counts: dict[str, Any]) -> dict[str, Any]:
    """Add Precision and Recall rates to one raw count object."""
    result = dict(counts)
    result["precision"] = ratio(counts["matched"], counts["reported"])
    result["recall"] = ratio(counts["matched"], counts["expected"])
    return result


def category_counts(
    expected: list[dict[str, Any]],
    observed: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Calculate the same metrics independently for each specialist branch."""
    result: dict[str, dict[str, Any]] = {}
    for category in ISSUE_CATEGORIES:
        expected_items = [
            issue for issue in expected if issue.get("category") == category
        ]
        observed_items = [
            issue for issue in observed if issue.get("category") == category
        ]
        result[category] = add_metric_rates(
            calculate_detection_counts(expected_items, observed_items)
        )
    return result


def aggregate_case_results(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate completed case records into the three headline metrics."""
    totals = defaultdict(int)
    category_totals: dict[str, defaultdict[str, int]] = {
        category: defaultdict(int) for category in ISSUE_CATEGORIES
    }
    repair_cases = 0
    repaired_cases = 0

    for case in case_results:
        detection = case.get("detection", {})
        for field in ("expected", "reported", "matched"):
            value = detection.get(field, 0)
            if isinstance(value, int):
                totals[field] += value

        for category in ISSUE_CATEGORIES:
            category_result = case.get("category_detection", {}).get(category, {})
            for field in ("expected", "reported", "matched"):
                value = category_result.get(field, 0)
                if isinstance(value, int):
                    category_totals[category][field] += value

        if case.get("requires_fix") is True:
            repair_cases += 1
            if case.get("fix_success") is True:
                repaired_cases += 1

    by_category = {}
    for category, counts in category_totals.items():
        by_category[category] = {
            "expected": counts["expected"],
            "reported": counts["reported"],
            "matched": counts["matched"],
            "precision": ratio(counts["matched"], counts["reported"]),
            "recall": ratio(counts["matched"], counts["expected"]),
        }

    return {
        "case_count": len(case_results),
        "expected_issues": totals["expected"],
        "reported_issues": totals["reported"],
        "matched_issues": totals["matched"],
        "precision": ratio(totals["matched"], totals["reported"]),
        "recall": ratio(totals["matched"], totals["expected"]),
        "repair_cases": repair_cases,
        "repaired_cases": repaired_cases,
        "fix_rate": ratio(repaired_cases, repair_cases),
        "by_category": by_category,
    }
