from evaluation.metrics import (
    add_metric_rates,
    aggregate_case_results,
    calculate_detection_counts,
    issue_matches,
)


def test_issue_match_uses_category_file_rule_and_line() -> None:
    expected = {
        "category": "quality",
        "rule_id": "F401",
        "file": "calculator.py",
        "line": 3,
    }
    observed = {
        "category": "quality",
        "rule_id": "F401",
        "file": "C:/temporary/project/calculator.py",
        "line": 3,
    }

    assert issue_matches(expected, observed) is True


def test_detection_counts_keep_false_positives_and_misses_separate() -> None:
    expected = [
        {"category": "quality", "rule_id": "F401", "file": "app.py"},
        {"category": "test", "test_name": "test_subtract"},
    ]
    observed = [
        {
            "category": "quality",
            "rule_id": "F401",
            "file": "app.py",
        },
        {
            "category": "quality",
            "rule_id": "F841",
            "file": "app.py",
        },
    ]

    result = add_metric_rates(calculate_detection_counts(expected, observed))

    assert result["matched"] == 1
    assert result["false_positives"] == 1
    assert result["missed"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5


def test_aggregate_uses_strict_case_level_fix_rate() -> None:
    cases = [
        {
            "requires_fix": True,
            "fix_success": True,
            "detection": {"expected": 2, "reported": 2, "matched": 2},
            "category_detection": {},
        },
        {
            "requires_fix": True,
            "fix_success": False,
            "detection": {"expected": 1, "reported": 2, "matched": 1},
            "category_detection": {},
        },
        {
            "requires_fix": False,
            "fix_success": False,
            "detection": {"expected": 0, "reported": 0, "matched": 0},
            "category_detection": {},
        },
    ]

    result = aggregate_case_results(cases)

    assert result["precision"] == 0.75
    assert result["recall"] == 1.0
    assert result["fix_rate"] == 0.5

