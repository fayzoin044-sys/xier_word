"""Run labelled review cases through the real A2A/LangGraph workflow."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import shutil
import sys
import tempfile
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import (
    add_metric_rates,
    aggregate_case_results,
    calculate_detection_counts,
    category_counts,
    extract_observed_issues,
)
from multi_agent_code_reviewer.launcher import local_a2a_services
from multi_agent_code_reviewer.workflows.review import (
    build_review_workflow,
    create_initial_state,
    snapshot_project_files,
)

CASES_ROOT = Path(__file__).resolve().parent / "cases"
DEFAULT_REPORTS_ROOT = Path(__file__).resolve().parent / "reports"
CACHE_NAMES = ("__pycache__", ".pytest_cache", ".ruff_cache", ".semgrep")


def load_case(case_directory: Path) -> dict[str, Any]:
    """Load and validate one labelled case without touching its project."""
    expected_path = case_directory / "expected.json"
    project_path = case_directory / "project"
    if not expected_path.is_file() or not project_path.is_dir():
        raise ValueError(
            f"Case must contain project/ and expected.json: {case_directory}"
        )

    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    if not isinstance(expected, dict):
        raise TypeError(f"expected.json must contain an object: {expected_path}")
    case_id = expected.get("id")
    if case_id != case_directory.name:
        raise ValueError(
            f"Case id must match its directory: {case_id!r} != {case_directory.name!r}"
        )
    issues = expected.get("issues")
    if not isinstance(issues, list) or not all(
        isinstance(issue, dict) for issue in issues
    ):
        raise TypeError(f"Case issues must be an object list: {expected_path}")
    if not isinstance(expected.get("requires_fix"), bool):
        raise TypeError(f"Case requires_fix must be a bool: {expected_path}")
    allowed_changed_files = expected.get("allowed_changed_files", [])
    if not isinstance(allowed_changed_files, list) or not all(
        isinstance(path, str) for path in allowed_changed_files
    ):
        raise TypeError(
            f"Case allowed_changed_files must be a string list: {expected_path}"
        )
    return {
        "id": case_id,
        "directory": case_directory,
        "project": project_path,
        "description": str(expected.get("description", "")),
        "requires_fix": expected["requires_fix"],
        "issues": issues,
        "allowed_changed_files": allowed_changed_files,
    }


def discover_cases(selected_ids: set[str] | None = None) -> list[dict[str, Any]]:
    """Discover cases in stable directory order."""
    if not CASES_ROOT.is_dir():
        raise FileNotFoundError(f"Evaluation cases directory is missing: {CASES_ROOT}")
    cases = []
    for directory in sorted(CASES_ROOT.iterdir()):
        if not directory.is_dir() or (selected_ids and directory.name not in selected_ids):
            continue
        cases.append(load_case(directory))
    if not cases:
        raise ValueError("No evaluation cases matched the selection")
    if selected_ids:
        found = {case["id"] for case in cases}
        missing = selected_ids - found
        if missing:
            raise ValueError(f"Unknown evaluation case(s): {sorted(missing)}")
    return cases


def copy_case_project(source: Path, temporary_root: Path, *, case_id: str) -> Path:
    """Create one disposable project copy."""
    target = temporary_root / case_id
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(*CACHE_NAMES),
    )
    return target


def state_has_review_results(state: object) -> bool:
    """Return whether all three initial branch results are available."""
    return isinstance(state, dict) and all(
        isinstance(state.get(field), dict)
        for field in ("security_result", "quality_result", "test_result")
    )


def get_patch_record(state: dict[str, Any]) -> dict[str, Any]:
    """Keep the public Patch boundary without A2A Task/history data."""
    patch_result = state.get("patch_result")
    if not isinstance(patch_result, dict):
        return {"summary": "", "changed_files": [], "patch": ""}
    data = patch_result.get("data")
    if not isinstance(data, dict):
        return {"summary": "", "changed_files": [], "patch": ""}
    changed_files = data.get("changed_files", [])
    return {
        "summary": str(data.get("summary", "")),
        "changed_files": changed_files if isinstance(changed_files, list) else [],
        "patch": str(data.get("patch", "")),
    }


def is_test_file(relative_path: str) -> bool:
    """Apply the same no-test-edit safety rule used by the Fix Agent."""
    path = PurePosixPath(relative_path)
    directory_parts = {part.lower() for part in path.parts[:-1]}
    filename = path.name.lower()
    return bool(
        directory_parts & {"test", "tests"}
        or filename.startswith("test_")
        or filename.endswith("_test.py")
    )


async def run_case(case: dict[str, Any], *, max_retries: int) -> dict[str, Any]:
    """Run one case on a temporary copy and auto-approve every Patch."""
    original_project = case["project"]
    original_before = snapshot_project_files(original_project)
    temporary_path: Path | None = None
    result: dict[str, Any] = {
        "id": case["id"],
        "description": case["description"],
        "requires_fix": case["requires_fix"],
        "expected_issues": case["issues"],
        "observed_issues": [],
        "detection": add_metric_rates(
            calculate_detection_counts(case["issues"], [])
        ),
        "category_detection": category_counts(case["issues"], []),
        "initial_decision": None,
        "approval_count": 0,
        "patch_result": {"summary": "", "changed_files": [], "patch": ""},
        "apply_result": {"success": False, "error": "", "changed_files": []},
        "final_decision": None,
        "final_status": "ERROR",
        "retry_count": 0,
        "fix_success": False,
        "safety_violations": [],
        "error": None,
    }

    with tempfile.TemporaryDirectory(prefix=f"code-review-eval-{case['id']}-") as temp:
        temporary_path = Path(temp)
        workspace = copy_case_project(
            original_project,
            temporary_path,
            case_id=case["id"],
        )
        workflow = build_review_workflow(checkpointer=InMemorySaver())
        thread_id = f"eval-{case['id']}-{uuid.uuid4().hex}"
        config = {"configurable": {"thread_id": thread_id}}
        first_state: dict[str, Any] = {}
        final_state: dict[str, Any] = {}

        try:
            workflow_result = await workflow.ainvoke(
                create_initial_state(workspace, max_retries=max_retries),
                config=config,
            )
            snapshot = await workflow.aget_state(config)
            first_state = copy.deepcopy(snapshot.values)

            if state_has_review_results(first_state):
                observed = extract_observed_issues(
                    first_state,
                    project_path=workspace,
                )
                result["observed_issues"] = observed
                result["detection"] = add_metric_rates(
                    calculate_detection_counts(case["issues"], observed)
                )
                result["category_detection"] = category_counts(
                    case["issues"],
                    observed,
                )
                result["initial_decision"] = first_state.get("decision")
                result["patch_result"] = get_patch_record(first_state)

            while "__interrupt__" in workflow_result:
                result["approval_count"] += 1
                workflow_result = await workflow.ainvoke(
                    Command(resume="APPROVE"),
                    config=config,
                )

            final_state = workflow_result
        except Exception as error:  # noqa: BLE001 - isolate failures per eval case.
            result["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            try:
                snapshot = await workflow.aget_state(config)
                final_state = copy.deepcopy(snapshot.values)
                if not first_state and state_has_review_results(final_state):
                    first_state = copy.deepcopy(final_state)
                    observed = extract_observed_issues(
                        first_state,
                        project_path=workspace,
                    )
                    result["observed_issues"] = observed
                    result["detection"] = add_metric_rates(
                        calculate_detection_counts(case["issues"], observed)
                    )
                    result["category_detection"] = category_counts(
                        case["issues"],
                        observed,
                    )
                    result["initial_decision"] = first_state.get("decision")
                    result["patch_result"] = get_patch_record(first_state)
            except Exception:  # noqa: BLE001 - preserve the original case error.
                final_state = {}

        if final_state:
            result["patch_result"] = get_patch_record(final_state)
            apply_result = final_state.get("apply_result")
            if isinstance(apply_result, dict):
                result["apply_result"] = apply_result
            result["final_decision"] = final_state.get("decision")
            result["final_status"] = final_state.get("final_status", "ERROR")
            retry_count = final_state.get("retry_count")
            result["retry_count"] = retry_count if isinstance(retry_count, int) else 0

        applied_files = result["apply_result"].get("changed_files", [])
        if isinstance(applied_files, list):
            test_edits = [path for path in applied_files if is_test_file(str(path))]
            if test_edits:
                result["safety_violations"].append(
                    f"Patch modified test files: {test_edits}"
                )
            allowed_files = set(case["allowed_changed_files"])
            unexpected_files = sorted(set(applied_files) - allowed_files)
            if unexpected_files:
                result["safety_violations"].append(
                    f"Patch modified files outside the case allowlist: {unexpected_files}"
                )

        result["fix_success"] = bool(
            case["requires_fix"]
            and result["apply_result"].get("success") is True
            and result["final_decision"] == "PASS"
            and result["final_status"] == "PASS"
            and not result["safety_violations"]
        )

    result["temporary_workspace_deleted"] = bool(
        temporary_path is not None and not temporary_path.exists()
    )
    if not result["temporary_workspace_deleted"]:
        result["safety_violations"].append("Temporary workspace was not deleted")
    if snapshot_project_files(original_project) != original_before:
        result["safety_violations"].append("Original labelled case was modified")
    result["safety_pass"] = not result["safety_violations"]
    return result


def print_case_summary(case: dict[str, Any]) -> None:
    """Print one concise case outcome while retaining full JSON in reports."""
    detection = case["detection"]
    print(f"\n=== {case['id']} ===")
    print(
        "Detection: "
        f"matched={detection['matched']}/{detection['expected']} | "
        f"reported={detection['reported']} | "
        f"precision={detection['precision']:.2%} | "
        f"recall={detection['recall']:.2%}"
    )
    print(
        "Repair: "
        f"status={case['final_status']} | "
        f"fix_success={case['fix_success']} | "
        f"approvals={case['approval_count']} | "
        f"retries={case['retry_count']}"
    )
    print(f"Safety: {'PASS' if case['safety_pass'] else 'FAIL'}")
    if case["error"]:
        print(f"Error: {case['error']['type']}: {case['error']['message']}")


def save_report(report: dict[str, Any], reports_root: Path) -> tuple[Path, Path]:
    """Write a timestamped result plus a convenient latest.json."""
    reports_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    timestamped = reports_root / f"evaluation-{timestamp}.json"
    latest = reports_root / "latest.json"
    content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    timestamped.write_text(content, encoding="utf-8")
    latest.write_text(content, encoding="utf-8")
    return timestamped, latest


async def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run selected cases sequentially under one set of A2A services."""
    selected = set(args.case) if args.case else None
    cases = discover_cases(selected)
    if any(case["requires_fix"] for case in cases) and not os.getenv(
        "DEEPSEEK_API_KEY"
    ):
        raise RuntimeError(
            "DEEPSEEK_API_KEY is required for Fix Agent evaluation. Add it to "
            "the run_evaluation PyCharm Run Configuration."
        )

    print(f"Discovered {len(cases)} evaluation case(s). Approval mode: AUTO")
    case_results: list[dict[str, Any]] = []
    with local_a2a_services():
        for index, case in enumerate(cases, start=1):
            print(f"\n[{index}/{len(cases)}] Running {case['id']} on a temporary copy")
            case_result = await run_case(case, max_retries=args.max_retries)
            case_results.append(case_result)
            print_case_summary(case_result)

    metrics = aggregate_case_results(case_results)
    safety_pass = all(case["safety_pass"] for case in case_results)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "approval_mode": "AUTO",
        "max_retries": args.max_retries,
        "metrics": metrics,
        "safety_gate": "PASS" if safety_pass else "FAIL",
        "cases": case_results,
    }
    timestamped, latest = save_report(report, args.reports_dir.resolve())

    print("\n=== Evaluation metrics ===")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"Fix rate:  {metrics['fix_rate']:.2%}")
    print(f"Safety:    {report['safety_gate']}")
    print(f"Report:    {timestamped}")
    print(f"Latest:    {latest}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the real review/fix workflow on labelled temporary copies."
    )
    parser.add_argument(
        "--case",
        action="append",
        help="Run one case id. Repeat the option for multiple cases; default is all.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum re-fix attempts after the initial Patch.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_ROOT,
        help="Directory for timestamped JSON reports and latest.json.",
    )
    arguments = parser.parse_args()
    if arguments.max_retries < 0:
        parser.error("--max-retries must be non-negative")
    return arguments


def main() -> None:
    report = asyncio.run(run(parse_args()))
    if report["safety_gate"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
