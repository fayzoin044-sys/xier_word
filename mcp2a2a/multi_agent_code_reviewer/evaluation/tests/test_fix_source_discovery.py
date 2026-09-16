"""Tests for deterministic failed-test to implementation-file discovery."""

from pathlib import Path

from multi_agent_code_reviewer.agents.fix import (
    FixPlanOutput,
    ProposedFileFix,
    build_unified_diff,
    resolve_test_import_files,
)


def _write(path: Path, content: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_resolves_top_level_module_imported_by_failed_test(tmp_path: Path) -> None:
    implementation = _write(tmp_path / "calculator.py", "def subtract(a, b):\n    return a + b\n")
    test_file = _write(
        tmp_path / "tests" / "test_calculator.py",
        "from calculator import subtract\n\ndef test_subtract():\n    assert subtract(5, 3) == 2\n",
    )

    resolved = resolve_test_import_files(
        project_path=tmp_path,
        test_file=test_file,
        test_content=test_file.read_text(encoding="utf-8"),
    )

    assert resolved == [implementation]


def test_resolves_module_from_src_layout(tmp_path: Path) -> None:
    implementation = _write(tmp_path / "src" / "shop" / "totals.py", "def total():\n    return 0\n")
    test_file = _write(
        tmp_path / "tests" / "test_totals.py",
        "from shop.totals import total\n",
    )

    resolved = resolve_test_import_files(
        project_path=tmp_path,
        test_file=test_file,
        test_content=test_file.read_text(encoding="utf-8"),
    )

    assert resolved == [implementation]


def test_ignores_external_and_test_module_imports(tmp_path: Path) -> None:
    test_helper = _write(tmp_path / "tests" / "helpers.py", "VALUE = 1\n")
    test_file = _write(
        tmp_path / "tests" / "test_example.py",
        "import json\nfrom tests import helpers\n",
    )

    resolved = resolve_test_import_files(
        project_path=tmp_path,
        test_file=test_file,
        test_content=test_file.read_text(encoding="utf-8"),
    )

    assert test_helper not in resolved
    assert resolved == []


def test_unified_diff_preserves_trailing_blank_context_line() -> None:
    original = "def subtract(a, b):\n    return a + b\n\n"
    plan = FixPlanOutput(
        summary="Repair subtraction.",
        files=[
            ProposedFileFix(
                path="calculator.py",
                updated_content="def subtract(a, b):\n    return a - b\n\n",
            )
        ],
    )

    patch = build_unified_diff(
        plan,
        source_by_path={"calculator.py": original},
    ).rstrip("\r\n")
    if patch:
        patch += "\n"

    assert patch.endswith("+    return a - b\n \n")
