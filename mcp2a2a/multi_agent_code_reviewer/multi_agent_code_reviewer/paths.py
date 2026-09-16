"""Stable filesystem locations shared across package modules."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
SAMPLE_PROJECTS_ROOT = PROJECT_ROOT / "sample_projects"
VULNERABLE_DEMO = SAMPLE_PROJECTS_ROOT / "vulnerable_demo"
QUALITY_TEST_DEMO = SAMPLE_PROJECTS_ROOT / "quality_test_demo"

