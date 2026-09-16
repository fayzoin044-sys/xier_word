# Labelled benchmark

This is a small, project-owned benchmark for the current Python review pipeline.
It is not a claim of general performance on arbitrary repositories.

## Method

- Date: 2026-08-24
- Cases: 14 independently runnable Python projects
- Labels: 14 manually declared diagnostics in `expected.json`
- Negative cases: 3 clean projects
- Repair cases: 11
- Detection tools: Semgrep CE 1.174.0, Ruff 0.16.4, Pytest 9.1.1
- Runtime: Python 3.11.15 in Conda environment `mcp1.29`
- Review agents: Ollama `qwen2.5:latest`
- Fix model: DeepSeek `deepseek-v4-pro`
- Transport/orchestration: stdio MCP, independent A2A services, LangGraph
- Approval: automatic only inside a temporary evaluation copy
- Success rule: Patch applied, all three review branches returned clean results,
  final decision/status were PASS, and the safety gate found no forbidden change

The labels are used only by the evaluator after the initial review. They are not
sent to any Agent. Every case is copied to a temporary directory before repair;
the original labelled projects are compared before/after and remain unchanged.

## Results

| Metric | Result | Raw count |
|---|---:|---:|
| Detection Precision | 100.00% | 14 matched / 14 reported |
| Detection Recall | 100.00% | 14 matched / 14 labelled |
| Verified Fix Rate | 81.82% | 9 passed / 11 repair cases |
| Safety gate | PASS | No original, test, or out-of-scope file was changed |

Detection by category:

| Category | Precision | Recall | Matched |
|---|---:|---:|---:|
| Security | 100.00% | 100.00% | 3/3 |
| Quality | 100.00% | 100.00% | 6/6 |
| Test | 100.00% | 100.00% | 5/5 |

## Case outcomes

| Case | Labels | Repair outcome |
|---|---:|---|
| `clean_demo` | 0 | Clean PASS |
| `clean_inventory_demo` | 0 | Clean PASS |
| `clean_text_demo` | 0 | Clean PASS |
| `mixed_demo` | 2 | PASS |
| `mixed_multi_file_demo` | 2 | Failed before apply: generated multi-file diff did not pass `git apply --check` |
| `quality_demo` | 1 | PASS |
| `quality_redefined_import_demo` | 2 | PASS |
| `quality_unused_variable_demo` | 1 | PASS |
| `security_demo` | 1 | Failed before apply: model returned `content` instead of required `updated_content` |
| `security_ping_demo` | 1 | PASS |
| `security_sql_demo` | 1 | PASS |
| `test_discount_demo` | 1 | PASS |
| `test_failure_demo` | 1 | PASS |
| `test_src_layout_demo` | 1 | PASS |

The two repair failures are intentionally retained in the benchmark. The
underlying proposed code changes were directionally correct, but the strict
structured-output/Patch gates rejected them, so they count as repair failures.

## Reproduce

```powershell
conda activate mcp1.29
$env:DEEPSEEK_API_KEY = "your-api-key"
python evaluation/run_evaluation.py
```

The measured full-run report is
[`baseline/evaluation-20260824T234040Z.json`](baseline/evaluation-20260824T234040Z.json), preserved with local user/source paths replaced by placeholders. This is a historical run, not a rerun during publication. Runtime reports are ignored
by Git; rerunning the command writes a timestamped report and updates
`evaluation/reports/latest.json`.

## Resume-safe wording

> Built a multi-agent Python code-review and Patch-verification workflow using
> MCP, A2A and LangGraph, integrating Semgrep, Ruff, Pytest and DeepSeek V4 Pro;
> achieved 100% detection precision, 100% recall and an 81.8% verified fix rate
> on a 14-project, 14-diagnostic self-authored benchmark, with the safety gate
> preserving all original and test files.

Always keep the qualifier “self-authored benchmark” and the raw sample counts.
The dataset is intentionally small and does not establish production-wide model
accuracy.
