# Labelled benchmark

This evaluation measures the current Python review pipeline on a labelled evaluation set.
It is not a claim of general performance on arbitrary repositories.

## Method

- Date: 2026-08-24
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

| Metric | Result |
|---|---:|
| Detection Precision | 100.00% |
| Detection Recall | 100.00% |
| Verified Fix Rate | 81.82% |
| Safety gate | PASS |

Results are specific to this evaluation set, toolchain, and model configuration. The safety gate verifies that original projects, tests and out-of-scope files remain unchanged.

## Failure analysis

Repair failures included a generated multi-file diff rejected by `git apply --check`, and structured output using `content` instead of the required `updated_content`. Failed verification counts as a repair failure even when the proposed changes are directionally correct.

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

## Reporting scope

The recorded evaluation achieved 100% detection precision, 100% recall and an 81.82% verified fix rate. These are evaluation-specific historical results, not a general accuracy guarantee or a new run during publication. The archived report retains the underlying measurements for traceability.
