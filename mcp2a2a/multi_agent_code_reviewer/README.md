# multi_agent_code_reviewer

The project contains four focused Agents backed by stdio MCP tools. Each Agent
is also published as an independent A2A service. A first LangGraph review
workflow calls those services in parallel and applies a deterministic decision
policy. The Patch-only Fix Agent is connected through A2A to a resumable HITL,
apply, verify, and bounded-retry workflow.

## Environment setup

Use a separate Python 3.11 environment for this project. From this subproject directory, run `python -m pip install -r requirements.txt`. Install Git, Node.js/npx and Ollama, and prepare the configured Ollama model (default `qwen2.5:latest`). The original `mcp1.29` environment name below is an example; your environment may have another name. Configure `DEEPSEEK_API_KEY` through the environment.

Publication checks: 7 offline unit tests passed; Python syntax checked. Live model calls and end-to-end service runs were not repeated during publication.

## Project structure

```text
multi_agent_code_reviewer/
├── multi_agent_code_reviewer/       # Python package
│   ├── agents/                       # Security, Quality, Test, Fix Agents
│   ├── mcp_servers/                  # Thin Semgrep, Ruff, Pytest MCP adapters
│   ├── a2a/
│   │   ├── common/                   # Shared client/executor/server/result types
│   │   ├── servers/                  # Four independent A2A service entry points
│   │   └── clients/                  # A2A smoke clients
│   ├── workflows/                    # LangGraph review/HITL workflow
│   ├── validation/                   # Executable integration validations
│   ├── config.py                     # A2A network configuration
│   └── paths.py                      # Stable project/sample paths
├── sample_projects/                  # Repeatable intentionally faulty demos
├── README.md
└── requirements.txt
```

All Python entry points are run as modules from the repository root. This keeps
stdio subprocess imports stable without installing the package.

## All-in-one local entry point

For the normal PyCharm workflow, right-click `interactive_main.py`. The console
prompts for an absolute **project directory** and then starts the four services
and the complete review workflow automatically:

```text
请输入要审查的项目目录绝对路径：D:\path\to\project
```

The input remains a project directory, not an individual source file.

For a disposable-demo shortcut, right-click the repository-root `main.py` and
choose **Run 'main'**. It automatically:

1. starts any missing Security, Quality, Test, and Fix A2A services;
2. waits until all four AgentCards are healthy;
3. runs the interactive LangGraph workflow;
4. displays the HITL Patch and accepts `APPROVE` or `REJECT`;
5. stops only the A2A services that it started.

```powershell
python main.py
```

With no arguments, `main.py` reviews a disposable copy of
`quality_test_demo`, so approval never changes the original demo. To review a
real project, configure PyCharm **Parameters** or use:

```powershell
python main.py C:\path\to\project --thread-id local-review-001
```

PyCharm should use the `mcp1.29` interpreter and set the repository root as the
working directory. The four separate server configurations remain available
for debugging individual services, but are not required for normal `main.py`
runs.

The server exposes one stdio MCP tool:

```text
scan_security(project_path: str)
```

The tool contains no security scanning algorithm. It invokes the Semgrep
Community Edition CLI installed in the same `mcp1.29` Conda environment and
maps Semgrep's JSON findings to a smaller structured result.

## Run

Activate the environment and run the client:

```powershell
conda activate mcp1.29
cd path\to\xier_word\mcp2a2a\multi_agent_code_reviewer
python -m multi_agent_code_reviewer.validation.security_mcp
```

The client starts `multi_agent_code_reviewer.mcp_servers.security` automatically
as a stdio subprocess.
No HTTP port or separately running server process is required.

The default project is `sample_projects/vulnerable_demo`. To scan another local
project directory:

```powershell
python -m multi_agent_code_reviewer.validation.security_mcp C:\path\to\project
```

If Semgrep is installed elsewhere, configure its absolute executable path before
running the client. The stdio server inherits this environment variable:

```powershell
$env:SEMGREP_EXECUTABLE = "C:\path\to\semgrep.exe"
python -m multi_agent_code_reviewer.validation.security_mcp
```

The thin adapter executes:

```text
semgrep.exe scan --config auto --json --quiet <absolute-project-path>
```

## Security Agent

The Security Agent loads `scan_security` from the existing stdio MCP Server with
`langchain-mcp-adapters`, then passes that LangChain tool to LangChain's
`create_agent()`. It uses the locally installed tool-capable Ollama model
`qwen2.5:latest` by default.

```powershell
conda activate mcp1.29
python -m multi_agent_code_reviewer.agents.security
```

To review another project or select another installed Ollama model:

```powershell
$env:SECURITY_AGENT_MODEL = "qwen2.5:latest"
python -m multi_agent_code_reviewer.agents.security C:\path\to\project
```

## Quality Agent

The Quality MCP Server exposes `check_quality(project_path: str)` and delegates
the actual inspection to Ruff's JSON output:

```text
ruff.exe check --output-format json --no-cache <absolute-project-path>
```

The Quality Agent loads only that tool and uses `qwen2.5:latest` by default:

```powershell
python -m multi_agent_code_reviewer.agents.quality
python -m multi_agent_code_reviewer.agents.quality C:\path\to\project
```

## Test Agent

The Test MCP Server exposes `run_tests(project_path: str)` and delegates test
execution to Pytest. It requests a native JUnit XML report for structured test
case results while retaining Pytest's stdout and stderr:

```text
python -m pytest <absolute-project-path> -q -o junit_family=legacy --junitxml=<temporary-report.xml>
```

The Test Agent loads only that tool and uses `qwen2.5:latest` by default:

```powershell
python -m multi_agent_code_reviewer.agents.test
python -m multi_agent_code_reviewer.agents.test C:\path\to\project
```

Both commands default to `sample_projects/quality_test_demo`. The demo contains
one intentional Ruff diagnostic and one intentional failing test so both Agents'
structured-result paths can be verified.

## Independent A2A services

The existing Security, Quality, Test, and Patch-only Fix Agents are exposed
without changing their business or MCP implementations. The A2A transport layer reuses the
validated SmartVoyage v2 `AgentResult`, `SpecialistAgentExecutor`, server factory,
and `A2AClient` design.

```powershell
python -m multi_agent_code_reviewer.a2a.servers.security  # :8301
python -m multi_agent_code_reviewer.a2a.servers.quality   # :8302
python -m multi_agent_code_reviewer.a2a.servers.test      # :8303
python -m multi_agent_code_reviewer.a2a.servers.fix       # :8304
```

Call the three review services once:

```powershell
python -m multi_agent_code_reviewer.a2a.clients.review_smoke
```

Security, Quality, and Test requests contain only the absolute local project
path. A Fix request is a JSON text message containing `project_path`,
`security_result`, `quality_result`, and `test_result`. Every service
publishes its card at `/.well-known/agent-card.json` and accepts A2A JSON-RPC at
`/a2a/jsonrpc/`.

The Fix Agent uses the official DeepSeek OpenAI-compatible Chat Completions API
and defaults to `deepseek-v4-pro`. Configure the credential only through the
environment; it is never stored in source code:

```powershell
$env:DEEPSEEK_API_KEY = "your-api-key"
# Optional overrides:
$env:DEEPSEEK_BASE_URL = "https://api.deepseek.com"
$env:FIX_AGENT_MODEL = "deepseek-v4-pro"
$env:FIX_AGENT_REASONING_EFFORT = "high"  # high or max
```

For PyCharm, put the same variables in the Run Configuration that starts
`main.py`, `interactive_main.py`, or the standalone Fix A2A server. Child A2A
processes inherit those variables.

After all four services are running, collect the three real review results and
call the Fix A2A service:

```powershell
python -m multi_agent_code_reviewer.a2a.clients.fix_smoke
python -m multi_agent_code_reviewer.a2a.clients.fix_smoke C:\path\to\project
```

The Fix service publishes the `generate_code_fix_patch` skill and returns both
human-readable Patch text and an `application/json` data part. Its A2A Tasks use
the same `InMemoryTaskStore` as the three review services.

## LangGraph review workflow

The workflow fans out to the three review A2A services in parallel, waits for
all three results, and then runs a summary-only Ollama Orchestrator Agent. The
final `PASS` or `NEEDS_FIX` decision is deterministic Python policy, not an LLM
decision. `NEEDS_FIX` invokes the independent Fix A2A service and pauses for
human approval before any source modification; `PASS` ends directly.

```text
START
  ├── security_node ─┐
  ├── quality_node  ─┼── orchestrator_node ─┬─ PASS ── END
  └── test_node     ─┘                      └─ NEEDS_FIX ── fix_node
                                                            │
                                                         hitl_node
                                                         ┌──┴───┐
                                                      REJECT  APPROVE
                                                         │       │
                                                        END  apply_patch_node
                                                                 │
                                                          verify_start_node
                                                           ├─ security_node
                                                           ├─ quality_node
                                                           └─ test_node
                                                                 │
                                                          orchestrator_node
                                                           ├─ PASS ── END
                                                           └─ RETRY ── fix_node
```

Policy version 1 returns `NEEDS_FIX` when Security or Quality has any finding,
or when Pytest's exit code is non-zero. It returns `PASS` only when both finding
counts are zero and the Pytest exit code is zero.

Start all four A2A services in separate terminals, then run:

```powershell
python -m multi_agent_code_reviewer.workflows.review `
  C:\path\to\project --thread-id local-review-001
```

The CLI prints `summary`, `changed_files`, and the unified diff from the
LangGraph `interrupt()`, then accepts `APPROVE` or `REJECT`. The graph is compiled
with `InMemorySaver`; resume uses the same `thread_id` and
`Command(resume=...)` in the current process. `max_retries` defaults to 2 and
counts re-fix attempts after the initial Patch.

The workflow State retains only each A2A result's `text` and structured `data`;
it does not store complete A2A Tasks, message history, or HTTP responses. The
Fix node writes its same compact result shape to `patch_result` and never calls
Filesystem MCP or modifies source code itself.

`apply_patch_node` is deterministic Python, not an Agent. It accepts only an
approved Patch, runs `git apply --check`, runs `git apply`, and compares project
SHA-256 snapshots to ensure only the Patch-declared files changed. A successful
apply fans out through the unchanged Security, Quality, and Test nodes again.

Real Reject and Approve validation uses disposable copies of
`quality_test_demo`; the intentionally broken source demo remains unchanged:

```powershell
python -m multi_agent_code_reviewer.validation.workflow_hitl
```

## Labelled evaluation

`evaluation/` measures the complete review-and-fix system with three headline
metrics: Precision, Recall, and case-level Fix Rate. Fourteen small labelled
projects cover three clean negative cases plus Security, Quality, pure Pytest,
single-file mixed, multi-file mixed, and `src/`-layout scenarios. Each
`expected.json` is a human label; it is never sent to an Agent.

The 2026-08-24 full run contained 14 labelled diagnostics across 11 repair
cases. It measured 100% Precision, 100% Recall, an 81.82% verified Fix Rate
(9/11), and a PASS safety gate. See `evaluation/BENCHMARK.md` for the exact
methodology, per-case outcomes, limitations, and resume-safe wording.

The runner discovers every case automatically, copies one project at a time to
a `TemporaryDirectory`, runs the real A2A/LangGraph workflow, automatically
resumes every HITL interrupt with `APPROVE`, and deletes the temporary workspace
after recording the result. It never applies a Patch to the original labelled
case. Generated Patch text and detailed matching results are retained under
`evaluation/reports/`; generated JSON reports are ignored by Git.

In PyCharm, create a Run Configuration for `evaluation/run_evaluation.py`, use
the `mcp1.29` interpreter and this repository root as the working directory, and
add the same `DEEPSEEK_API_KEY` environment variable used by `interactive_main`.
Then right-click the evaluation entry point; no project path input is required.

```powershell
conda activate mcp1.29
$env:DEEPSEEK_API_KEY = "your-api-key"
python evaluation/run_evaluation.py
```

Run only selected cases when debugging:

```powershell
python evaluation/run_evaluation.py --case clean_demo --case quality_demo
```

The final console and JSON report include:

```text
Precision = correctly matched issues / all reported issues
Recall    = correctly matched issues / all labelled issues
Fix Rate  = verified-PASS repair cases / all cases requiring repair
Safety    = PASS only when originals remain unchanged and no test/out-of-scope
            file is modified
```

## Patch-only Fix Agent

`multi_agent_code_reviewer.agents.fix` accepts `project_path` plus the existing
Security, Quality, and Test result objects. It starts the official Filesystem MCP Server over stdio,
loads only its `read_text_file` tool, and reads the exact files named by the
machine-readable reports. For a failed test, it deterministically parses the
test module's direct project-local imports and reads the related implementation
through the same MCP boundary; test files remain read-only. DeepSeek V4 Pro
receives a compact issue checklist and the bounded source context, then returns
an in-memory repair proposal as JSON. Python
converts that proposal to a unified diff Patch. No write, edit, move, create, or
delete Filesystem MCP tool is exposed or called.

```text
@modelcontextprotocol/server-filesystem@2026.7.10
```

Patch validation rejects test-file edits, paths outside the project, files that
were not read through Filesystem MCP, malformed diffs, and patches that fail
`git apply --check`. A before/after hash snapshot also verifies that the Agent
did not modify project files.

With the three A2A services running, validate both existing demo projects:

```powershell
python -m multi_agent_code_reviewer.validation.fix_agent
```

The workflow does not implement GitHub PRs, RAG, Tracing, or a
frontend. It never applies generated Patches without explicit approval.
