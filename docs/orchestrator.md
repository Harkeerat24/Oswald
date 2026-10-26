# Running the minimal orchestrator locally

The orchestrator is ordinary async Python. It creates a task ID and an isolated Git
worktree, starts a Docker sandbox, and runs Planner → Coder → Tester → Reviewer.
It saves validated outputs, reports the terminal stage and reason, saves a cumulative
patch, stops the container, and removes the task worktree and branch.

Planner is still pending on this branch. The integration contract and a temporary
saved-plan adapter let you verify everything around it now. Scripted model responses
in tests exercise the real Coder, Tester, and Reviewer implementations; they do not
prove a live model will successfully implement arbitrary requests.

## Verify without an API key

From this branch's checkout, in Bash/WSL:

```bash
uv sync --locked --all-groups
export LITELLM_LOCAL_MODEL_COST_MAP=True
uv run pytest tests/unit/test_orchestrator.py tests/unit/test_orchestrator_cli.py tests/integration/test_orchestrator_flow.py -m 'not integration' -q
```

The environment variable makes LiteLLM use its bundled model-cost data, avoiding a
network fetch during imports. This check uses real Git worktrees and real agent
classes with scripted model responses. Its local test command runs against a small
fixture repository, without Docker or external services.

For the real Docker check, ensure Docker Desktop/Engine is running and your current
shell can run `docker info`, then run:

```bash
docker pull python:3.12-slim
uv run pytest tests/integration/test_orchestrator_flow.py -m integration -v
```

Expect the Docker case to **pass**, rather than skip. It deliberately implements the
wrong answer on attempt one, runs a failing `unittest` command inside Docker, sends
findings back to Coder, and passes on attempt two. It also checks that Reviewer sees
the test-file diff, the final patch passes `git apply --check`, and the task branch,
worktree, and container are cleaned up. No model API key is used.

For the repository checks:

```bash
uv run ruff format --check
uv run ruff check
uv run mypy src tests
uv run lint-imports
uv run pytest
```

## Run a task through the CLI before Planner lands

The CLI is `uv run python -m oswald.cli`. You must select either `--plan` (temporary
saved-plan adapter) or `--planner` (a future real Planner factory). There is no silent
fallback pretending to plan a request.

The following Bash example creates a separate demo repository. Keep Oswald as your
current directory so `uv run` uses Oswald's environment:

```bash
demo_dir=$(mktemp -d)
git -C "$demo_dir" init -b main
printf 'def value():\n    return 1\n' > "$demo_dir/app.py"
printf '__pycache__/\n.oswald/\n' > "$demo_dir/.gitignore"
git -C "$demo_dir" add .
git -C "$demo_dir" -c user.name=Demo -c user.email=demo@example.com commit -m 'Demo baseline'
uv run python - "$demo_dir" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1])
base_commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
plan = {
    "base_commit": base_commit,
    "spec": {
        "task_id": "template",
        "summary": "Make value return 2",
        "target_files": ["app.py"],
        "acceptance_criteria": ["value() returns 2"],
        "content": "Change app.py so value() returns 2. Verify with Python unittest, which is available in the sandbox.",
    },
}
(repo.parent / (repo.name + "-plan.json")).write_text(json.dumps(plan), encoding="utf-8")
print("Plan:", repo.parent / (repo.name + "-plan.json"))
PY
```

Set `LLM_PROVIDER_KEY`, `CODER_MODEL`, `TESTER_MODEL`, and `REVIEWER_MODEL` in your
environment or Oswald's `.env` to your provider's key and supported LiteLLM model
identifiers. This command makes real model API calls:

```bash
uv run python -m oswald.cli 'Make value() return 2 and verify it with unittest' \
  --repo "$demo_dir" \
  --plan "${demo_dir}-plan.json" \
  --image python:3.12-slim \
  --output .oswald/runs
```

The saved plan must match the exact base commit. The adapter assigns the generated
run ID to the plan. It does not inspect the repository or replace the eventual
Planner implementation. The normal runtime binds the same isolated worktree at
`/workspace`; Planner's input, Coder's tools, and Tester's Docker commands all refer
to that same snapshot. Uncommitted changes in the source checkout are excluded.

Exit code `0` means tests passed and Reviewer approved. Exit code `1` means stopped
or failed. Ctrl-C records cancellation and exits `130` after cleanup. JSON output
includes the task ID, stage, reason, attempts, base commit, artifact references, and
patch path. Logs go to stderr.

Inspect `.oswald/runs/<task-id>/`:

- `request.json`: original request, workspace, and pinned base commit.
- `attempt-0/spec.json`: validated Planner output.
- `attempt-1/` and optionally `attempt-2/`: Coder, Tester, cumulative review input,
  and Reviewer JSON reports. Their `content` fields contain the Markdown reports.
- `spec/`, `code_changes/`, `test_results/`, `review_changes/`, `review/`: existing
  content-addressed artifact storage with lineage. Identical payloads are deduplicated.
- `logs/`: copied Docker command stdout/stderr logs.
- `final.patch`: cumulative implementation and test changes, also saved on failures.
- `run.json`: terminal outcome and any cleanup errors.

Review the patch before applying it to the original repository:

```bash
git -C "$demo_dir" apply --check /absolute/path/to/final.patch
git -C "$demo_dir" apply /absolute/path/to/final.patch
```

The orchestrator does not commit, push, merge, or publish changes. Approval here
means the automated Reviewer approved; it does not substitute for human approval.
If patch/log saving or container teardown fails, the result records the failure
and retains the worktree for recovery. Do not remove it until you recover the changes.

## Connect the future Planner

Provide a synchronous factory `module:factory` accepting the isolated `Path` and
returning an object with:

```python
async def run(self, data: PlannerInput) -> PlanSpec: ...
```

Import `PlannerInput` from `oswald.core.orchestrator`. Its fields are `task_id`,
`request`, `workspace_root`, and `base_commit`. Build any repository index inside
this isolated snapshot and inspect it read-only. Return the same task ID. The
orchestrator validates and saves `PlanSpec`; the Planner adapter need not persist it.
Configure the Planner's own LLM client/model in your factory. For example:

```bash
uv run python -m oswald.cli 'Your task request' \
  --repo /absolute/path/to/repo \
  --planner your_package.planner_adapter:create_planner \
  --image your-prebuilt-project-test-image
```

For programmatic use, inject an `Agents` factory and a `SandboxRuntime` factory into
`Orchestrator`, then `await orchestrator.run(request, base_ref="HEAD")`. The CLI uses
`Settings` for the existing three agents and `DockerSandboxRuntime` for execution.
No API server, database, Redis, Taskiq, or orchestration framework is needed.

## Routing and limits

| Result | Action |
| --- | --- |
| Planning questions in either control field | Stop at Planner before Coder |
| Coder escalation | Stop at Coder |
| Agent exception, timeout, invalid output, foreign task ID, persistence or sandbox error | Fail with stage and reason |
| Tester infrastructure/agent failure | Fail at Tester; no correction |
| Reviewer `REJECTED` | Stop at Reviewer; no correction |
| Failed tests or Reviewer `CHANGES_REQUESTED` | Send findings to Coder once, then rerun Tester and Reviewer |
| Second unsuccessful attempt | Stop; never run a third Coder attempt |
| Passing tests plus Reviewer `APPROVED` with no blockers | Succeed |

Reviewer runs after normal test failures to supply additional findings. A rejection
still ends the run. The per-stage timeout defaults to 900 seconds and is configurable
with `--stage-timeout`; agent turn ceilings remain in effect.

Docker starts a task container using an already running Docker daemon. It does not
start the daemon or install your project dependencies. The default image supports
the standard-library demo above. For a real project, use `--image` with a prebuilt
image containing its test tools and dependencies: the sandbox has networking disabled
and a read-only root filesystem. Native Windows execution has not been verified;
the documented commands and integration checks target Bash/WSL/Linux.
