# Running the minimal orchestrator locally

The orchestrator is ordinary async Python. It creates a task ID and an isolated Git
worktree, starts a Docker sandbox, and runs Planner → Coder → Tester → Reviewer.
It saves validated outputs, reports the terminal stage and reason, saves a cumulative
patch, stops the container, and removes the task worktree and branch.

The CLI uses the real Planner, Coder, Tester, and Reviewer implementations by
default. Planner is integrated from upstream prototype PR #21 (`488d630`). A small
adapter maps the task request to Planner's `prompt` field, passes the pinned base
commit as context, and checks that Planner is bound to the task worktree.

The automated flow tests script model responses while exercising all four real
agents, their tools, and the production CLI composition. They verify the integration
without API keys; they do not prove a live model will implement arbitrary requests.

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

Expect **two Docker cases to pass**, rather than skip: first-attempt success and
one correction. The correction case deliberately implements the wrong answer,
runs a failing `unittest` command inside Docker, sends findings back to Coder, and
passes on attempt two. In both cases, Planner uses its actual list/read tools on
the committed task snapshot, even though the source checkout has a dirty file.
Tests also check that Reviewer sees the test-file diff, the final patch passes
`git apply --check`, and the task branch, worktree, and container are cleaned up.
No model API key is used.

For the repository checks:

```bash
uv run ruff format --check
uv run ruff check
uv run mypy src tests
uv run lint-imports
uv run pytest
```

## Run a task through the CLI with real models

The CLI is `uv run python -m oswald.cli`. The real Planner runs by default;
no saved plan or custom factory is needed.

Create `.env` in this checkout (or export the corresponding environment variables)
with `LLM_PROVIDER_KEY`, `PLANNER_MODEL`, `CODER_MODEL`, `TESTER_MODEL`, and
`REVIEWER_MODEL`. Use your provider's key and supported LiteLLM model identifiers.
A Git worktree does not automatically inherit another checkout's ignored `.env`.

The following Bash example creates a separate demo repository. Keep Oswald as your
current directory so `uv run` uses Oswald's environment:

```bash
demo_dir=$(mktemp -d)
git -C "$demo_dir" init -b main
printf 'def value():\n    return 1\n' > "$demo_dir/app.py"
printf '__pycache__/\n.oswald/\n' > "$demo_dir/.gitignore"
git -C "$demo_dir" add .
git -C "$demo_dir" -c user.name=Demo -c user.email=demo@example.com commit -m 'Demo baseline'
uv run python -m oswald.cli \
  'Make value() in app.py return 2. Verify it with Python unittest, which is available in the sandbox.' \
  --repo "$demo_dir" \
  --image python:3.12-slim \
  --output .oswald/runs
```

This command makes real model API calls. The automated tests do not require model
credentials. For your own project, replace `--repo` with its absolute path and
`--image` with a prebuilt image containing its test dependencies.

The runtime binds the isolated worktree at `/workspace`; Planner's read tools,
Coder's edit tools, and Tester's Docker commands all use the same repository
snapshot. Uncommitted changes in the source checkout are excluded. `--base-ref`
selects the commit/branch to snapshot (defaults to `HEAD`).

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

## Programmatic use and optional Planner overrides

For programmatic use, inject an `Agents` factory and a `SandboxRuntime` factory into
`Orchestrator`, then `await orchestrator.run(request, base_ref="HEAD")`. Wrap a
`PlannerAgent` with `PlannerAdapter`. Import the orchestration types from
`oswald.core.orchestrator` and `PlannerAgent` from `oswald.core.agents`. Construct
`PlannerAgent` with the `root` path supplied
to your agent factory, as the default CLI does.

`PlannerInput` from `oswald.core.orchestrator` contains `task_id`, `request`,
`workspace_root`, and `base_commit`. `PlannerAdapter` translates it to the agent's
`PlannerInput` (`task_id`, `prompt`, and `context`). All four CLI agents use
`Settings` for model configuration. No API server, database, Redis, Taskiq, or
orchestration framework is needed.

The CLI retains two optional overrides for diagnostics or custom integrations:

- `--plan /absolute/path/to/plan.json` uses a saved plan instead of calling Planner.
  The file contains `{"base_commit": "<exact commit SHA>", "spec": {<PlanSpec fields>}}`.
  A mismatched commit is rejected; the adapter assigns the new task ID to the plan.
- `--planner module:factory` loads a synchronous factory receiving the isolated
  workspace `Path`. Return an object with
  `async run(data: oswald.core.orchestrator.PlannerInput) -> PlanSpec`.

These overrides are mutually exclusive and optional. The orchestrator validates
and persists the returned specification regardless of which Planner is used.

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
