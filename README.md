<p align="center">
  <img src="docs/assets/logo.svg" alt="OSWALD Logo" width="550" />
</p>

<p align="center">
  <strong>Deterministic, multi-agent pipeline for autonomous software engineering</strong>
</p>

<p align="center">
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv" /></a>
  <a href="https://docs.astral.sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff" /></a>
  <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+" />
  <img src="https://img.shields.io/badge/types-mypy%20strict-blue.svg" alt="mypy strict" />
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT" /></a>
</p>

<br>

**OSWALD** coordinates four agent roles through an async Python pipeline. The local CLI creates an isolated Git worktree, uses Docker for checks and tests, passes validated artifacts between agents, and saves a patch for human review. The API and worker architecture is described in [the architecture specification](docs/architecture.md); the current CLI stops after saving its reports and patch.

---

## Table of Contents

- [Pipeline Lifecycle](#pipeline-lifecycle)
- [Tech Stack](#tech-stack)
- [Quickstart](#quickstart)
- [Run the Orchestrator](#run-the-orchestrator)
- [Developer Workflow](#developer-workflow)
- [Architectural Boundaries](#architectural-boundaries)
- [Documentation](#documentation)
- [Community](#community)
- [Contributing](#contributing)

---

## Pipeline Lifecycle

OSWALD transitions deterministically through four specialized roles, passing immutable Pydantic artifacts between stages:

```text
[ Task Request ] ──► ( Planner ) ──► spec.md ──► ( Coder ) ──► changes.md + diff
                                                                   │
[ Git PR / Push ] ◄── [ Human Approval Gate ] ◄── review.md ◄─────┴──► ( Tester ) ──► test-results.md ──► ( Reviewer )
```

| Stage             | Role                                             | Tool Permissions               | Output Artifact       |
| :---------------- | :----------------------------------------------- | :----------------------------- | :-------------------- |
| **1. Planner**    | Analyzes problem statement, resolves ambiguities | Read-only workspace inspection | `spec.md`             |
| **2. Coder**      | Implements specification in Docker sandbox       | File edit, bash in sandbox     | `changes.md` + diff   |
| **3. Tester**     | Authors and runs test suites inside sandbox      | Test runner in sandbox         | `test-results.md`     |
| **4. Reviewer**   | Evaluates code against specification & coverage  | Read-only diff & spec          | `review.md` (verdict) |
| **Approval Gate** | **Human Technical Sign-off**                     | Human reviewer inspects diff   | PR creation unblocked |

> **Security Rule:** Agents are strictly forbidden from pushing branches or creating PRs directly. All VCS mutations require explicit human approval.

---

## Tech Stack

- **Language & Runtime:** Python 3.12+, native `asyncio`
- **Package Management:** [`uv`](https://docs.astral-sh/uv/) (blazing fast dependency sync & pinned `uv.lock`)
- **API & Job Queue:** [FastAPI](https://fastapi.tiangolo.com/) + [Taskiq](https://taskiq-python.github.io/) (Redis broker)
- **LLM Engine:** LiteLLM adapter via the provider-neutral `LLMClient` protocol
- **Data Contracts:** [Pydantic v2](https://docs.pydantic.dev/) (immutable schemas, zero ORM coupling)
- **Backing Services:** PostgreSQL 16 + Redis 7 (orchestrated via Docker Compose)
- **Quality Tooling:** `ruff` (lint/format), `mypy` (strict types), `import-linter` (boundary contracts), `pytest`

---

## Quickstart

For the local async agent pipeline and verification commands, see [Running the minimal orchestrator](docs/orchestrator.md). The CLI connects Planner, Coder, Tester, and Reviewer in an isolated task workspace.

### Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- Docker & Docker Compose

### 1. Clone & Setup

```bash
git clone https://github.com/Oswald-Agent/Oswald.git
cd Oswald

# Copy environment template
cp .env.example .env

# Install dependencies and sync virtual environment
uv sync
```

### 2. Start Backing Services

```bash
# Starts local PostgreSQL and Redis containers in background
make up
```

### 3. Run Verification Suite

```bash
# Runs ruff, mypy, import-linter, and unit tests
make check
```

---

## Run the Orchestrator

The CLI runs **Planner → Coder → Tester → Reviewer** sequentially. Planner reads the
same committed snapshot that Coder edits and Docker mounts at `/workspace` for test
execution. Coder changes production files; Tester authors and runs tests; Reviewer
assesses the cumulative diff and test evidence. No API server, PostgreSQL, Redis,
Taskiq worker, or `make up` is required for this local flow.

Use a checkout containing the orchestrator implementation and run the following
commands from its root. The examples below use Bash/WSL/Linux.

### 1. Install and configure

```bash
uv sync --locked --all-groups
docker info
docker pull python:3.12-slim
```

Create `.env` from `.env.example` if it does not already exist, then edit it in this
checkout. Do not overwrite an existing configuration. For example, with a DeepSeek
API key and access to the indicated model:

```dotenv
LLM_PROVIDER_KEY=replace-with-your-deepseek-api-key
LLM_BASE_URL=https://api.deepseek.com
PLANNER_MODEL=deepseek/deepseek-v4-flash
CODER_MODEL=deepseek/deepseek-v4-flash
TESTER_MODEL=deepseek/deepseek-v4-flash
REVIEWER_MODEL=deepseek/deepseek-v4-flash
LLM_TIMEOUT_SECONDS=300
LLM_MAX_RETRIES=1
```

All four model settings must match the provider for your key. For OpenRouter, use
its key, `LLM_BASE_URL=https://openrouter.ai/api/v1`, and LiteLLM model identifiers
such as `openrouter/<provider>/<model-id>`. Merely setting `DEFAULT_MODEL` does not
override the four role-specific settings. Exported environment variables override
`.env`; another checkout's ignored `.env` is not copied into this worktree.

The model-request deadline defaults to **300 seconds**, with one transient retry.
The deadline for each complete agent stage defaults to **1800 seconds** and is
configurable with `--stage-timeout`. These limits bound waiting; provider rate
limits, credentials, unavailable models, and network failures can still stop a run.

### 2. Create a small repository to work on

This demo uses only Python's standard library, so the default sandbox image has
all the runtime dependencies it needs. Keep these commands in the same shell so
`demo_dir` remains available. Continue running `uv run` from the Oswald checkout.

```bash
demo_dir=$(mktemp -d /tmp/oswald-demo.XXXXXX)
git -C "$demo_dir" init -b main
printf 'def value():\n    return 1\n' > "$demo_dir/app.py"
printf '__pycache__/\n.oswald/\n' > "$demo_dir/.gitignore"
git -C "$demo_dir" add .
git -C "$demo_dir" -c user.name=Demo -c user.email=demo@example.com commit -m 'Demo baseline'
echo "$demo_dir"
```

The target must have a Git commit. The orchestrator snapshots `HEAD` by default;
`--base-ref` can select another commit or branch. Uncommitted changes in the target
checkout are excluded from the task snapshot.

### 3. Submit the task

This command makes real model API calls. It writes the final JSON result to a file
while progress logs remain visible on stderr:

```bash
uv run python -m oswald.cli \
  'Change value() in app.py to return 2 instead of 1. Add a Python unittest verifying the result, run the tests, and review the changes.' \
  --repo "$demo_dir" \
  --image python:3.12-slim \
  --stage-timeout 1800 \
  --output .oswald/runs \
  > /tmp/oswald-demo-result.json

cat /tmp/oswald-demo-result.json
```

A successful result has `status: "SUCCEEDED"`, `stage: "reviewer"`, and the reason
`Tests passed and Reviewer approved`. It also contains the task ID, attempt count,
base commit, saved patch path, artifact references, and cleanup errors.

Exit code `0` means success; `1` means the run stopped or failed. Planning questions,
Coder escalation, and Reviewer rejection stop the run. Test failures or review
changes can trigger **one correction attempt**, for **two total Coder attempts**.
Exceptions and timeouts report the stage and reason. Automated Reviewer approval
does not authorize publishing, pushing, merging, or deploying changes.

### 4. Find the patch and reports

From the Oswald checkout, use the task ID from the JSON result:

```bash
task_id=$(uv run python -c 'import json; print(json.load(open("/tmp/oswald-demo-result.json"))["task_id"])')
run_dir="$(pwd)/.oswald/runs/$task_id"
cat "$run_dir/run.json"
less "$run_dir/final.patch"
ls "$run_dir"
```

| Saved output | Purpose |
| --- | --- |
| `request.json` | Task request, isolated workspace path, and pinned base commit |
| `attempt-0/spec.json` | Validated Planner specification |
| `attempt-1/`, optionally `attempt-2/` | Coder, Tester, cumulative review input, and Reviewer JSON reports; `content` contains Markdown |
| `spec/`, `code_changes/`, `test_results/`, `review_changes/`, `review/` | Versioned artifact payloads and lineage; identical payloads are deduplicated |
| `logs/` | Saved Docker command stdout/stderr logs |
| `final.patch` | Cumulative production and test-file changes |
| `run.json` | Terminal status, stage, reason, and cleanup details |

The original target repository is **not modified**. Normally the container, task
worktree, and task branch are removed after the patch and logs are saved. The
`workspace_root` in the result therefore usually names a directory that no longer
exists. If saving or teardown fails, inspect `cleanup_errors` and any retained
worktree before deleting it. Failed runs may contain only a partial patch and reports.

### 5. Apply the successful result and test it locally

After inspecting the reports and patch, apply it to the original demo repository:

```bash
git -C "$demo_dir" apply --check "$run_dir/final.patch"
git -C "$demo_dir" apply "$run_dir/final.patch"
cd "$demo_dir"
cat app.py
python3 -B -m unittest discover -v
```

Expect `value()` to return `2` and the generated unittest suite to pass. If you open
a new terminal, set `demo_dir` to the path printed in step 2; shell variables do not
persist into a new session. `--repo` can point at your own project instead, but
review its saved patch before applying it.

### Other projects and verification without API calls

`python:3.12-slim` does not include Node/npm, pytest, Git, ruff, or mypy. For a
React/Node project, use a sandbox image with Node/npm and the project's dependencies
already available inside `/workspace`. The sandbox disables networking and has a
read-only root filesystem; it cannot install missing dependencies during the run.
A longer model timeout does not supply those tools. After applying a generated
web-project patch locally, use `npm run` to inspect its actual scripts before
choosing a dev server, build, or test command.

To verify the pipeline without model API calls, run these from the Oswald checkout:

```bash
export LITELLM_LOCAL_MODEL_COST_MAP=True
uv run pytest tests/unit/test_orchestrator.py tests/unit/test_orchestrator_cli.py tests/integration/test_orchestrator_flow.py -m 'not integration' -q
uv run pytest tests/integration/test_orchestrator_flow.py -m integration -v
```

The tests use scripted model responses with the real agent implementations and
Git worktrees. The second command also uses real Docker; expect two Docker cases to
pass rather than skip when the daemon and image are available. These tests make no
paid model requests. Native Windows execution has not been verified.

See [the full orchestrator guide](docs/orchestrator.md) for programmatic use,
`--plan` replay, custom Planner factories, routing details, and additional checks.

---

## Developer Workflow

All common developer tasks are encapsulated in the root `Makefile`:

```bash
make install          # Sync virtual environment dependencies via uv sync
make format           # Automatically format codebase using ruff
make lint             # Run ruff code quality checks
make typecheck        # Run strict static type checking via mypy
make imports          # Verify architectural boundary contracts via import-linter
make test             # Run fast unit tests (excludes @pytest.mark.integration)
make test-integration # Run integration tests against Postgres & Redis
make check            # Run full verification suite (lint, format, types, imports, tests)
make up               # Start local Postgres and Redis containers
make down             # Stop local Postgres and Redis containers
```

---

## Architectural Boundaries

OSWALD enforces hard architectural boundaries via `import-linter` (enforced on every `make check` and CI run):

1. **`core/` Framework Isolation:** Contains pure business logic. Forbidden from importing `oswald.api`, `oswald.workers`, or `oswald.db`.
2. **`core/artifacts/` Pydantic Purity:** Immutable pipeline handoffs. Forbidden from importing `sqlalchemy` or ORM models.
3. **Single Entry Point:** `core/services` is the exclusive public interface into `core/orchestrator`.
4. **VCS Segregation:** Only `core/vcs` is authorized to run git commands.

---

## Documentation

- [Orchestrator Guide](docs/orchestrator.md) — CLI examples, saved outputs, correction limits, and local verification.
- [Architecture Specification](docs/architecture.md) — System architecture, orchestrator state machine, and component boundaries.
- [Contributing Guidelines](CONTRIBUTING.md) — Developer setup, Makefile workflow, and PR conventions.

---

## Community

Connect with the OSWALD community on Slack to ask questions, share ideas, and discuss contributions.

[![Slack](https://img.shields.io/badge/chat-on_slack-purple.svg?style=for-the-badge&logo=slack)](https://join.slack.com/t/oswaldheadquarters/shared_invite/zt-4e7y8ltcy-oppiIxdJ0CBRrm6MwPkFHw)

---

## Contributing

We welcome open-source contributions.

1. Check out our [Contributing Guidelines](CONTRIBUTING.md) for local setup and PR conventions.
2. Ensure `make check` passes cleanly before opening a pull request.
3. Follow Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).
