# Datus-Agent Project Instructions

## Overview

AI-powered data-analysis agent: NL → SQL, multi-DB, RAG knowledge base, MCP protocol.

- **Stack**: Python 3.12+, OpenAI Agents SDK + LiteLLM, LanceDB, FastAPI, FastMCP, Streamlit
- **Package manager**: uv · **License**: Apache-2.0

## Build & Run

```bash
uv sync                                                # Install dependencies
uv run python ci/run-pr-tests.py upstream/main         # PR CI harness: acceptance + impacted unit tests + coverage
uv run pytest -m nightly tests/                        # Nightly (needs API keys)
uv run pytest -m "nightly or regression" tests/        # Full regression
uv run ruff format datus/ tests/ && uv run ruff check --fix datus/ tests/  # Lint & format
bash build_scripts/build_test_data.sh                  # Build test KB
```

## Coding Conventions

- **ruff**: format + lint, line-length 120, `extend-exclude = mcp/`, isort groups stdlib → third-party → `datus.*`
- **Types**: hints throughout; Pydantic for data structures
- **Logging**: `from datus.utils.loggings import get_logger` — never `print()`
- **Errors**: raise `DatusException(ErrorCode.XXX, ...)` from `datus.utils.exceptions`. Code ranges: 1xxxxx common, 2xxxxx node, 3xxxxx model, 4xxxxx tool/storage, 5xxxxx database, 6xxxxx semantic
- **English only** in code, comments, commit/PR text — Chinese only in user-facing docs explicitly targeted to a Chinese audience

### CLI UI

All colours/symbols/helpers live in `datus/cli/cli_styles.py` — use the `print_*` helpers (`print_error`, `print_success`, `print_warning`, `print_info`, `print_status`, `print_usage`, `print_empty_set`) instead of inline Rich markup. Constraints:

- Colours never `bold`; `bold` is reserved for headers/prompt labels
- Unicode `✓`/`✗` only — no emoji in new code
- Closing tags use short form `[/]`
- Tables: `header_style=TABLE_HEADER_STYLE`; prefer `build_row_table()` from `_render_utils.py`
- Code rendering: `CODE_THEME = "monokai"` for all `Syntax()`
- Interactive selectors: import `CLR_CURSOR` / `CLR_CURRENT` from `cli_styles`

For full-screen TUI components, follow `ModelApp` (`model_app.py`): wrap `app.run()` in `tui_app.suspend_input()`, never nest `asyncio.run()`, use `DynamicContainer` + `Condition` guards, exit via `app.exit(result=Selection(...))`.

### Async tests

Use `@pytest.mark.asyncio` and `pytest_asyncio.fixture`. Event-loop helpers (esp. Windows): `datus/utils/async_utils.py`.

## Architecture

### Storage Layout

- **Per-project (CWD)**:
  - `./subject/{semantic_models, sql_summaries}/` — KB content, anchored to project root
  - `./.datus/skills/` — project skills, override `~/.datus/skills`
  - `./.datus/config.yml` — project overrides for `target` (provider/model), `default_datasource`, `project_name`. Whitelisted keys only; written by the `/model` slash command
- **Global, sharded by project**:
  - `~/.datus/sessions/{project}/{session_id}.db`
  - `~/.datus/data/{project}/datus_db/` (LanceDB, document stores)
  - `~/.datus/{conf, logs, cache, template, run, benchmark, workspace, skills}` — shared
- **`project_name`**: derived from CWD via `_normalize_project_name` (`agent_config.py`); long paths get an md5 suffix
- **`agent.knowledge_base_home` is removed** — KB is always under `{project_root}/subject/`; the YAML field is silently ignored

### LLM Configuration

Two-tier provider model:

1. **Provider-level** (`agent.providers.<name>` in `agent.yml`) — preferred. Credentials only; available models come from `conf/providers.yml`. The `/model` CLI command switches without YAML edits.
2. **Custom/legacy** (`agent.models.<name>`) — for self-hosted endpoints not in `providers.yml`.

Active selection persists in `./.datus/config.yml`:
```yaml
target: { provider: openai, model: gpt-4.1 }
```
Resolution order: `.datus/config.yml` → `agent.target` in `agent.yml`.

### Extension Points

- **New Node**: file in `datus/agent/node/`, inherit `Node` or `AgenticNode`, register type in `datus/configuration/node_type.py`, add factory mapping in `Node.new_instance()` (`node.py`)
- **New LLM provider (existing interface)**: add entry to `conf/providers.yml` and `datus/conf/providers.yml`; optionally add `model_specs`. No Python needed
- **New LLM model (new SDK/auth)**: file in `datus/models/`, inherit `LLMBaseModel` (`base.py`), register in `MODEL_TYPE_MAP`, add to `PROVIDER_MODELS` in `tests/regression/test_regression_llm.py`
- **New MCP tool**: function in `datus/tools/func_tool/`, register in MCP server tool list

## Guardrails

- **No direct DB imports**: use `ConnectorRegistry` / `db_manager_instance`
- **No hardcoded LLM calls in nodes**: go through `LLMBaseModel`
- **No external deps in CI tests**: zero API keys, zero pre-built data, zero network
- **No secrets in code**: env vars or `${ENV_VAR}` substitution in `agent.yml`
- **Config via YAML**: new tunable parameters belong in `agent.yml`, not hardcoded constants

## PR Conventions

### Title

Must start with one of: `[BugFix]` `[Enhancement]` `[Feature]` `[Refactor]` `[UT]` `[Doc]` `[Tool]` `[Others]`. CI rejects untyped titles.

### Body — must follow `.github/PULL_REQUEST_TEMPLATE.md`

**Non-negotiable.** Every PR body uses the template verbatim with all three sections filled in:

1. **`## Why`** — problem solved; link issues if any
2. **`## Solution`** — approach, key decisions, tradeoffs
3. **`## Test Cases`** — added/changed integration/nightly tests; if none, justify

When using `gh pr create --body`, copy `.github/PULL_REQUEST_TEMPLATE.md` as the starting point. PRs with empty/missing sections must be revised before review.

## Commit Workflow

Run the same gates that protect ordinary PRs before pushing, and keep extra full-suite runs targeted to high-risk changes.

1. **Pre-format**: `uv run ruff format datus/ tests/ && uv run ruff check --fix datus/ tests/` before staging. CI checks the same paths with `ruff format --check datus/ tests/` and `ruff check datus/ tests/`.
2. **PR coverage harness**: `uv run python ci/run-pr-tests.py upstream/main`. This runs the fixed acceptance harness, impacted unit-test targets selected from the diff, Cobertura coverage, and diff coverage. Inspect `ci/test-report.md` and `ci/diff-cover-report.md` when it fails.
3. **Test-quality audit**: `uv run python ci/audit_tests.py --repo-root . --diff-only upstream/main` — must report **`P0=0`**. P0 hard-fails CI; P1 is warn-only but should be addressed. Use `--all` for a full scan when you've touched many test files. Honor noqa with `# audit-noqa: <rule>` only when justified.
4. **Merge-queue rehearsal**: run `uv run python ci/run-merge-queue-tests.py` when changing acceptance harness targets, CI scripts, or code likely to affect merge-queue-only integration coverage.
5. **Pre-commit hooks**: never use `--no-verify`; auto-fix and retry until they pass.
6. **Push**: only to `origin`, never to `upstream`.
7. **PR body**: see **PR Conventions → Body** above.

## Testing Rules

### Tiers and mocking

| Tier | Marker | Mock policy |
|------|--------|-------------|
| CI | PR acceptance harness plus impacted `tests/unit_tests/`; <5 s/test, deterministic | **Must** mock all external calls (LLM, remote DBs, network, optional packages) |
| Nightly | `@pytest.mark.nightly` | Real LLM APIs OK; mock unstable services |
| Regression | `@pytest.mark.regression` | Real services; gate missing keys with `@pytest.mark.skipif` |

CI runs without optional packages (`datus-bi-superset`, `datus-bi-grafana`, …). Tests touching code that imports them must work whether or not the package is installed. (`datus-bi-core` is a hard dependency and always available.)

### File naming and location

| Location | Pattern |
|----------|---------|
| `tests/unit_tests/` | `test_{module}.py`, **mirroring** source path: `datus/a/b/c.py` → `tests/unit_tests/a/b/test_c.py` (e.g. `datus/utils/json_utils.py` → `tests/unit_tests/utils/test_json_utils.py`) |
| `tests/integration/` | `test_{scenario}.py` |
| `tests/regression/` | `test_regression_{dimension}.py` |

Create intermediate `__init__.py` when adding new subdirs. Common patterns: `@pytest.mark.skipif(not os.getenv("KEY"), reason=...)` for missing API keys; `@pytest.mark.parametrize("db_type", [DBType.SQLITE, DBType.DUCKDB])` to fan out across DBs.

### Tests required when modifying these modules

Unit tests follow the mapping rule above. The table lists **additional** integration/regression tests that aren't obvious from the mapping:

| Modified module | Additional tests |
|---|---|
| `datus/models/{provider}_model.py` | `integration/models/test_*_model.py`, `regression/test_regression_llm.py` |
| `datus/agent/node/` | `unit_tests/agent/node/test_node.py`, `test_schema_linking.py`, `test_date_parser_*.py` |
| `datus/cli/repl.py` | `integration/cli/test_cli_commands.py`, `regression/test_regression_web_e2e.py` |
| `datus/tools/func_tool/` | `integration/tools/test_func_tools_db.py`, `integration/tools/test_mcp_server.py` |
| `datus/tools/skill_tools/` | `unit_tests/tools/skill_tools/test_skill_*.py` (config, registry_unit, manager_unit, bash_tool, func_tool) |
| `datus/tools/permission/` | `unit_tests/tools/permission/test_permission_*.py` |
| `datus/mcp_server.py` | `unit_tests/test_mcp_server.py`, `integration/tools/test_mcp_server.py` |
| `datus/storage/reference_template/` | `unit_tests/storage/reference_template/test_*.py`, `integration/tools/test_reference_template.py` |
| `datus/storage/document/` | `integration/storage/test_doc_search.py`, `integration/storage/test_platform_doc.py` |

### Test quality (beyond coverage)

Beyond happy paths, exercise: **input format variants** (all valid shapes, not just the common one); **return-type contracts** (every branch returns the same structure); **cross-component contracts** (consume the producer's real output); **adversarial inputs** for regex/SQL/path sandboxes; **recursive/nested structures** at depth ≥ 3; **spec compliance** for standards (`.gitignore`, SQL dialects).
