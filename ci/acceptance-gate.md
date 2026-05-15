# Acceptance Gate Selection

Acceptance tests are the smallest deterministic representative set for product-critical cross-component contracts. A test should be promoted to the merge queue acceptance gate only after it passes the basic CI requirements and protects a core chain.

Basic requirements:

- No external API keys.
- No public network access.
- No real LLM calls.
- Stable temporary directory and cache isolation.
- Clear per-test timeout and bounded total runtime.

Core-chain criteria:

| Question | Strong signal for acceptance |
| --- | --- |
| Is this a user-visible entry point? | CLI, API, MCP, gateway, or a core subagent path |
| Does it cross component boundaries? | Config to tool registry to connector, API to workflow to node, storage path to document store to search tool |
| Would a break block a main product capability? | Querying data, running workflows, exposing MCP tools, retrieving context, rendering reference templates |
| Would unit tests miss the failure mode? | Registration, config resolution, path normalization, real local SQLite or DuckDB behavior, tool schemas |
| Is it representative rather than exhaustive? | One happy path plus one important boundary or failure case for each core chain |

Good merge queue acceptance candidates:

- API workflow execution over local fixtures.
- CLI commands that exercise real command dispatch and tool behavior.
- MCP tool registration and one or two real client calls over local datasources.
- Database tools over SQLite and DuckDB fixtures.
- Reference template search, get, and render over prebuilt local data.
- Platform document local fetch, parse, chunk, store, and search.
- Deterministic agentic harness tests with fake or mocked LLMs that verify tool wiring, output path, and state contracts.

Do not promote these to merge queue acceptance:

- Real provider model calls.
- External website or GitHub document fetching.
- Full real-LLM product E2E tests.
- Slow adapter smoke tests that depend on live services.
- Broad parameter matrices that duplicate lower-level coverage.
