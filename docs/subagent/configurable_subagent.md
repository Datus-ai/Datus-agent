# Configurable subagents

## 1. User definitions

`node_class: configurable` creates an instance of `ConfigurableAgenticNode`.
No Python subclass or SQL-specific parent is required. Existing typed nodes keep
their current behavior and defaults.

This release provides YAML/AgentConfig definitions, runtime factories, CLI direct
selection and task/chat execution. Existing TUI and `/agent` CRUD editor forms
retain their typed-agent contracts; SaaS/Backend editor integration follows
separately.

```yaml
agent:
  agentic_nodes:
    finance_review:
      node_class: configurable
      description: Review sales using selected evidence.
      instructions: Explain the result and cite the supporting evidence.
      skills:
        - finance-review
      tools:
        - db_tools.execute_sql
        - filesystem_tools.read_file
      mcp:
        - connection_ref: company_docs
          tools:
            - search
      scoped_context:
        datasource: warehouse
        tables: orders,customers
        metrics: Finance.Revenue
        sqls: Finance.Reports
        project_files:
          - reports/**
      output:
        format: markdown
        instructions: State any gaps in the evidence.
```

| Field | Meaning |
|---|---|
| `name`, `description` | Identity and discovery description. An optional name must match the registration key. |
| `instructions` | Behavior without a custom prompt-template file. |
| `skills` | Exact names of installed, authorized skills. Loading a skill grants no tools or resource access. |
| `tools` | Explicit native tool IDs, preferably `category.method`. Omission selects no native business tools. |
| `mcp` | Project-owned connection references plus explicit tool lists. Credentials and connection details remain with the host. |
| `scoped_context` | Existing datasource, table, metric and reference SQL scopes, extended with `project_files`. See the execution rules below. |
| `output` | `format: text`, `markdown` or `json`, and optional instructions. Invalid JSON fails the turn. |

Chat selects delegates using the capability descriptions in its `task` tool.
Describe the agent's purpose, appropriate tasks, boundaries and deliverables in
`description`; the host does not infer these from its instructions, skills or
implementation. Common Chat delegation principles govern direct answers,
workflow coordination and continuation of existing artifacts. Business-specific
agent selection belongs to the advertised descriptions.

An explicit plugin source replaces the same-named builtin's advertised
description. On each Chat turn the host refreshes the mounted task description
without replacing its permission or proxy invocation wrappers. Capability
descriptions do not expand tool permissions or `scoped_context`.

Skills, native tools, MCP tool selections and project files use YAML lists;
there is no default selection or selection mode. Existing `tables`, `metrics`
and `sqls` retain their comma/newline-separated string formats.
`datus.agent.node.configurable_tools.configurable_tool_catalog()` lists native
tools with registered contracts and common factories. Specialized nodes' private
helpers and wildcard tool selections are unavailable. Unambiguous bare method
names are also accepted for a user definition.

Use existing entry points: `datus --subagent finance_review -p "..."`, the
interactive `/agent finance_review` command, or `task(type="finance_review", ...)` from an
authorized parent. Delegated nodes retain the two-level limit. Interactive
clarification is an internal control channel.

## 2. Execution boundaries

The host intersects the resolved instance definition with the caller's
project/table scopes. SaaS validates its template and role restrictions before
building this definition. Workflow runs retain the caller's permission profile.
Filesystem reads, writes, glob and grep enforce selected paths and the existing
path policy, including traversal and symlink protection. File imports, SQL file
references and exports also check file selection; exports require an explicit
allowed output path.

New configurable agents require explicit scopes: no `datasource` or
`tables_by_datasource` means no business data tools; no `project_files` means
no explicit project file access. A single `datasource` is sufficient to select
that connection, including when it differs from the caller's current connection.
Within selected datasources, omitted `tables`, `metrics` or `sqls` add no further
filter; configured filters use the existing scope machinery.

For multiple sources, the keys below are project datasource names. `datasource`
selects the default connection and must belong to the allowed source set. Use
`*` to explicitly allow all tables in one source.

```yaml
scoped_context:
  datasource: warehouse
  tables_by_datasource:
    warehouse: orders,customers
    crm: customers
  metrics: Finance.Revenue
  sqls: Finance.Reports
  project_files:
    - reports/**
```

Semantic tools retain existing model discovery under
`subject/semantic_models/<datasource>/`, including existing ontology loading.
There is no separate semantic model selection field. `metrics` selects logical
metric subjects; `project_files` controls explicit filesystem operations rather
than the semantic engine's model discovery. Reading a YAML file directly requires
a matching file scope and the corresponding native tool.

These explicit defaults apply only to `node_class: configurable`. Existing
`gen_sql` / `ask_metrics` scope defaults and inheritance remain unchanged.

MCP filters intersect the project connection's filter, agent selection and live
permissions. The external service enforces its resource scope; SaaS must provide
the appropriate authorized connection. AgentSpec cannot introduce a connection,
URL, command or credentials.

`prepare_turn` can narrow `scoped_context`, skills and MCP tools. Omitted scope
fields retain the static bounds; datasource sets and table/subject filters may
not be widened. Dynamic file filters are intersected with static filters.
Data and retrieval tools are rebound to the effective scope before execution.
Host calls during preparation/finalization use the same scope and permission
gates as model calls.

## 3. Private plugin definitions

Extend an installed package's existing `datus-plugin.yml` with `agents`. Its
entry-point group remains `datus.plugins`; the value names the package. Package
the YAML and instruction assets in its wheel. No public Datus-Plugins repository
change is required. Private packages can live under `Datus-backend/plugins/`
and be installed only in SaaS images, using existing project plugin activation.

```toml
[project.entry-points."datus.plugins"]
private-research = "private_research"

[tool.setuptools.package-data]
private_research = ["datus-plugin.yml", "instructions.md"]
```

```yaml
manifest_version: 1
agents:
  concept_review:
    protocol_version: 1
    description: Review concepts with instance-local business state.
    instructions_file: instructions.md
    runtime_factory: private_research.runtime:create_runtime
    tool_definitions:
      - tool_id: concept_tools.record_proposal
        description: Record a proposed concept for this review.
        access: write
        input_schema:
          type: object
          properties:
            name:
              type: string
          required:
            - name
          additionalProperties: false
```

The publisher can declare all common behavior fields, optional `config` defaults
and a JSON `config_schema` for business parameters. Plugin tools use a separate
permission family such as `concept_tools`. The host reserves its own tool
families, `load_skill` and names starting with `mcp__`. The host reserves the
`ontology_tools` family for the `datus-semantic-plugin` entry-point name and grants
permission to publisher tools through ordinary permission configuration.

```yaml
agent:
  agentic_nodes:
    project_concepts:
      node_class: configurable
      source:
        plugin: private-research
        agent: concept_review
      runtime_config:
        review_label: Sales concepts
```

Instance fields override publisher defaults; omitted fields keep the publisher's
explicit definition. Projects bind their own authorized resource names so a
package can be reused across projects. Runtime preparation can only narrow the
resolved instance definition. User definitions cannot supply executable
factory references. Discovery reads metadata without importing plugin code;
runtime loading is lazy. Duplicate names or missing/disabled sources fail with
a configuration error. Replacing a built-in such as `semantic_modeling` requires an explicit `source`,
without an implicit fallback. `ontology_analyst` remains a built-in agent.

## 4. Optional runtime

```python
class ReviewRuntime:
    def __init__(self, config, host):
        self.host = host
        self.label = config["review_label"]
        self.proposals = []

    def prepare_turn(self, context):
        return {
            "context": {"proposal_count": len(self.proposals)},
            "tool_bindings": {"concept_tools.record_proposal": self.record},
        }

    def record(self, name):
        self.proposals.append(name)
        return {"proposals": list(self.proposals)}

    async def finalize(self, result, context):
        return {"review": {"label": self.label, "proposals": list(self.proposals)}}

    def close(self):
        self.proposals.clear()


def create_runtime(config, host):
    return ReviewRuntime(config, host)
```

| Stage | Contract |
|---|---|
| `create_runtime(config, host)` | Once per execution instance. Return a plain runtime object; business configuration is copied per instance. |
| `prepare_turn(context)` | Before each user turn. Return optional `instructions`, JSON `context`, declared `tool_bindings`, and narrowing `skills`, `mcp`, `scoped_context`. |
| `finalize(result, context)` | After normal model completion. Return fields to merge or `None`; exceptions fail the turn. Skipped after model failure/cancellation. |
| `close()` | Optional, once on instance release, including failed/cancelled runs; not after every successful turn. Release plugin-owned resources. |

No `datus`, Backend or shared SDK import is needed. Methods may be synchronous
or asynchronous. Sync callbacks must be nonblocking; use async handlers for I/O.
Callbacks sharing a runtime are serialized and execute locally. Publisher code
is trusted Python; this interface is not a sandbox for user-submitted code.

Context contains `agent_name`, `session_id`, JSON `input` and `scoped_context`;
finalization additionally sees prepared `context`. `host.scoped_context` exposes
current selections. `await host.call_tool(tool_id, arguments)` invokes a selected
native tool with permission/resource checks and decodes JSON results. The host
does not expose AgentConfig, connectors or the node.

Common result fields include `response`, `data`, `success`, `error` and
`tokens_used`. Additional JSON business fields survive node serialization and
`task()` success/failure envelopes. For JSON output, the host parses the model
reply into `data` before finalization. The finalizer can return a display summary
in `response` and update the machine-readable payload in `data`. A successful
`--print` run emits the finalized `data` as a JSON content block. Dynamic prompt
content and selected skill contents invalidate stale system-prompt snapshots.

Configuration-only and static plugin agents need no runtime. Backend packages
the enhanced semantic-modeling runtime for SaaS and On-Prem. Agent provides
the built-in semantic modeler and `ontology_analyst`.


## 5. Shared semantic authoring

The built-in semantic modeler and configurable agents reuse an Agent-owned
`SemanticAuthoringSession`. Select the needed native authoring tools explicitly;
no authoring tools are added by default. For example:

```yaml
agent:
  agentic_nodes:
    semantic_modeling:
      node_class: configurable
      source:
        plugin: datus-semantic-plugin
        agent: semantic_modeling
      tools:
        - semantic_tools.list_existing_osi_semantic_models
        - semantic_tools.plan_osi_semantic_model_target
        - semantic_tools.bind_osi_semantic_model_target
        - semantic_tools.upsert_osi_datasets
        - semantic_tools.upsert_osi_metrics
        - semantic_tools.validate_semantic
        - semantic_tools.write_semantic_model_plan_file
        - filesystem_tools.read_file
      scoped_context:
        datasource: warehouse
        tables: orders,customers
        project_files:
          - subject/semantic_models/warehouse/**
          - subject/ontology/**
          - .datus/semantic-model-plans/**
```

This example requires an installed, activated private package. The public Agent
provides the host capability; it does not ship the private enhanced runtime.
Backend installs the package for SaaS and On-Prem deployments.
The example assumes the affected models are all under `warehouse`. When an
ontology is shared across datasources, grant each affected model path in
`project_files` so the host can validate the complete set.

A private runtime uses the injected host by duck typing, with no `datus` import
or shared SDK. Existing `host.call_tool(tool_id, arguments)` calls invoke only
selected native tools through ordinary permissions. Semantic target inventory,
selection and edits additionally honor `project_files`. The required
`dosi-semantic-authoring` specification is injected by the host; its references
are read-only runtime documentation, separate from business file grants.

```python
async def finalize(self, result, context):
    # prepare_turn starts isolation with await host.semantic_authoring.begin_draft().
    # Prepare UID before validating the exact document. These methods return
    # the complete document, model path, revision and business-validation data.
    await self.host.semantic_authoring.prepare()
    checked = await self.host.semantic_authoring.validate()
    # files contains project-relative ontology/mapping paths and UTF-8 text.
    await self.host.semantic_authoring.inspect_bundle(files, mapping_file)
    await self.host.semantic_authoring.publish(files, expected_revision=checked["artifact_sha256"])
    reconciled = await self.host.semantic_authoring.reconcile()
    if reconciled.get("code") == "kb_sync_failed":
        return reconciled
    completed = await self.host.semantic_authoring.complete()
    return {"semantic_models": [completed["semantic_model_file"]]}
```

`snapshot()` reads the complete selected model without model-facing preview
truncation. `prepare()` applies host-owned identities; `validate()` checks current
artifact and compile evidence. `reconcile()` prepares and revalidates the current
revision before KB synchronization, retaining the rollback checkpoint.
`complete()` releases it only after that exact revision was reconciled.
Lifecycle writes require a selected, permitted semantic mutation tool as well
as a permitted datasource and target file. Purely configured agents without a
private runtime receive the same deterministic completion automatically after
successful authoring.

Target selection remains limited to one model per turn. Artifact mutation after
reconciliation requires a new turn. Before successful reconciliation, terminal
failure or cancellation restores the target snapshot. After successful KB sync,
a later private-runtime failure is reported and the matching YAML is retained;
the host does not claim a transaction across files and the KB.

An installed publisher can call `begin_draft()` once at turn preparation, before
selecting a target. The host snapshots authorized `project_files` and directs
native semantic edits to an isolated project. `inspect_bundle()` uses production
discovery and Dosi lowering on those exact bytes, including every affected
mapping. `publish()` checks the validated digests and live input revisions under
shared project/model locks, then publishes model and contributed ontology files
together. Failures before reconciliation restore the complete bundle; a KB sync
failure retains the validated files and reports `kb_sync_failed` with recovery
required, because the KB may already be partially updated. `state()` reports
selection/mutation state even before a target exists; `snapshot()` still requires
one selected model. Business edit rules stay in the private runtime; scope,
permissions, project I/O, publication and native validation stay in the host.

The original built-in retains its tools, statuses, datasets-only scope, UID
rules, planning/review flow, validation and KB reconciliation. Plan acceptance
retains prepare/commit revision checks and the existing Web review adapter;
`review_status: applied` alone is not human approval.
