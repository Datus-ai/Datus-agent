# Package Command

The `datus package` command exports the current project directory as a **self-contained zip**. The receiver unzips it, exports a few environment variables, runs one setup script, and has a working Datus project — no shared `~/.datus`, no manual config editing, and no credentials travelling in the archive.

## Design guarantees

| Guarantee | How it is enforced |
|---|---|
| **Self-contained** | The generated `conf/agent.yml` pins `home: .` and a fixed `project_name`, so the unzipped directory is the entire runtime. The receiver's `~/.datus` is never read or written. |
| **Zero secrets** | `conf/agent.yml` and `conf/.mcp.json` are *generated*, never copied. Every credential-bearing field is replaced with a `${VAR}` placeholder. A final content scan over the staged files fails the build if anything secret-looking survives — there is no bypass flag. |
| **Sources, not indexes** | Metric / semantic-model / reference-SQL **YAML sources** ship together with a generated `scripts/rebuild_kb.sh`. Binary LanceDB indexes never ship — the receiver rebuilds them locally. |

## Usage

```bash
# Interactive wizard (the normal path)
datus package

# Non-interactive: package everything with defaults
datus package -y
```

All parameters are collected through the wizard — `-y/--yes` is the only flag, and it exists as the scripting / non-TTY escape hatch. Run the command from the project root; the current working directory is what gets packaged.

Press `Ctrl+C` at any step to abort. Nothing is written, and if the interrupt lands during zip assembly the partially written archive is deleted.

### Options

| Option | Description |
|--------|-------------|
| `-y`, `--yes` | Skip the wizard and package everything with defaults. Required when stdin/stdout is not a TTY. |

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Package built |
| `1` | Build failed (including a secret detected by the final scan) |
| `2` | Usage error, or non-interactive terminal without `--yes` |
| `3` | No agent configuration found (`./conf/agent.yml` or `~/.datus/conf/agent.yml`) |
| `130` | Cancelled with `Ctrl+C` |

## Wizard steps

The wizard runs as a linear flow. Empty categories are skipped silently.

| Step | What it asks |
|---|---|
| **Output path** | Where to write the zip. Defaults to `./<project_name>.zip`; must end in `.zip`. Prompts before overwriting an existing file. |
| **File scope** | Package all files, or supply comma-separated **include / exclude regex patterns**. Patterns are validated on the spot and re-asked if invalid. |
| **Subagents** | Which agentic nodes from `agent.yml` to carry. Their prompt templates are pulled in automatically. |
| **Skills** | Project skills (`./.datus/skills`) and global skills (`~/.datus/skills`); the source of each is shown. |
| **Metric datasources** | Which `subject/semantic_models/<datasource>` trees to ship. |
| **Subject areas** | A two-level subject tree. Gates **both** metric documents and reference-SQL summaries — see below. |
| **Plugins** | Installed plugins to record in `scripts/install_plugins.sh`. |
| **Reports** / **Dashboards** | Which artifacts under `reports/` and `dashboards/` to include. |
| **Report dist** | Only asked when reports are selected: bundle the `web-artifact-render` dist so `index.html` opens over `file://`, or leave it loading from the CDN (default). |
| **Summary** | A table of every choice, then a final confirm. Declining exits with `130` and writes nothing. |

Each multi-select screen starts fully selected: `Space` toggles, `a` toggles all, `Enter` confirms. Deselecting an entire category asks for confirmation, so a stray `Ctrl+C` cannot silently drop one.

### Subject-area tree

Subject areas are read from the project's subject tree in the vector store and rendered **two levels deep** — roots plus their direct children, with anything deeper folded into its depth-2 parent:

```
[✓] 营销分析 (24 reference SQL)
[✓]   └ 活动统计 (15 reference SQL)
[-]   └ 预算分析 (2 reference SQL)
[ ] 运营 (22 metrics)
[ ]   └ 活动 (22 metrics)
```

- Counts **roll up**, so a root shows the full cost of taking everything beneath it.
- Checkboxes **cascade**: toggling a parent toggles all of its children, and a parent becomes checked exactly when all of its children are. A partially selected parent is marked `[-]`.
- Selection matches by path prefix, so picking `营销分析` keeps its whole subtree while `营销分析/活动统计` narrows to that branch.

A metric document is matched through its `subject_tree:` tag and filtered **per document**, not per file — one metrics YAML can span several subject areas and only the matching documents travel. Reference SQL is matched through the `subject_tree` field of each summary in `subject/sql_summaries/`.

Metric documents and summaries carrying no subject tag are packaged anyway, with a warning — they belong to no subject area and would otherwise vanish from every filtered package.

## Package layout

```
<project_name>.zip
├── README.md                    # generated: quickstart + required env vars
├── requirements.txt             # generated: pinned datus packages
├── package_manifest.json        # generated: format, selections, per-file sha256
├── conf/
│   ├── agent.yml                # generated: home: ., ${VAR} placeholders
│   └── .mcp.json                # generated (only when MCP servers are configured)
├── .datus/
│   ├── config.yml               # generated: pinned project_name / default datasource
│   └── skills/                  # selected project skills
├── scripts/
│   ├── init.sh                  # generated: dependencies → plugins → knowledge base
│   ├── install_plugins.sh       # generated: datus plugin install --force, per plugin
│   └── rebuild_kb.sh            # generated: bootstrap-kb per source YAML
├── subject/
│   ├── semantic_models/<ds>/    # selected semantic models + metric documents
│   └── sql_summaries/           # selected reference-SQL summaries
├── template/                    # prompt templates for the selected subagents
├── reports/ · dashboards/       # selected artifacts
└── ...                          # the rest of your project files
```

`package_manifest.json` records the package format version, the exact selections, the required environment variables, and a `sha256` plus a `generated` / `project` provenance flag for every file.

### Never packaged

| Category | Entries |
|---|---|
| Runtime state (top level) | `sessions/` `data/` `logs/` `run/` `cache/` `save/` `trajectory/` `output*/` `.venv/` `.git/`, and the REPL `history` file |
| Secrets and OS/editor litter | `.env` `.DS_Store` `._*` (macOS AppleDouble sidecars) `__MACOSX/` `__pycache__/` `*.swp` `*.swo` `*~` `*.duckdb.wal`, plus volume metadata such as `.Spotlight-V100/` |
| Binary indexes | LanceDB data under `data/` — rebuilt by `scripts/rebuild_kb.sh` |

`reports/`, `dashboards/` and `template/` are owned by the selectors: only what you pick in the wizard ships, even when "package all files" is chosen.

## Secrets

Every credential field in `agent.yml` is rewritten to a `${VAR}` placeholder before the config is written into the package:

```yaml
agent:
  home: .
  project_name: baisheng
  providers:
    deepseek:
      api_key: ${DEEPSEEK_API_KEY}
  services:
    datasources:
      starrocks:
        host: ${STARROCKS_HOST:-127.0.0.1}
        port: ${STARROCKS_PORT:-9030}
        password: ${STARROCKS_PASSWORD}
```

- Fields that already used `${VAR}` or `${VAR:-default}` in the source config keep their variable name and default.
- Detection is **schema-driven** — the field's role decides, not the value, because a plaintext key and an ordinary string are indistinguishable.
- Database URIs have only the password component replaced, so host, port and database name survive: `postgresql://svc:${DATUS_DS_PG_URI_PASSWORD}@db.example.com/warehouse`.
- Plugin profiles are sanitized from each plugin's config schema. A plugin whose schema cannot be loaded degrades to replacing *every* string field, with a warning — safe by default.

After staging, a content scan runs over the whole package. If any real-looking secret is found the build **fails** and the offending file plus locator is printed; fix the source config (or exclude the file) and retry.

The generated `README.md` lists every required variable and where it is used:

| Variable | Used by |
|---|---|
| `DEEPSEEK_API_KEY` | providers.deepseek.api_key |
| `STARROCKS_PASSWORD` | services.datasources.starrocks.password |

## Receiving a package

```bash
unzip baisheng.zip -d baisheng && cd baisheng
export DEEPSEEK_API_KEY=... STARROCKS_PASSWORD=...   # see README.md
bash scripts/init.sh
datus-api        # or `datus` for the interactive console
```

`init.sh` installs dependencies, then plugins, then rebuilds the knowledge base. It is safe to re-run: pip is idempotent, plugin installs pass `--force`, and the KB steps overwrite. Each step is also available on its own — `scripts/rebuild_kb.sh` is worth re-running after editing the subject YAML.

!!! note
    `.env` files are **not** auto-loaded when datus-agent is installed via pip. Export the variables in your shell, or run `set -a; source .env; set +a`.

`init.sh` uses `$PYTHON` (default `python3`) and prefers `uv` over `pip`, so it works inside a uv-created virtualenv that has no `pip` module. Set `PYTHON=/path/to/python` to target a specific interpreter.

## Example session

```
$ datus package
Packaging project 'baisheng' from /Users/me/baisheng-project
Output zip path [/Users/me/baisheng-project/baisheng.zip]:
Package all files? [Y/n]: y
Subagents: Space toggles, 'a' toggles all, Enter confirms
...
Subject areas: Space toggles, 'a' toggles all, Enter confirms
...
             Package summary
  Item                Value
  Project             baisheng
  Output              /Users/me/baisheng-project/baisheng.zip
  Subject areas       营销分析/活动统计
  Metric datasources  starrocks
  ...
Build the package now? [Y/n]: y
✓ Package built: /Users/me/baisheng-project/baisheng.zip
  163 files, 4.2 MB uncompressed
  Subject areas: 营销分析/活动统计 → 15 reference-SQL summaries
  Receiver must export: DEEPSEEK_API_KEY, STARROCKS_PASSWORD
```

The result always spells out what the selection produced. Counting zip entries by hand is unreliable — `unzip -l` wraps long or CJK filenames onto several lines, which makes a correctly filtered package look unfiltered.

## Related

- [Knowledge Base — Reference SQL](../knowledge_base/reference_sql.md)
- [Knowledge Base — Metrics](../knowledge_base/metrics.md)
- [Configuration — Agent](../configuration/agent.md)
