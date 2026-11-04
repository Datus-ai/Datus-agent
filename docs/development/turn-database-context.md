# Database switching between chat turns

Each chat turn resolves one immutable execution context before constructing its
tools. It contains a datasource, catalog, physical database and schema. The next
turn may select another target, while saved report/dashboard queries retain their
own coordinates.

## Request resolution

An omitted coordinate (`null`/absent) inherits from the preceding turn only while
its parent coordinates are unchanged. An explicit empty string clears that level.
Changing a datasource invalidates catalog/database/schema; changing a catalog
invalidates database/schema; changing a database invalidates schema. Connection
defaults only apply under their configured parent coordinates. A catalog-only
switch must select a database before opening a connector if its old configured
database belongs to another catalog; it must not silently reuse that database.

Resolved execution contexts require a nonempty database when creating a SQL
connector. Clearing the database invalidates the old selection; execution is
refused until a database is explicitly selected again. This avoids restoring a
configured database, choosing the first file in a glob, or relying on a driver's
implicit database default. Context-free discovery/legacy calls may still use
their configured defaults.

CLI argument overrides are partial: absent flags (including argparse's empty
defaults) keep the selected datasource's configured catalog/database/schema.
The CLI merges those effective defaults before pinning a complete runtime
snapshot. API resolved empty fields continue to mean explicit clears.

The resolved snapshot preserves empty coordinates through node input and semantic
runtime configuration. Empty schema removes the configured schema override, so
the engine selects its default schema. Native DuckDB connections initialize the
selected schema before entering the cache. Query cursors inherit the native
connection's effective search path, because DuckDB cursors are separate sessions.

Metric identity within an artifact includes datasource, subject path and name.
Different sources may define metrics with the same path/name and different model
revisions; conflicting revisions within one source remain invalid.

## Deployment and rollback

This change has no relational database migration, but **does change artifact file
contracts**. New SQL/metric query metadata may contain `execution_context`.
The new reader accepts old files. Old readers with `extra="forbid"` reject files
written by the new writer; compatibility is not symmetric.

Before enabling these writers or reopening the frontend selector, replace **all**
Studio, publication, sandbox and other artifact-reader replicas sharing a project
volume with compatible versions. Deploy Agent through the Backend pin, then SaaS.
Do not run old and new readers against newly written artifact bundles. During a
rolling upgrade, keep writes quiesced until every reader is compatible.

Once new-format artifacts exist, rolling back only the Backend/Agent binary is
unsupported. Either retain compatible readers or quiesce writes and restore the
matching pre-upgrade artifact-volume backup along with the old application set.
Do not strip coordinates to make files parse: doing so can retarget queries.
Take the volume backup before permitting new writes; this PR provides no lossy
artifact downgrade/migration command.

## Observability

Session acknowledgments/history and successful SQL/metric tool results expose
non-secret execution coordinates. SQL dashboard query responses also return the
actual connector coordinates alongside the saved query target and result rows.
Correlate these per turn; a browser selector screenshot alone does not establish
which target reached a historical request.

Prompt snapshot fingerprints include static Jinja include/import/extends
dependencies. Dynamic references conservatively fingerprint every template
available to the tenant's loader. Template changes invalidate persisted prompts
on the next turn.
