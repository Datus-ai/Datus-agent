# Export complete query results

Use `execute_sql` to inspect query results. Use `export_query` when the complete
result must be consumed by Python or downloaded from the project Explorer:

```python
export_query(
    sql="sql/order_match/batch_001.sql",
    output_file="exports/order_match/parts/001.parquet",
    datasource="warehouse",  # optional
    database="analytics",  # optional
)
```

SQL text is also accepted. Like `execute_sql`, optional `datasource` and
`database` select the connection; omitting them uses the current datasource and
database. Selecting another datasource without a database uses that datasource's
configured default. A fixed single-connector instance refuses selectors that
would require switching its bound connection. An omitted output path generates `exports/<id>.csv`. Output paths and SQL files
are resolved inside the node's project filesystem root. Existing outputs are
never overwritten. CSV and Parquet are supported; Python can create a final XLSX.

The tool returns only a project-relative path, byte/row counts, SQL hashes,
policy scope and completion status. Query rows never enter the tool response.
SQL read permissions and the uploads catalog file-access restrictions also apply
to exports, including SQL read from a file or rewritten by a policy.
The adjacent `<filename>.meta.json` contains the same completion metadata.
`complete=true` means the authorized query was fully consumed, not that a larger
multi-query task is complete. Policy changes to row limits are refused rather
than silently publishing a truncated result as complete.
`read_mode` reports `streaming` or `buffered`. Streaming is preferred when both
the adapter and result-policy hooks support it. Otherwise the tool fetches the
complete query result into Agent memory, applies whole-result policies, and
writes bounded batches. Neither mode returns query rows to the model.

## Excel matching workflow

1. Inspect the workbook with Python through `bash`. Read IDs as strings and
   preserve leading zeros. Report numeric cells which may already have lost
   precision, empty IDs and duplicates. Print only small statistics.
2. Persist normalized IDs, a batch manifest and one `.sql` file per ID batch.
   Check the datasource schema before choosing column names. Match IDs exactly;
   ranges are not substitutes for exact membership.
3. The Agent invokes `export_query` for each SQL file. Python does not need a DB
   password or an HTTP authentication bridge. Use Parquet for typed intermediates.
4. Python reads only completed outputs whose SQL hashes match the manifest.
   Reconcile expected batches, exported row counts, matched IDs and unmatched IDs.
   Retain multiple source records per ID; deduplicate only the query's input keys.
5. Write final results under `exports/<task>/`. File delivery uses the existing
   Explorer behavior and its 50 MiB read limit. Oversized files still fail to
   download through that path. Chat reports paths, sizes and statistics, not the
   result contents or a promise that every generated file is downloadable.

Resume from manifest entries without valid completion metadata. A file alone
does not prove that an export completed. Report conflicting or incomplete files
instead of overwriting them or silently incorporating them into the merge.

## Host limits and policies

```yaml
agent:
  bulk_query:
    batch_rows: 5000
    max_rows: 1000000
    max_bytes: 1073741824
    max_batch_bytes: 33554432
```

These limits apply to file exports during source consumption and staging,
before output publication. For buffered reads, source row/byte limits can only
be checked after the full result arrives; they do not cap the driver's peak
memory allocation. Failed/cancelled exports do not publish a successful
file. SDK export calls wait asynchronously before entering a worker thread,
with at most two operations per event loop and two readers per connector.
Export tool waits are bounded at 300 seconds. A timed-out export keeps its slot until its
worker exits, so repeated attempts cannot fill the thread pool. A driver
operation may outlive the wait; cancellation prevents late file publication
without claiming the database was killed. `batch_rows` bounds source batches in
streaming mode and file-staging batches in both modes.

SQL-file reads, Arrow staging, export publication, compressed diagnostics and
cleanup all use pinned project directory descriptors. File opens refuse
subsequently swapped symlinks; mkdir/link/unlink operate relative to those
descriptors. Publication also checks that its directory still occupies the
authorized path, so a replaced or renamed directory cannot produce a false
completion reference. Project-local file links remain supported. This secure
IO path requires POSIX directory descriptors (Linux/macOS); other platforms
fail closed instead of falling back to path-only checks.

Adapters can implement `supports_streaming_read` and `iter_query_batches` to
enable streaming. Without this capability, exports use the existing
`execute_query(..., result_format="arrow")` API and buffer the full result.
SQLAlchemy-backed adapters require an unbuffered driver (or a native cursor such
as SQLite/Oracle); DuckDB uses Arrow batches. Abandoned SQLAlchemy streams
invalidate their connection before cursor cleanup so MySQL cannot drain unread rows.
Staging reconciles null-only batch schemas on disk without a full-result DataFrame.
Plugins with post-read result hooks can declare `supports_batch_results=True`
or implement `after_read_batch` to enable streaming. Whole-result-only policies
select buffered mode and run once over the complete result before file staging.
Batch-only hooks are also enforced on the buffered result, preserving plugin
order. Policy denials, result limits and query rewrites are never bypassed.
The execution mode is selected before querying; a failed stream is not replayed
using a buffered read.

Adapter and SQL-policy upgrades are optional streaming improvements rather than
requirements for exporting files. SQLAlchemy streaming uses
`datus-db-core>=0.1.11` and `datus-sqlalchemy>=0.1.13`.
Existing Backend/SaaS download behavior and limits
are preserved. This change does not alter deployment pins.

## Existing transfer protections

`transfer_query_result` retains its existing DataFrame read, dialect-specific
writer or generated INSERT path, replace/append behavior, one-million-row limit
and `deliverable_target` reconciliation. This change does not add a general
streaming migration API or change adapter write implementations.

The tool requires an integer `batch_size` between 1 and 10,000. Generated INSERT
statements must fit within 1 MiB of UTF-8 text. All such statement sizes are
checked before replace clears target data; exceeding a limit fails rather than
silently changing batch sizes. The source is still materialized in memory.

The response omits full source SQL and generated CREATE TABLE SQL, retaining
only a source SQL hash and transfer metadata. Existing partial-write errors
retain their confirmed row count in the short error message. Check the target
before retrying a failed transfer. Use `export_query` for a file deliverable;
SQL file inputs are supported by that tool, not by `transfer_query_result`.

Errors are limited to 2 KiB and complete tool metadata to 16 KiB. Detailed errors
live as compressed files under `exports/.diagnostics/`. Inspect them with Python
and print a bounded summary. Explorer downloads retain their existing limits.
Tool results entering model context, events, history and tracing are bounded.
Bash sessions retain their existing file-backed output archiving; scripts
should print only summaries and paths.

The database-free integration test `tests/integration_tests/test_query_export_workflow.py`
exercises 114,444 synthetic IDs, 23 SQL-file batches and XLSX reconciliation.
This is not live StarRocks acceptance; validate its driver, execution plans and
latency separately against the intended StarRocks datasource.
