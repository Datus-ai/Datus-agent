# Embedded Dosi integration

Datus Agent depends directly on `dosi-engine`. `dosi/runtime.py` keeps the Agent-facing catalog and query contract while calling the native Python binding. It loads one engine per OSI model file, refreshes changed files, and maps native results and errors to `SemanticTools` and Explorer responses.

Only Agent-used behavior lives here:

- `dosi/engine.py`, `model.py`, `dialects.py`, and `errors.py`: model discovery, connection setup, native calls, and error mapping.
- `dosi/runtime.py`: semantic model and metric discovery, dimensions, query, attribution, validation, lineage, and Explorer metric source methods.
- `dosi/authoring.py`, `metric_author.py`, and `authoring.py`: native validated OSI YAML metric editing for Explorer.
- `dosi/authoring_spec.py`: prompt contract from the installed engine plus the OSI core authoring reference.

Knowledge Base synchronization runs through the OSI source indexer after validated file edits.

`SemanticTools.runtime` exposes the embedded `DosiRuntime` to callers using `lineage_graph`.

Use `uv sync --locked --group ci` for development and `uv run --locked pytest` for verification.
