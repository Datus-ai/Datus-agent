"""Per-invocation init metadata, scope restrictions and ordinary cancellation."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from agents import FunctionTool

from datus.cli.init_context import InitContext
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.tool_result_store import ACTIVE_RESULT_STORE, ToolResultStore


@pytest.fixture
def starrocks_capabilities(monkeypatch):
    """Unit tests declare the adapter contract without requiring its package."""
    from datus.tools.db_tools import connector_registry

    monkeypatch.setitem(connector_registry._capabilities, "starrocks", {"catalog", "database"})


@pytest.fixture
def run(tmp_path):
    context = InitContext(tmp_path, {"project": "test", "datasource": "db"}, "orders")
    yield context
    context.close()


class Metadata:
    def __init__(self):
        self.calls = 0

    def describe_table(self, table_name: str) -> FuncToolResult:
        """Describe live structure."""
        self.calls += 1
        return FuncToolResult(result={"columns": [{"name": "id", "type": "BIGINT", "pk": True}]})

    def search_table(self, query_text: str) -> FuncToolResult:
        """Search indexed metadata."""
        self.calls += 1
        return FuncToolResult(result={"metadata": [{"table_name": "orders", "sample_rows": ["secret" * 10000]}]})


def wrapped(context, method):
    return context.wrap_tools([trans_to_function_tool(method)])[0]


@pytest.mark.asyncio
async def test_cache_and_metadata_only_tool_surface(run):
    metadata = Metadata()
    tool = wrapped(run, metadata.describe_table)
    a = await tool.on_invoke_tool(None, '{"table_name":"orders"}')
    b = await tool.on_invoke_tool(None, '{"table_name":"orders"}')
    assert a == b and metadata.calls == 1
    await tool.on_invoke_tool(None, '{"table_name":"customers"}')
    assert metadata.calls == 2
    unsafe = FunctionTool(
        name="execute_sql", description="SQL", params_json_schema={}, on_invoke_tool=tool.on_invoke_tool
    )
    assert run.wrap_tools([unsafe]) == []


@pytest.mark.asyncio
async def test_cached_samples_cannot_reenter_archive(run, tmp_path):
    store = ToolResultStore(SimpleNamespace(dir=tmp_path / "archive"))
    token = ACTIVE_RESULT_STORE.set(store)
    metadata = Metadata()
    tool = wrapped(run, metadata.search_table)
    try:
        for call_id in ("first", "cached"):
            result = await tool.on_invoke_tool(SimpleNamespace(tool_call_id=call_id), '{"query_text":"orders"}')
            projected = store.project(call_id, json.dumps(result), force=True)
            assert "secret" not in json.dumps(projected)
            assert "sample_rows" not in json.dumps(projected)
        assert metadata.calls == 1
    finally:
        ACTIVE_RESULT_STORE.reset(token)


def test_each_invocation_uses_current_scope_and_existing_documents(tmp_path):
    (tmp_path / "AGENTS.md").write_text("# Project\nExisting user knowledge.")
    # Historical init files are unrelated inputs, even if they are corrupt.
    legacy = tmp_path / ".datus/init/old/state.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("not json")
    first = InitContext(tmp_path, {}, "orders")
    first.record_user_updates(["Orders only"])
    first.close()
    second = InitContext(tmp_path, {}, "customers")
    try:
        assert second.state["scope"] == "customers" and second.user_context() == []
        assert second.calls == 0 and second._cache == {}
        assert (tmp_path / "AGENTS.md").read_text() == "# Project\nExisting user knowledge."
        assert legacy.read_text() == "not json"
    finally:
        second.close()


def test_project_paths(run):
    with pytest.raises(ValueError, match="inside"):
        run._path("../outside.sql")
    with pytest.raises(ValueError, match="only"):
        run._output_path(".datus/config.yml")


@pytest.mark.asyncio
async def test_parallel_duplicate_calls_use_one_database_call(run):
    metadata = Metadata()
    tool = wrapped(run, metadata.describe_table)
    results = await asyncio.gather(*(tool.on_invoke_tool(None, '{"table_name":"orders"}') for _ in range(8)))
    assert all(result["success"] == 1 for result in results)
    assert metadata.calls == 1


@pytest.mark.asyncio
async def test_failed_metadata_is_retried_and_namespace_is_bound(run):
    async def failure(context, arguments):
        return {"success": 0, "error": "unavailable"}

    for _ in range(3):
        result = await run.invoke("describe_table", failure, None, '{"table_name":"orders"}')
    assert result["error"] == "unavailable" and run.calls == 3
    assert (await run.invoke("describe_table", failure, None, '{"table_name":"orders","datasource":"other"}'))[
        "success"
    ] == 0
    assert (await run.invoke("query_lineage", failure, None, "{}"))["success"] == 0


@pytest.mark.asyncio
async def test_read_can_recover_after_missing_file_is_created(run):
    from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

    files = FilesystemFuncTool(root_path=str(run.root))
    read = wrapped(run, files.read_file)
    args = json.dumps({"path": "knowledge/sales.md"})
    for _ in range(2):
        assert (await read.on_invoke_tool(None, args))["success"] == 0
    write = wrapped(run, files.write_file)
    assert (await write.on_invoke_tool(None, json.dumps({"path": "knowledge/sales.md", "content": "Sales"})))[
        "success"
    ] == 1
    result = await read.on_invoke_tool(None, args)
    assert result["success"] == 1 and "Sales" in str(result["result"])


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["describe_table", "get_table_ddl"])
@pytest.mark.parametrize(
    "table_name",
    ["other.orders", "`other`.`orders`", "other_catalog.allowed.orders", "`other_catalog`.`allowed`.`orders`"],
)
async def test_qualified_identifier_cannot_override_bound_namespace(
    tmp_path, starrocks_capabilities, tool_name, table_name
):
    run = InitContext(tmp_path, {"dialect": "starrocks", "catalog": "default_catalog", "database": "allowed"}, "")

    async def unexpected(context, arguments):
        pytest.fail("A cross-namespace lookup reached the database")

    try:
        result = await run.invoke(
            tool_name, unexpected, None, json.dumps({"table_name": table_name, "database": "allowed"})
        )
        assert result["success"] == 0 and "database context" in result["error"]
        assert run.calls == 0
    finally:
        run.close()


@pytest.mark.asyncio
async def test_qualified_cache_and_quoted_dotted_table_use_effective_target(tmp_path, starrocks_capabilities):
    run = InitContext(tmp_path, {"dialect": "starrocks", "database": "allowed", "catalog": "default_catalog"}, "")
    received = []

    async def describe(context, arguments):
        received.append(json.loads(arguments))
        return {"success": 1, "result": {"columns": [{"name": "id"}]}}

    try:
        for name in ("orders", "allowed.orders", "`default_catalog`.`allowed`.`orders`"):
            result = await run.invoke("describe_table", describe, None, json.dumps({"table_name": name}))
            assert result["success"] == 1
        assert len(received) == 1
        result = await run.invoke("describe_table", describe, None, '{"table_name":"`allowed`.`other.orders`"}')
        assert result["success"] == 1
        assert received[-1]["table_name"] == "`allowed`.`other.orders`"
        assert len(received) == 2
    finally:
        run.close()


@pytest.mark.asyncio
async def test_literal_dot_and_qualified_name_use_distinct_metadata_cache(tmp_path):
    run = InitContext(tmp_path, {"dialect": "duckdb", "database": "allowed"}, "")
    received = []

    async def describe(context, arguments):
        received.append(json.loads(arguments)["table_name"])
        return {"success": 1, "result": {"columns": [{"name": str(len(received))}]}}

    try:
        literal = await run.invoke("describe_table", describe, None, json.dumps({"table_name": '"other.orders"'}))
        qualified = await run.invoke("describe_table", describe, None, json.dumps({"table_name": "other.orders"}))
        assert literal["success"] == qualified["success"] == 1
        assert literal["result"] != qualified["result"]
        assert received == ['"other.orders"', "other.orders"]
    finally:
        run.close()


@pytest.mark.asyncio
async def test_metadata_cache_uses_shared_canonical_arguments(run):
    class Catalog:
        calls = 0

        def list_tables(self, include_views: bool = True) -> FuncToolResult:
            self.calls += 1
            return FuncToolResult(result=[{"qualified_name": "orders", "include_views": include_views}])

    catalog = Catalog()
    tool = wrapped(run, catalog.list_tables)
    first = await tool.on_invoke_tool(None, '{"include_views":"false"}')
    second = await tool.on_invoke_tool(None, '{"include_views":false}')
    assert first == second and first["success"] == 1
    assert catalog.calls == 1


@pytest.mark.asyncio
async def test_same_table_names_in_different_schemas_both_query_live_metadata(run):
    async def describe(context, arguments):
        return {"success": 1, "result": {"columns": [{"name": "id"}]}}

    a = await run.invoke("describe_table", describe, None, '{"table_name":"orders","schema_name":"a"}')
    b = await run.invoke("describe_table", describe, None, '{"table_name":"orders","schema_name":"b"}')
    assert a["success"] == b["success"] == 1
    assert run.calls == 2


@pytest.mark.asyncio
async def test_final_tool_filter_is_idempotent(run):
    metadata = Metadata()
    tool = run.wrap_tools([wrapped(run, metadata.describe_table)])[0]
    assert (await tool.on_invoke_tool(None, '{"table_name":"orders"}'))["success"] == 1
    assert metadata.calls == 1


@pytest.mark.asyncio
async def test_interruption_preserves_published_documents(run):
    target = run.root / "AGENTS.md"
    target.write_text("# Previously published knowledge\n")
    entered = asyncio.Event()

    async def actions():
        entered.set()
        await asyncio.Event().wait()
        yield "unreachable"

    async def consume():
        async for _ in run.stream(actions()):
            pass

    task = asyncio.create_task(consume())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    run.close()
    assert run.state["status"] == "stopped"
    assert target.read_text() == "# Previously published knowledge\n"


@pytest.mark.asyncio
async def test_tool_owned_timeout_stops_further_exploration(run):
    async def timeout(context, arguments):
        raise TimeoutError("Database statement timeout")

    result = await run.invoke("describe_table", timeout, None, '{"table_name":"orders"}')
    assert result["success"] == 0 and "timed out" in result["error"]
    assert "timed out" in run.state["stop_reason"]


@pytest.mark.asyncio
async def test_late_search_result_never_registers_raw_samples(run, tmp_path):
    store = ToolResultStore(SimpleNamespace(dir=tmp_path / "archive"))
    token = ACTIVE_RESULT_STORE.set(store)
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed(context, arguments):
        entered.set()
        await release.wait()
        active = ACTIVE_RESULT_STORE.get()
        if active:
            active.register("late", {"sample_rows": ["secret"]})
        return {"success": 1, "result": {"sample_rows": ["secret"]}}

    try:
        task = asyncio.create_task(run.invoke("search_table", delayed, SimpleNamespace(tool_call_id="late"), "{}"))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        run.close()
        release.set()
        await asyncio.gather(*run._inflight)
        await asyncio.sleep(0)
        assert not store._registered
    finally:
        release.set()
        ACTIVE_RESULT_STORE.reset(token)


@pytest.mark.asyncio
async def test_ordinary_file_tools_edit_scoped_knowledge_across_invocations(run):
    from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

    files = FilesystemFuncTool(root_path=str(run.root))
    write = wrapped(run, files.write_file)
    args = json.dumps({"path": "knowledge/sales.md", "content": "# Sales\nExisting user explanation.\n"})
    assert (await write.on_invoke_tool(None, args))["success"] == 1
    assert (run.root / "knowledge/sales.md").read_text() == "# Sales\nExisting user explanation.\n"
    assert run.state["outputs"] == ["knowledge/sales.md"]
    run.close()

    second = InitContext(run.root, run.binding, "Add invoicing guidance to orders")
    edit = wrapped(second, files.edit_file)
    existing = "Existing user explanation."
    try:
        result = await edit.on_invoke_tool(
            None,
            json.dumps(
                {
                    "path": "knowledge/sales.md",
                    "old_string": existing,
                    "new_string": existing + "\nPayment must settle before invoicing (rules.md).",
                }
            ),
        )
        assert result["success"] == 1
        content = (run.root / "knowledge/sales.md").read_text()
        assert existing in content and "Payment must settle" in content
        failed = await edit.on_invoke_tool(
            None, json.dumps({"path": "knowledge/sales.md", "old_string": "No such text", "new_string": "Replacement"})
        )
        assert failed["success"] == 0
        assert (run.root / "knowledge/sales.md").read_text() == content
        assert second.state["outputs"] == ["knowledge/sales.md"]
        assert not (run.root / ".datus/init").exists()
    finally:
        second.close()


@pytest.mark.asyncio
async def test_knowledge_writes_keep_project_boundaries(run, tmp_path):
    from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

    files = FilesystemFuncTool(root_path=str(run.root))
    write = wrapped(run, files.write_file)
    (run.root / "knowledge").mkdir()
    outside = tmp_path.parent / (tmp_path.name + "-outside.md")
    (run.root / "knowledge/linked.md").symlink_to(outside)
    for path in ["README.md", "lineage/lineage.json", "knowledge/linked.md", str(outside)]:
        result = await write.on_invoke_tool(None, json.dumps({"path": path, "content": "Do not write"}))
        assert result["success"] == 0
    assert run.state["outputs"] == [] and not outside.exists()
