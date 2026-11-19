"""Native pagination preserves rows and binds snapshots to the active scope."""

from types import SimpleNamespace

import pytest

from datus.utils.result_pagination import SnapshotStore, list_page, validate_page
from datus.utils.tool_result_store import ACTIVE_RESULT_STORE, ToolResultStore


def test_page_size_has_no_hard_cap():
    rows = [{"name": str(i)} for i in range(2400)]
    first = list_page(rows, "id", 1800, 0)
    second = list_page(rows, "id", 1800, first["extra"]["next_offset"])
    assert len(first["items"]) == 1800
    assert first["items"] + second["items"] == rows
    assert second["has_more"] is False and second["extra"]["next_offset"] is None


@pytest.mark.parametrize("size,offset", [(0, 0), (-1, 0), (True, 0), (1, -1), (1, True), (1.5, 0)])
def test_invalid_page_arguments(size, offset):
    with pytest.raises(ValueError):
        validate_page(size, offset)


def test_snapshot_is_immutable_and_rejects_changed_scope_or_filters():
    store = SnapshotStore()
    rows = [{"name": "orders"}]
    rid = store.save("list_tables", rows, {"tables": ["orders"]}, {"include_views": True})
    rows[0]["name"] = "changed"
    assert store.read(rid, "list_tables", {"tables": ["orders"]}, {}) == [{"name": "orders"}]
    for tool, scope, query in [
        ("list_schemas", {"tables": ["orders"]}, {}),
        ("list_tables", {"tables": []}, {}),
        ("list_tables", {"tables": ["orders"]}, {"include_views": False}),
    ]:
        with pytest.raises(ValueError):
            store.read(rid, tool, scope, query)


def test_snapshot_survives_store_recreation_but_not_session_change(tmp_path):
    active = ToolResultStore(SimpleNamespace(dir=tmp_path / "session-a"))
    token = ACTIVE_RESULT_STORE.set(active)
    try:
        rid = SnapshotStore().save("list_tables", [1, 2], {}, {})
        assert SnapshotStore().read(rid, "list_tables", {}, {}) == [1, 2]
        path = SnapshotStore._path(rid)
        assert path.stat().st_mode & 0o777 == 0o600
        other = ToolResultStore(SimpleNamespace(dir=tmp_path / "session-b"))
        ACTIVE_RESULT_STORE.set(other)
        with pytest.raises(ValueError, match="unavailable"):
            SnapshotStore().read(rid, "list_tables", {}, {})
        ACTIVE_RESULT_STORE.set(active)
        path.unlink()
        target = tmp_path / "outside"
        target.write_text("{}")
        path.symlink_to(target)
        with pytest.raises(ValueError, match="unavailable"):
            SnapshotStore().read(rid, "list_tables", {}, {})
    finally:
        ACTIVE_RESULT_STORE.reset(token)
