# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus/tools/func_tool/_visual_artifact_helpers.py``.

Covers the three analysis-directory filesystem helpers plus the
``coerce_uses_arg`` normalizer that bridges between the LLM tool call
shape (``dict`` or ``None``) and the strict :class:`SubjectRefs`
schema. All filesystem mutations use ``tmp_path`` — these helpers are
pure I/O so we want deterministic, isolated state for each case.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from datus.schemas.analysis_artifacts import QueryBrief, SubjectAssetRef, SubjectRefs
from datus.schemas.artifact_manifest import ArtifactManifest
from datus.tools.func_tool._visual_artifact_helpers import (
    append_intent_section,
    coerce_uses_arg,
    staged_query_files,
    touch_manifest_after_save,
    utc_now_iso,
    write_query_brief,
)

# --------------------------------------------------------------------------- #
# append_intent_section                                                       #
# --------------------------------------------------------------------------- #


class TestAppendIntentSection:
    def test_creates_file_with_first_section(self, tmp_path: Path):
        analysis_dir = tmp_path / "analysis"
        err = append_intent_section(
            analysis_dir,
            user_message="Please summarise Q1 east-region sales.",
            mode="new",
            timestamp="2026-05-14T10:00:00Z",
        )
        assert err is None
        path = analysis_dir / "intent.md"
        assert path.is_file()
        text = path.read_text(encoding="utf-8")
        # Heading + standard 3-backtick fence wrapping the verbatim message.
        assert text == ("### [2026-05-14T10:00:00Z] mode: new\n```\nPlease summarise Q1 east-region sales.\n```\n")

    def test_appends_second_section_preserving_first(self, tmp_path: Path):
        analysis_dir = tmp_path / "analysis"
        append_intent_section(
            analysis_dir,
            user_message="initial prompt",
            mode="new",
            timestamp="2026-05-14T10:00:00Z",
        )
        append_intent_section(
            analysis_dir,
            user_message="follow-up edit",
            mode="edit",
            timestamp="2026-05-14T11:30:00Z",
        )
        text = (analysis_dir / "intent.md").read_text(encoding="utf-8")
        # Both sections must be present, original first.
        assert text.count("###") == 2
        first_idx = text.find("mode: new")
        second_idx = text.find("mode: edit")
        assert 0 <= first_idx < second_idx
        # Each prompt sits on its own line between fences.
        assert "\ninitial prompt\n" in text
        assert "\nfollow-up edit\n" in text
        # A blank line should separate the sections.
        assert "\n\n###" in text

    @pytest.mark.parametrize("user_message", ["", "   ", "\n\n\t"])
    def test_whitespace_only_message_skipped(self, tmp_path: Path, user_message: str):
        analysis_dir = tmp_path / "analysis"
        err = append_intent_section(
            analysis_dir,
            user_message=user_message,
            mode="new",
            timestamp="2026-05-14T10:00:00Z",
        )
        assert err is None
        # Nothing was written.
        assert not (analysis_dir / "intent.md").exists()

    @pytest.mark.parametrize(
        "noise_message",
        [
            # Renderer / compiler error reports forwarded into the loop.
            "Error: Failed to compile render/app.jsx: Unexpected token (152:158)",
            "ReferenceError: Cell is not defined\n    at eval (render/chart.jsx:89:49)",
            "TypeError: Cannot read properties of undefined (reading 'map')",
            "Uncaught Traceback (most recent call last):\n  File 'x.py', line 1",
        ],
    )
    def test_error_reports_dropped(self, tmp_path: Path, noise_message: str):
        """Renderer / compiler errors are system→LLM signals, not user
        intent. Recording them pollutes the file the follow-up ask agent
        reads as the canonical user voice."""
        analysis_dir = tmp_path / "analysis"
        err = append_intent_section(
            analysis_dir,
            user_message=noise_message,
            mode="edit",
            timestamp="2026-05-14T10:00:00Z",
        )
        assert err is None
        assert not (analysis_dir / "intent.md").exists()

    @pytest.mark.parametrize(
        "passthrough",
        [
            # Used to be hard-killed by the phrase set; now passes through
            # so the finalize-stage LLM curator can make a semantic call.
            "continue",
            "keep going",
            "please continue",
            "next",
            "ok",
            "sure",
            "proceed",
            "go ahead",
            # Real-but-short intent that exact-match heuristics historically
            # almost false-killed; it must pass through.
            "focus on risk",
        ],
    )
    def test_short_prompts_pass_append_filter(self, tmp_path: Path, passthrough: str):
        """Append-time filter only blocks mechanical noise (traceback /
        Error:). Semantic 'placeholder vs real direction' judgment runs
        later in the finalize LLM call, which has multi-prompt context
        to tell ``next`` (placeholder) apart from ``focus on risk`` (real
        direction shift) reliably."""
        analysis_dir = tmp_path / "analysis"
        err = append_intent_section(
            analysis_dir,
            user_message=passthrough,
            mode="edit",
            timestamp="2026-05-14T10:00:00Z",
        )
        assert err is None
        text = (analysis_dir / "intent.md").read_text(encoding="utf-8")
        assert passthrough in text

    def test_multiline_message_preserved_verbatim_inside_fence(self, tmp_path: Path):
        analysis_dir = tmp_path / "analysis"
        message = "first line\n\nthird line"
        append_intent_section(
            analysis_dir,
            user_message=message,
            mode="new",
            timestamp="2026-05-14T10:00:00Z",
        )
        text = (analysis_dir / "intent.md").read_text(encoding="utf-8")
        lines = text.splitlines()
        # Header + opening fence + 3 content lines (blank line preserved
        # verbatim, no prefix) + closing fence.
        assert lines[0] == "### [2026-05-14T10:00:00Z] mode: new"
        assert lines[1] == "```"
        assert lines[2] == "first line"
        assert lines[3] == ""
        assert lines[4] == "third line"
        assert lines[5] == "```"

    def test_fence_grows_when_message_contains_triple_backticks(self, tmp_path: Path):
        """If the user's prompt itself contains a ``` fence (or longer
        run of backticks), the wrapper fence must be longer so the
        section round-trips losslessly. CommonMark closes on the first
        equal-or-longer same-character fence line."""
        analysis_dir = tmp_path / "analysis"
        message = "describe this snippet:\n```sql\nSELECT 1\n```\nplease"
        append_intent_section(
            analysis_dir,
            user_message=message,
            mode="new",
            timestamp="2026-05-14T10:00:00Z",
        )
        text = (analysis_dir / "intent.md").read_text(encoding="utf-8")
        # Wrapper must be 4+ backticks because the body contains a run of 3.
        assert text == (
            "### [2026-05-14T10:00:00Z] mode: new\n````\ndescribe this snippet:\n```sql\nSELECT 1\n```\nplease\n````\n"
        )

    def test_fence_grows_for_longer_internal_backtick_runs(self, tmp_path: Path):
        """A six-backtick run inside the body forces a seven-backtick
        wrapper. Guards against assuming "3 → 4 is enough"."""
        analysis_dir = tmp_path / "analysis"
        message = "weird: ``````"
        append_intent_section(
            analysis_dir,
            user_message=message,
            mode="new",
            timestamp="2026-05-14T10:00:00Z",
        )
        text = (analysis_dir / "intent.md").read_text(encoding="utf-8")
        assert text == ("### [2026-05-14T10:00:00Z] mode: new\n```````\nweird: ``````\n```````\n")

    def test_returns_error_string_on_oserror(self, tmp_path: Path, monkeypatch):
        """Surface a failed file write as a warning string instead of raising."""
        analysis_dir = tmp_path / "analysis"

        # Force an OSError on the underlying atomic write — we want the
        # helper to surface it as a string rather than re-raise.
        from datus.tools.func_tool import _visual_artifact_helpers as helpers

        def _boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(helpers, "_atomic_write_text", _boom)
        err = append_intent_section(
            analysis_dir,
            user_message="anything",
            mode="new",
            timestamp="2026-05-14T10:00:00Z",
        )
        assert isinstance(err, str)
        assert "Failed to append intent section" in err
        assert "disk full" in err


# --------------------------------------------------------------------------- #
# touch_manifest_after_save                                                  #
# --------------------------------------------------------------------------- #


def _seed_manifest(tmp_path: Path, **overrides) -> Path:
    payload = {
        "slug": "demo",
        "name": "demo report",
        "description": "Demo for the manifest upsert helper unit tests.",
        "kind": "report",
        "created_at": "2026-05-14T10:00:00Z",
    }
    payload.update(overrides)
    # Make sure the payload validates so we're testing the helper, not a
    # broken fixture.
    ArtifactManifest.model_validate(payload)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestTouchManifestAfterSave:
    def test_updates_timestamp_preserving_derived_sources(self, tmp_path: Path):
        """Refresh recency while preserving the existing derived datasource index."""
        manifest_path = _seed_manifest(tmp_path, datasources=["primary_pg"])
        err = touch_manifest_after_save(manifest_path, timestamp="2026-05-14T11:00:00Z")
        assert err is None
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert data["datasources"] == ["primary_pg"]
        assert data["updated_at"] == "2026-05-14T11:00:00Z"

    def test_repeated_saves_preserve_derived_sources(self, tmp_path: Path):
        """Keep repeated query saves from incrementally changing derived source metadata."""
        manifest_path = _seed_manifest(tmp_path, datasources=["pg"])
        touch_manifest_after_save(manifest_path, timestamp="2026-05-14T11:00:00Z")
        touch_manifest_after_save(manifest_path, timestamp="2026-05-14T12:00:00Z")
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Timestamp refreshes leave the derived source index intact.
        assert data["datasources"] == ["pg"]
        # updated_at always bumps to the latest timestamp.
        assert data["updated_at"] == "2026-05-14T12:00:00Z"

    def test_always_bumps_updated_at(self, tmp_path: Path):
        """Refresh the manifest timestamp on every successful query save."""
        manifest_path = _seed_manifest(tmp_path)
        touch_manifest_after_save(manifest_path, timestamp="2026-05-14T11:00:00Z")
        # Another query mutation, distinct timestamp.
        touch_manifest_after_save(manifest_path, timestamp="2026-05-14T13:00:00Z")
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert data["updated_at"] == "2026-05-14T13:00:00Z"

    def test_missing_manifest_returns_error_string(self, tmp_path: Path):
        """Return a warning string when the manifest does not exist."""
        missing = tmp_path / "missing_manifest.json"
        err = touch_manifest_after_save(missing, timestamp="t")
        assert isinstance(err, str)
        assert "manifest missing" in err

    def test_corrupt_manifest_returns_error_string(self, tmp_path: Path):
        """Return a warning string when the manifest cannot be parsed."""
        corrupt = tmp_path / "manifest.json"
        corrupt.write_text("{not-json", encoding="utf-8")
        err = touch_manifest_after_save(corrupt, timestamp="t")
        assert isinstance(err, str)
        assert "corrupt" in err

    def test_schema_invalid_manifest_returns_error_string(self, tmp_path: Path):
        """Report invalid manifest schema as a warning without failing the saved query."""
        invalid = tmp_path / "manifest.json"
        invalid.write_text('{"slug":"demo"}', encoding="utf-8")  # missing required fields
        err = touch_manifest_after_save(invalid, timestamp="t")
        assert isinstance(err, str)
        assert "schema validation failed" in err

    def test_initial_empty_source_index_is_preserved(self, tmp_path: Path):
        """Leave an empty manifest source index for finalize to populate."""
        manifest_path = _seed_manifest(tmp_path)
        err = touch_manifest_after_save(manifest_path, timestamp="2026-05-14T14:00:00Z")
        assert err is None
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Saving a query updates recency while finalize owns the source index.
        assert data["datasource_ids"] == []
        assert "datasources" not in data
        assert data["updated_at"] == "2026-05-14T14:00:00Z"


# --------------------------------------------------------------------------- #
# write_query_brief                                                           #
# --------------------------------------------------------------------------- #


class TestWriteQueryBrief:
    def test_writes_file_that_round_trips(self, tmp_path: Path):
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        err = write_query_brief(
            queries_dir,
            name="sales_by_store",
            hypothesis="high-risk signups cluster around promotional campaigns",
            uses=SubjectRefs(metrics=[SubjectAssetRef(path=["Signups", "Risk"], name="high_risk_signups")]),
            caveats="Excludes test accounts.",
        )
        assert err is None
        path = queries_dir / "sales_by_store.brief.json"
        assert path.is_file()
        # Re-validate the file through the schema to prove the round-trip.
        data = json.loads(path.read_text(encoding="utf-8"))
        brief = QueryBrief.model_validate(data)
        assert brief.name == "sales_by_store"
        assert brief.uses.metrics[0].path == ["Signups", "Risk"]
        assert brief.uses.metrics[0].name == "high_risk_signups"
        assert brief.caveats == "Excludes test accounts."

    def test_invalid_slug_returns_error(self, tmp_path: Path):
        """Report an invalid artifact slug before writing query metadata."""
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        err = write_query_brief(
            queries_dir,
            name="Has-Hyphen",  # invalid per ANALYSIS_SLUG_PATTERN
            hypothesis="hypothesis",
            uses=SubjectRefs(),
            caveats="",
        )
        assert isinstance(err, str)
        assert "schema validation failed" in err
        # No file gets written when validation fails.
        assert not (queries_dir / "Has-Hyphen.brief.json").exists()

    def test_empty_hypothesis_returns_error(self, tmp_path: Path):
        """Reject missing query rationale before saving its brief."""
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        err = write_query_brief(
            queries_dir,
            name="empty_hypothesis",
            hypothesis="",
            uses=SubjectRefs(),
            caveats="",
        )
        assert isinstance(err, str)
        assert "schema validation failed" in err


# --------------------------------------------------------------------------- #
# coerce_uses_arg                                                             #
# --------------------------------------------------------------------------- #


class TestCoerceUsesArg:
    def test_none_returns_empty_buckets(self):
        refs = coerce_uses_arg(None)
        assert isinstance(refs, SubjectRefs)
        assert refs.metrics == []
        assert refs.reference_sql == []

    def test_full_dict_round_trips(self):
        payload = {
            "metrics": [
                {"path": ["Commerce", "Orders"], "name": "aov"},
                {"path": ["Commerce", "Orders"], "name": "order_count"},
            ],
            "reference_sql": [{"path": ["Templates"], "name": "top_q"}],
        }
        refs = coerce_uses_arg(payload)
        assert [(m.path, m.name) for m in refs.metrics] == [
            (["Commerce", "Orders"], "aov"),
            (["Commerce", "Orders"], "order_count"),
        ]
        assert refs.reference_sql[0].name == "top_q"

    def test_passthrough_for_subject_refs(self):
        original = SubjectRefs(metrics=[SubjectAssetRef(path=["A"], name="x")])
        refs = coerce_uses_arg(original)
        assert refs is original

    def test_non_dict_raises_value_error(self):
        with pytest.raises(ValueError) as exc:
            coerce_uses_arg([{"path": ["A"], "name": "x"}])  # type: ignore[arg-type]
        assert "uses must be a JSON object" in str(exc.value)

    def test_legacy_string_id_form_rejected(self):
        """The old ``["metric:Sales/Revenue.gross"]`` LLM-drift shape must
        now hard-fail at the write boundary so a malformed brief never
        lands on disk to poison ``subject_refs.json``."""
        with pytest.raises(ValueError) as exc:
            coerce_uses_arg({"metrics": ["metric:Sales/Revenue.gross"]})
        assert "schema validation failed" in str(exc.value)

    def test_unknown_bucket_rejected(self):
        """``SubjectRefs`` uses ``extra=forbid``; an LLM hallucination
        of a brand-new bucket name surfaces immediately."""
        with pytest.raises(ValueError) as exc:
            coerce_uses_arg({"metrics": [], "future_kind": []})
        assert "schema validation failed" in str(exc.value)

    @pytest.mark.parametrize(
        "bad_entry",
        [
            {"name": "x"},  # missing path
            {"path": ["A"]},  # missing name
            {"path": [], "name": "x"},  # empty path
            {"path": ["A"], "name": ""},  # empty name
            {"path": "A.B", "name": "x"},  # path is a string, not a list
        ],
    )
    def test_malformed_entries_rejected(self, bad_entry: dict):
        with pytest.raises(ValueError) as exc:
            coerce_uses_arg({"metrics": [bad_entry]})
        assert "schema validation failed" in str(exc.value)


# --------------------------------------------------------------------------- #
# utc_now_iso                                                                 #
# --------------------------------------------------------------------------- #


class TestUtcNowIso:
    """Sanity check the ISO format used as the timestamp for every file."""

    def test_format_is_seconds_precision_zulu(self):
        ts = utc_now_iso()
        # YYYY-MM-DDTHH:MM:SSZ — exactly 20 chars.
        assert len(ts) == 20
        assert ts[4] == "-" and ts[7] == "-"
        assert ts[10] == "T"
        assert ts[13] == ":" and ts[16] == ":"
        assert ts.endswith("Z")


@pytest.mark.parametrize("existing", [False, True])
def test_failed_query_publish_restores_previous_files(tmp_path, monkeypatch, existing):
    queries = tmp_path / "queries"
    queries.mkdir()
    old = {"q.metric.json": b"old recipe"}
    if existing:
        old.update({"q.sql": b"old SQL", "q.json": b"old result"})
    for name, content in old.items():
        (queries / name).write_bytes(content)
    replace = os.replace
    published = []

    def fail_second_replace(src, dst):
        if dst.parent == queries and src.parent.name != "backup":
            published.append(dst)
            if len(published) == 2:
                raise OSError("injected publish failure")
        replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="injected publish failure"):
        with staged_query_files(queries, "q", remove_suffixes=(".metric.json",)) as staging:
            (staging / "q.sql").write_bytes(b"new SQL")
            (staging / "q.json").write_bytes(b"new result")
    assert {p.name: p.read_bytes() for p in queries.iterdir()} == old


def test_retry_keeps_first_mode_and_separates_new_turns(tmp_path):
    args = dict(
        user_message="Check net sales\r\nby category", timestamp="2026-06-01T00:00:00.123456Z", deduplicate=True
    )
    assert append_intent_section(tmp_path, mode="new", **args) is None
    original = (tmp_path / "intent.md").read_bytes()
    assert append_intent_section(tmp_path, mode="edit", **args) is None
    assert (tmp_path / "intent.md").read_bytes() == original
    assert append_intent_section(tmp_path, mode="edit", **{**args, "timestamp": "2026-06-02T00:00:00Z"}) is None
    assert append_intent_section(tmp_path, mode="edit", **{**args, "user_message": "Show gross sales"}) is None
    text = (tmp_path / "intent.md").read_text()
    assert text.count("Check net sales") == 2
    assert text.count("Show gross sales") == 1


def test_rollback_retains_unrestored_backup_and_restores_remaining_files(tmp_path, monkeypatch):
    queries = tmp_path / "queries"
    queries.mkdir()
    for name in ("a", "b", "c"):
        (queries / name).write_text("old")
    replace = os.replace
    published = []
    unrestored = []

    def fail_publish_and_rollback(src, dst):
        if dst.parent == queries:
            if src.parent.name == "backup":
                if not unrestored:
                    unrestored.append(src)
                    raise OSError("injected restore failure")
            else:
                if len(published) == 2:
                    raise OSError("injected publish failure")
                published.append(dst)
        replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_publish_and_rollback)
    with pytest.raises(OSError) as exc:
        with staged_query_files(queries, "q", remove_suffixes=()) as staging:
            for name in ("a", "b", "c"):
                (staging / name).write_text("new")
    assert str(staging) in str(exc.value)
    assert unrestored[0].read_text() == "old"
    assert published[0].read_text() == "old"


@pytest.mark.parametrize("operation", ["edit_file", "write_file", "delete_file"])
def test_direct_intent_mutation_excludes_append(tmp_path, monkeypatch, operation):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

    fs = FilesystemFuncTool(root_path=str(tmp_path))
    relative = "dashboards/demo/analysis/intent.md"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text("Initial content")
    entered, release, appending = Event(), Event(), Event()
    method = {"edit_file": "read_text", "write_file": "write_text", "delete_file": "unlink"}[operation]
    original = getattr(Path, method)

    def pause_once(target, *args, **kwargs):
        result = original(target, *args, **kwargs)
        if target == path and not entered.is_set():
            entered.set()
            release.wait(timeout=5)
        return result

    def append():
        appending.set()
        return append_intent_section(path.parent, user_message="Review revenue", timestamp="t", mode="edit")

    monkeypatch.setattr(Path, method, pause_once)
    args = {"edit_file": ("Initial", "Revised"), "write_file": ("Revised content",), "delete_file": ()}[operation]
    with ThreadPoolExecutor(max_workers=2) as pool:
        write = pool.submit(getattr(fs, operation), relative, *args)
        try:
            assert entered.wait(5)
            add = pool.submit(append)
            assert appending.wait(5)
            with pytest.raises(TimeoutError):
                add.result(timeout=0.25)
        finally:
            release.set()
        assert write.result().success == 1 and add.result() is None
    text = path.read_text()
    assert "Review revenue" in text
    assert ("Revised content" in text) == (operation != "delete_file")


def test_windows_intent_writers_use_one_persistent_lock_file(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock

    from datus.tools.func_tool import _visual_artifact_helpers as helpers
    from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

    fs = FilesystemFuncTool(root_path=str(tmp_path))
    relative = "dashboards/demo/analysis/intent.md"
    path = tmp_path / relative
    locks = []
    windows_os = Mock(wraps=os)
    windows_os.name = "nt"
    monkeypatch.setattr(helpers, "os", windows_os)
    monkeypatch.setitem(
        sys.modules,
        "msvcrt",
        SimpleNamespace(
            LK_LOCK=1, LK_UNLCK=0, locking=lambda fd, mode, n: locks.append((os.fstat(fd).st_ino, mode, n))
        ),
    )
    assert fs.write_file(relative, "original").success == 1
    assert fs.edit_file(relative, "original", "updated").success == 1
    assert append_intent_section(path.parent, user_message="Review sales", timestamp="t", mode="edit") is None
    assert fs.delete_file(relative).success == 1
    inode = path.with_name(".intent.md.lock").stat().st_ino
    assert locks == [(inode, 1, 1), (inode, 0, 1)] * 4
