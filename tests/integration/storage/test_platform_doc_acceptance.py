# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Focused acceptance coverage for the local platform-doc ingest pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from datus.configuration.agent_config import DocumentConfig
from datus.storage.document.doc_init import init_platform_docs
from datus.storage.document.store import document_store
from datus.tools.search_tools.search_tool import SearchTool
from tests.conftest import load_acceptance_config

pytestmark = pytest.mark.acceptance


@pytest.fixture(autouse=True)
def clear_document_store_cache():
    document_store.cache_clear()
    yield
    document_store.cache_clear()


def _first_leaf_name(nodes: list[dict]) -> str:
    for node in nodes:
        if "tree" in node:
            name = _first_leaf_name(node["tree"])
            if name:
                return name
            continue
        children = node.get("children") or []
        if not children:
            return str(node.get("name") or "")
        name = _first_leaf_name(children)
        if name:
            return name
    return ""


def test_local_platform_doc_pipeline_ingests_and_serves_nav_search_and_readback(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "guide.md").write_text(
        """# Datus Local Platform Guide

## Installation

Install Datus and configure a local SQLite datasource.

## Query Workflow

Use the query workflow to inspect schema metadata and generate SQL.
""",
        encoding="utf-8",
    )
    (docs_dir / "api.html").write_text(
        """<!doctype html>
<html>
<body>
  <h1>Datus API Reference</h1>
  <h2>Authentication</h2>
  <p>Use bearer tokens for API requests.</p>
  <h2>Query Endpoint</h2>
  <p>The query endpoint streams action events for the web chatbot.</p>
</body>
</html>
""",
        encoding="utf-8",
    )

    platform = "acceptance_local_docs"
    agent_config = load_acceptance_config(home=str(tmp_path / "datus-home"))
    cfg = DocumentConfig(type="local", source=str(docs_dir), version="v1", chunk_size=256)

    try:
        result = init_platform_docs(platform=platform, cfg=cfg, build_mode="overwrite", pool_size=1)

        assert result.success is True, result.errors
        assert result.total_docs == 2
        assert result.total_chunks >= 2
        assert result.version == "v1"

        tool = SearchTool(agent_config=agent_config)
        nav = tool.list_document_nav(platform=platform, version="v1")
        assert nav.success is True, nav.error
        assert nav.total_docs == 2
        assert nav.nav_tree != []
        leaf_title = _first_leaf_name(nav.nav_tree)
        assert leaf_title in {"Datus API Reference", "Datus Local Platform Guide"}

        search = tool.search_document(platform=platform, keywords=["query workflow"], version="v1", top_n=3)
        assert search.success is True, search.error
        assert search.doc_count > 0
        assert any("query" in chunk["chunk_text"].lower() for chunks in search.docs.values() for chunk in chunks)

        document = tool.get_document(platform=platform, titles=[leaf_title], version="v1")
        assert document.success is True, document.error
        assert document.chunk_count > 0
        assert all(chunk["doc_path"] for chunk in document.chunks)
    finally:
        document_store(platform).delete_docs(version=None)
