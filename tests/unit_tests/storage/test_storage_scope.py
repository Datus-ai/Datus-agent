# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for centralized storage namespace derivation."""

from types import SimpleNamespace

import pytest

from datus.storage.scope import (
    DATASOURCE_SCOPED_KB_STORES,
    PROJECT_SCOPED_STORES,
    datasource_storage_namespace,
    project_storage_namespace,
    safe_storage_namespace_token,
)
from datus.utils.exceptions import DatusException, ErrorCode


def test_safe_storage_namespace_token_normalizes_and_hashes_unsafe_values():
    token = safe_storage_namespace_token("Sales DB/2026")

    assert token.startswith("Sales_DB_2026_")
    assert "/" not in token
    assert " " not in token
    assert token != "Sales_DB_2026"


def test_safe_storage_namespace_token_prefixes_digit_start():
    assert safe_storage_namespace_token("2026-prod").startswith("n_2026_prod_")


def test_datasource_namespace_combines_project_and_datasource():
    cfg = SimpleNamespace(project_name="workspace", current_datasource="sales db")

    namespace = datasource_storage_namespace(cfg)

    assert namespace.startswith("workspace__ds__sales_db_")
    assert namespace != project_storage_namespace(cfg)


def test_datasource_namespace_without_datasource_raises():
    cfg = SimpleNamespace(project_name="workspace", current_datasource="")

    with pytest.raises(DatusException) as exc:
        datasource_storage_namespace(cfg)
    assert exc.value.code == ErrorCode.STORAGE_INVALID_ARGUMENT
    assert "datasource is required" in str(exc.value)


def test_storage_scope_classification_documents_kb_and_project_stores():
    assert {"metric", "semantic_model", "reference_sql", "subject_tree"} <= DATASOURCE_SCOPED_KB_STORES
    assert {"task", "feedback", "document"} <= PROJECT_SCOPED_STORES
    assert DATASOURCE_SCOPED_KB_STORES.isdisjoint(PROJECT_SCOPED_STORES)
