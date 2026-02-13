# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus.storage.ext_knowledge.init_utils module."""

import tempfile

import pytest

from datus.storage.embedding_models import get_db_embedding_model
from datus.storage.ext_knowledge.init_utils import exists_ext_knowledge, gen_ext_knowledge_id
from datus.storage.ext_knowledge.store import ExtKnowledgeStore


class TestGenExtKnowledgeId:
    """Tests for gen_ext_knowledge_id function."""

    def test_gen_ext_knowledge_id_basic(self):
        """Test basic ID generation with subject_path and search_text."""
        result = gen_ext_knowledge_id(["Finance", "Revenue"], "quarterly_sales")
        assert result == "Finance/Revenue/quarterly_sales"
        assert "/" in result

    def test_gen_ext_knowledge_id_single_path(self):
        """Test ID generation with a single path component."""
        result = gen_ext_knowledge_id(["Finance"], "annual_report")
        assert result == "Finance/annual_report"

    def test_gen_ext_knowledge_id_empty_path(self):
        """Test ID generation with empty subject_path."""
        result = gen_ext_knowledge_id([], "standalone_term")
        assert result == "standalone_term"

    def test_gen_ext_knowledge_id_special_characters(self):
        """Test that slashes in path parts and search_text are replaced with underscores."""
        result = gen_ext_knowledge_id(["Finance/Dept", "Revenue/Q1"], "sales/report")
        assert "Finance_Dept" in result
        assert "Revenue_Q1" in result
        assert result.endswith("sales_report")

    @pytest.mark.parametrize(
        "subject_path,search_text,expected",
        [
            (["A", "B", "C"], "term", "A/B/C/term"),
            (["X"], "y", "X/y"),
            ([], "only_text", "only_text"),
        ],
    )
    def test_gen_ext_knowledge_id_parametrized(self, subject_path, search_text, expected):
        """Test ID generation with various input combinations."""
        result = gen_ext_knowledge_id(subject_path, search_text)
        assert result == expected


class TestExistsExtKnowledge:
    """Tests for exists_ext_knowledge function."""

    def test_overwrite_mode_returns_empty_set(self):
        """Test that overwrite mode always returns an empty set regardless of stored data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = ExtKnowledgeStore(db_path=tmp_dir, embedding_model=get_db_embedding_model())
            storage.batch_store_knowledge(
                [
                    {
                        "subject_path": ["Finance", "Banking"],
                        "name": "APR",
                        "search_text": "APR",
                        "explanation": "Annual Percentage Rate",
                    }
                ]
            )
            result = exists_ext_knowledge(storage, build_mode="overwrite")
            assert result == set()
            assert isinstance(result, set)

    def test_incremental_mode_returns_existing_ids(self):
        """Test that incremental mode returns IDs of existing knowledge entries."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = ExtKnowledgeStore(db_path=tmp_dir, embedding_model=get_db_embedding_model())
            storage.batch_store_knowledge(
                [
                    {
                        "subject_path": ["Finance", "Banking"],
                        "name": "APR",
                        "search_text": "APR",
                        "explanation": "Annual Percentage Rate",
                    },
                    {
                        "subject_path": ["Finance", "Investment"],
                        "name": "ROI",
                        "search_text": "ROI",
                        "explanation": "Return on Investment",
                    },
                ]
            )
            result = exists_ext_knowledge(storage, build_mode="incremental")
            assert len(result) == 2
            assert isinstance(result, set)
            # Verify IDs match what gen_ext_knowledge_id would produce
            expected_id1 = gen_ext_knowledge_id(["Finance", "Banking"], "APR")
            expected_id2 = gen_ext_knowledge_id(["Finance", "Investment"], "ROI")
            assert expected_id1 in result
            assert expected_id2 in result

    def test_incremental_mode_empty_store(self):
        """Test incremental mode with no existing knowledge returns empty set."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = ExtKnowledgeStore(db_path=tmp_dir, embedding_model=get_db_embedding_model())
            result = exists_ext_knowledge(storage, build_mode="incremental")
            assert result == set()
