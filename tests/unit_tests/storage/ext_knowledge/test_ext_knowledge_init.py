# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus.storage.ext_knowledge.ext_knowledge_init module."""

import argparse

import pytest

from datus.storage.embedding_models import get_db_embedding_model
from datus.storage.ext_knowledge.ext_knowledge_init import (
    init_ext_knowledge,
    init_success_story_knowledge,
    process_knowledge_line,
    process_row,
)
from datus.storage.ext_knowledge.store import ExtKnowledgeStore
from tests.unit_tests.mock_llm_model import build_simple_response


@pytest.fixture
def ext_store(tmp_path):
    """Create an ExtKnowledgeStore instance with real LanceDB."""
    return ExtKnowledgeStore(db_path=str(tmp_path / "db"), embedding_model=get_db_embedding_model())


@pytest.fixture
def csv_file(tmp_path):
    """Create a sample CSV file for testing."""
    csv_path = str(tmp_path / "knowledge.csv")
    with open(csv_path, "w") as f:
        f.write("subject_path,name,search_text,explanation\n")
        f.write("Finance/Banking,APR,Annual Percentage Rate,The yearly cost of borrowing\n")
        f.write("Finance/Investment,ROI,Return on Investment,A measure of profitability\n")
        f.write("Technology/AI,NLP,Natural Language Processing,AI subfield for human language\n")
    return csv_path


@pytest.fixture
def csv_file_with_missing_fields(tmp_path):
    """Create a CSV file with some rows having empty subject_path (invalid path components)."""
    csv_path = str(tmp_path / "knowledge_partial.csv")
    with open(csv_path, "w") as f:
        f.write("subject_path,name,search_text,explanation\n")
        f.write("Finance/Banking,APR,Annual Percentage Rate,The yearly cost of borrowing\n")
        f.write("///,badname,badtext,badexpl\n")
        f.write("Technology/AI,NLP,Natural Language Processing,AI subfield for human language\n")
    return csv_path


@pytest.fixture
def csv_file_missing_columns(tmp_path):
    """Create a CSV file missing required columns."""
    csv_path = str(tmp_path / "knowledge_bad.csv")
    with open(csv_path, "w") as f:
        f.write("subject_path,name\n")
        f.write("Finance/Banking,APR\n")
    return csv_path


# ============================================================
# process_row
# ============================================================


class TestProcessRow:
    """Tests for process_row function."""

    def test_process_row_success(self, ext_store):
        """Test processing a valid row stores knowledge and returns 'processed'."""
        from threading import Lock

        row = {
            "subject_path": "Finance/Banking",
            "name": "APR",
            "search_text": "Annual Percentage Rate",
            "explanation": "The yearly cost of borrowing money",
        }
        existing = set()
        lock = Lock()
        result = process_row(ext_store, row, 0, existing, lock)
        assert result == "processed"
        # Verify stored
        knowledge = ext_store.search_all_knowledge(subject_path=["Finance", "Banking"])
        assert len(knowledge) == 1
        assert knowledge[0]["name"] == "APR"

    def test_process_row_skips_existing(self, ext_store):
        """Test that process_row skips entries already in the existing set."""
        from threading import Lock

        row = {
            "subject_path": "Finance/Banking",
            "name": "APR",
            "search_text": "Annual Percentage Rate",
            "explanation": "The yearly cost of borrowing money",
        }
        # Pre-populate existing set with the generated ID
        existing = {"Finance/Banking/Annual Percentage Rate"}
        lock = Lock()
        result = process_row(ext_store, row, 0, existing, lock)
        assert result == "skipped"

    def test_process_row_skips_empty_fields(self, ext_store):
        """Test that process_row skips rows with empty required fields."""
        from threading import Lock

        row = {
            "subject_path": "",
            "name": "",
            "search_text": "",
            "explanation": "",
        }
        existing = set()
        lock = Lock()
        result = process_row(ext_store, row, 0, existing, lock)
        assert result == "skipped"

    def test_process_row_skips_invalid_path(self, ext_store):
        """Test that process_row skips rows with invalid subject_path (no valid components)."""
        from threading import Lock

        row = {
            "subject_path": "///",
            "name": "test",
            "search_text": "test",
            "explanation": "test",
        }
        existing = set()
        lock = Lock()
        result = process_row(ext_store, row, 0, existing, lock)
        assert result == "skipped"

    def test_process_row_adds_to_existing_set(self, ext_store):
        """Test that process_row adds the new knowledge ID to the existing set."""
        from threading import Lock

        row = {
            "subject_path": "Science/Physics",
            "name": "Gravity",
            "search_text": "Gravitational Force",
            "explanation": "Fundamental force of nature",
        }
        existing = set()
        lock = Lock()
        process_row(ext_store, row, 0, existing, lock)
        assert len(existing) == 1


# ============================================================
# init_ext_knowledge
# ============================================================


class TestInitExtKnowledge:
    """Tests for init_ext_knowledge function."""

    def test_init_ext_knowledge_overwrite_mode(self, ext_store, csv_file):
        """Test init_ext_knowledge loads all entries in overwrite mode."""
        args = argparse.Namespace(ext_knowledge=csv_file)
        init_ext_knowledge(ext_store, args, build_mode="overwrite", pool_size=1)
        results = ext_store.search_all_knowledge()
        assert len(results) == 3

    def test_init_ext_knowledge_incremental_mode(self, ext_store, csv_file):
        """Test init_ext_knowledge in incremental mode skips existing entries."""
        args = argparse.Namespace(ext_knowledge=csv_file)
        # First load
        init_ext_knowledge(ext_store, args, build_mode="overwrite", pool_size=1)
        first_count = len(ext_store.search_all_knowledge())
        assert first_count == 3
        # Second load in incremental mode should skip duplicates
        init_ext_knowledge(ext_store, args, build_mode="incremental", pool_size=1)
        second_count = len(ext_store.search_all_knowledge())
        assert second_count == first_count

    def test_init_ext_knowledge_no_ext_knowledge_attr(self, ext_store):
        """Test init_ext_knowledge returns early when args has no ext_knowledge attribute."""
        args = argparse.Namespace()
        init_ext_knowledge(ext_store, args)
        results = ext_store.search_all_knowledge()
        assert results == []

    def test_init_ext_knowledge_empty_ext_knowledge(self, ext_store):
        """Test init_ext_knowledge returns early when ext_knowledge is empty string."""
        args = argparse.Namespace(ext_knowledge="")
        init_ext_knowledge(ext_store, args)
        results = ext_store.search_all_knowledge()
        assert results == []

    def test_init_ext_knowledge_file_not_found(self, ext_store):
        """Test init_ext_knowledge returns early when CSV file does not exist."""
        args = argparse.Namespace(ext_knowledge="/nonexistent/path/knowledge.csv")
        init_ext_knowledge(ext_store, args)
        results = ext_store.search_all_knowledge()
        assert results == []

    def test_init_ext_knowledge_missing_columns(self, ext_store, csv_file_missing_columns):
        """Test init_ext_knowledge raises ValueError when CSV is missing required columns."""
        args = argparse.Namespace(ext_knowledge=csv_file_missing_columns)
        with pytest.raises(ValueError, match="Missing required columns"):
            init_ext_knowledge(ext_store, args, build_mode="overwrite", pool_size=1)

    def test_init_ext_knowledge_skips_invalid_rows(self, ext_store, csv_file_with_missing_fields):
        """Test init_ext_knowledge skips rows with missing fields but processes valid ones."""
        args = argparse.Namespace(ext_knowledge=csv_file_with_missing_fields)
        init_ext_knowledge(ext_store, args, build_mode="overwrite", pool_size=1)
        results = ext_store.search_all_knowledge()
        assert len(results) == 2

    def test_init_ext_knowledge_parallel(self, ext_store, csv_file):
        """Test init_ext_knowledge works with parallel processing (pool_size > 1)."""
        args = argparse.Namespace(ext_knowledge=csv_file)
        init_ext_knowledge(ext_store, args, build_mode="overwrite", pool_size=2)
        results = ext_store.search_all_knowledge()
        assert len(results) == 3


# ============================================================
# process_knowledge_line
# ============================================================


class TestProcessKnowledgeLine:
    """Tests for async process_knowledge_line function."""

    @pytest.mark.asyncio
    async def test_process_knowledge_line_success(self, real_agent_config, mock_llm_create):
        """Test successful processing of a knowledge line with mock LLM."""
        mock_llm_create.reset(
            responses=[
                build_simple_response("Generated knowledge for SAT scores"),
            ]
        )

        row = {
            "question": "What are the average SAT scores?",
            "sql": "SELECT AVG(AvgScrMath) FROM satscores",
            "subject_path": "Education/SAT",
        }
        result = await process_knowledge_line(row, real_agent_config, subject_tree=["Education"])
        assert result["successful"] is True
        assert result["error"] == ""

    @pytest.mark.asyncio
    async def test_process_knowledge_line_missing_question(self, real_agent_config, mock_llm_create):
        """Test that missing question field returns failure."""
        row = {
            "question": "",
            "sql": "SELECT 1",
            "subject_path": "Test",
        }
        result = await process_knowledge_line(row, real_agent_config)
        assert result["successful"] is False
        assert "Missing question field" in result["error"]

    @pytest.mark.asyncio
    async def test_process_knowledge_line_no_question_key(self, real_agent_config, mock_llm_create):
        """Test that row without question key returns failure."""
        row = {
            "sql": "SELECT 1",
            "subject_path": "Test",
        }
        result = await process_knowledge_line(row, real_agent_config)
        assert result["successful"] is False
        assert "Missing question field" in result["error"]

    @pytest.mark.asyncio
    async def test_process_knowledge_line_no_sql(self, real_agent_config, mock_llm_create):
        """Test processing a line without SQL still succeeds."""
        mock_llm_create.reset(
            responses=[
                build_simple_response("Generated knowledge without SQL reference"),
            ]
        )

        row = {
            "question": "What is revenue?",
            "sql": "",
            "subject_path": "",
        }
        result = await process_knowledge_line(row, real_agent_config)
        assert result["successful"] is True
        assert result["error"] == ""

    @pytest.mark.asyncio
    async def test_process_knowledge_line_with_subject_tree(self, real_agent_config, mock_llm_create):
        """Test processing with a predefined subject_tree."""
        mock_llm_create.reset(
            responses=[
                build_simple_response("Knowledge generated with subject tree"),
            ]
        )

        row = {
            "question": "How many schools are there?",
            "sql": "SELECT COUNT(*) FROM schools",
            "subject_path": "Education/Schools",
        }
        result = await process_knowledge_line(row, real_agent_config, subject_tree=["Education", "Finance"])
        assert result["successful"] is True
        assert result["error"] == ""

    @pytest.mark.asyncio
    async def test_process_knowledge_line_exception_handling(self, real_agent_config, mock_llm_create):
        """Test that exceptions during execution are caught and returned as error."""
        # Reset with no responses so MockLLMModel returns empty
        # The node should still complete without raising
        mock_llm_create.reset(responses=[])

        row = {
            "question": "What is something?",
            "sql": "",
            "subject_path": "",
        }
        # Even with empty responses, the node should handle gracefully
        result = await process_knowledge_line(row, real_agent_config)
        # Should either succeed (empty response handled) or fail gracefully
        assert "successful" in result
        assert "error" in result


# ============================================================
# init_success_story_knowledge
# ============================================================


class TestInitSuccessStoryKnowledge:
    """Tests for init_success_story_knowledge function."""

    def test_init_success_story_file_not_found(self, real_agent_config, mock_llm_create):
        """Test returns failure when CSV file does not exist."""
        args = argparse.Namespace(success_story="/nonexistent/path/stories.csv")
        success, error_msg = init_success_story_knowledge(args, real_agent_config)
        assert success is False
        assert "not found" in error_msg

    def test_init_success_story_single_row(self, tmp_path, real_agent_config, mock_llm_create):
        """Test processing a single-row success story CSV."""
        csv_path = str(tmp_path / "stories.csv")
        with open(csv_path, "w") as f:
            f.write("question,sql,subject_path\n")
            f.write("What is the total enrollment?,SELECT SUM(enroll12) FROM satscores,Education/Enrollment\n")

        mock_llm_create.reset(
            responses=[
                build_simple_response("Knowledge generated for enrollment"),
            ]
        )

        args = argparse.Namespace(success_story=csv_path)
        success, error_msg = init_success_story_knowledge(args, real_agent_config, subject_tree=["Education"])
        assert success is True
        assert error_msg == ""

    def test_init_success_story_multiple_rows(self, tmp_path, real_agent_config, mock_llm_create):
        """Test processing multiple rows in success story CSV."""
        csv_path = str(tmp_path / "stories.csv")
        with open(csv_path, "w") as f:
            f.write("question,sql,subject_path\n")
            f.write("What is enrollment?,SELECT SUM(enroll12) FROM satscores,Education\n")
            f.write("How many schools?,SELECT COUNT(*) FROM schools,Education\n")

        mock_llm_create.reset(
            responses=[
                build_simple_response("Knowledge for enrollment"),
                build_simple_response("Knowledge for school count"),
            ]
        )

        args = argparse.Namespace(success_story=csv_path)
        success, error_msg = init_success_story_knowledge(args, real_agent_config)
        assert success is True
        assert error_msg == ""

    def test_init_success_story_with_missing_question(self, tmp_path, real_agent_config, mock_llm_create):
        """Test that rows with missing question are reported as errors."""
        csv_path = str(tmp_path / "stories.csv")
        with open(csv_path, "w") as f:
            f.write("question,sql,subject_path\n")
            f.write(",SELECT 1,Test\n")

        args = argparse.Namespace(success_story=csv_path)
        success, error_msg = init_success_story_knowledge(args, real_agent_config)
        # With one failing row and no successful rows, success should be False
        assert success is False
        # pandas reads empty CSV value as NaN, which causes Pydantic validation error
        assert "Error processing row 1" in error_msg

    def test_init_success_story_partial_failure(self, tmp_path, real_agent_config, mock_llm_create):
        """Test that partial failures still return success if some rows succeed."""
        csv_path = str(tmp_path / "stories.csv")
        with open(csv_path, "w") as f:
            f.write("question,sql,subject_path\n")
            f.write("What is enrollment?,SELECT SUM(enroll12) FROM satscores,Education\n")
            f.write(",SELECT 1,Test\n")

        mock_llm_create.reset(
            responses=[
                build_simple_response("Knowledge for enrollment"),
            ]
        )

        args = argparse.Namespace(success_story=csv_path)
        success, error_msg = init_success_story_knowledge(args, real_agent_config)
        # One row succeeds, one fails - overall should be True (at least one success)
        assert success is True
        assert "Error processing row 2" in error_msg
