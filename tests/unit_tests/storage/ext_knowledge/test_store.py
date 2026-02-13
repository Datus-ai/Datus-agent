# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus.storage.ext_knowledge.store module."""

import pytest

from datus.storage.embedding_models import get_db_embedding_model
from datus.storage.ext_knowledge.store import ExtKnowledgeRAG, ExtKnowledgeStore, gen_subject_item_id


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary directory for LanceDB storage."""
    return str(tmp_path)


@pytest.fixture
def ext_store(tmp_db):
    """Create an ExtKnowledgeStore instance with real LanceDB."""
    return ExtKnowledgeStore(db_path=tmp_db, embedding_model=get_db_embedding_model())


@pytest.fixture
def sample_knowledge():
    """Sample knowledge entries for testing."""
    return [
        {
            "subject_path": ["Finance", "Banking", "Retail"],
            "name": "APR",
            "search_text": "APR",
            "explanation": "Annual Percentage Rate - the yearly cost of a loan",
        },
        {
            "subject_path": ["Finance", "Investment", "Stocks"],
            "name": "PE_Ratio",
            "search_text": "P/E Ratio",
            "explanation": "Price-to-earnings ratio - a valuation metric for stocks",
        },
        {
            "subject_path": ["Finance", "Banking", "Corporate"],
            "name": "LTV",
            "search_text": "Loan-to-Value",
            "explanation": "Loan-to-Value ratio used in mortgage lending",
        },
    ]


# ============================================================
# gen_subject_item_id
# ============================================================


class TestGenSubjectItemId:
    """Tests for the gen_subject_item_id helper function."""

    def test_basic_id_generation(self):
        """Test basic ID generation from subject_path and name."""
        result = gen_subject_item_id(["Finance", "Revenue"], "total_revenue")
        assert result == "Finance/Revenue/total_revenue"

    def test_single_path_component(self):
        """Test ID generation with a single path component."""
        result = gen_subject_item_id(["Root"], "item")
        assert result == "Root/item"

    def test_empty_path(self):
        """Test ID generation with empty subject_path."""
        result = gen_subject_item_id([], "name_only")
        assert result == "name_only"

    @pytest.mark.parametrize(
        "path,name,expected",
        [
            (["A"], "b", "A/b"),
            (["A", "B"], "c", "A/B/c"),
            (["A", "B", "C"], "d", "A/B/C/d"),
        ],
    )
    def test_parametrized(self, path, name, expected):
        """Test various path lengths produce correct IDs."""
        assert gen_subject_item_id(path, name) == expected


# ============================================================
# ExtKnowledgeStore
# ============================================================


class TestExtKnowledgeStoreInit:
    """Tests for ExtKnowledgeStore initialization."""

    def test_store_creation(self, ext_store):
        """Test that store initializes correctly with expected attributes."""
        assert ext_store.table_name == "ext_knowledge"
        assert ext_store.subject_tree is not None

    def test_store_empty_initially(self, ext_store):
        """Test that a new store has no entries."""
        results = ext_store.search_all_knowledge()
        assert results == []


class TestExtKnowledgeStoreOperations:
    """Tests for ExtKnowledgeStore CRUD operations."""

    def test_store_knowledge_single(self, ext_store):
        """Test storing a single knowledge entry."""
        ext_store.store_knowledge(
            subject_path=["Finance", "Banking"],
            name="APR",
            search_text="Annual Percentage Rate",
            explanation="The yearly cost of a loan expressed as a percentage",
        )
        results = ext_store.search_all_knowledge(subject_path=["Finance", "Banking"])
        assert len(results) == 1
        assert results[0]["name"] == "APR"
        assert results[0]["search_text"] == "Annual Percentage Rate"

    def test_batch_store_knowledge(self, ext_store, sample_knowledge):
        """Test storing multiple knowledge entries in batch."""
        ext_store.batch_store_knowledge(sample_knowledge)
        results = ext_store.search_all_knowledge()
        assert len(results) == 3

    def test_batch_store_knowledge_empty(self, ext_store):
        """Test batch store with empty list does nothing."""
        ext_store.batch_store_knowledge([])
        results = ext_store.search_all_knowledge()
        assert results == []

    def test_batch_store_knowledge_skips_invalid(self, ext_store):
        """Test batch store skips entries with missing required fields."""
        entries = [
            {
                "subject_path": ["Valid"],
                "name": "valid_entry",
                "search_text": "valid",
                "explanation": "valid explanation",
            },
            {
                "subject_path": [],
                "name": "",
                "search_text": "",
                "explanation": "",
            },
        ]
        ext_store.batch_store_knowledge(entries)
        results = ext_store.search_all_knowledge()
        assert len(results) == 1
        assert results[0]["name"] == "valid_entry"

    def test_upsert_knowledge_insert(self, ext_store):
        """Test upsert inserts a new entry when it does not exist."""
        ext_store.upsert_knowledge(
            subject_path=["Tech", "AI"],
            name="NLP",
            search_text="Natural Language Processing",
            explanation="AI subfield dealing with human language",
        )
        results = ext_store.search_all_knowledge(subject_path=["Tech", "AI"])
        assert len(results) == 1
        assert results[0]["name"] == "NLP"
        assert results[0]["explanation"] == "AI subfield dealing with human language"

    def test_upsert_knowledge_update(self, ext_store):
        """Test upsert updates an existing entry with the same id."""
        ext_store.upsert_knowledge(
            subject_path=["Tech", "AI"],
            name="NLP",
            search_text="Natural Language Processing",
            explanation="Original explanation",
        )
        ext_store.upsert_knowledge(
            subject_path=["Tech", "AI"],
            name="NLP",
            search_text="Natural Language Processing v2",
            explanation="Updated explanation",
        )
        results = ext_store.search_all_knowledge(subject_path=["Tech", "AI"])
        assert len(results) == 1
        assert results[0]["explanation"] == "Updated explanation"

    def test_batch_upsert_knowledge(self, ext_store):
        """Test batch upsert with multiple entries."""
        entries = [
            {
                "subject_path": ["Science", "Physics"],
                "name": "gravity",
                "search_text": "Gravity",
                "explanation": "Fundamental force of attraction",
            },
            {
                "subject_path": ["Science", "Chemistry"],
                "name": "atom",
                "search_text": "Atom",
                "explanation": "Basic unit of matter",
            },
        ]
        upserted_ids = ext_store.batch_upsert_knowledge(entries)
        assert len(upserted_ids) == 2
        assert "Science/Physics/gravity" in upserted_ids
        assert "Science/Chemistry/atom" in upserted_ids

    def test_batch_upsert_knowledge_empty(self, ext_store):
        """Test batch upsert with empty list returns empty."""
        result = ext_store.batch_upsert_knowledge([])
        assert result == []

    def test_batch_upsert_knowledge_skips_invalid(self, ext_store):
        """Test batch upsert skips entries with missing required fields."""
        entries = [
            {
                "subject_path": ["Valid"],
                "name": "valid",
                "search_text": "valid_text",
                "explanation": "valid_exp",
            },
            {
                "subject_path": [],
                "name": "",
                "search_text": "",
                "explanation": "",
            },
        ]
        upserted_ids = ext_store.batch_upsert_knowledge(entries)
        assert len(upserted_ids) == 1


class TestExtKnowledgeStoreSearch:
    """Tests for ExtKnowledgeStore search operations."""

    def test_search_knowledge_by_query(self, ext_store, sample_knowledge):
        """Test vector search returns relevant results."""
        ext_store.batch_store_knowledge(sample_knowledge)
        results = ext_store.search_knowledge(query_text="financial metrics", top_n=2)
        assert len(results) <= 2
        assert len(results) > 0

    def test_search_knowledge_by_subject_path(self, ext_store, sample_knowledge):
        """Test search filtered by subject_path."""
        ext_store.batch_store_knowledge(sample_knowledge)
        results = ext_store.search_knowledge(subject_path=["Finance", "Banking"], top_n=10)
        assert len(results) == 2
        for r in results:
            assert r["subject_path"][0] == "Finance"
            assert r["subject_path"][1] == "Banking"

    def test_search_all_knowledge_no_filter(self, ext_store, sample_knowledge):
        """Test search_all_knowledge returns all entries."""
        ext_store.batch_store_knowledge(sample_knowledge)
        results = ext_store.search_all_knowledge()
        assert len(results) == 3

    def test_search_all_knowledge_with_subject_filter(self, ext_store, sample_knowledge):
        """Test search_all_knowledge with subject_path filter."""
        ext_store.batch_store_knowledge(sample_knowledge)
        results = ext_store.search_all_knowledge(subject_path=["Finance", "Investment"])
        assert len(results) == 1
        assert results[0]["name"] == "PE_Ratio"

    def test_search_knowledge_with_select_fields(self, ext_store, sample_knowledge):
        """Test search with specific field selection."""
        ext_store.batch_store_knowledge(sample_knowledge)
        results = ext_store.search_knowledge(
            query_text="banking",
            select_fields=["name", "search_text"],
            top_n=2,
        )
        assert len(results) > 0
        for r in results:
            assert "name" in r
            assert "search_text" in r


class TestExtKnowledgeStoreDelete:
    """Tests for ExtKnowledgeStore delete operations."""

    def test_delete_knowledge_success(self, ext_store):
        """Test successful deletion of a knowledge entry."""
        ext_store.store_knowledge(
            subject_path=["Test", "Delete"],
            name="to_delete",
            search_text="Delete me",
            explanation="This will be deleted",
        )
        result = ext_store.delete_knowledge(subject_path=["Test", "Delete"], name="to_delete")
        assert result is True
        remaining = ext_store.search_all_knowledge(subject_path=["Test", "Delete"])
        assert len(remaining) == 0

    def test_delete_knowledge_not_found(self, ext_store):
        """Test deletion of non-existent entry returns False."""
        ext_store.store_knowledge(
            subject_path=["Test", "Keep"],
            name="existing",
            search_text="Existing entry",
            explanation="This stays",
        )
        result = ext_store.delete_knowledge(subject_path=["Test", "Keep"], name="nonexistent")
        assert result is False


class TestExtKnowledgeStoreAfterInit:
    """Tests for after_init (index creation)."""

    def test_after_init_creates_indices(self, ext_store, sample_knowledge):
        """Test that after_init creates indices without errors."""
        ext_store.batch_store_knowledge(sample_knowledge)
        ext_store.after_init()
        # Verify search still works after index creation
        results = ext_store.search_all_knowledge()
        assert len(results) == 3


class TestExtKnowledgeStoreCreateIndices:
    """Tests for create_indices method."""

    def test_create_indices(self, ext_store, sample_knowledge):
        """Test create_indices creates scalar and FTS indices."""
        ext_store.batch_store_knowledge(sample_knowledge)
        ext_store.create_indices()
        # Verify search still works after creating indices
        results = ext_store.search_knowledge(query_text="banking", top_n=2)
        assert len(results) > 0


# ============================================================
# ExtKnowledgeRAG
# ============================================================


class TestExtKnowledgeRAGInit:
    """Tests for ExtKnowledgeRAG initialization."""

    def test_rag_init(self, real_agent_config):
        """Test RAG initializes with real agent config."""
        rag = ExtKnowledgeRAG(agent_config=real_agent_config)
        assert rag.store is not None


class TestExtKnowledgeRAGOperations:
    """Tests for ExtKnowledgeRAG CRUD operations."""

    def test_query_knowledge_empty(self, real_agent_config):
        """Test querying when no knowledge exists returns empty list."""
        rag = ExtKnowledgeRAG(agent_config=real_agent_config)
        results = rag.query_knowledge(query_text="anything", top_n=5)
        assert results == []

    def test_get_knowledge_size_empty(self, real_agent_config):
        """Test knowledge size is zero when store is empty."""
        rag = ExtKnowledgeRAG(agent_config=real_agent_config)
        size = rag.get_knowledge_size()
        assert size == 0

    def test_parse_subject_path_string(self, real_agent_config):
        """Test parsing subject_path from string format."""
        rag = ExtKnowledgeRAG(agent_config=real_agent_config)
        result = rag._parse_subject_path("Finance/Revenue/Q1")
        assert result == ["Finance", "Revenue", "Q1"]

    def test_parse_subject_path_list(self, real_agent_config):
        """Test parsing subject_path from list format (passthrough)."""
        rag = ExtKnowledgeRAG(agent_config=real_agent_config)
        result = rag._parse_subject_path(["Finance", "Revenue"])
        assert result == ["Finance", "Revenue"]

    def test_parse_subject_path_invalid_type(self, real_agent_config):
        """Test parsing subject_path with invalid type raises ValueError."""
        rag = ExtKnowledgeRAG(agent_config=real_agent_config)
        with pytest.raises(ValueError, match="must be string or list"):
            rag._parse_subject_path(123)

    def test_parse_subject_path_strips_whitespace(self, real_agent_config):
        """Test parsing string path strips whitespace from components."""
        rag = ExtKnowledgeRAG(agent_config=real_agent_config)
        result = rag._parse_subject_path(" Finance / Revenue / Q1 ")
        assert result == ["Finance", "Revenue", "Q1"]

    def test_parse_subject_path_skips_empty_components(self, real_agent_config):
        """Test parsing string path skips empty components from double slashes."""
        rag = ExtKnowledgeRAG(agent_config=real_agent_config)
        result = rag._parse_subject_path("Finance//Revenue")
        assert result == ["Finance", "Revenue"]

    def test_store_and_query_knowledge(self, real_agent_config):
        """Test full round-trip: store via store, query via RAG."""
        rag = ExtKnowledgeRAG(agent_config=real_agent_config)
        rag.store.batch_store_knowledge(
            [
                {
                    "subject_path": ["Finance", "Banking"],
                    "name": "APR",
                    "search_text": "Annual Percentage Rate",
                    "explanation": "The yearly cost of borrowing",
                },
            ]
        )
        results = rag.query_knowledge(query_text="banking rate", top_n=5)
        assert len(results) >= 1

    def test_get_knowledge_detail(self, real_agent_config):
        """Test getting knowledge detail by path and name."""
        rag = ExtKnowledgeRAG(agent_config=real_agent_config)
        rag.store.batch_store_knowledge(
            [
                {
                    "subject_path": ["Tech", "AI"],
                    "name": "ML",
                    "search_text": "Machine Learning",
                    "explanation": "Subset of AI focused on learning from data",
                },
            ]
        )
        results = rag.get_knowledge_detail(subject_path=["Tech", "AI"], name="ML")
        assert len(results) == 1
        assert results[0]["name"] == "ML"

    def test_delete_knowledge(self, real_agent_config):
        """Test deleting knowledge entry via RAG."""
        rag = ExtKnowledgeRAG(agent_config=real_agent_config)
        rag.store.batch_store_knowledge(
            [
                {
                    "subject_path": ["Test", "Delete"],
                    "name": "item",
                    "search_text": "to delete",
                    "explanation": "will be deleted",
                },
            ]
        )
        deleted = rag.delete_knowledge(subject_path=["Test", "Delete"], name="item")
        assert deleted is True
        results = rag.get_knowledge_detail(subject_path=["Test", "Delete"], name="item")
        assert len(results) == 0

    def test_get_knowledge_batch(self, real_agent_config):
        """Test getting multiple knowledge entries by paths."""
        rag = ExtKnowledgeRAG(agent_config=real_agent_config)
        rag.store.batch_store_knowledge(
            [
                {
                    "subject_path": ["A", "B"],
                    "name": "item1",
                    "search_text": "Item 1",
                    "explanation": "First item",
                },
                {
                    "subject_path": ["C", "D"],
                    "name": "item2",
                    "search_text": "Item 2",
                    "explanation": "Second item",
                },
            ]
        )
        results = rag.get_knowledge_batch(
            paths=[
                ["A", "B", "item1"],
                ["C", "D", "item2"],
            ]
        )
        assert len(results) == 2

    def test_get_knowledge_batch_skips_empty(self, real_agent_config):
        """Test get_knowledge_batch skips empty paths."""
        rag = ExtKnowledgeRAG(agent_config=real_agent_config)
        results = rag.get_knowledge_batch(paths=[[], ["nonexistent"]])
        assert isinstance(results, list)
