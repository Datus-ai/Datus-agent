# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from unittest.mock import MagicMock, patch

import pytest

from datus.storage.reference_template.store import ReferenceTemplateRAG, ReferenceTemplateStorage
from datus.utils.exceptions import DatusException


class TestReferenceTemplateStorage:
    @patch("datus.storage.reference_template.store.BaseSubjectEmbeddingStore.__init__", return_value=None)
    def test_init(self, mock_super):
        embedding_model = MagicMock()
        embedding_model.dim_size = 128
        ReferenceTemplateStorage(embedding_model)
        mock_super.assert_called_once()
        call_kwargs = mock_super.call_args
        assert call_kwargs.kwargs["table_name"] == "reference_template"
        assert call_kwargs.kwargs["vector_source_name"] == "search_text"
        assert call_kwargs.kwargs["unique_columns"] == ["id"]

    @patch("datus.storage.reference_template.store.BaseSubjectEmbeddingStore.__init__", return_value=None)
    def test_create_indices(self, mock_super):
        embedding_model = MagicMock()
        embedding_model.dim_size = 128
        storage = ReferenceTemplateStorage(embedding_model)
        storage._ensure_table_ready = MagicMock()
        storage._create_scalar_index = MagicMock()
        storage.create_subject_index = MagicMock()
        storage.create_fts_index = MagicMock()

        storage.create_indices()

        storage._ensure_table_ready.assert_called_once()
        assert storage._create_scalar_index.call_count == 3
        storage.create_subject_index.assert_called_once()
        storage.create_fts_index.assert_called_once()

    @patch("datus.storage.reference_template.store.BaseSubjectEmbeddingStore.__init__", return_value=None)
    def test_batch_store_templates_empty(self, mock_super):
        embedding_model = MagicMock()
        embedding_model.dim_size = 128
        storage = ReferenceTemplateStorage(embedding_model)
        storage.batch_store = MagicMock()

        storage.batch_store_templates([])
        storage.batch_store.assert_not_called()

    @patch("datus.storage.reference_template.store.BaseSubjectEmbeddingStore.__init__", return_value=None)
    def test_batch_store_templates_valid(self, mock_super):
        embedding_model = MagicMock()
        embedding_model.dim_size = 128
        storage = ReferenceTemplateStorage(embedding_model)
        storage.batch_store = MagicMock()

        items = [
            {
                "name": "test",
                "template": "SELECT {{x}}",
                "summary": "test summary",
                "search_text": "test search",
                "subject_path": ["Sales"],
            }
        ]
        storage.batch_store_templates(items)
        storage.batch_store.assert_called_once_with(items)

    @patch("datus.storage.reference_template.store.BaseSubjectEmbeddingStore.__init__", return_value=None)
    def test_batch_store_templates_missing_fields(self, mock_super):
        embedding_model = MagicMock()
        embedding_model.dim_size = 128
        storage = ReferenceTemplateStorage(embedding_model)
        storage.batch_store = MagicMock()

        items = [
            {
                "name": "test",
                "template": "SELECT {{x}}",
                # missing summary, search_text, subject_path
            }
        ]
        storage.batch_store_templates(items)
        # Should skip the invalid item
        storage.batch_store.assert_called_once_with([])

    @patch("datus.storage.reference_template.store.BaseSubjectEmbeddingStore.__init__", return_value=None)
    def test_batch_upsert_templates_empty(self, mock_super):
        embedding_model = MagicMock()
        embedding_model.dim_size = 128
        storage = ReferenceTemplateStorage(embedding_model)
        storage.batch_upsert = MagicMock()

        storage.batch_upsert_templates([])
        storage.batch_upsert.assert_not_called()

    @patch("datus.storage.reference_template.store.BaseSubjectEmbeddingStore.__init__", return_value=None)
    def test_batch_upsert_templates_missing_subject_path(self, mock_super):
        embedding_model = MagicMock()
        embedding_model.dim_size = 128
        storage = ReferenceTemplateStorage(embedding_model)
        storage.batch_upsert = MagicMock()

        items = [{"id": "abc", "template": "SELECT 1"}]
        with pytest.raises(DatusException):
            storage.batch_upsert_templates(items)

    @patch("datus.storage.reference_template.store.BaseSubjectEmbeddingStore.__init__", return_value=None)
    def test_batch_upsert_templates_valid(self, mock_super):
        embedding_model = MagicMock()
        embedding_model.dim_size = 128
        storage = ReferenceTemplateStorage(embedding_model)
        storage.batch_upsert = MagicMock()

        items = [{"id": "abc", "template": "SELECT 1", "subject_path": ["Sales"]}]
        storage.batch_upsert_templates(items)
        storage.batch_upsert.assert_called_once_with(items, on_column="id")

    @patch("datus.storage.reference_template.store.BaseSubjectEmbeddingStore.__init__", return_value=None)
    def test_search_reference_templates(self, mock_super):
        embedding_model = MagicMock()
        embedding_model.dim_size = 128
        storage = ReferenceTemplateStorage(embedding_model)
        storage.search_with_subject_filter = MagicMock(return_value=[{"name": "test"}])

        result = storage.search_reference_templates(query_text="test", top_n=3)
        storage.search_with_subject_filter.assert_called_once()
        assert len(result) == 1

    @patch("datus.storage.reference_template.store.BaseSubjectEmbeddingStore.__init__", return_value=None)
    def test_search_all_reference_templates(self, mock_super):
        embedding_model = MagicMock()
        embedding_model.dim_size = 128
        storage = ReferenceTemplateStorage(embedding_model)
        storage.search_with_subject_filter = MagicMock(return_value=[])

        result = storage.search_all_reference_templates(subject_path=["Sales"])
        storage.search_with_subject_filter.assert_called_once()
        assert result == []

    @patch("datus.storage.reference_template.store.BaseSubjectEmbeddingStore.__init__", return_value=None)
    def test_delete_reference_template(self, mock_super):
        embedding_model = MagicMock()
        embedding_model.dim_size = 128
        storage = ReferenceTemplateStorage(embedding_model)
        storage.delete_entry = MagicMock(return_value=True)

        result = storage.delete_reference_template(["Sales"], "test")
        assert result is True
        storage.delete_entry.assert_called_once()


class TestReferenceTemplateRAG:
    def _create_rag(self, mock_get_storage, mock_filter=None, filter_value=None):
        mock_storage = MagicMock()
        mock_get_storage.return_value = mock_storage
        if mock_filter:
            mock_filter.return_value = filter_value

        config = MagicMock()
        config.current_datasource = "test_ns"
        return ReferenceTemplateRAG(config), mock_storage

    @patch("datus.storage.rag_scope._build_sub_agent_filter", return_value=None)
    @patch("datus.storage.registry.get_storage")
    def test_init(self, mock_get_storage, mock_filter):
        rag, _ = self._create_rag(mock_get_storage)
        assert rag.datasource_id == "test_ns"
        mock_get_storage.assert_called_once()

    @patch("datus.storage.rag_scope._build_sub_agent_filter", return_value=None)
    @patch("datus.storage.registry.get_storage")
    def test_truncate(self, mock_get_storage, mock_filter):
        rag, mock_storage = self._create_rag(mock_get_storage)
        rag.truncate()
        mock_storage.truncate_scoped.assert_called_once()

    @patch("datus.storage.rag_scope._build_sub_agent_filter", return_value=None)
    @patch("datus.storage.registry.get_storage")
    def test_store_batch(self, mock_get_storage, mock_filter):
        rag, mock_storage = self._create_rag(mock_get_storage)
        items = [{"name": "test"}]
        rag.store_batch(items)
        mock_storage.batch_store_templates.assert_called_once_with(items)

    @patch("datus.storage.rag_scope._build_sub_agent_filter", return_value=None)
    @patch("datus.storage.registry.get_storage")
    def test_upsert_batch(self, mock_get_storage, mock_filter):
        rag, mock_storage = self._create_rag(mock_get_storage)
        items = [{"id": "abc"}]
        rag.upsert_batch(items)
        mock_storage.batch_upsert_templates.assert_called_once_with(items)

    @patch("datus.storage.rag_scope._build_sub_agent_filter", return_value=None)
    @patch("datus.storage.registry.get_storage")
    def test_get_reference_template_size_no_filter(self, mock_get_storage, mock_filter):
        rag, mock_storage = self._create_rag(mock_get_storage)
        mock_storage._count_rows.return_value = 5
        assert rag.get_reference_template_size() == 5

    @patch("datus.storage.rag_scope._build_sub_agent_filter")
    @patch("datus.storage.registry.get_storage")
    def test_get_reference_template_size_with_filter(self, mock_get_storage, mock_filter):
        rag, mock_storage = self._create_rag(mock_get_storage, mock_filter, filter_value=MagicMock())
        mock_storage._count_rows.return_value = 3
        assert rag.get_reference_template_size() == 3

    @patch("datus.storage.rag_scope._build_sub_agent_filter", return_value=None)
    @patch("datus.storage.registry.get_storage")
    def test_search_reference_templates(self, mock_get_storage, mock_filter):
        rag, mock_storage = self._create_rag(mock_get_storage)
        mock_storage.search_reference_templates.return_value = [{"name": "tpl"}]
        result = rag.search_reference_templates("query", top_n=3)
        assert len(result) == 1

    @patch("datus.storage.rag_scope._build_sub_agent_filter", return_value=None)
    @patch("datus.storage.registry.get_storage")
    def test_get_reference_template_detail(self, mock_get_storage, mock_filter):
        rag, mock_storage = self._create_rag(mock_get_storage)
        mock_storage.search_all_reference_templates.return_value = [{"name": "tpl"}]
        result = rag.get_reference_template_detail(["Sales"], "tpl")
        assert len(result) == 1
        mock_storage.search_all_reference_templates.assert_called_once()

    @patch("datus.storage.rag_scope._build_sub_agent_filter", return_value=None)
    @patch("datus.storage.registry.get_storage")
    def test_delete_reference_template(self, mock_get_storage, mock_filter):
        rag, mock_storage = self._create_rag(mock_get_storage)
        mock_storage.delete_reference_template.return_value = True
        result = rag.delete_reference_template(["Sales"], "tpl")
        assert result is True

    @patch("datus.storage.rag_scope._build_sub_agent_filter", return_value=None)
    @patch("datus.storage.registry.get_storage")
    def test_after_init(self, mock_get_storage, mock_filter):
        rag, mock_storage = self._create_rag(mock_get_storage)
        rag.after_init()
        mock_storage.create_indices.assert_called_once()

    @patch("datus.storage.rag_scope._build_sub_agent_filter", return_value=None)
    @patch("datus.storage.registry.get_storage")
    def test_search_all_reference_templates(self, mock_get_storage, mock_filter):
        rag, mock_storage = self._create_rag(mock_get_storage)
        mock_storage.search_all_reference_templates.return_value = []
        result = rag.search_all_reference_templates()
        assert result == []
