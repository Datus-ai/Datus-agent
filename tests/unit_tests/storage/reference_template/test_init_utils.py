# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from unittest.mock import MagicMock

from datus.storage.reference_template.init_utils import exists_reference_templates, gen_reference_template_id


class TestGenReferenceTemplateId:
    def test_deterministic(self):
        template = "SELECT * FROM t WHERE x = {{val}}"
        id1 = gen_reference_template_id(template)
        id2 = gen_reference_template_id(template)
        assert id1 == id2

    def test_different_templates_different_ids(self):
        id1 = gen_reference_template_id("SELECT 1")
        id2 = gen_reference_template_id("SELECT 2")
        assert id1 != id2

    def test_returns_hex_string(self):
        result = gen_reference_template_id("SELECT 1")
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result), f"Not valid hex: {result}"


class TestExistsReferenceTemplates:
    def test_overwrite_mode_returns_empty(self):
        mock_storage = MagicMock()
        result = exists_reference_templates(mock_storage, build_mode="overwrite")
        assert result == set()
        mock_storage.search_all_reference_templates.assert_not_called()

    def test_incremental_mode_returns_existing_ids(self):
        mock_storage = MagicMock()
        mock_storage.search_all_reference_templates.return_value = [
            {"id": "abc123"},
            {"id": "def456"},
        ]
        result = exists_reference_templates(mock_storage, build_mode="incremental")
        assert result == {"abc123", "def456"}
