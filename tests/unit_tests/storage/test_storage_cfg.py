# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.storage_cfg."""

import os

import pytest

from datus.storage.embedding_models import DEFAULT_MODEL_CONFIG, EmbeddingModel
from datus.storage.storage_cfg import (
    _find_config_differences,
    check_storage_config,
    load_storage_config,
    save_storage_config,
    save_storage_configs,
)
from datus.utils.constants import EmbeddingProvider
from datus.utils.exceptions import DatusException

# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


class TestSaveAndLoadStorageConfig:
    """Tests for save_storage_config and load_storage_config round-trips."""

    def test_save_and_load_round_trip(self, tmp_path):
        """Saving a config and loading it back should produce matching data."""
        rag_path = str(tmp_path / "rag")
        model = EmbeddingModel(model_name="all-MiniLM-L6-v2", dim_size=384)
        save_storage_config("database", rag_path, config=model)

        loaded = load_storage_config(rag_path)
        assert "database" in loaded
        assert loaded["database"]["model_name"] == "all-MiniLM-L6-v2"
        assert loaded["database"]["dim_size"] == "384"

    def test_save_removes_section_when_config_is_none(self, tmp_path):
        """Passing config=None should remove the section from the cfg file."""
        rag_path = str(tmp_path / "rag")
        model = EmbeddingModel(model_name="test-model", dim_size=128)
        save_storage_config("database", rag_path, config=model)
        # Verify it was saved
        assert "database" in load_storage_config(rag_path)
        # Remove it
        save_storage_config("database", rag_path, config=None)
        loaded = load_storage_config(rag_path)
        assert "database" not in loaded

    def test_save_multiple_sections(self, tmp_path):
        """Multiple config sections should all be persisted."""
        rag_path = str(tmp_path / "rag")
        model_a = EmbeddingModel(model_name="model-a", dim_size=256)
        model_b = EmbeddingModel(model_name="model-b", dim_size=768)
        save_storage_config("database", rag_path, config=model_a)
        save_storage_config("document", rag_path, config=model_b)

        loaded = load_storage_config(rag_path)
        assert "database" in loaded
        assert "document" in loaded
        assert loaded["database"]["model_name"] == "model-a"
        assert loaded["document"]["model_name"] == "model-b"

    def test_load_missing_file_returns_empty(self, tmp_path):
        """Loading from a non-existent path should return an empty dict."""
        rag_path = str(tmp_path / "nonexistent")
        loaded = load_storage_config(rag_path)
        assert loaded == {}


# ---------------------------------------------------------------------------
# save_storage_configs
# ---------------------------------------------------------------------------


class TestSaveStorageConfigs:
    """Tests for save_storage_configs with raw dicts."""

    def test_save_configs_creates_directory(self, tmp_path):
        """save_storage_configs should auto-create missing directories."""
        rag_path = str(tmp_path / "a" / "b" / "c")
        configs = {"database": {"model_name": "test", "dim_size": "128"}}
        save_storage_configs(configs, rag_path)
        assert os.path.isfile(os.path.join(rag_path, "datus_db.cfg"))

    def test_save_configs_enum_serialization(self, tmp_path):
        """Enum values should be serialized as their .value string."""
        rag_path = str(tmp_path / "rag")
        configs = {
            "database": {
                "model_name": "test",
                "dim_size": "128",
                "registry_name": EmbeddingProvider.OPENAI,
            }
        }
        save_storage_configs(configs, rag_path)
        loaded = load_storage_config(rag_path)
        assert loaded["database"]["registry_name"] == EmbeddingProvider.OPENAI.value

    def test_save_and_load_multiple_configs(self, tmp_path):
        """Multiple sections should all be preserved in the config file."""
        rag_path = str(tmp_path / "rag")
        configs = {
            "database": {"model_name": "m1", "dim_size": "256"},
            "document": {"model_name": "m2", "dim_size": "512"},
            "metric": {"model_name": "m3", "dim_size": "768"},
        }
        save_storage_configs(configs, rag_path)
        loaded = load_storage_config(rag_path)
        assert set(loaded.keys()) == {"database", "document", "metric"}


# ---------------------------------------------------------------------------
# _find_config_differences
# ---------------------------------------------------------------------------


class TestFindConfigDifferences:
    """Tests for _find_config_differences."""

    def test_identical_configs_no_differences(self):
        """Identical configs should return an empty list."""
        cfg = {"model_name": "all-MiniLM-L6-v2", "dim_size": "384", "registry_name": "sentence-transformers"}
        diffs = _find_config_differences("database", cfg, dict(cfg))
        assert diffs == []

    def test_value_mismatch_detected(self):
        """A differing value should produce a mismatch message."""
        existing = {"model_name": "model-a", "dim_size": "256", "registry_name": "sentence-transformers"}
        new = {"model_name": "model-b", "dim_size": "256", "registry_name": "sentence-transformers"}
        diffs = _find_config_differences("database", existing, new)
        assert len(diffs) == 1
        assert "model_name" in diffs[0]

    def test_missing_key_detected(self):
        """A key present in existing but missing in new should be reported."""
        existing = {
            "model_name": "model-a",
            "dim_size": "256",
            "registry_name": "sentence-transformers",
            "extra": "val",
        }
        new = {"model_name": "model-a", "dim_size": "256", "registry_name": "sentence-transformers"}
        diffs = _find_config_differences("database", existing, new)
        assert any("extra" in d for d in diffs)

    def test_none_existing_uses_default(self):
        """When existing is None, DEFAULT_MODEL_CONFIG should be used."""
        new = dict(DEFAULT_MODEL_CONFIG)
        new["registry_name"] = EmbeddingProvider.SENTENCE_TRANSFORMERS
        diffs = _find_config_differences("database", None, new)
        assert diffs == []

    def test_none_new_uses_default(self):
        """When new is None, DEFAULT_MODEL_CONFIG should be used."""
        existing = dict(DEFAULT_MODEL_CONFIG)
        existing["registry_name"] = EmbeddingProvider.SENTENCE_TRANSFORMERS
        diffs = _find_config_differences("database", existing, None)
        assert diffs == []

    def test_enum_values_compared_correctly(self):
        """Enum values should be compared via .value, not repr."""
        existing = {"model_name": "m", "dim_size": "384", "registry_name": EmbeddingProvider.OPENAI}
        new = {"model_name": "m", "dim_size": "384", "registry_name": "openai"}
        diffs = _find_config_differences("database", existing, new)
        assert diffs == []

    def test_both_none_no_differences(self):
        """Both None should both default to DEFAULT_MODEL_CONFIG, no diffs."""
        diffs = _find_config_differences("database", None, None)
        assert diffs == []


# ---------------------------------------------------------------------------
# check_storage_config
# ---------------------------------------------------------------------------


class TestCheckStorageConfig:
    """Tests for check_storage_config."""

    def test_check_saves_config_on_fresh_path(self, tmp_path):
        """On a fresh path, check_storage_config should save the config."""
        rag_path = str(tmp_path / "rag")
        cfg = {"model_name": "all-MiniLM-L6-v2", "dim_size": "384", "registry_name": "sentence-transformers"}
        check_storage_config("database", cfg, rag_path)
        loaded = load_storage_config(rag_path)
        assert "database" in loaded

    def test_check_no_error_on_matching_config(self, tmp_path):
        """Re-checking with the same config should not raise."""
        rag_path = str(tmp_path / "rag")
        cfg = {"model_name": "all-MiniLM-L6-v2", "dim_size": "384", "registry_name": "sentence-transformers"}
        check_storage_config("database", cfg, rag_path)
        # Should not raise
        check_storage_config("database", cfg, rag_path)

    def test_check_raises_on_mismatch(self, tmp_path):
        """Mismatching config should raise DatusException."""
        rag_path = str(tmp_path / "rag")
        cfg_a = {"model_name": "model-a", "dim_size": "256", "registry_name": "sentence-transformers"}
        check_storage_config("database", cfg_a, rag_path)

        cfg_b = {"model_name": "model-b", "dim_size": "256", "registry_name": "sentence-transformers"}
        with pytest.raises(DatusException):
            check_storage_config("database", cfg_b, rag_path)

    def test_check_save_config_false_skips_write(self, tmp_path):
        """save_config=False should validate but not write to disk."""
        rag_path = str(tmp_path / "rag")
        cfg = {"model_name": "m", "dim_size": "384", "registry_name": "sentence-transformers"}
        check_storage_config("database", cfg, rag_path, save_config=False)
        # No config file should be created since there was no prior config
        # and save_config=False was passed.
        # Note: check_storage_config only saves when save_config=True
        loaded = load_storage_config(rag_path)
        assert loaded == {}

    def test_check_none_config_uses_default(self, tmp_path):
        """Passing None as storage_config should save the default config."""
        rag_path = str(tmp_path / "rag")
        check_storage_config("database", None, rag_path)
        loaded = load_storage_config(rag_path)
        assert "database" in loaded
        assert loaded["database"]["model_name"] == DEFAULT_MODEL_CONFIG["model_name"]
