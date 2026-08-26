"""Unit tests for embedding_diagnostics classification and formatting helpers."""

from datus.storage.embedding_diagnostics import (
    format_context_unavailable,
    is_datasource_scope_error,
)

# The exact shape produced by datasource_scope.resolve_datasource_id when no
# datasource is selected — the bug this guards against surfaced this string
# wrapped in an "embedding model is unavailable" warning.
DATASOURCE_SCOPE_ERROR = (
    "error_code=410007, error_message=Invalid storage argument: datasource is required for datasource-scoped storage"
)


class TestIsDatasourceScopeError:
    def test_matches_resolve_datasource_id_error(self):
        assert is_datasource_scope_error(DATASOURCE_SCOPE_ERROR) is True

    def test_matches_exception_instances(self):
        assert is_datasource_scope_error(RuntimeError(DATASOURCE_SCOPE_ERROR)) is True

    def test_rejects_none_and_empty(self):
        assert is_datasource_scope_error(None) is False
        assert is_datasource_scope_error("") is False

    def test_rejects_other_storage_invalid_argument_errors(self):
        # Same 410007 code, different cause — must not classify as "no datasource".
        assert is_datasource_scope_error("error_code=410007: business id is required to build storage_key") is False

    def test_rejects_embedding_errors(self):
        assert is_datasource_scope_error("error_code=300019 embedding download failed") is False


class TestFormatContextUnavailable:
    def test_embedding_error_keeps_embedding_remediation(self):
        message = format_context_unavailable("error_code=300019: embedding download failed")
        assert "embedding model is unavailable" in message
        assert "Hugging Face" in message

    def test_datasource_error_never_blames_embeddings(self):
        message = format_context_unavailable(DATASOURCE_SCOPE_ERROR)
        assert "embedding" not in message.lower()
        assert "Hugging Face" not in message
        assert "no datasource is selected" in message

    def test_generic_error_keeps_details_without_embedding_remediation(self):
        message = format_context_unavailable("lance table corrupted")
        assert "Context search and @ references are disabled" in message
        assert "lance table corrupted" in message
        assert "embedding" not in message.lower()
        assert "Hugging Face" not in message

    def test_exception_instances_accepted(self):
        message = format_context_unavailable(RuntimeError(DATASOURCE_SCOPE_ERROR))
        assert "no datasource is selected" in message

    def test_none_yields_generic_message_without_details(self):
        message = format_context_unavailable(None)
        assert "Context search and @ references are disabled" in message
        assert "Details:" not in message
