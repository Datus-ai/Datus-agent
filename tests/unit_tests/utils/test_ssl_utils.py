# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/utils/ssl_utils.py. Pure, no network."""

import pytest

from datus.utils.exceptions import DatusException
from datus.utils.ssl_utils import (
    is_ssl_cert_verification_error,
    normalize_ssl_verify,
    ssl_verify_to_env,
)


class TestNormalizeSslVerify:
    @pytest.mark.parametrize("value", [True, False])
    def test_bool_passthrough(self, value):
        assert normalize_ssl_verify(value) is value

    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "  true  "])
    def test_true_strings_coerced(self, value):
        assert normalize_ssl_verify(value) is True

    @pytest.mark.parametrize("value", ["false", "False", "FALSE", " false "])
    def test_false_strings_coerced(self, value):
        assert normalize_ssl_verify(value) is False

    def test_path_string_returned_stripped(self):
        assert normalize_ssl_verify("  /etc/ssl/ca.pem ") == "/etc/ssl/ca.pem"

    def test_path_string_not_coerced_to_bool(self):
        # A path that merely contains "true"/"false" substrings stays a path.
        assert normalize_ssl_verify("/etc/ssl/true-ca.pem") == "/etc/ssl/true-ca.pem"

    def test_disable_logs_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            assert normalize_ssl_verify(False) is False
        assert any("disabled" in r.message.lower() for r in caplog.records)

    @pytest.mark.parametrize("value", [1, 0, None, ["x"], {"a": 1}])
    def test_invalid_type_raises(self, value):
        with pytest.raises(DatusException):
            normalize_ssl_verify(value)

    @pytest.mark.parametrize("value", ["", "   "])
    def test_empty_string_raises(self, value):
        with pytest.raises(DatusException):
            normalize_ssl_verify(value)


class TestSslVerifyToEnv:
    def test_true(self):
        assert ssl_verify_to_env(True) == "true"

    def test_false(self):
        assert ssl_verify_to_env(False) == "false"

    def test_path(self):
        assert ssl_verify_to_env("/etc/ssl/ca.pem") == "/etc/ssl/ca.pem"

    def test_round_trip_path(self):
        # Rendering a normalized path and re-normalizing yields the same path.
        v = normalize_ssl_verify("/etc/ssl/ca.pem")
        assert normalize_ssl_verify(ssl_verify_to_env(v)) == "/etc/ssl/ca.pem"


class TestIsSslCertVerificationError:
    @pytest.mark.parametrize(
        "msg",
        [
            "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate",
            "litellm.InternalServerError: AnthropicException - ... CERTIFICATE_VERIFY_FAILED ...",
            "certificate verify failed: self signed certificate in certificate chain",
        ],
    )
    def test_direct_message_detected(self, msg):
        assert is_ssl_cert_verification_error(Exception(msg)) is True

    def test_detected_through_cause_chain(self):
        # Native Anthropic path: top-level is generic, SSL detail nested in __cause__.
        inner = Exception("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate")
        try:
            try:
                raise inner
            except Exception as cause:
                raise ValueError("Connection error.") from cause
        except Exception as wrapped:
            assert is_ssl_cert_verification_error(wrapped) is True

    @pytest.mark.parametrize("msg", ["429 rate limit exceeded", "Connection error.", "401 unauthorized"])
    def test_unrelated_errors_not_detected(self, msg):
        assert is_ssl_cert_verification_error(Exception(msg)) is False

    def test_self_referential_chain_terminates(self):
        # A cyclic __context__ must not loop forever.
        e = Exception("boom")
        e.__context__ = e
        assert is_ssl_cert_verification_error(e) is False
