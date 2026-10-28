# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/utils/ssl_utils.py. Pure, no network."""

import os
import ssl

import pytest

from datus.utils.exceptions import DatusException
from datus.utils.ssl_utils import (
    is_pem_cert_content,
    is_ssl_cert_verification_error,
    materialize_ca_bundle,
    normalize_ssl_verify,
    resolve_ssl_verify_for_httpx,
    ssl_verify_to_env,
)


def _self_signed_ca_pem() -> str:
    """A throwaway, valid self-signed CA certificate (PEM), generated once."""
    from datetime import datetime, timedelta, timezone

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "datus-test-ca")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc) - timedelta(days=1))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    return cert.public_bytes(serialization.Encoding.PEM).decode("utf-8")


_CA_PEM = _self_signed_ca_pem()


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


class TestIsPemCertContent:
    def test_pem_content_detected(self):
        assert is_pem_cert_content(_CA_PEM) is True

    @pytest.mark.parametrize("value", ["/etc/ssl/ca.pem", "true", "false", "", True, False, None, 1])
    def test_non_pem_rejected(self, value):
        assert is_pem_cert_content(value) is False


class TestResolveSslVerifyForHttpx:
    def test_pem_content_becomes_ssl_context(self):
        result = resolve_ssl_verify_for_httpx(_CA_PEM)
        assert isinstance(result, ssl.SSLContext)

    @pytest.mark.parametrize("value", [True, False])
    def test_bool_delegates_to_normalize(self, value):
        assert resolve_ssl_verify_for_httpx(value) is value

    def test_path_delegates_to_normalize(self):
        assert resolve_ssl_verify_for_httpx("/etc/ssl/ca.pem") == "/etc/ssl/ca.pem"

    def test_malformed_pem_raises(self):
        bad = "-----BEGIN CERTIFICATE-----\nnot-valid-base64\n-----END CERTIFICATE-----"
        with pytest.raises(ssl.SSLError):
            resolve_ssl_verify_for_httpx(bad)


class TestMaterializeCaBundle:
    # Each test uses its own certificate so the content-addressed paths are
    # distinct and per-test cleanup cannot collide (even under parallel runs).
    def test_writes_content_to_file(self):
        pem = _self_signed_ca_pem()
        path = materialize_ca_bundle(pem)
        try:
            assert os.path.exists(path)
            with open(path, encoding="utf-8") as f:
                assert f.read() == pem
        finally:
            os.remove(path)

    def test_content_addressed_dedup(self):
        pem = _self_signed_ca_pem()
        p1 = materialize_ca_bundle(pem)
        p2 = materialize_ca_bundle(pem)
        try:
            assert p1 == p2
        finally:
            os.remove(p1)

    def test_different_content_different_path(self):
        p1 = materialize_ca_bundle(_self_signed_ca_pem())
        p2 = materialize_ca_bundle(_self_signed_ca_pem())
        try:
            assert p1 != p2
        finally:
            os.remove(p1)
            os.remove(p2)
