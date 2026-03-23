# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for PKCE utilities."""

import base64
import hashlib

from datus.auth.pkce import generate_pkce_pair, generate_state


class TestGeneratePkcePair:
    def test_returns_two_strings(self):
        verifier, challenge = generate_pkce_pair()
        assert isinstance(verifier, str)
        assert isinstance(challenge, str)

    def test_verifier_length(self):
        verifier, _ = generate_pkce_pair()
        # 64 random bytes -> base64url without padding should be ~86 chars
        assert len(verifier) >= 43

    def test_challenge_is_sha256_of_verifier(self):
        verifier, challenge = generate_pkce_pair()
        expected = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
        )
        assert challenge == expected

    def test_no_padding_characters(self):
        verifier, challenge = generate_pkce_pair()
        assert "=" not in verifier
        assert "=" not in challenge

    def test_url_safe_characters(self):
        verifier, challenge = generate_pkce_pair()
        # base64url uses only A-Z, a-z, 0-9, -, _
        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
        assert all(c in allowed for c in verifier)
        assert all(c in allowed for c in challenge)

    def test_pairs_are_unique(self):
        pair1 = generate_pkce_pair()
        pair2 = generate_pkce_pair()
        assert pair1[0] != pair2[0]
        assert pair1[1] != pair2[1]


class TestGenerateState:
    def test_returns_string(self):
        state = generate_state()
        assert isinstance(state, str)

    def test_length(self):
        state = generate_state()
        # 32 random bytes -> base64url without padding should be ~43 chars
        assert len(state) >= 32

    def test_no_padding(self):
        state = generate_state()
        assert "=" not in state

    def test_states_are_unique(self):
        states = {generate_state() for _ in range(10)}
        assert len(states) == 10
