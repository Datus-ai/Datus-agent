# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for token storage."""

import json
import os
import stat
from datetime import datetime, timedelta, timezone

import pytest

from datus.auth.oauth_config import TOKEN_REFRESH_INTERVAL_SECONDS
from datus.auth.token_storage import TokenStorage


@pytest.fixture
def token_file(tmp_path):
    """Return a temporary path for token storage."""
    return str(tmp_path / "auth.json")


@pytest.fixture
def storage(token_file):
    return TokenStorage(path=token_file)


class TestSave:
    def test_creates_file(self, storage, token_file):
        storage.save({"access_token": "tok123"})
        assert os.path.exists(token_file)

    def test_file_permissions(self, storage, token_file):
        storage.save({"access_token": "tok123"})
        mode = os.stat(token_file).st_mode
        assert mode & stat.S_IRUSR  # owner read
        assert mode & stat.S_IWUSR  # owner write
        assert not (mode & stat.S_IRGRP)  # no group read
        assert not (mode & stat.S_IROTH)  # no other read

    def test_sets_last_refresh(self, storage, token_file):
        storage.save({"access_token": "tok123"})
        with open(token_file) as f:
            data = json.load(f)
        assert "last_refresh" in data

    def test_preserves_existing_last_refresh(self, storage, token_file):
        custom_ts = "2025-01-01T00:00:00+00:00"
        storage.save({"access_token": "tok", "last_refresh": custom_ts})
        with open(token_file) as f:
            data = json.load(f)
        assert data["last_refresh"] == custom_ts

    def test_creates_parent_directories(self, tmp_path):
        nested_path = str(tmp_path / "a" / "b" / "auth.json")
        s = TokenStorage(path=nested_path)
        s.save({"access_token": "tok"})
        assert os.path.exists(nested_path)


class TestLoad:
    def test_returns_none_when_missing(self, storage):
        assert storage.load() is None

    def test_loads_saved_tokens(self, storage):
        storage.save({"access_token": "tok123", "refresh_token": "rt_abc"})
        tokens = storage.load()
        assert tokens["access_token"] == "tok123"
        assert tokens["refresh_token"] == "rt_abc"

    def test_returns_none_on_invalid_json(self, storage, token_file):
        os.makedirs(os.path.dirname(token_file), exist_ok=True)
        with open(token_file, "w") as f:
            f.write("not json")
        assert storage.load() is None


class TestClear:
    def test_removes_file(self, storage, token_file):
        storage.save({"access_token": "tok"})
        storage.clear()
        assert not os.path.exists(token_file)

    def test_no_error_when_missing(self, storage):
        storage.clear()  # should not raise


class TestNeedsRefresh:
    def test_true_when_no_tokens(self, storage):
        assert storage.needs_refresh() is True

    def test_true_when_no_last_refresh(self, storage):
        storage.save({"access_token": "tok"})
        # Remove last_refresh
        tokens = storage.load()
        del tokens["last_refresh"]
        with open(storage.path, "w") as f:
            json.dump(tokens, f)
        assert storage.needs_refresh() is True

    def test_false_when_recently_refreshed(self, storage):
        storage.save({"access_token": "tok"})
        assert storage.needs_refresh() is False

    def test_true_when_expired(self, storage):
        old_time = datetime.now(timezone.utc) - timedelta(seconds=TOKEN_REFRESH_INTERVAL_SECONDS + 100)
        storage.save({"access_token": "tok", "last_refresh": old_time.isoformat()})
        assert storage.needs_refresh() is True

    def test_true_on_invalid_timestamp(self, storage):
        storage.save({"access_token": "tok", "last_refresh": "not-a-date"})
        assert storage.needs_refresh() is True


class TestIsExpired:
    def test_true_when_no_tokens(self, storage):
        assert storage.is_expired(None) is True

    def test_false_with_valid_expires_at(self, storage):
        future = datetime.now(timezone.utc).timestamp() + 3600
        assert storage.is_expired({"access_token": "tok", "expires_at": future}) is False

    def test_true_with_past_expires_at(self, storage):
        past = datetime.now(timezone.utc).timestamp() - 100
        assert storage.is_expired({"access_token": "tok", "expires_at": past}) is True

    def test_true_within_safety_buffer(self, storage):
        # expires_at is 30 seconds from now (within 60s safety buffer)
        near_future = datetime.now(timezone.utc).timestamp() + 30
        assert storage.is_expired({"access_token": "tok", "expires_at": near_future}) is True

    def test_falls_back_to_last_refresh(self, storage):
        old_time = datetime.now(timezone.utc) - timedelta(seconds=TOKEN_REFRESH_INTERVAL_SECONDS + 100)
        tokens = {"access_token": "tok", "last_refresh": old_time.isoformat()}
        assert storage.is_expired(tokens) is True

    def test_not_expired_with_recent_last_refresh(self, storage):
        recent = datetime.now(timezone.utc).isoformat()
        tokens = {"access_token": "tok", "last_refresh": recent}
        assert storage.is_expired(tokens) is False


class TestSaveExpiresAt:
    def test_computes_expires_at_from_expires_in(self, storage, token_file):
        storage.save({"access_token": "tok", "expires_in": 3600})
        with open(token_file) as f:
            data = json.load(f)
        assert "expires_at" in data
        expected = datetime.now(timezone.utc).timestamp() + 3600
        assert abs(data["expires_at"] - expected) < 5

    def test_preserves_existing_expires_at(self, storage, token_file):
        storage.save({"access_token": "tok", "expires_in": 3600, "expires_at": 99999})
        with open(token_file) as f:
            data = json.load(f)
        assert data["expires_at"] == 99999
