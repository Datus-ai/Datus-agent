from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.integration.adapters.snowflake_secret_helpers import delete_owned_secret, identifier_name


@pytest.mark.parametrize(
    "value,expected",
    [
        ("analytics", "ANALYTICS"),
        ("COMPUTE_WH", "COMPUTE_WH"),
        ('"Case Sensitive"', "Case Sensitive"),
        ('"a""b"', 'a"b'),
    ],
)
def test_identifier_expectation_preserves_quoted_names(value, expected):
    assert identifier_name(value) == expected


class SecretNotFound(Exception):
    pass


@pytest.fixture
def secret_client():
    client = Mock()
    client.exceptions = SimpleNamespace(ResourceNotFoundException=SecretNotFound)
    client.delete_secret.return_value = {"ARN": "owned-arn", "DeletionDate": datetime(2026, 1, 1, tzinfo=timezone.utc)}
    return client


@pytest.mark.parametrize("physically_deleted", [False, True])
def test_owned_secret_with_extra_tags_is_deleted_without_a_physical_removal_deadline(secret_client, physically_deleted):
    secret_client.describe_secret.side_effect = [
        {"Tags": [{"Key": "datus-test-run", "Value": "run"}, {"Key": "cost-center", "Value": "automation"}]},
        SecretNotFound() if physically_deleted else {"ARN": "owned-arn"},
    ]
    result = delete_owned_secret(secret_client, "owned-arn", "run")
    assert result == {
        "arn": "owned-arn",
        "state": "deleted" if physically_deleted else "deletion_requested",
        "deletion_date": "2026-01-01T00:00:00+00:00",
    }
    secret_client.delete_secret.assert_called_once_with(SecretId="owned-arn", ForceDeleteWithoutRecovery=True)


@pytest.mark.parametrize("tags", [[], [{"Key": "datus-test-run", "Value": "another-run"}]])
def test_secret_without_matching_ownership_is_preserved(secret_client, tags):
    secret_client.describe_secret.return_value = {"Tags": tags}
    with pytest.raises(AssertionError, match="matching ownership"):
        delete_owned_secret(secret_client, "owned-arn", "run")
    secret_client.delete_secret.assert_not_called()


def test_delete_api_failure_is_not_reported_as_cleanup_success(secret_client):
    secret_client.describe_secret.return_value = {"Tags": [{"Key": "datus-test-run", "Value": "run"}]}
    secret_client.delete_secret.side_effect = RuntimeError("DeleteSecret denied")
    with pytest.raises(RuntimeError, match="DeleteSecret denied"):
        delete_owned_secret(secret_client, "owned-arn", "run")
