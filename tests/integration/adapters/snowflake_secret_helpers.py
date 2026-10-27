"""Assertions and ownership-scoped cleanup for Snowflake cloud acceptance."""


def identifier_name(value: str) -> str:
    """Resolve an unquoted identifier or unescape an explicitly quoted name."""
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1].replace('""', '"')
    return value.upper()


def delete_owned_secret(client, arn: str, run_id: str) -> dict[str, str]:
    """Verify ownership and record accepted asynchronous deletion separately from absence."""
    tags = {tag["Key"]: tag["Value"] for tag in client.describe_secret(SecretId=arn).get("Tags", [])}
    assert tags.get("datus-test-run") == run_id, "Refusing to delete a Secret without matching ownership"
    response = client.delete_secret(SecretId=arn, ForceDeleteWithoutRecovery=True)
    assert response["ARN"] == arn and response.get("DeletionDate") is not None
    result = {"arn": arn, "state": "deletion_requested", "deletion_date": response["DeletionDate"].isoformat()}
    try:
        client.describe_secret(SecretId=arn)
    except client.exceptions.ResourceNotFoundException:
        result["state"] = "deleted"
    return result
