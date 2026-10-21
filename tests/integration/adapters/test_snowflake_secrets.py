"""Live YAML -> DBManager -> AWS Secret -> native Snowflake login acceptance.

Opt in with ADAPTERS_SNOWFLAKE_SECRETS=1 and install datus-snowflake[aws]>=0.1.10.
Requires SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_WAREHOUSE,
SNOWFLAKE_PRIVATE_KEY_FILE, SNOWFLAKE_DATABASE and optional
SNOWFLAKE_PRIVATE_KEY_FILE_PWD, SNOWFLAKE_SCHEMA, SNOWFLAKE_ROLE. AWS uses the SDK chain,
DATUS_SECRET_AWS_REGION (default us-east-1), and optional DATUS_SECRET_AWS_PROFILE.
Creates one owned temporary AWS Secret and verifies its deletion. No Snowflake
users or keys are changed by this test. See adapters/README.md.
"""

import json
import os
import time
import uuid
from pathlib import Path

import pytest
import yaml

from tests.nightly_requirements import import_required, require_opt_in_env

require_opt_in_env("ADAPTERS_SNOWFLAKE_SECRETS", "tests/integration/adapters/README.md")
import_required("datus_snowflake", reason="Install datus-snowflake[aws]>=0.1.10")
boto3 = import_required("boto3", reason="Install datus-snowflake[aws]>=0.1.10")

from datus.configuration.agent_config import DbConfig  # noqa: E402
from datus.tools.db_tools.db_manager import DBManager  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.nightly]


def test_secret_profile_connects_from_yaml_and_reopens(tmp_path):
    required = (
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_WAREHOUSE",
        "SNOWFLAKE_PRIVATE_KEY_FILE",
        "SNOWFLAKE_DATABASE",
    )
    missing = [key for key in required if not os.getenv(key)]
    assert not missing, f"Required Snowflake configuration missing: {missing}"
    run_id = uuid.uuid4().hex
    region = os.getenv("DATUS_SECRET_AWS_REGION", "us-east-1")
    profile_name = os.getenv("DATUS_SECRET_AWS_PROFILE")
    client = boto3.Session(region_name=region, profile_name=profile_name).client("secretsmanager")
    payload = {
        "sf_user": os.environ["SNOWFLAKE_USER"],
        "sf_key": Path(os.environ["SNOWFLAKE_PRIVATE_KEY_FILE"]).read_text(),
    }
    mapping = {"username": "sf_user", "private_key": "sf_key"}
    if os.getenv("SNOWFLAKE_PRIVATE_KEY_FILE_PWD"):
        payload["sf_passphrase"] = os.environ["SNOWFLAKE_PRIVATE_KEY_FILE_PWD"]
        mapping["private_key_file_pwd"] = "sf_passphrase"
    arn = client.create_secret(
        Name=f"datus/cmp-secret-01/agent-{run_id}",
        SecretString=json.dumps(payload),
        Tags=[{"Key": "datus-test-run", "Value": run_id}],
    )["ARN"]
    manager = None
    try:
        profile = {
            "type": "snowflake",
            "account": "${SNOWFLAKE_ACCOUNT}",
            "warehouse": "${SNOWFLAKE_WAREHOUSE}",
            "auth_mode": "key_pair",
            "secret_ref": arn,
            "secret_key_mapping": mapping,
            "secret_cache_ttl_seconds": 300,
            "aws": {"region": region},
        }
        if profile_name:
            profile["aws"]["profile"] = profile_name
        for field in ("database", "schema", "role"):
            if os.getenv(f"SNOWFLAKE_{field.upper()}"):
                profile[field] = "${SNOWFLAKE_" + field.upper() + "}"
        path = tmp_path / "agent.yml"
        path.write_text(yaml.safe_dump({"agent": {"services": {"datasources": {"secret": profile}}}}))
        parsed = yaml.safe_load(path.read_text())["agent"]["services"]["datasources"]["secret"]
        manager = DBManager({"secret": DbConfig.filter_kwargs(DbConfig, parsed)})
        connector = manager.get_conn("secret")
        for _ in range(2):
            result = connector.execute_query(
                'SELECT CURRENT_USER() AS "user", CURRENT_WAREHOUSE() AS "warehouse", \'数据\' AS "text"',
                result_format="list",
            )
            assert result.success is True
            assert result.sql_return == [
                {"user": os.environ["SNOWFLAKE_USER"], "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"], "text": "数据"}
            ]
            connector.close()
        assert os.environ["SNOWFLAKE_DATABASE"] in connector.get_databases()
    finally:
        try:
            if manager is not None:
                manager.close()
        finally:
            try:
                tags = {tag["Key"]: tag["Value"] for tag in client.describe_secret(SecretId=arn)["Tags"]}
                assert tags == {"datus-test-run": run_id}
                client.delete_secret(SecretId=arn, ForceDeleteWithoutRecovery=True)
                for _ in range(60):
                    try:
                        client.describe_secret(SecretId=arn)
                    except client.exceptions.ResourceNotFoundException:
                        break
                    time.sleep(0.5)
                else:
                    pytest.fail("Owned Agent test Secret still exists after deletion")
            finally:
                client.close()
