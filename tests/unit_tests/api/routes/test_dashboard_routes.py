from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from datus.api.models.dashboard_models import DashboardQueryRequest
from datus.api.routes.dashboard_routes import run_dashboard_query


@pytest.mark.asyncio
async def test_run_dashboard_query_forwards_request_policy_context():
    expected = object()
    dashboard = SimpleNamespace(run_query=AsyncMock(return_value=expected))
    svc = SimpleNamespace(
        agent_config=SimpleNamespace(project_root="/project"),
        dashboard=dashboard,
    )
    policy_context = {"row_filter": {"access_mode": "unrestricted"}}
    ctx = SimpleNamespace(policy_context=policy_context)
    body = DashboardQueryRequest(dashboard_slug="sales", query_slug="by_region", params={"region": "APAC"})

    result = await run_dashboard_query(body, svc, ctx)

    assert result is expected
    assert dashboard.run_query.await_args.kwargs["policy_context"] is policy_context
