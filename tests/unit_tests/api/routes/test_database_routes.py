# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/api/routes/database_routes.py — list_catalogs endpoint."""

from typing import Optional
from unittest.mock import ANY, MagicMock, patch

import pytest

from datus.api.models.base_models import Result
from datus.api.models.database_models import DatabaseInfo, DatabasesData, ListDatabasesData, ListDatabasesInput
from datus.api.routes.database_routes import _DB_IO_TIMEOUT, list_catalogs


def _make_db_info(name: str = "main") -> DatabaseInfo:
    return DatabaseInfo(
        name=name,
        uri=f"sqlite:///{name}.db",
        type="sqlite",
        current=True,
        connection_status="connected",
    )


def _make_svc(
    list_databases_return: Optional[Result[ListDatabasesData]] = None,
    current_datasource: str = "default_ds",
) -> MagicMock:
    svc = MagicMock()
    svc.datasource.current_datasource = current_datasource
    if list_databases_return is not None:
        svc.datasource.list_databases.return_value = list_databases_return
    return svc


async def _timeout_wait_for(awaitable, timeout):
    """Async stub for asyncio.wait_for that closes the awaitable before raising TimeoutError."""
    if hasattr(awaitable, "close"):
        awaitable.close()
    raise TimeoutError


async def _call(
    svc: MagicMock,
    datasource_id: str = "",
    catalog_name: Optional[str] = None,
    database_name: str = "",
    schema_name: str = "",
    include_sys_schemas: bool = False,
) -> Result[DatabasesData]:
    """Call list_catalogs with explicit defaults to bypass FastAPI Query() object resolution."""
    return await list_catalogs(
        svc,
        datasource_id=datasource_id,
        catalog_name=catalog_name,
        database_name=database_name,
        schema_name=schema_name,
        include_sys_schemas=include_sys_schemas,
    )


class TestListCatalogs:
    """list_catalogs wraps list_databases in a thread, maps to DatabasesData, and handles timeout."""

    @pytest.mark.asyncio
    async def test_success_returns_databases_data(self):
        db = _make_db_info("main")
        list_result = Result[ListDatabasesData](
            success=True,
            data=ListDatabasesData(databases=[db], total_count=1, current_database="main"),
        )
        svc = _make_svc(list_databases_return=list_result)

        result = await _call(svc)

        assert result.success is True
        assert isinstance(result.data, DatabasesData)
        assert len(result.data.databases) == 1
        assert result.data.databases[0].name == "main"

    @pytest.mark.asyncio
    async def test_success_empty_list(self):
        list_result = Result[ListDatabasesData](
            success=True,
            data=ListDatabasesData(databases=[], total_count=0, current_database=None),
        )
        svc = _make_svc(list_databases_return=list_result)

        result = await _call(svc)

        assert result.success is True
        assert isinstance(result.data, DatabasesData)
        assert result.data.databases == []

    @pytest.mark.asyncio
    async def test_timeout_returns_request_timeout_error(self):
        svc = _make_svc()

        with patch("datus.api.routes.database_routes.asyncio.wait_for", side_effect=_timeout_wait_for) as mock_wf:
            result = await _call(svc)

        assert result.success is False
        assert result.errorCode == "REQUEST_TIMEOUT"
        assert result.errorMessage == "Datasource query timed out"
        mock_wf.assert_called_once_with(ANY, timeout=_DB_IO_TIMEOUT)

    @pytest.mark.asyncio
    async def test_timeout_result_type_is_result(self):
        svc = _make_svc()

        with patch("datus.api.routes.database_routes.asyncio.wait_for", side_effect=_timeout_wait_for) as mock_wf:
            result = await _call(svc)

        assert isinstance(result, Result)
        assert result.data is None
        mock_wf.assert_called_once_with(ANY, timeout=_DB_IO_TIMEOUT)

    @pytest.mark.asyncio
    async def test_service_error_propagates_error_code(self):
        list_result = Result[ListDatabasesData](
            success=False,
            errorCode="DATASOURCE_NOT_FOUND",
            errorMessage="Datasource not found",
        )
        svc = _make_svc(list_databases_return=list_result)

        result = await _call(svc)

        assert result.success is False
        assert result.errorCode == "DATASOURCE_NOT_FOUND"
        assert result.errorMessage == "Datasource not found"

    @pytest.mark.asyncio
    async def test_success_true_but_data_none_returns_error(self):
        list_result = Result[ListDatabasesData](success=True, data=None)
        svc = _make_svc(list_databases_return=list_result)

        result = await _call(svc)

        assert result.success is False

    @pytest.mark.asyncio
    async def test_uses_current_datasource_when_datasource_id_empty(self):
        list_result = Result[ListDatabasesData](
            success=True,
            data=ListDatabasesData(databases=[], total_count=0, current_database=None),
        )
        svc = _make_svc(list_databases_return=list_result, current_datasource="my_ds")

        await _call(svc, datasource_id="")

        call_arg = svc.datasource.list_databases.call_args[0][0]
        assert isinstance(call_arg, ListDatabasesInput)
        assert call_arg.datasource_id == "my_ds"

    @pytest.mark.asyncio
    async def test_uses_explicit_datasource_id_when_provided(self):
        list_result = Result[ListDatabasesData](
            success=True,
            data=ListDatabasesData(databases=[], total_count=0, current_database=None),
        )
        svc = _make_svc(list_databases_return=list_result, current_datasource="other_ds")

        await _call(svc, datasource_id="explicit_ds")

        call_arg = svc.datasource.list_databases.call_args[0][0]
        assert isinstance(call_arg, ListDatabasesInput)
        assert call_arg.datasource_id == "explicit_ds"

    @pytest.mark.asyncio
    async def test_multiple_databases_all_returned(self):
        dbs = [_make_db_info(f"db_{i}") for i in range(3)]
        list_result = Result[ListDatabasesData](
            success=True,
            data=ListDatabasesData(databases=dbs, total_count=3, current_database="db_0"),
        )
        svc = _make_svc(list_databases_return=list_result)

        result = await _call(svc)

        assert result.success is True
        assert len(result.data.databases) == 3
        assert result.data.databases[0].name == "db_0"
        assert result.data.databases[2].name == "db_2"
