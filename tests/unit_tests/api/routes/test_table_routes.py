# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/api/routes/table_routes.py — table endpoints."""

from unittest.mock import MagicMock

import pytest

from datus.api.models.base_models import Result
from datus.api.models.table_models import (
    ColumnInfo,
    GetTablesColumnsData,
    GetTablesColumnsInput,
    TableColumns,
)
from datus.api.routes.table_routes import get_table_detail, get_tables_columns


class TestGetTableDetail:
    @pytest.mark.asyncio
    async def test_delegates_to_service(self):
        svc = MagicMock()
        svc.datasource.get_table_schema.return_value = Result(success=True)

        result = await get_table_detail(svc, table="db.public.orders")

        assert result.success is True
        svc.datasource.get_table_schema.assert_called_once_with("db.public.orders")


class TestGetTablesColumns:
    @pytest.mark.asyncio
    async def test_delegates_to_service_with_tables(self):
        svc = MagicMock()
        data = GetTablesColumnsData(
            tables=[TableColumns(table="db.public.orders", columns=[ColumnInfo(name="id", type="INT", nullable=False)])]
        )
        svc.datasource.get_tables_columns.return_value = Result(success=True, data=data)

        request = GetTablesColumnsInput(tables=["db.public.orders", "db.public.users"])
        result = await get_tables_columns(request, svc)

        assert result.success is True
        assert [t.table for t in result.data.tables] == ["db.public.orders"]
        svc.datasource.get_tables_columns.assert_called_once_with(["db.public.orders", "db.public.users"])
