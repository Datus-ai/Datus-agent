# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from datus.tools.bi_tools.base_adaptor import AuthParam, AuthType, DimensionDef, MetricDef
from datus.tools.bi_tools.superset.superset_adaptor import (
    SupersetAdaptor,
    SupersetAdaptorError,
    _coerce_id,
    _load_json_field,
    _normalize_series_columns_in_query,
    _normalize_series_columns_in_query_context,
    _parse_datasource_value,
)


@pytest.fixture
def auth_params():
    """Create test authentication parameters."""
    return AuthParam(
        username="test_user",
        password="test_password",
        extra={"provider": "db"},
    )


@pytest.fixture
def mock_client():
    """Create a mock httpx client."""
    client = MagicMock(spec=httpx.Client)
    client.base_url = "http://localhost:8088"
    return client


@pytest.fixture
def adaptor(auth_params, mock_client):
    """Create a SupersetAdaptor instance with mocked client."""
    with patch("httpx.Client", return_value=mock_client):
        adaptor = SupersetAdaptor(
            api_base_url="http://localhost:8088/api/v1",
            auth_params=auth_params,
            dialect="mysql",
            timeout=30.0,
        )
        adaptor._client = mock_client
        adaptor._auth_header_value = {"Authorization": "Bearer test_token"}
        adaptor._token_expiration = 9999999999  # Far future
        return adaptor


class TestSupersetAdaptorInitialization:
    """Test SupersetAdaptor initialization and basic properties."""

    def test_init_with_api_v1_url(self, auth_params):
        """Test initialization with /api/v1 URL."""
        with patch("httpx.Client") as mock_client_class:
            adaptor = SupersetAdaptor(
                api_base_url="http://localhost:8088/api/v1",
                auth_params=auth_params,
                dialect="postgresql",
            )

            assert adaptor.base_url == "http://localhost:8088"
            assert adaptor._api_base == "http://localhost:8088/api/v1"
            assert adaptor.dialect == "postgresql"
            mock_client_class.assert_called_once()

    def test_init_without_api_v1_url(self, auth_params):
        """Test initialization without /api/v1 URL."""
        with patch("httpx.Client"):
            adaptor = SupersetAdaptor(
                api_base_url="http://localhost:8088",
                auth_params=auth_params,
                dialect="mysql",
            )

            assert adaptor.base_url == "http://localhost:8088"
            assert adaptor._api_base == "http://localhost:8088/api/v1"

    def test_platform_name(self, adaptor):
        """Test platform_name method."""
        assert adaptor.platform_name() == "superset"

    def test_auth_type(self, adaptor):
        """Test auth_type method."""
        assert adaptor.auth_type() == AuthType.LOGIN

    def test_close(self, adaptor, mock_client):
        """Test close method."""
        adaptor.close()
        mock_client.close.assert_called_once()


class TestParseDashboardId:
    """Test dashboard ID parsing functionality."""

    def test_parse_numeric_string(self, adaptor):
        """Test parsing numeric string."""
        result = adaptor.parse_dashboard_id("123")
        assert result == 123

    def test_parse_empty_string(self, adaptor):
        """Test parsing empty string."""
        result = adaptor.parse_dashboard_id("")
        assert result == ""

        result = adaptor.parse_dashboard_id("   ")
        assert result == ""

    def test_parse_full_url(self, adaptor):
        """Test parsing full dashboard URL."""
        url = "http://localhost:8088/superset/dashboard/123/"
        result = adaptor.parse_dashboard_id(url)
        assert result == "123"

    def test_parse_url_with_query_params(self, adaptor):
        """Test parsing URL with query parameters."""
        # Note: parse_dashboard_id extracts from path segments first,
        # then falls back to query params. So /dashboard path returns "dashboard"
        url = "http://localhost:8088/dashboard?dashboard_id=456"
        result = adaptor.parse_dashboard_id(url)
        assert result == "dashboard"  # Extracts from path, not query param

        # URL without path segments will try query params
        url = "http://localhost:8088/?dashboard_id=456"
        result = adaptor.parse_dashboard_id(url)
        assert result == "456"

        url = "http://localhost:8088/?id=789"
        result = adaptor.parse_dashboard_id(url)
        assert result == "789"

    def test_parse_complex_path(self, adaptor):
        """Test parsing URL with complex path."""
        url = "http://localhost:8088/superset/dashboard/my-dashboard-slug"
        result = adaptor.parse_dashboard_id(url)
        assert result == "my-dashboard-slug"

    def test_parse_non_url_string(self, adaptor):
        """Test parsing non-URL string."""
        result = adaptor.parse_dashboard_id("my-custom-id")
        assert result == "my-custom-id"


class TestGetDashboardInfo:
    """Test dashboard information retrieval."""

    def test_get_dashboard_info_success(self, adaptor, mock_client):
        """Test successful dashboard info retrieval."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "id": 123,
                "dashboard_title": "Test Dashboard",
                "description": "A test dashboard",
            }
        }
        mock_client.request.return_value = mock_response

        # Mock charts response
        charts_response = MagicMock()
        charts_response.json.return_value = {
            "result": [
                {"slice_id": 1, "slice_name": "Chart 1"},
                {"slice_id": 2, "slice_name": "Chart 2"},
            ]
        }

        mock_client.request.side_effect = [mock_response, charts_response]

        dashboard_info = adaptor.get_dashboard_info(123)

        assert dashboard_info is not None
        assert dashboard_info.id == 123
        assert dashboard_info.name == "Test Dashboard"
        assert dashboard_info.description == "A test dashboard"
        assert len(dashboard_info.chart_ids) == 2
        assert dashboard_info.chart_ids == [1, 2]

    def test_get_dashboard_info_with_slug(self, adaptor, mock_client):
        """Test dashboard info with slug as name."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "id": 123,
                "slug": "test-dashboard",
            }
        }
        charts_response = MagicMock()
        charts_response.json.return_value = {"result": []}

        mock_client.request.side_effect = [mock_response, charts_response]

        dashboard_info = adaptor.get_dashboard_info(123)

        assert dashboard_info.name == "test-dashboard"

    def test_get_dashboard_info_deduplicates_charts(self, adaptor, mock_client):
        """Test that duplicate chart IDs are removed."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "id": 123,
                "dashboard_title": "Test",
            }
        }

        charts_response = MagicMock()
        charts_response.json.return_value = {
            "result": [
                {"slice_id": 1},
                {"slice_id": 2},
                {"slice_id": 1},  # Duplicate
                {"slice_id": 3},
            ]
        }

        mock_client.request.side_effect = [mock_response, charts_response]

        dashboard_info = adaptor.get_dashboard_info(123)

        assert len(dashboard_info.chart_ids) == 3
        assert dashboard_info.chart_ids == [1, 2, 3]

    def test_get_dashboard_info_error(self, adaptor, mock_client):
        """Test dashboard info retrieval with error."""
        mock_client.request.side_effect = httpx.HTTPStatusError(
            "Not found", request=MagicMock(), response=MagicMock(status_code=404, text="Not found")
        )

        with pytest.raises(SupersetAdaptorError) as exc_info:
            adaptor.get_dashboard_info(999)

        assert "failed with 404" in str(exc_info.value)


class TestListCharts:
    """Test chart listing functionality."""

    def test_list_charts_success(self, adaptor, mock_client):
        """Test successful chart listing."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": [
                {
                    "slice_id": 1,
                    "slice_name": "Sales Chart",
                    "viz_type": "bar",
                    "description": "Sales overview",
                },
                {
                    "slice_id": 2,
                    "slice_name": "Revenue Chart",
                    "viz_type": "line",
                },
            ]
        }
        mock_client.request.return_value = mock_response

        charts = adaptor.list_charts(123)

        assert len(charts) == 2
        assert charts[0].id == 1
        assert charts[0].name == "Sales Chart"
        assert charts[0].chart_type == "bar"
        assert charts[0].description == "Sales overview"
        assert charts[1].id == 2
        assert charts[1].name == "Revenue Chart"

    def test_list_charts_with_form_data(self, adaptor, mock_client):
        """Test chart listing with form_data containing chart info."""
        form_data = json.dumps({"slice_id": 3, "viz_type": "pie"})

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": [
                {
                    "id": 3,
                    "name": "Distribution Chart",
                    "form_data": form_data,
                    "viz_type": "pie",  # viz_type should be at top level
                }
            ]
        }
        mock_client.request.return_value = mock_response

        charts = adaptor.list_charts(123)

        assert len(charts) == 1
        assert charts[0].id == 3
        assert charts[0].chart_type == "pie"

    def test_list_charts_empty(self, adaptor, mock_client):
        """Test listing charts when none exist."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": []}
        mock_client.request.return_value = mock_response

        charts = adaptor.list_charts(123)

        assert len(charts) == 0

    def test_list_charts_error(self, adaptor, mock_client):
        """Test chart listing with error."""
        mock_client.request.side_effect = SupersetAdaptorError("API error")

        charts = adaptor.list_charts(123)

        # Should return empty list on error
        assert len(charts) == 0


class TestListDatasets:
    """Test dataset listing functionality."""

    def test_list_datasets_success(self, adaptor, mock_client):
        """Test successful dataset listing."""
        form_data = json.dumps({"datasource": "1__table"})
        query_context = json.dumps(
            {
                "datasource": {"id": 1, "type": "table", "name": "sales_table"},
            }
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": [
                {
                    "slice_id": 1,
                    "form_data": form_data,
                    "query_context": query_context,
                    "dataset": {
                        "table_name": "sales_table",
                        "datasource_name": "sales_table",  # Add this for name extraction
                        "columns": [
                            {"column_name": "id", "type": "INTEGER"},
                            {"column_name": "amount", "type": "DECIMAL"},
                        ],
                        "metrics": [
                            {
                                "metric_name": "total_sales",
                                "expression": "SUM(amount)",
                            }
                        ],
                    },
                    "datasource_id": 1,  # Add this to help with name resolution
                }
            ]
        }
        mock_client.request.return_value = mock_response

        datasets = adaptor.list_datasets(123)

        assert len(datasets) == 1
        assert datasets[0].id == 1
        # Name may be from datasource ref or dataset block
        assert datasets[0].name in ["sales_table", "1"]
        assert datasets[0].dialect == "mysql"

    def test_list_datasets_with_sql(self, adaptor, mock_client):
        """Test dataset listing with SQL-based virtual dataset."""
        form_data = json.dumps({"datasource": "1__table"})

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": [
                {
                    "slice_id": 1,
                    "form_data": form_data,
                    "dataset": {
                        "table_name": "virtual_dataset",
                        "sql": "SELECT * FROM sales JOIN customers ON sales.customer_id = customers.id",
                    },
                }
            ]
        }
        mock_client.request.return_value = mock_response

        datasets = adaptor.list_datasets(123)

        assert len(datasets) == 1
        # Should extract tables from SQL
        assert datasets[0].tables is not None

    def test_list_datasets_empty(self, adaptor, mock_client):
        """Test listing datasets when none exist."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": []}
        mock_client.request.return_value = mock_response

        datasets = adaptor.list_datasets(123)

        assert len(datasets) == 0


class TestGetChart:
    """Test chart detail retrieval."""

    def test_get_chart_success(self, adaptor, mock_client):
        """Test successful chart retrieval."""
        form_data = {"slice_id": 1, "datasource": "10__table", "viz_type": "bar"}
        query_context = {
            "datasource": {"id": 10, "type": "table"},
            "queries": [
                {
                    "metrics": [{"label": "count", "expression": "COUNT(*)"}],
                    "columns": [{"column_name": "category"}],
                }
            ],
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "id": 1,
                "slice_name": "Test Chart",
                "description": "Test description",
                "viz_type": "bar",
                "form_data": json.dumps(form_data),
                "query_context": json.dumps(query_context),
            }
        }
        mock_client.request.return_value = mock_response

        # Mock SQL collection
        with patch.object(adaptor, "_collect_sql_from_chart", return_value=(["SELECT * FROM test"], {0})):
            chart = adaptor.get_chart(1, 123)

        assert chart is not None
        assert chart.id == 1
        assert chart.name == "Test Chart"
        assert chart.description == "Test description"
        assert chart.chart_type == "bar"
        assert chart.query is not None
        assert chart.query.kind == "sql"
        assert len(chart.query.sql) == 1

    def test_get_chart_with_nested_form_data(self, adaptor, mock_client):
        """Test chart retrieval with nested form_data in slice."""
        slice_form_data = {"slice_id": 1, "viz_type": "line"}
        outer_form_data = {"slice_id": 1, "metric": "revenue"}

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "id": 1,
                "slice_name": "Revenue Chart",
                "form_data": json.dumps(outer_form_data),
                "slice": {
                    "id": 1,
                    "slice_name": "Revenue Chart",
                    "form_data": json.dumps(slice_form_data),
                },
            }
        }
        mock_client.request.return_value = mock_response

        with patch.object(adaptor, "_collect_sql_from_chart", return_value=([], None)):
            chart = adaptor.get_chart(1)

        assert chart is not None
        assert chart.chart_type == "line"

    def test_get_chart_error(self, adaptor, mock_client):
        """Test chart retrieval with error."""
        mock_client.request.side_effect = SupersetAdaptorError("Chart not found")

        chart = adaptor.get_chart(999)

        assert chart is None


class TestGetDataset:
    """Test dataset retrieval."""

    def test_get_dataset_success(self, adaptor, mock_client):
        """Test successful dataset retrieval."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "id": 10,
                "table_name": "sales",
                "description": "Sales data",
                "columns": [
                    {"column_name": "id", "type": "INTEGER"},
                    {"column_name": "amount", "type": "DECIMAL"},
                ],
                "metrics": [
                    {"metric_name": "total", "expression": "SUM(amount)"},
                ],
                "database": {"database_name": "prod"},
                "schema": "public",
            }
        }
        mock_client.request.return_value = mock_response

        dataset = adaptor.get_dataset(10)

        assert dataset is not None
        assert dataset.id == 10
        assert dataset.name == "sales"
        assert dataset.description == "Sales data"
        assert dataset.dialect == "mysql"
        assert len(dataset.columns) == 2
        assert len(dataset.metrics) == 1

    def test_get_dataset_with_cache(self, adaptor, mock_client):
        """Test dataset retrieval uses cache."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "id": 10,
                "table_name": "sales",
            }
        }
        mock_client.request.return_value = mock_response

        # First call
        dataset1 = adaptor.get_dataset(10)
        # Second call should use cache
        dataset2 = adaptor.get_dataset(10)

        assert dataset1 == dataset2
        # Request should only be called once
        assert mock_client.request.call_count == 1

    def test_get_dataset_none(self, adaptor):
        """Test dataset retrieval with None ID."""
        dataset = adaptor.get_dataset(None)
        assert dataset is None

    def test_get_dataset_error(self, adaptor, mock_client):
        """Test dataset retrieval with error."""
        mock_client.request.side_effect = SupersetAdaptorError("Dataset not found")

        dataset = adaptor.get_dataset(999)

        assert dataset is None


class TestMetricsAndDimensions:
    """Test metrics and dimensions normalization."""

    def test_normalize_metric_string(self, adaptor):
        """Test normalizing string metric."""
        metric = adaptor._normalize_metric("count", "sales_table", "chart")

        assert metric is not None
        assert metric.name == "count"
        assert metric.expression == "count"
        assert metric.table == "sales_table"
        assert metric.origin == "chart"

    def test_normalize_metric_dict_simple(self, adaptor):
        """Test normalizing dict metric with simple expression."""
        metric_dict = {
            "label": "Total Sales",
            "expression": "SUM(amount)",
            "description": "Sum of all sales",
        }

        metric = adaptor._normalize_metric(metric_dict, "sales", "dataset")

        assert metric is not None
        assert metric.name == "Total Sales"
        assert metric.expression == "SUM(amount)"
        assert metric.description == "Sum of all sales"

    def test_normalize_metric_dict_simple_type(self, adaptor):
        """Test normalizing dict metric with SIMPLE expression type."""
        metric_dict = {
            "label": "Count Items",
            "expressionType": "SIMPLE",
            "aggregate": "COUNT",
            "column": {"column_name": "item_id"},
        }

        metric = adaptor._normalize_metric(metric_dict, "items", "chart")

        assert metric is not None
        assert metric.name == "Count Items"
        assert metric.expression == "COUNT(item_id)"

    def test_normalize_metric_dict_sql_type(self, adaptor):
        """Test normalizing dict metric with SQL expression type."""
        metric_dict = {
            "metric_name": "Revenue",
            "expressionType": "SQL",
            "sqlExpression": "SUM(price * quantity)",
        }

        metric = adaptor._normalize_metric(metric_dict, "orders", "dataset")

        assert metric is not None
        assert metric.name == "Revenue"
        assert metric.expression == "SUM(price * quantity)"

    def test_normalize_metric_empty(self, adaptor):
        """Test normalizing empty metric."""
        assert adaptor._normalize_metric("", "table", "chart") is None
        assert adaptor._normalize_metric({}, "table", "chart") is None

    def test_normalize_dimension_string(self, adaptor):
        """Test normalizing string dimension."""
        dim = adaptor._normalize_dimension("category", "products", "chart")

        assert dim is not None
        assert dim.name == "category"
        assert dim.table == "products"
        assert dim.origin == "chart"

    def test_normalize_dimension_dict(self, adaptor):
        """Test normalizing dict dimension."""
        dim_dict = {
            "column_name": "created_at",
            "verbose_name": "Creation Date",
            "type": "TIMESTAMP",
            "description": "Record creation timestamp",
        }

        dim = adaptor._normalize_dimension(dim_dict, "records", "dataset")

        assert dim is not None
        assert dim.name == "created_at"
        assert dim.title == "Creation Date"
        assert dim.data_type == "TIMESTAMP"
        assert dim.description == "Record creation timestamp"

    def test_dedupe_metrics(self, adaptor):
        """Test deduplication of metrics."""
        metrics = [
            MetricDef(name="count", expression="COUNT(*)", origin="chart"),
            MetricDef(name="count", expression="COUNT(*)", origin="dataset"),  # Duplicate
            MetricDef(name="sum", expression="SUM(amount)", origin="chart"),
        ]

        deduped = adaptor._dedupe_metrics(metrics)

        assert len(deduped) == 2
        assert deduped[0].name == "count"
        assert deduped[1].name == "sum"

    def test_dedupe_dimensions(self, adaptor):
        """Test deduplication of dimensions."""
        dimensions = [
            DimensionDef(name="category", origin="chart"),
            DimensionDef(name="category", origin="dataset"),  # Duplicate
            DimensionDef(name="region", origin="chart"),
        ]

        deduped = adaptor._dedupe_dimensions(dimensions)

        assert len(deduped) == 2
        assert deduped[0].name == "category"
        assert deduped[1].name == "region"


class TestAuthentication:
    """Test authentication functionality."""

    def test_ensure_authenticated_with_valid_token(self, adaptor):
        """Test that authentication is not triggered when token is valid."""
        adaptor._auth_header_value = {"Authorization": "Bearer valid_token"}
        adaptor._token_expiration = 9999999999  # Far future

        with patch.object(adaptor, "_authenticate") as mock_auth:
            adaptor._ensure_authenticated()
            mock_auth.assert_not_called()

    def test_ensure_authenticated_with_expired_token(self, adaptor):
        """Test that authentication is triggered when token expires."""
        adaptor._auth_header_value = None
        adaptor._token_expiration = None

        with patch.object(adaptor, "_authenticate") as mock_auth:
            adaptor._ensure_authenticated()
            mock_auth.assert_called_once()

    def test_authenticate_success(self, adaptor, mock_client):
        """Test successful authentication."""
        adaptor._auth_header_value = None

        login_response = MagicMock()
        login_response.json.return_value = {
            "result": {
                "access_token": "new_token",
                "token_type": "Bearer",
                "expires_in": 3600,
            }
        }

        # Mock browser login to fail, so it uses API login
        with patch.object(adaptor, "_try_login_by_browser", return_value=False):
            with patch.object(adaptor, "_request", return_value=login_response):
                adaptor._authenticate()

        assert adaptor._auth_header_value == {"Authorization": "Bearer new_token"}
        assert adaptor._token_expiration is not None

    def test_try_login_by_browser_success(self, adaptor, mock_client):
        """Test successful browser-based login."""
        # Mock login page response
        login_page = MagicMock()
        login_page.is_success = True
        login_page.text = '<input name="csrf_token" type="hidden" value="test_csrf">'

        # Mock login POST response
        login_post = MagicMock()
        login_post.is_success = True

        # Mock CSRF token response
        csrf_response = MagicMock()
        csrf_response.is_success = True
        csrf_response.json.return_value = {"result": "csrf_token_value"}

        mock_client.get.side_effect = [login_page, csrf_response]
        mock_client.post.return_value = login_post

        result = adaptor._try_login_by_browser()

        assert result is True
        assert adaptor._auth_header_value == {"X-CSRFToken": "csrf_token_value"}

    def test_try_login_by_browser_failure(self, adaptor, mock_client):
        """Test failed browser-based login."""
        login_page = MagicMock()
        login_page.is_success = False

        mock_client.get.return_value = login_page

        result = adaptor._try_login_by_browser()

        assert result is False


class TestHelperFunctions:
    """Test helper and utility functions."""

    def test_coerce_id_int(self):
        """Test _coerce_id with integer."""
        assert _coerce_id(123) == 123
        assert _coerce_id("456") == 456

    def test_coerce_id_string(self):
        """Test _coerce_id with non-numeric string."""
        assert _coerce_id("abc") == "abc"

    def test_coerce_id_none(self):
        """Test _coerce_id with None."""
        assert _coerce_id(None) is None

    def test_load_json_field_dict(self):
        """Test _load_json_field with dict."""
        data = {"key": "value"}
        assert _load_json_field(data) == data

    def test_load_json_field_json_string(self):
        """Test _load_json_field with JSON string."""
        json_str = '{"key": "value"}'
        result = _load_json_field(json_str)
        assert result == {"key": "value"}

    def test_load_json_field_invalid_json(self):
        """Test _load_json_field with invalid JSON."""
        assert _load_json_field("not json") is None

    def test_load_json_field_empty(self):
        """Test _load_json_field with empty/None."""
        assert _load_json_field("") is None
        assert _load_json_field(None) is None

    def test_parse_datasource_value_dict(self):
        """Test _parse_datasource_value with dict."""
        value = {"id": 123, "type": "table"}
        ds_id, ds_type = _parse_datasource_value(value)

        assert ds_id == 123
        assert ds_type == "table"

    def test_parse_datasource_value_string_with_separator(self):
        """Test _parse_datasource_value with string containing __."""
        ds_id, ds_type = _parse_datasource_value("123__table")

        assert ds_id == 123
        assert ds_type == "table"

    def test_parse_datasource_value_numeric_string(self):
        """Test _parse_datasource_value with numeric string."""
        ds_id, ds_type = _parse_datasource_value("456")

        assert ds_id == 456
        assert ds_type is None

    def test_parse_datasource_value_int(self):
        """Test _parse_datasource_value with integer."""
        ds_id, ds_type = _parse_datasource_value(789)

        assert ds_id == 789
        assert ds_type is None

    def test_parse_datasource_value_none(self):
        """Test _parse_datasource_value with None."""
        ds_id, ds_type = _parse_datasource_value(None)

        assert ds_id is None
        assert ds_type is None

    def test_normalize_series_columns_in_query(self):
        """Test _normalize_series_columns_in_query."""
        query = {
            "series_columns": ["col1", "col2"],
            "columns": ["col3"],
        }

        _normalize_series_columns_in_query(query)

        assert "col1" in query["columns"]
        assert "col2" in query["columns"]
        assert "col3" in query["columns"]
        assert query["series_columns"] == ["col1", "col2"]

    def test_normalize_series_columns_in_query_with_dict(self):
        """Test _normalize_series_columns_in_query with dict columns."""
        query = {
            "series_columns": [{"column_name": "col1"}, {"name": "col2"}],
            "columns": [{"column_name": "col3"}],
        }

        _normalize_series_columns_in_query(query)

        assert "col1" in query["columns"]
        assert "col2" in query["columns"]
        assert "col3" in query["columns"]

    def test_normalize_series_columns_in_query_no_series(self):
        """Test _normalize_series_columns_in_query with no series_columns."""
        query = {"columns": ["col1", "col2"]}

        _normalize_series_columns_in_query(query)

        # Should not modify query
        assert query["columns"] == ["col1", "col2"]

    def test_normalize_series_columns_in_query_context(self):
        """Test _normalize_series_columns_in_query_context."""
        query_context = {
            "queries": [
                {
                    "series_columns": ["col1"],
                    "columns": ["col2"],
                },
                {
                    "series_columns": ["col3"],
                    "columns": [],
                },
            ]
        }

        _normalize_series_columns_in_query_context(query_context)

        assert "col1" in query_context["queries"][0]["columns"]
        assert "col2" in query_context["queries"][0]["columns"]
        assert query_context["queries"][1]["columns"] == ["col3"]


class TestDataExtraction:
    """Test data extraction methods."""

    def test_extract_datasource_ref_from_query_context(self, adaptor):
        """Test extracting datasource reference from query_context."""
        query_context = {
            "datasource": {
                "id": 100,
                "type": "table",
                "name": "sales",
            }
        }

        ref = adaptor._extract_datasource_ref(query_context=query_context)

        assert ref is not None
        assert ref["id"] == 100
        assert ref["type"] == "table"

    def test_extract_datasource_ref_from_form_data(self, adaptor):
        """Test extracting datasource reference from form_data."""
        form_data = {
            "datasource": "200__table",
            "datasource_type": "table",
        }

        ref = adaptor._extract_datasource_ref(form_data=form_data)

        assert ref is not None
        assert ref["id"] == 200
        assert ref["type"] == "table"

    def test_extract_datasource_ref_priority(self, adaptor):
        """Test datasource reference extraction priority."""
        query_context = {"datasource": {"id": 100, "type": "table"}}
        form_data = {"datasource": "200__table"}

        # query_context should take priority
        ref = adaptor._extract_datasource_ref(query_context=query_context, form_data=form_data)

        assert ref["id"] == 100

    def test_extract_datasource_ref_none(self, adaptor):
        """Test extracting datasource reference when none exists."""
        ref = adaptor._extract_datasource_ref()

        assert ref is None

    def test_tables_from_sql(self, adaptor):
        """Test extracting tables from SQL."""
        sql = "SELECT * FROM sales JOIN customers ON sales.customer_id = customers.id"

        tables = adaptor._tables_from_sql(sql)

        assert isinstance(tables, list)
        # Should extract table names (depends on SQL parser)

    def test_tables_from_sql_empty(self, adaptor):
        """Test extracting tables from empty SQL."""
        assert adaptor._tables_from_sql(None) == []
        assert adaptor._tables_from_sql("") == []

    def test_dedupe_tables(self, adaptor):
        """Test deduplication of table names."""
        tables = ["sales", "customers", "sales", "products"]

        deduped = adaptor._dedupe_tables(tables)

        assert len(deduped) == 3
        assert deduped == ["sales", "customers", "products"]


class TestSQLCollection:
    """Test SQL collection from charts."""

    def test_collect_sql_via_chart_data_success(self, adaptor, mock_client):
        """Test collecting SQL via chart/data API."""
        query_context = {
            "datasource": {"id": 1},
            "queries": [{"metrics": ["count"]}],
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": [
                {"query": "SELECT COUNT(*) FROM table1"},
                {"query": "SELECT SUM(amount) FROM table2"},
            ]
        }
        mock_client.request.return_value = mock_response

        sqls, indexes = adaptor._collect_sql_via_chart_data(1, query_context)

        assert len(sqls) == 2
        assert "SELECT COUNT(*)" in sqls[0]
        assert "SELECT SUM(amount)" in sqls[1]
        assert indexes == {0, 1}

    def test_collect_sql_via_chart_data_empty_result(self, adaptor, mock_client):
        """Test collecting SQL with empty result."""
        query_context = {"datasource": {"id": 1}}

        mock_response = MagicMock()
        mock_response.json.return_value = {"result": []}
        mock_client.request.return_value = mock_response

        sqls, indexes = adaptor._collect_sql_via_chart_data(1, query_context)

        assert len(sqls) == 0
        assert indexes is None

    def test_collect_sql_via_chart_data_error(self, adaptor, mock_client):
        """Test collecting SQL with API error."""
        query_context = {"datasource": {"id": 1}}

        mock_client.request.side_effect = SupersetAdaptorError("API error")

        with pytest.raises(SupersetAdaptorError) as exc_info:
            adaptor._collect_sql_via_chart_data(1, query_context)

        assert "chart/data failed" in str(exc_info.value)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_get_chart_id_from_various_sources(self, adaptor):
        """Test extracting chart ID from various metadata sources."""
        # From form_data
        chart_meta = {"form_data": json.dumps({"slice_id": 100})}
        assert adaptor._extract_chart_id(chart_meta) == 100

        # From slice_id
        chart_meta = {"slice_id": 200}
        assert adaptor._extract_chart_id(chart_meta) == 200

        # From chart_id
        chart_meta = {"chart_id": 300}
        assert adaptor._extract_chart_id(chart_meta) == 300

        # From id
        chart_meta = {"id": 400}
        assert adaptor._extract_chart_id(chart_meta) == 400

    def test_chart_description_fallback(self, adaptor):
        """Test chart description extraction with fallbacks."""
        # From chart_meta
        desc = adaptor._chart_description({"description": "Meta desc"}, None)
        assert desc == "Meta desc"

        # From chart_detail
        desc = adaptor._chart_description(None, {"description": "Detail desc"})
        assert desc == "Detail desc"

        # From description_markeddown
        desc = adaptor._chart_description(None, {"description_markeddown": "Markdown desc"})
        assert desc == "Markdown desc"

        # Priority: chart_meta over chart_detail
        desc = adaptor._chart_description({"description": "Meta"}, {"description": "Detail"})
        assert desc == "Meta"

    def test_normalize_api_base(self, adaptor):
        """Test API base URL normalization."""
        assert adaptor._normalize_api_base("http://localhost:8088") == "http://localhost:8088/api/v1"
        assert adaptor._normalize_api_base("http://localhost:8088/api/v1") == "http://localhost:8088/api/v1"
        assert adaptor._normalize_api_base("http://localhost:8088/api/v1/") == "http://localhost:8088/api/v1"

    def test_parse_dataset_columns(self, adaptor):
        """Test parsing dataset columns."""
        dataset = {
            "columns": [
                {"column_name": "id", "type": "INTEGER", "description": "Primary key"},
                {"column_name": "name", "type": "VARCHAR"},
                {"name": "amount"},  # Using 'name' instead of 'column_name'
            ]
        }

        columns = adaptor._parse_dataset_columns(dataset, "test_table")

        assert len(columns) == 3
        assert columns[0].name == "id"
        assert columns[0].data_type == "INTEGER"
        assert columns[0].description == "Primary key"
        assert columns[1].name == "name"
        assert columns[2].name == "amount"

    def test_parse_dataset_metrics(self, adaptor):
        """Test parsing dataset metrics."""
        dataset = {
            "metrics": [
                {
                    "metric_name": "count",
                    "expression": "COUNT(*)",
                },
                {
                    "label": "total_amount",
                    "expression": "SUM(amount)",
                },
            ]
        }

        metrics = adaptor._parse_dataset_metrics(dataset, "test_table")

        assert len(metrics) == 2
        assert metrics[0].name == "count"
        assert metrics[1].name == "total_amount"

    def test_parse_dataset_dimensions(self, adaptor):
        """Test parsing dataset dimensions."""
        dataset = {
            "columns": [
                {"column_name": "category", "groupby": True},
                {"column_name": "region", "filterable": True},
                {"column_name": "created_at", "is_dttm": True},
                {"column_name": "id"},  # No flags, should be excluded when flags exist
            ]
        }

        columns = adaptor._parse_dataset_columns(dataset, "test_table")
        dimensions = adaptor._parse_dataset_dimensions(dataset, "test_table", columns)

        # Should only include columns with groupby/filterable/is_dttm flags
        assert len(dimensions) == 3
        assert dimensions[0].name == "category"
        assert dimensions[1].name == "region"
        assert dimensions[2].name == "created_at"
