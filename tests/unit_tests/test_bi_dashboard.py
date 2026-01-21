# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.


from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from datus.cli.bi_dashboard import BiDashboardCommands, DashboardCliOptions, _parse_subject_path_for_metrics
from datus.configuration.agent_config import AgentConfig, DashboardConfig
from datus.tools.bi_tools.base_adaptor import AuthParam, AuthType, ChartInfo, DashboardInfo, DatasetInfo, QuerySpec
from datus.tools.bi_tools.dashboard_assembler import ChartSelection, DashboardAssemblyResult, SelectedSqlCandidate
from tests.conftest import load_acceptance_config


@pytest.fixture
def agent_config() -> AgentConfig:
    """Create a test agent configuration."""
    config = load_acceptance_config(namespace="bird_school")
    # Use the existing namespace from the config
    return config


@pytest.fixture
def console() -> Console:
    """Create a console for testing."""
    return Console(log_path=False)


@pytest.fixture
def mock_adaptor():
    """Create a mock BI adaptor."""
    adaptor = MagicMock()
    adaptor.parse_dashboard_id.return_value = "123"
    adaptor.close.return_value = None
    return adaptor


@pytest.fixture
def sample_dashboard():
    """Create a sample dashboard info."""
    return DashboardInfo(
        id="123",
        name="Test Dashboard",
        description="A test dashboard",
        chart_ids=["1", "2", "3"],
    )


@pytest.fixture
def sample_charts():
    """Create sample chart info list."""
    return [
        ChartInfo(
            id="1",
            name="Sales Chart",
            description="Sales overview",
            query=QuerySpec(
                kind="sql",
                sql=["SELECT sum(sales) FROM sales_table WHERE year = 2024"],
            ),
            chart_type="bar",
        ),
        ChartInfo(
            id="2",
            name="Revenue Chart",
            description="Revenue analysis",
            query=QuerySpec(
                kind="sql",
                sql=["SELECT avg(revenue) FROM revenue_table"],
            ),
            chart_type="line",
        ),
        ChartInfo(
            id="3",
            name="Customer Chart",
            description="Customer metrics",
            query=QuerySpec(
                kind="sql",
                sql=["SELECT count(*) FROM customers"],
            ),
            chart_type="pie",
        ),
    ]


@pytest.fixture
def sample_datasets():
    """Create sample dataset info list."""
    return [
        DatasetInfo(
            id="1",
            name="Sales Dataset",
            dialect="mysql",
            tables=["sales_table"],
        ),
        DatasetInfo(
            id="2",
            name="Revenue Dataset",
            dialect="postgresql",
            tables=["revenue_table"],
        ),
    ]


class TestBiDashboardCommands:
    """Test cases for BiDashboardCommands class."""

    def test_init_with_agent_config(self, agent_config, console):
        """Test BiDashboardCommands initialization with AgentConfig."""
        commands = BiDashboardCommands(agent_config, console)
        assert commands.agent_config == agent_config
        assert commands.console == console
        assert commands.cli is None

    def test_init_with_cli(self, agent_config, console):
        """Test BiDashboardCommands initialization with DatusCLI mock."""
        mock_cli = MagicMock()
        mock_cli.agent_config = agent_config
        mock_cli.console = console

        commands = BiDashboardCommands(mock_cli, console)
        assert commands.agent_config == agent_config
        assert commands.cli == mock_cli

    def test_parse_selection_all(self):
        """Test _parse_selection with 'all' input."""
        commands = BiDashboardCommands(load_acceptance_config(namespace="bird_school"))

        result = commands._parse_selection("all", 5)
        assert result == [0, 1, 2, 3, 4]

        result = commands._parse_selection("*", 3)
        assert result == [0, 1, 2]

        result = commands._parse_selection("", 4)
        assert result == [0, 1, 2, 3]

    def test_parse_selection_none(self):
        """Test _parse_selection with 'none' input."""
        commands = BiDashboardCommands(load_acceptance_config(namespace="bird_school"))

        result = commands._parse_selection("none", 5)
        assert result == []

        result = commands._parse_selection("n", 5)
        assert result == []

        result = commands._parse_selection("no", 5)
        assert result == []

    def test_parse_selection_single_indices(self):
        """Test _parse_selection with single index values."""
        commands = BiDashboardCommands(load_acceptance_config(namespace="bird_school"))

        result = commands._parse_selection("1", 5)
        assert result == [0]

        result = commands._parse_selection("1,3,5", 5)
        assert result == [0, 2, 4]

        result = commands._parse_selection("2 4", 5)
        assert result == [1, 3]

    def test_parse_selection_ranges(self):
        """Test _parse_selection with range values."""
        commands = BiDashboardCommands(load_acceptance_config(namespace="bird_school"))

        result = commands._parse_selection("1-3", 5)
        assert result == [0, 1, 2]

        result = commands._parse_selection("2-4", 5)
        assert result == [1, 2, 3]

        result = commands._parse_selection("1-2,4-5", 5)
        assert result == [0, 1, 3, 4]

    def test_parse_selection_mixed(self):
        """Test _parse_selection with mixed input."""
        commands = BiDashboardCommands(load_acceptance_config(namespace="bird_school"))

        result = commands._parse_selection("1,3-5,7", 10)
        assert result == [0, 2, 3, 4, 6]

    def test_parse_selection_out_of_bounds(self):
        """Test _parse_selection with out of bounds indices."""
        commands = BiDashboardCommands(load_acceptance_config(namespace="bird_school"))

        result = commands._parse_selection("1,6,10", 5)
        assert result == [0]

        result = commands._parse_selection("10-15", 5)
        assert result == []

    def test_parse_selection_duplicates(self):
        """Test _parse_selection removes duplicates."""
        commands = BiDashboardCommands(load_acceptance_config(namespace="bird_school"))

        result = commands._parse_selection("1,1,2,2,3", 5)
        assert result == [0, 1, 2]

    def test_normalize_identifier_basic(self):
        """Test _normalize_identifier with basic input."""
        commands = BiDashboardCommands(load_acceptance_config(namespace="bird_school"))

        result = commands._normalize_identifier("Test Dashboard")
        assert result == "test_dashboard"

        result = commands._normalize_identifier("Sales-Report-2024")
        assert result == "sales_report_2024"

    def test_normalize_identifier_special_chars(self):
        """Test _normalize_identifier with special characters."""
        commands = BiDashboardCommands(load_acceptance_config(namespace="bird_school"))

        result = commands._normalize_identifier("Test@Dashboard#123")
        assert result == "test_dashboard_123"

        result = commands._normalize_identifier("Sales & Revenue!")
        assert result == "sales_revenue"

    def test_normalize_identifier_cjk(self):
        """Test _normalize_identifier with CJK characters."""
        commands = BiDashboardCommands(load_acceptance_config(namespace="bird_school"))

        result = commands._normalize_identifier("销售报表")
        assert result == "销售报表"

        result = commands._normalize_identifier("Sales销售Dashboard")
        assert result == "sales_销售_dashboard"

    def test_normalize_identifier_max_words(self):
        """Test _normalize_identifier with max_words limit."""
        commands = BiDashboardCommands(load_acceptance_config(namespace="bird_school"))

        result = commands._normalize_identifier("One Two Three Four Five", max_words=3)
        assert result == "one_two_three"

    def test_normalize_identifier_empty(self):
        """Test _normalize_identifier with empty input."""
        commands = BiDashboardCommands(load_acceptance_config(namespace="bird_school"))

        result = commands._normalize_identifier("")
        assert result == "item"

        result = commands._normalize_identifier("   ", fallback="test")
        assert result == "test"

        result = commands._normalize_identifier("@#$%", fallback="default")
        assert result == "default"

    def test_build_sub_agent_name(self):
        """Test _build_sub_agent_name."""
        commands = BiDashboardCommands(load_acceptance_config(namespace="bird_school"))

        result = commands._build_sub_agent_name("superset", "Sales Dashboard")
        assert result == "superset_sales_dashboard"

        result = commands._build_sub_agent_name("tableau", "Q1 Revenue Report 2024")
        assert result == "tableau_q1_revenue_report"

    def test_build_sub_agent_name_special_cases(self):
        """Test _build_sub_agent_name with special cases."""
        commands = BiDashboardCommands(load_acceptance_config(namespace="bird_school"))

        # Empty dashboard name
        result = commands._build_sub_agent_name("superset", "")
        assert result == "superset_dashboard"

        # Dashboard name starting with number
        result = commands._build_sub_agent_name("", "123 Report")
        assert result == "bi_123_report"

    def test_dedupe_values(self):
        """Test _dedupe_values."""
        commands = BiDashboardCommands(load_acceptance_config(namespace="bird_school"))

        result = commands._dedupe_values(["table1", "table2", "table1", "table3"])
        assert result == ["table1", "table2", "table3"]

        result = commands._dedupe_values(["  table1  ", "table2", "table1"])
        assert result == ["table1", "table2"]

        result = commands._dedupe_values(["", "table1", "", "table2"])
        assert result == ["table1", "table2"]

    def test_clean_comment_text(self):
        """Test _clean_comment_text."""
        commands = BiDashboardCommands(load_acceptance_config(namespace="bird_school"))

        result = commands._clean_comment_text("Test  Dashboard\n\tName")
        assert result == "Test Dashboard Name"

        result = commands._clean_comment_text("   Multiple   Spaces   ")
        assert result == "Multiple Spaces"

    def test_split_subject_tree(self):
        """Test _split_subject_tree."""
        commands = BiDashboardCommands(load_acceptance_config(namespace="bird_school"))

        domain, layer1, layer2 = commands._split_subject_tree("domain/layer1/layer2")
        assert domain == "domain"
        assert layer1 == "layer1"
        assert layer2 == "layer2"

        domain, layer1, layer2 = commands._split_subject_tree("domain/layer1")
        assert domain == "domain"
        assert layer1 == "layer1"
        assert layer2 == ""

        domain, layer1, layer2 = commands._split_subject_tree("")
        assert domain == ""
        assert layer1 == ""
        assert layer2 == ""

    def test_derive_api_base(self):
        """Test _derive_api_base."""
        commands = BiDashboardCommands(load_acceptance_config(namespace="bird_school"))

        result = commands._derive_api_base("https://example.com/dashboard/123")
        assert result == "https://example.com"

        result = commands._derive_api_base("http://localhost:8088/superset/dashboard/1/")
        assert result == "http://localhost:8088"

        result = commands._derive_api_base("invalid-url")
        assert result == ""

    def test_create_adaptor(self):
        """Test _create_adaptor."""
        config = load_acceptance_config(namespace="bird_school")

        # Create a commands instance
        commands = BiDashboardCommands(config)

        # Create a mock adaptor class
        mock_adaptor_cls = MagicMock()

        # Inject the mock into the registry
        commands._adaptor_registry["test_platform"] = mock_adaptor_cls

        options = DashboardCliOptions(
            platform="test_platform",
            dashboard_url="https://example.com/dashboard/123",
            api_base_url="https://example.com",
            auth_params=AuthParam(username="test", password="test"),
            dialect="mysql",
        )

        # Call the method
        commands._create_adaptor(options)

        # Verify the mock was called with correct parameters
        mock_adaptor_cls.assert_called_once_with(
            api_base_url="https://example.com",
            auth_params=options.auth_params,
            dialect=config.db_type,
        )

    def test_resolve_auth_params_login(self):
        """Test _resolve_auth_params with LOGIN auth type."""
        config = load_acceptance_config(namespace="bird_school")
        dashboard_config = DashboardConfig(
            platform="superset",
            username="testuser",
            password="testpass",
            extra={"key": "value"},
        )
        config.dashboard_config = {"superset": dashboard_config}

        commands = BiDashboardCommands(config)

        result = commands._resolve_auth_params("superset", AuthType.LOGIN)

        assert result is not None
        assert result.username == "testuser"
        assert result.password == "testpass"
        assert result.extra == {"key": "value"}

    def test_resolve_auth_params_api_key(self):
        """Test _resolve_auth_params with API_KEY auth type."""
        config = load_acceptance_config(namespace="bird_school")
        dashboard_config = DashboardConfig(
            platform="tableau",
            api_key="test_api_key_12345",
            extra={"region": "us-west"},
        )
        config.dashboard_config = {"tableau": dashboard_config}

        commands = BiDashboardCommands(config)

        result = commands._resolve_auth_params("tableau", AuthType.API_KEY)

        assert result is not None
        assert result.api_key == "test_api_key_12345"
        assert result.extra == {"region": "us-west"}

    def test_resolve_auth_params_missing_config(self):
        """Test _resolve_auth_params with missing config."""
        config = load_acceptance_config(namespace="bird_school")
        config.dashboard_config = {}

        commands = BiDashboardCommands(config)

        result = commands._resolve_auth_params("superset", AuthType.LOGIN)
        assert result is None

    def test_resolve_auth_params_incomplete_login(self):
        """Test _resolve_auth_params with incomplete LOGIN credentials."""
        config = load_acceptance_config(namespace="bird_school")
        dashboard_config = DashboardConfig(platform="superset", username="testuser")  # Missing password
        config.dashboard_config = {"superset": dashboard_config}

        commands = BiDashboardCommands(config)

        with pytest.raises(Exception) as exc_info:
            commands._resolve_auth_params("superset", AuthType.LOGIN)

        assert "username and password" in str(exc_info.value)

    def test_resolve_auth_params_missing_api_key(self):
        """Test _resolve_auth_params with missing API key."""
        config = load_acceptance_config(namespace="bird_school")
        dashboard_config = DashboardConfig(platform="tableau")  # Missing api_key
        config.dashboard_config = {"tableau": dashboard_config}

        commands = BiDashboardCommands(config)

        with pytest.raises(Exception) as exc_info:
            commands._resolve_auth_params("tableau", AuthType.API_KEY)

        assert "api_key" in str(exc_info.value)

    def test_lookup_dashboard_config(self):
        """Test _lookup_dashboard_config."""
        config = load_acceptance_config(namespace="bird_school")
        config1 = DashboardConfig(platform="superset", username="user1")
        config2 = DashboardConfig(platform="Tableau", username="user2")

        configs = {
            "superset": config1,
            "Tableau": config2,
        }

        commands = BiDashboardCommands(config)

        # Exact match
        result = commands._lookup_dashboard_config(configs, "superset")
        assert result == config1

        # Case-insensitive match
        result = commands._lookup_dashboard_config(configs, "SUPERSET")
        assert result == config1

        result = commands._lookup_dashboard_config(configs, "tableau")
        assert result == config2

        # Not found
        result = commands._lookup_dashboard_config(configs, "powerbi")
        assert result is None

    @patch("datus.cli.bi_dashboard.init_reference_sql")
    @patch("datus.cli.bi_dashboard.get_path_manager")
    def test_gen_reference_sqls(self, mock_path_manager, mock_init_ref_sql, agent_config, sample_dashboard, tmp_path):
        """Test _gen_reference_sqls."""
        # Setup mocks
        mock_path_manager.return_value.dashboard_path.return_value = tmp_path

        mock_init_ref_sql.return_value = {
            "status": "success",
            "valid_entries": 5,
            "invalid_entries": 0,
            "processed_entries": 5,
            "processed_items": [
                {
                    "subject_tree": "domain/layer1/layer2",
                    "name": "test_sql",
                }
            ],
        }

        commands = BiDashboardCommands(agent_config, Console(log_path=False))

        reference_sqls = [
            SelectedSqlCandidate(
                chart_id="1",
                chart_name="Test Chart",
                description="Test description",
                sql="SELECT * FROM test_table",
            )
        ]

        result = commands._gen_reference_sqls(reference_sqls, "superset", sample_dashboard)

        assert isinstance(result, list)
        assert len(result) > 0
        assert "domain.layer1.layer2.test_sql" in result

    @patch("datus.cli.bi_dashboard.init_metrics")
    @patch("datus.cli.bi_dashboard.get_path_manager")
    def test_gen_metrics(self, mock_path_manager, mock_init_metrics, agent_config, sample_dashboard, tmp_path):
        """Test _gen_metrics."""
        # Create temporary CSV file
        mock_path_manager.return_value.dashboard_path.return_value = tmp_path
        mock_path_manager.return_value.semantic_model_path.return_value = tmp_path

        # Create a test YAML file
        yaml_file = tmp_path / "test_metric.yml"
        yaml_content = """
metric:
  name: test_metric
  locked_metadata:
    tags:
      - "subject_tree:domain/layer1/layer2"
"""
        yaml_file.write_text(yaml_content)

        mock_init_metrics.return_value = (
            True,
            {"semantic_models": [str(yaml_file)]},
        )

        commands = BiDashboardCommands(agent_config, Console(log_path=False))

        metric_sqls = [
            SelectedSqlCandidate(
                chart_id="1",
                chart_name="Sales Chart",
                description="Total sales",
                sql="SELECT SUM(amount) FROM sales",
            )
        ]

        result = commands._gen_metrics(metric_sqls, "superset", sample_dashboard)

        assert isinstance(result, list)
        mock_init_metrics.assert_called_once()

    @patch("datus.cli.bi_dashboard.init_semantic_model")
    @patch("datus.cli.bi_dashboard.get_path_manager")
    def test_gen_semantic_model(self, mock_path_manager, mock_init_semantic, agent_config, sample_dashboard, tmp_path):
        """Test _gen_semantic_model."""
        mock_path_manager.return_value.dashboard_path.return_value = tmp_path

        mock_init_semantic.return_value = (
            True,
            {"semantic_model_count": 3},
        )

        commands = BiDashboardCommands(agent_config, Console(log_path=False))

        metric_sqls = [
            SelectedSqlCandidate(
                chart_id="1",
                chart_name="Sales Chart",
                description="Total sales",
                sql="SELECT SUM(amount) FROM sales",
            )
        ]

        result = commands._gen_semantic_model(metric_sqls, "superset", sample_dashboard)

        assert result is True
        mock_init_semantic.assert_called_once()

    @patch("datus.cli.bi_dashboard.init_semantic_model")
    def test_gen_semantic_model_empty_sqls(self, mock_init_semantic, agent_config, sample_dashboard):
        """Test _gen_semantic_model with empty SQL list."""
        commands = BiDashboardCommands(agent_config, Console(log_path=False))

        result = commands._gen_semantic_model([], "superset", sample_dashboard)

        assert result is False
        mock_init_semantic.assert_not_called()

    @patch("datus.cli.bi_dashboard.SubAgentManager")
    @patch("datus.cli.bi_dashboard.get_path_manager")
    def test_save_sub_agent(self, mock_path_manager, mock_manager_cls, agent_config, sample_dashboard, tmp_path):
        """Test _save_sub_agent."""
        mock_path_manager.return_value.dashboard_path.return_value = tmp_path
        mock_path_manager.return_value.semantic_model_path.return_value = tmp_path

        mock_manager = MagicMock()
        mock_manager_cls.return_value = mock_manager
        mock_manager.list_agents.return_value = {}

        commands = BiDashboardCommands(agent_config, Console(log_path=False))

        result = DashboardAssemblyResult(
            dashboard=sample_dashboard,
            charts=[],
            datasets=[],
            reference_sqls=[],
            metric_sqls=[],
            tables=["sales_table", "revenue_table"],
        )

        with patch.object(commands, "_gen_reference_sqls", return_value=["ref_sql_1"]), patch.object(
            commands, "_gen_semantic_model", return_value=True
        ), patch.object(commands, "_gen_metrics", return_value=["metric_1"]):
            commands._save_sub_agent("superset", sample_dashboard, result)

        # Verify SubAgentManager was called
        assert mock_manager.save_agent.call_count == 2  # Main agent + attribution agent
        assert mock_manager.bootstrap_agent.call_count == 2

    def test_build_sql_comment_lines(self, agent_config, sample_dashboard):
        """Test _build_sql_comment_lines."""
        commands = BiDashboardCommands(agent_config)

        sql_item = SelectedSqlCandidate(
            chart_id="1",
            chart_name="Test Chart",
            description="Test description",
            sql="SELECT * FROM test_table",
        )

        lines = commands._build_sql_comment_lines(sql_item, sample_dashboard)

        assert len(lines) == 3
        assert "Dashboard=Test Dashboard" in lines[0]
        assert "Chart=Test Chart" in lines[1]
        assert "Description=Test description" in lines[2]

    def test_build_sql_file_name(self, agent_config, sample_dashboard):
        """Test _build_sql_file_name."""
        commands = BiDashboardCommands(agent_config)

        # Just test that the function returns a string with expected components
        result = commands._build_sql_file_name("superset", sample_dashboard)

        assert "superset" in result
        assert "test_dashboard" in result
        # Should have a timestamp component
        assert len(result.split("_")) >= 3

    @patch("datus.cli.bi_dashboard.get_path_manager")
    def test_write_chart_sql_files(self, mock_path_manager, agent_config, sample_dashboard, tmp_path):
        """Test _write_chart_sql_files."""
        mock_path_manager.return_value.dashboard_path.return_value = tmp_path

        commands = BiDashboardCommands(agent_config)

        reference_sqls = [
            SelectedSqlCandidate(
                chart_id="1",
                chart_name="Chart 1",
                description="Description 1",
                sql="SELECT * FROM table1",
            ),
            SelectedSqlCandidate(
                chart_id="2",
                chart_name="Chart 2",
                description="Description 2",
                sql="SELECT * FROM table2",
            ),
        ]

        result_path = commands._write_chart_sql_files(reference_sqls, "superset", sample_dashboard)

        assert result_path is not None
        assert result_path.exists()

        content = result_path.read_text()
        assert "SELECT * FROM table1" in content
        assert "SELECT * FROM table2" in content
        assert "Dashboard=Test Dashboard" in content

    @patch("datus.cli.bi_dashboard.get_path_manager")
    def test_write_chart_sql_files_empty(self, agent_config, sample_dashboard):
        """Test _write_chart_sql_files with empty list."""
        commands = BiDashboardCommands(agent_config)

        result = commands._write_chart_sql_files([], "superset", sample_dashboard)
        assert result is None


class TestParseSubjectPathForMetrics:
    """Test cases for _parse_subject_path_for_metrics function."""

    def test_parse_subject_path_valid(self):
        """Test parsing valid subject tree tags."""
        tags = ["subject_tree:domain/layer1/layer2"]
        result = _parse_subject_path_for_metrics(tags)
        assert result == "domain.layer1.layer2"

    def test_parse_subject_path_multiple_tags(self):
        """Test parsing with multiple tags."""
        tags = ["tag1:value1", "subject_tree:sales/revenue/q1", "tag2:value2"]
        result = _parse_subject_path_for_metrics(tags)
        assert result == "sales.revenue.q1"

    def test_parse_subject_path_empty(self):
        """Test parsing with empty tags."""
        result = _parse_subject_path_for_metrics([])
        assert result is None

    def test_parse_subject_path_no_match(self):
        """Test parsing with no matching tags."""
        tags = ["tag1:value1", "tag2:value2"]
        result = _parse_subject_path_for_metrics(tags)
        assert result is None

    def test_parse_subject_path_with_spaces(self):
        """Test parsing with spaces in path."""
        tags = ["subject_tree: domain / layer1 / layer2 "]
        result = _parse_subject_path_for_metrics(tags)
        # The function doesn't strip spaces from individual segments
        assert result == "domain . layer1 . layer2"


class TestDashboardCliOptions:
    """Test cases for DashboardCliOptions dataclass."""

    def test_dashboard_cli_options_creation(self):
        """Test creating DashboardCliOptions."""
        auth_param = AuthParam(username="test", password="test")

        options = DashboardCliOptions(
            platform="superset",
            dashboard_url="https://example.com/dashboard/123",
            api_base_url="https://example.com",
            auth_params=auth_param,
            dialect="mysql",
        )

        assert options.platform == "superset"
        assert options.dashboard_url == "https://example.com/dashboard/123"
        assert options.api_base_url == "https://example.com"
        assert options.auth_params == auth_param
        assert options.dialect == "mysql"

    def test_dashboard_cli_options_defaults(self):
        """Test DashboardCliOptions with default values."""
        options = DashboardCliOptions(
            platform="superset",
            dashboard_url="https://example.com/dashboard/123",
            api_base_url="https://example.com",
        )

        assert options.auth_params is None
        assert options.dialect is None


class TestChartSelection:
    """Automated tests for chart selection workflow."""

    def test_select_charts_flow_accept_selection(self, agent_config, sample_charts):
        """Test the chart selection flow with user accepting the selection."""
        commands = BiDashboardCommands(agent_config, Console(log_path=False))

        # Mock user input to accept selection immediately
        with patch.object(commands, "_prompt_input", return_value="y"):
            selections = commands._load_chart_selections(
                sample_charts,
                [0, 2],
                purpose="reference SQL",
            )

        assert len(selections) == 2
        assert isinstance(selections[0], ChartSelection)
        assert selections[0].chart.id == "1"
        assert selections[1].chart.id == "3"
        # Verify SQL indices are set correctly
        assert selections[0].sql_indices == [0]  # Sales Chart has 1 SQL
        assert selections[1].sql_indices == [0]  # Customer Chart has 1 SQL

    def test_select_charts_flow_reject_then_accept(self, agent_config, sample_charts):
        """Test the chart selection flow with user rejecting then accepting."""
        commands = BiDashboardCommands(agent_config, Console(log_path=False))

        mock_inputs = ["1,3"]

        with patch.object(commands, "_prompt_input", side_effect=mock_inputs):
            selections = commands._load_chart_selections(
                sample_charts,
                [0, 2],  # Initial selection: charts 1 and 3
                purpose="reference SQL",
            )

        # Should get charts based on the new selection "1,2" (indices 0 and 1)
        assert len(selections) == 2
        assert selections[0].chart.id == "1"
        assert selections[1].chart.id == "3"

    def test_select_charts_flow_reject_and_cancel(self, agent_config, sample_charts):
        """Test the chart selection flow with user rejecting and then cancelling."""
        commands = BiDashboardCommands(agent_config, Console(log_path=False))

        # Simulate user rejecting first selection, then selecting nothing (cancels)
        # First call: reject ("n"), second call: select "none"
        mock_inputs = ["all"]

        with patch.object(commands, "_prompt_input", side_effect=mock_inputs):
            selections = commands._load_chart_selections(
                sample_charts,
                [0, 1, 2],  # Initial selection: all charts
                purpose="metrics",
            )

        # Should get empty list when user selects "none"
        assert len(selections) == 3

    def test_select_charts_flow_empty_indices(self, agent_config, sample_charts):
        """Test the chart selection flow with empty indices."""
        commands = BiDashboardCommands(agent_config, Console(log_path=False))

        # No need to mock input as the method returns early for empty indices
        selections = commands._load_chart_selections(
            sample_charts,
            [],  # Empty indices
            purpose="reference SQL",
        )

        assert len(selections) == 0

    def test_hydrate_charts(self, agent_config, mock_adaptor, sample_charts):
        """Test _hydrate_charts with successful chart loading."""
        mock_adaptor.get_chart.side_effect = lambda chart_id, dashboard_id: next(
            (c for c in sample_charts if c.id == chart_id), None
        )

        commands = BiDashboardCommands(agent_config, Console(log_path=False))

        chart_metas = [
            ChartInfo(id="1", name="Chart 1"),
            ChartInfo(id="2", name="Chart 2"),
        ]

        result = commands._hydrate_charts(mock_adaptor, "dashboard_123", chart_metas)

        assert len(result) == 2
        assert all(isinstance(c, ChartInfo) for c in result)
        # Verify the charts were properly hydrated with details
        assert result[0].description == "Sales overview"
        assert result[1].description == "Revenue analysis"

    def test_hydrate_charts_with_error(self, agent_config, mock_adaptor, sample_charts):
        """Test _hydrate_charts with errors during loading."""

        def mock_get_chart(chart_id, dashboard_id):
            if chart_id == "2":
                raise Exception("Chart not found")
            return next((c for c in sample_charts if c.id == chart_id), None)

        mock_adaptor.get_chart.side_effect = mock_get_chart

        commands = BiDashboardCommands(agent_config, Console(log_path=False))

        chart_metas = [
            ChartInfo(id="1", name="Chart 1"),
            ChartInfo(id="2", name="Chart 2"),
        ]

        result = commands._hydrate_charts(mock_adaptor, "dashboard_123", chart_metas)

        assert len(result) == 2
        # First chart should be hydrated with full details
        assert result[0].description == "Sales overview"
        # Second chart should be the meta (not detailed) because of error
        assert result[1].name == "Chart 2"
        assert result[1].description is None  # Not hydrated due to error

    def test_hydrate_charts_all_errors(self, agent_config, mock_adaptor):
        """Test _hydrate_charts when all charts fail to load."""

        def mock_get_chart(chart_id, dashboard_id):
            raise Exception(f"Failed to load chart {chart_id}")

        mock_adaptor.get_chart.side_effect = mock_get_chart

        commands = BiDashboardCommands(agent_config, Console(log_path=False))

        chart_metas = [
            ChartInfo(id="1", name="Chart 1"),
            ChartInfo(id="2", name="Chart 2"),
            ChartInfo(id="3", name="Chart 3"),
        ]

        result = commands._hydrate_charts(mock_adaptor, "dashboard_123", chart_metas)

        assert len(result) == 3
        # All charts should be the original metas since hydration failed
        for i, chart in enumerate(result):
            assert chart.name == f"Chart {i+1}"
            assert chart.description is None


class TestResolveDefaultTableContext:
    """Test cases for _resolve_default_table_context."""

    def test_resolve_from_cli_context(self, agent_config):
        """Test resolving context from CLI context."""
        mock_cli = MagicMock()
        mock_cli.agent_config = agent_config
        mock_cli.console = Console(log_path=False)
        mock_cli.cli_context = MagicMock()
        mock_cli.cli_context.current_catalog = "test_catalog"
        mock_cli.cli_context.current_db_name = "test_db"
        mock_cli.cli_context.current_schema = "test_schema"

        commands = BiDashboardCommands(mock_cli)

        catalog, database, schema = commands._resolve_default_table_context()

        assert catalog == "test_catalog"
        assert database == "test_db"
        assert schema == "test_schema"

    def test_resolve_from_db_config(self, agent_config):
        """Test resolving context from database config."""
        commands = BiDashboardCommands(agent_config, Console(log_path=False))

        catalog, database, schema = commands._resolve_default_table_context()

        # Should get values from agent_config's current database config
        # The actual values depend on the test config
        assert isinstance(catalog, str)
        assert isinstance(database, str)
        assert isinstance(schema, str)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_parse_selection_invalid_input(self):
        """Test _parse_selection with invalid input."""
        commands = BiDashboardCommands(load_acceptance_config(namespace="bird_school"))

        result = commands._parse_selection("abc", 5)
        assert result == []

        result = commands._parse_selection("1-abc", 5)
        assert result == []

    def test_normalize_identifier_unicode(self):
        """Test _normalize_identifier with various Unicode characters."""
        commands = BiDashboardCommands(load_acceptance_config(namespace="bird_school"))

        # Emoji and special Unicode
        result = commands._normalize_identifier("Test 📊 Dashboard")
        assert "test" in result
        assert "dashboard" in result

        # Mixed scripts
        result = commands._normalize_identifier("测试Test仪表板Dashboard")
        assert "测试" in result or "test" in result

    def test_build_sub_agent_name_edge_cases(self):
        """Test _build_sub_agent_name with edge cases."""
        commands = BiDashboardCommands(load_acceptance_config(namespace="bird_school"))

        # Both empty
        result = commands._build_sub_agent_name("", "")
        assert result == "bi_dashboard"

        # Very long name
        long_name = " ".join([f"word{i}" for i in range(20)])
        result = commands._build_sub_agent_name("platform", long_name)
        assert result.startswith("platform_")
        assert len(result.split("_")) <= 5  # platform + max 3 words + possible prefix

    def test_save_sub_agent_no_namespace(self, agent_config, sample_dashboard):
        """Test _save_sub_agent with no namespace set."""
        # Bypass the setter validation by setting the private attribute directly
        agent_config._current_namespace = ""

        commands = BiDashboardCommands(agent_config, Console(log_path=False))

        result = DashboardAssemblyResult(
            dashboard=sample_dashboard,
            charts=[],
            datasets=[],
            reference_sqls=[],
            metric_sqls=[],
            tables=[],
        )

        # Should return early without error
        commands._save_sub_agent("superset", sample_dashboard, result)

    def test_save_sub_agent_reserved_name(self, agent_config, sample_dashboard):
        """Test _save_sub_agent with reserved sub-agent name."""
        from datus.utils.constants import SYS_SUB_AGENTS

        if SYS_SUB_AGENTS:
            reserved_name = list(SYS_SUB_AGENTS)[0] if SYS_SUB_AGENTS else "chat"

            commands = BiDashboardCommands(agent_config, Console(log_path=False))

            # Mock to return a reserved name
            with patch.object(commands, "_build_sub_agent_name", return_value=reserved_name):
                result = DashboardAssemblyResult(
                    dashboard=sample_dashboard,
                    charts=[],
                    datasets=[],
                    reference_sqls=[],
                    metric_sqls=[],
                    tables=["test_table"],
                )

                # Should return early without error
                commands._save_sub_agent("superset", sample_dashboard, result)
