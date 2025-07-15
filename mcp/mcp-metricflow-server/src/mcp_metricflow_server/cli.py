"""CLI for testing MetricFlow Server functionality."""

import argparse
import os
import sys
from pathlib import Path

from .config import MetricFlowConfig
from .server import MetricFlowServer


class MetricFlowCli:
    """CLI for testing MetricFlow Server functionality."""

    def __init__(self):
        """Initialize the CLI with MetricFlow server."""
        # Use the specific path provided by the user
        config = MetricFlowConfig(
            mf_path=os.getenv("MF_PATH", "/path/to/mf"),
            project_dir=Path.cwd(),
            verbose=False,  # Disable verbose mode by default
        )
        self.server = MetricFlowServer(config)

    def run_validate_configs(self):
        """Test validate-configs command."""
        print("Testing validate-configs...")
        result = self.server.validate_configs()
        print(f"Result: {result}")
        print("-" * 50)

    def run_list_metrics(self):
        """Test list-metrics command."""
        print("Testing list-metrics...")
        result = self.server.list_metrics()
        print(f"Result: {result}")
        print("-" * 50)

    def run_list_dimensions_with_metrics(self):
        """Test list-dimensions --metric-names transactions command."""
        print("Testing list-dimensions --metric-names transactions...")
        result = self.server.get_dimensions(metrics=["transactions"])
        print(f"Result: {result}")
        print("-" * 50)

    def run_query_with_dimensions_and_limit(self):
        """Test query --metrics transactions --dimensions ds --limit 5 command."""
        print("Testing query --metrics transactions --dimensions ds --limit 5...")
        result = self.server.query_metrics(metrics=["transactions"], group_by=["ds"], limit=5)
        print(f"Result: {result}")
        print("-" * 50)

    def run_query_with_metric_time_and_explain(self):
        """Test query --metrics transactions --dimensions metric_time --order metric_time --explain command."""
        print("Testing query --metrics transactions --dimensions metric_time --order metric_time --explain...")
        result = self.server.query_metrics(
            metrics=["transactions"], group_by=["metric_time"], order_by=["metric_time"], explain=True
        )
        print(f"Result: {result}")
        print("-" * 50)

    def run_all_tests(self):
        """Run all test commands."""
        print("Running all MetricFlow CLI tests...")
        print("=" * 50)

        try:
            self.run_validate_configs()
            self.run_list_metrics()
            self.run_list_dimensions_with_metrics()
            self.run_query_with_dimensions_and_limit()
            self.run_query_with_metric_time_and_explain()

            print("All tests completed!")

        except Exception as e:
            print(f"Error during testing: {e}")
            sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="MetricFlow CLI Testing Tool")
    parser.add_argument(
        "--command",
        choices=[
            "validate-configs",
            "list-metrics",
            "list-dimensions-transactions",
            "query-ds-limit",
            "query-metric-time-explain",
            "all",
        ],
        default="all",
        help="Command to test",
    )

    args = parser.parse_args()
    cli = MetricFlowCli()

    if args.command == "validate-configs":
        cli.run_validate_configs()
    elif args.command == "list-metrics":
        cli.run_list_metrics()
    elif args.command == "list-dimensions-transactions":
        cli.run_list_dimensions_with_metrics()
    elif args.command == "query-ds-limit":
        cli.run_query_with_dimensions_and_limit()
    elif args.command == "query-metric-time-explain":
        cli.run_query_with_metric_time_and_explain()
    elif args.command == "all":
        cli.run_all_tests()


if __name__ == "__main__":
    main()
