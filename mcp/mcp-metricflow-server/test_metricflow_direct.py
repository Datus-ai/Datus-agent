"""Direct test for MetricFlow Server functionality."""

import asyncio
import os
from pathlib import Path

from src.mcp_metricflow_server.config import MetricFlowConfig
from src.mcp_metricflow_server.server import MetricFlowServer


class MetricFlowDirectTester:
    """Direct tester for MetricFlow Server functionality."""

    def __init__(self):
        """Initialize the tester."""
        # Use the specific path and current directory
        mf_path = os.getenv("MF_PATH", "/path/to/mf")
        self.config = MetricFlowConfig(mf_path=mf_path, project_dir=Path.cwd(), verbose=False)
        self.server = MetricFlowServer(self.config)
        self.test_results = []

    def test_validate_configs(self) -> bool:
        """Test validate-configs command."""
        print("🧪 Testing validate-configs...")
        try:
            result = self.server.validate_configs()
            success = "Successfully validated" in result
            print(f"✅ Validate configs: {'PASS' if success else 'FAIL'}")
            if success:
                print(f"   Result snippet: {result[:100]}...")
            self.test_results.append(("validate_configs", success))
            return success
        except Exception as e:
            print(f"❌ Validate configs: FAIL - {e}")
            self.test_results.append(("validate_configs", False))
            return False

    def test_list_metrics(self) -> bool:
        """Test list-metrics command."""
        print("\n🧪 Testing list-metrics...")
        try:
            result = self.server.list_metrics()
            success = "metrics" in result.lower() or "transactions" in result
            print(f"✅ List metrics: {'PASS' if success else 'FAIL'}")
            if success:
                print(f"   Result snippet: {result[:100]}...")
            self.test_results.append(("list_metrics", success))
            return success
        except Exception as e:
            print(f"❌ List metrics: FAIL - {e}")
            self.test_results.append(("list_metrics", False))
            return False

    def test_get_dimensions(self) -> bool:
        """Test get-dimensions command."""
        print("\n🧪 Testing get-dimensions for transactions...")
        try:
            result = self.server.get_dimensions(metrics=["transactions"])
            success = "dimensions" in result.lower() or "ds" in result
            print(f"✅ Get dimensions: {'PASS' if success else 'FAIL'}")
            if success:
                print(f"   Result snippet: {result[:100]}...")
            self.test_results.append(("get_dimensions", success))
            return success
        except Exception as e:
            print(f"❌ Get dimensions: FAIL - {e}")
            self.test_results.append(("get_dimensions", False))
            return False

    def test_query_metrics_simple(self) -> bool:
        """Test simple query-metrics command."""
        print("\n🧪 Testing query-metrics (transactions, ds, limit 3)...")
        try:
            result = self.server.query_metrics(metrics=["transactions"], group_by=["ds"], limit=3)
            success = "transactions" in result and ("2022" in result or "Success" in result)
            print(f"✅ Query metrics (simple): {'PASS' if success else 'FAIL'}")
            if success:
                print(f"   Result snippet: {result[:100]}...")
            self.test_results.append(("query_metrics_simple", success))
            return success
        except Exception as e:
            print(f"❌ Query metrics (simple): FAIL - {e}")
            self.test_results.append(("query_metrics_simple", False))
            return False

    def test_query_metrics_explain(self) -> bool:
        """Test query-metrics with explain."""
        print("\n🧪 Testing query-metrics with explain...")
        try:
            result = self.server.query_metrics(
                metrics=["transactions"], group_by=["metric_time"], order_by=["metric_time"], explain=True
            )
            success = "SQL" in result or "SELECT" in result
            print(f"✅ Query metrics (explain): {'PASS' if success else 'FAIL'}")
            if success:
                print(f"   Result snippet: {result[:100]}...")
            self.test_results.append(("query_metrics_explain", success))
            return success
        except Exception as e:
            print(f"❌ Query metrics (explain): FAIL - {e}")
            self.test_results.append(("query_metrics_explain", False))
            return False

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("DIRECT METRICFLOW TEST SUMMARY")
        print("=" * 60)

        passed = sum(1 for _, success in self.test_results if success)
        total = len(self.test_results)

        for test_name, success in self.test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{test_name:25}: {status}")

        print("-" * 60)
        print(f"TOTAL: {passed}/{total} tests passed")

        if passed == total:
            print("🎉 All direct tests passed!")
        else:
            print("⚠️  Some direct tests failed")

    async def run_all_tests(self):
        """Run all direct tests."""
        print("🚀 Starting Direct MetricFlow Server Tests")
        print("=" * 60)

        try:
            self.test_validate_configs()
            self.test_list_metrics()
            self.test_get_dimensions()
            self.test_query_metrics_simple()
            self.test_query_metrics_explain()

        except Exception as e:
            print(f"❌ Test execution failed: {e}")

        finally:
            self.print_summary()


async def main():
    """Main test execution function."""
    tester = MetricFlowDirectTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
