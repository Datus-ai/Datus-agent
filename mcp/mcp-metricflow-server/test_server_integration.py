"""Integration tests for MCP MetricFlow Server."""

import asyncio
import json
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict


class MetricFlowServerTester:
    """Class for testing MCP MetricFlow Server integration."""

    def __init__(self):
        """Initialize the tester."""
        self.server_process = None
        self.test_results = []
        self.request_id = 1
        self.response_queue = queue.Queue()
        self.reader_thread = None
        self.stop_reading = False

    def _read_responses(self):
        """Continuously read responses from server in a separate thread."""
        while not self.stop_reading and self.server_process and self.server_process.poll() is None:
            try:
                line = self.server_process.stdout.readline()
                if line:
                    line = line.strip()
                    if line:  # Only process non-empty lines
                        try:
                            response = json.loads(line)
                            self.response_queue.put(response)
                        except json.JSONDecodeError:
                            # Log server messages but don't put them in queue
                            print(f"📝 Server log: {line}")
                else:
                    time.sleep(0.1)
            except Exception as e:
                if not self.stop_reading:
                    print(f"Error reading from server: {e}")
                break

    async def start_server(self) -> subprocess.Popen:
        """Start the MCP MetricFlow server using uv."""
        print("Starting MCP MetricFlow server...")

        # Start server using uv
        process = subprocess.Popen(
            ["uv", "run", "python", "-m", "mcp_metricflow_server.main"],
            cwd=Path.cwd(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
        )

        # Give server time to start
        await asyncio.sleep(3)

        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise Exception(f"Server failed to start. STDOUT: {stdout}, STDERR: {stderr}")

        self.server_process = process

        # Start response reader thread
        self.reader_thread = threading.Thread(target=self._read_responses, daemon=True)
        self.reader_thread.start()

        print("✅ Server started successfully")
        return process

    def send_mcp_request(self, method: str, params: Dict[str, Any] = None, timeout: int = 15) -> Dict[str, Any]:
        """Send an MCP request to the server and wait for response."""
        if not self.server_process or self.server_process.poll() is not None:
            raise Exception("Server is not running")

        request_id = self.request_id
        self.request_id += 1

        request = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}

        request_json = json.dumps(request) + "\n"

        try:
            # Clear any old responses
            while not self.response_queue.empty():
                try:
                    self.response_queue.get_nowait()
                except queue.Empty:
                    break

            # Send request
            self.server_process.stdin.write(request_json)
            self.server_process.stdin.flush()

            # Wait for response
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = self.response_queue.get(timeout=0.5)
                    if response.get("id") == request_id:
                        return response
                    else:
                        # Put it back if it's not our response
                        self.response_queue.put(response)
                except queue.Empty:
                    continue

            raise Exception(f"Timeout waiting for response to {method} (request_id: {request_id})")

        except Exception as e:
            raise Exception(f"Failed to communicate with server: {e}")

    async def test_initialize(self) -> bool:
        """Test server initialization."""
        print("\n🧪 Testing server initialization...")

        try:
            response = self.send_mcp_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0.0"},
                },
            )

            success = "result" in response and "capabilities" in response["result"]
            if success:
                print(f"   Server capabilities: {list(response['result']['capabilities'].keys())}")

            result = f"✅ Initialize: {'PASS' if success else 'FAIL'}"
            print(result)
            self.test_results.append(("initialize", success))

            # Send initialized notification
            if success:
                initialized_notification = {"jsonrpc": "2.0", "method": "notifications/initialized"}
                self.server_process.stdin.write(json.dumps(initialized_notification) + "\n")
                self.server_process.stdin.flush()
                await asyncio.sleep(0.5)  # Give server time to process

            return success

        except Exception as e:
            result = f"❌ Initialize: FAIL - {e}"
            print(result)
            self.test_results.append(("initialize", False))
            return False

    async def test_list_tools(self) -> bool:
        """Test listing available tools."""
        print("\n🧪 Testing list tools...")

        try:
            response = self.send_mcp_request("tools/list")

            success = "result" in response and "tools" in response["result"]
            if success:
                tools = response["result"]["tools"]
                print(f"   Found {len(tools)} tools:")
                for tool in tools:
                    print(f"     - {tool['name']}: {tool.get('description', 'No description')[:50]}...")
            else:
                print(f"   Response: {response}")

            result = f"✅ List tools: {'PASS' if success else 'FAIL'}"
            print(result)
            self.test_results.append(("list_tools", success))
            return success

        except Exception as e:
            result = f"❌ List tools: FAIL - {e}"
            print(result)
            self.test_results.append(("list_tools", False))
            return False

    async def test_validate_configs_tool(self) -> bool:
        """Test validate_configs tool."""
        print("\n🧪 Testing validate_configs tool...")

        try:
            response = self.send_mcp_request("tools/call", {"name": "validate_configs", "arguments": {}})

            success = "result" in response and "content" in response["result"]
            if success:
                content = response["result"]["content"]
                print(f"   Tool response length: {len(content[0]['text'])} characters")
                print(f"   Response snippet: {content[0]['text'][:100]}...")
            else:
                print(f"   Response: {response}")

            result = f"✅ Validate configs: {'PASS' if success else 'FAIL'}"
            print(result)
            self.test_results.append(("validate_configs", success))
            return success

        except Exception as e:
            result = f"❌ Validate configs: FAIL - {e}"
            print(result)
            self.test_results.append(("validate_configs", False))
            return False

    async def test_list_metrics_tool(self) -> bool:
        """Test list_metrics tool."""
        print("\n🧪 Testing list_metrics tool...")

        try:
            response = self.send_mcp_request("tools/call", {"name": "list_metrics", "arguments": {}})

            success = "result" in response and "content" in response["result"]
            if success:
                content = response["result"]["content"]
                print(f"   Tool response length: {len(content[0]['text'])} characters")
                print(f"   Response snippet: {content[0]['text'][:100]}...")
            else:
                print(f"   Response: {response}")

            result = f"✅ List metrics: {'PASS' if success else 'FAIL'}"
            print(result)
            self.test_results.append(("list_metrics", success))
            return success

        except Exception as e:
            result = f"❌ List metrics: FAIL - {e}"
            print(result)
            self.test_results.append(("list_metrics", False))
            return False

    async def test_query_metrics_tool(self) -> bool:
        """Test query_metrics tool."""
        print("\n🧪 Testing query_metrics tool...")

        try:
            response = self.send_mcp_request(
                "tools/call",
                {"name": "query_metrics", "arguments": {"metrics": ["transactions"], "group_by": ["ds"], "limit": 3}},
            )

            success = "result" in response and "content" in response["result"]
            if success:
                content = response["result"]["content"]
                print(f"   Tool response length: {len(content[0]['text'])} characters")
                print(f"   Response snippet: {content[0]['text'][:100]}...")
            else:
                print(f"   Response: {response}")

            result = f"✅ Query metrics: {'PASS' if success else 'FAIL'}"
            print(result)
            self.test_results.append(("query_metrics", success))
            return success

        except Exception as e:
            result = f"❌ Query metrics: FAIL - {e}"
            print(result)
            self.test_results.append(("query_metrics", False))
            return False

    def stop_server(self):
        """Stop the MCP server."""
        if self.server_process:
            print("\n🛑 Stopping server...")
            self.stop_reading = True

            # Wait for reader thread to finish
            if self.reader_thread and self.reader_thread.is_alive():
                self.reader_thread.join(timeout=1)

            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()
            print("✅ Server stopped")

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("INTEGRATION TEST SUMMARY")
        print("=" * 60)

        passed = sum(1 for _, success in self.test_results if success)
        total = len(self.test_results)

        for test_name, success in self.test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{test_name:25}: {status}")

        print("-" * 60)
        print(f"TOTAL: {passed}/{total} tests passed")

        if passed == total:
            print("🎉 All integration tests passed!")
        else:
            print("⚠️  Some integration tests failed")

    async def run_all_tests(self):
        """Run all integration tests."""
        print("🚀 Starting MCP MetricFlow Server Integration Tests")
        print("=" * 70)

        try:
            # Start server
            await self.start_server()

            # Run tests in sequence
            init_success = await self.test_initialize()

            if init_success:
                await self.test_list_tools()
                await self.test_validate_configs_tool()
                await self.test_list_metrics_tool()
                await self.test_query_metrics_tool()
            else:
                print("⚠️  Skipping remaining tests due to initialization failure")

        except Exception as e:
            print(f"❌ Test execution failed: {e}")

        finally:
            # Stop server and print summary
            self.stop_server()
            self.print_summary()


async def main():
    """Main test execution function."""
    tester = MetricFlowServerTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
