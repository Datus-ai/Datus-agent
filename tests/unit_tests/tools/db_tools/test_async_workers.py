# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Database waits must not turn into interpreter-exit joins."""

import subprocess
import sys
import textwrap


def test_blocked_database_operation_and_close_allow_gateway_process_exit():
    program = textwrap.dedent(
        '''
        import asyncio
        import threading
        from types import SimpleNamespace
        import datus.agent.node.agentic_node as nodes
        from datus.api.services.chat_task_manager import ChatTask
        from datus.gateway.runtime import DatusGateway
        from datus.tools.db_tools.connection_scope import connection_scope, drain_connections
        from datus.tools.func_tool.base import trans_to_function_tool
        from datus.tools.registry.contracts import tool_contract

        entered, never = threading.Event(), threading.Event()

        class Database:
            @tool_contract(access="read", direct_call=True, category="db_tools")
            @connection_scope
            def list_tables(self) -> dict:
                """List tables from the blocking test datasource."""
                entered.set()
                never.wait()
                return {}

            def close(self):
                drain_connections(self)

        async def main():
            nodes._DATABASE_CLOSE_TIMEOUT = 0.03
            gateway = DatusGateway(SimpleNamespace(), {})
            database = Database()
            node = SimpleNamespace(db_func_tool=database, _session=None, _session_manager=None)
            tool = trans_to_function_tool(database.list_tables)

            async def model():
                try:
                    await tool.on_invoke_tool(None, '{}')
                finally:
                    await nodes.AgenticNode.aclose(node)

            task = asyncio.create_task(model())
            gateway._task_manager._tasks['blocked'] = ChatTask('blocked', task)
            async with asyncio.timeout(2):
                while not entered.is_set():
                    await asyncio.sleep(0.01)
            await gateway.shutdown()
            print('gateway-shutdown-complete', flush=True)

        asyncio.run(main())
        print('process-exit-complete', flush=True)
        '''
    )
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=12)
    assert result.returncode == 0, result.stderr
    assert "gateway-shutdown-complete" in result.stdout
    assert "process-exit-complete" in result.stdout
