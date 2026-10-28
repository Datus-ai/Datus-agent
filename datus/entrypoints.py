# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Lightweight console-script entrypoints.

Multiprocessing policy must be selected before importing application modules
that initialize threaded runtimes such as LanceDB or embedding providers.
"""

from importlib import import_module
from typing import Any

from datus.utils.multiprocessing_utils import configure_multiprocessing_start_method


def _run(module_name: str) -> Any:
    configure_multiprocessing_start_method()
    return import_module(module_name).main()


def agent_main() -> Any:
    return _run("datus.main")


def cli_main() -> Any:
    return _run("datus.cli.main")


def api_main() -> Any:
    return _run("datus.api.main")


def mcp_main() -> Any:
    return _run("datus.mcp_server")


def gateway_main() -> Any:
    return _run("datus.gateway.main")
