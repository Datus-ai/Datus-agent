# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Process-level multiprocessing policy for Datus entrypoints."""

import multiprocessing


def configure_multiprocessing_start_method() -> str:
    """Select ``spawn`` once without overriding an embedding host.

    Datus entrypoints call this before starting application services. Library
    modules must not call it during import because multiprocessing policy is a
    process-wide choice owned by the executable (or by an embedding host).
    """

    current_method = multiprocessing.get_start_method(allow_none=True)
    if current_method:
        return current_method

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # Another thread or library may have selected a method between the
        # check and the setter. Respect that choice instead of forcing ours.
        return multiprocessing.get_start_method(allow_none=True) or "spawn"
    return "spawn"
