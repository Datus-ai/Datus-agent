#!/usr/bin/env python3
"""Backward-compatible entrypoint for the PR coverage runner.

`pull_request_target` jobs currently execute the base branch workflow, which
still invokes this legacy script path while checking out the PR head. Keep this
shim until the reusable workflow on `main` is updated to call
`ci/run-pr-tests.py` directly.
"""

from __future__ import annotations

import os
import runpy

if __name__ == "__main__":
    script_path = os.path.join(os.path.dirname(__file__), "run-pr-tests.py")
    runpy.run_path(script_path, run_name="__main__")
