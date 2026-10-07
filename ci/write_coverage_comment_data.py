#!/usr/bin/env python3

"""Serialize unprivileged PR coverage results for the trusted commenter."""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    data = {
        "pr_number": int(os.environ["PR_NUMBER"]),
        "head_sha": os.environ["PR_HEAD_SHA"],
        "head_repo_id": int(os.environ["PR_HEAD_REPO_ID"]),
        "head_ref": os.environ["PR_HEAD_REF"],
        "metrics": {
            "overall": os.environ["METRIC_OVERALL"],
            "diff": os.environ["METRIC_DIFF"],
            "test_outcome": os.environ["METRIC_TEST_OUTCOME"],
            "test_total": os.environ["METRIC_TEST_TOTAL"],
            "test_passed": os.environ["METRIC_TEST_PASSED"],
            "test_failed": os.environ["METRIC_TEST_FAILED"],
            "test_skipped": os.environ["METRIC_TEST_SKIPPED"],
        },
    }
    Path("ci/coverage-comment.json").write_text(json.dumps(data), encoding="utf-8")


if __name__ == "__main__":
    main()
