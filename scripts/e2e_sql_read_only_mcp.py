#!/usr/bin/env python
"""End-to-end check: does `agent.sql_read_only` gate the MCP `execute_sql` tool?

Drives a REAL `datus-mcp --transport stdio` subprocess over JSON-RPC using the
mcp SDK client. No mocks, no in-process shortcuts.

Runs the whole matrix TWICE, with the switch on and off, and compares. That
matters: "INSERT was refused" on its own proves nothing -- the write could be
failing for an unrelated reason (read-only sqlite file, bad SQL, missing table).
Only the OFF run establishes the write path works, which is what makes the ON
run meaningful.

The comparison also separates two things a single run conflates:
  * refused BY THE SWITCH        -- what we are testing
  * refused REGARDLESS of it     -- multi-statement input and writable PRAGMAs
                                    are rejected by _validate_read_sql either
                                    way; that is pre-existing behaviour, and
                                    reporting it as a switch effect would
                                    overstate what this change does.

Companion: scripts/e2e_sql_read_only_mcp_project.py runs the same check
against an existing project's conf/agent.yml instead of a synthetic one.

Usage:  <repo>/.venv/bin/python scripts/e2e_sql_read_only_mcp.py
Exit:   0 = switch behaves correctly, 1 = it does not

This is an operator tool, not library code: its verdict table on stdout IS the
deliverable, so it uses `print` rather than the `get_logger` the package
requires. Nothing under `datus/` imports it.
"""

import argparse
import asyncio
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class Probe(NamedTuple):
    """One statement to run under both flag settings.

    A NamedTuple rather than a bare tuple so the third field is self-describing
    at every use site -- `is_a_read` flips the whole verdict, and a positional
    bool is exactly the field a reader guesses wrong. Not a Pydantic model:
    these are literals in this file, there is no untrusted input to validate,
    and a standalone operator script should not grow a dependency to name three
    fields.
    """

    label: str
    sql: str
    is_a_read: bool


CASES = [
    Probe("SELECT", "SELECT COUNT(*) FROM t", True),
    Probe("SHOW-ish (PRAGMA table_info)", "PRAGMA table_info(t)", True),
    Probe("EXPLAIN", "EXPLAIN SELECT * FROM t", True),
    Probe("INSERT", "INSERT INTO t (v) VALUES ('written')", False),
    Probe("UPDATE", "UPDATE t SET v = 'mutated'", False),
    Probe("DELETE", "DELETE FROM t", False),
    Probe("CREATE TABLE", "CREATE TABLE t2 (id INT)", False),
    Probe("DROP TABLE", "DROP TABLE t", False),
    Probe("multi-statement", "SELECT 1; DROP TABLE t", False),
    Probe("writable PRAGMA", "PRAGMA journal_mode=WAL", False),
]

# Substring identifying a refusal that came from the read-only gate specifically,
# as opposed to the statement-shape checks that run regardless.
#
# It must match the gate's wording and NOT the statement-shape wording, which is
# the whole distinction the verdict table reports. That rules out both obvious
# alternatives:
#
#   "read-only"          too loose -- classify_read_only_violation also says
#                        "Only read-only queries (SELECT, SHOW, ...) are
#                        allowed", so gate refusals and shape refusals would
#                        become indistinguishable.
#   "agent.sql_read_only" never matches -- that string appears only in the REST
#                        route's message (cli_service), and these scripts probe
#                        the MCP execute_sql tool, whose refusal reads "This
#                        agent is read-only: ...". Using it would report every
#                        gated statement as having reached the database.
#
# If the tool's wording ever changes, the flag-on matrix reports the writes as
# executed and fails loudly; run_endpoint's zero-gated check catches the same
# drift. Neither mode can pass silently on a stale marker.
GATE_MARKER = "this agent is read-only"


def build_workspace(root: Path, *, sql_read_only: bool) -> Path:
    home = root / ("on" if sql_read_only else "off")
    home.mkdir(parents=True, exist_ok=True)

    db = home / "probe.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE t (v TEXT)")
    con.execute("INSERT INTO t (v) VALUES ('seed')")
    con.commit()
    con.close()

    agent = {
        "home": str(home),
        "target": "mock",
        "models": {
            "mock": {
                "type": "openai",
                "api_key": "mock-api-key",
                "model": "mock-model",
                "base_url": "http://localhost:0",
            }
        },
        "services": {
            "datasources": {"probe": {"type": "sqlite", "uri": str(db), "name": "probe", "default": True}},
            "semantic_layer": {},
        },
        "agentic_nodes": {},
    }
    if sql_read_only:
        agent["sql_read_only"] = True

    cfg = home / "agent.yml"
    cfg.write_text(yaml.safe_dump({"agent": agent}, sort_keys=False), encoding="utf-8")
    return cfg


async def run_matrix(cfg: Path) -> dict[str, tuple[bool, str]]:
    """Return {label: (succeeded, error_text)} for every case."""
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "datus.mcp_server",
            "--datasource",
            "probe",
            "--transport",
            "stdio",
            "--config",
            str(cfg),
        ],
        # cwd is the throwaway workspace, NOT a real project: running from a
        # datus project directory would pull in its .datus/config.yml override
        # and the server would fail to boot against this synthetic config.
        cwd=str(cfg.parent),
    )
    results: dict[str, tuple[bool, str]] = {}
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = {t.name for t in (await session.list_tools()).tools}
            assert "execute_sql" in names, f"execute_sql not exposed; saw {sorted(names)[:10]}"
            for label, sql, _ in CASES:
                res = await session.call_tool("execute_sql", {"sql": sql})
                raw = "".join(getattr(c, "text", "") for c in res.content)
                try:
                    payload = json.loads(raw)
                    ok = payload.get("success") == 1
                    err = payload.get("error") or ""
                except (json.JSONDecodeError, AttributeError):
                    ok, err = not res.isError, raw
                results[label] = (ok, (err or "").strip())
    return results


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        print("running matrix with sql_read_only: true  ...")
        on = await run_matrix(build_workspace(root, sql_read_only=True))
        print("running matrix with sql_read_only: false ...\n")
        off = await run_matrix(build_workspace(root, sql_read_only=False))

    hdr = f"{'statement':<30} {'flag off':<12} {'flag on':<12} verdict"
    print(hdr)
    print("-" * len(hdr))

    failures = []
    for label, _, is_read in CASES:
        off_ok, _ = off[label]
        on_ok, on_err = on[label]
        gated = GATE_MARKER in on_err.lower()

        if is_read:
            good = off_ok and on_ok
            verdict = "read allowed in both" if good else "READ BROKEN BY SWITCH"
        elif not off_ok:
            # Rejected even with the switch off -> a statement-shape rule, not
            # the switch. Correct, but not evidence for what we are testing.
            good = not on_ok
            verdict = "blocked either way (not the switch)"
        else:
            good = off_ok and on_ok is False and gated
            verdict = "SWITCH REFUSED IT" if good else "LEAKED THROUGH"

        if not good:
            failures.append(label)
        print(
            f"{label:<30} {'ok' if off_ok else 'refused':<12} "
            f"{'ok' if on_ok else 'refused':<12} {'' if good else '!! '}{verdict}"
        )

    print()
    if failures:
        print(f"FAIL: {', '.join(failures)}")
        for label in failures:
            print(f"  {label}: off={off[label]} on={on[label]}")
        return 1
    print("PASS - agent.sql_read_only gates the MCP execute_sql tool end to end")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
