#!/usr/bin/env python
"""End-to-end check of `agent.sql_read_only` against a REAL datus project config.

Companion to scripts/e2e_sql_read_only_mcp.py, which builds a throwaway
sqlite workspace instead of reading a project config.
This one drives an existing project's `conf/agent.yml` -- its real datasource
definitions, its real config shape -- through a real `datus-mcp --transport
stdio` subprocess.

TWO SAFETY PROPERTIES, both deliberate:

1. Nothing is written to the project. The config is copied to a temp directory
   with `home` rewritten there, and the server runs with that as its cwd, so
   session/index state lands in the temp dir and is discarded.

2. Every non-read probe targets a table that does not exist, and the DDL probe
   is CTAS from that missing table. Even with live credentials and a completely
   broken gate, none of these can destroy anything. This matters more than
   usual here: the thing under test IS a safety gate, so the failure mode being
   guarded against is precisely "the destructive statement executed".

   Never add a bare `CREATE TABLE x (...)` or a DROP of a real table here.

VERDICT LOGIC. Because the probes are designed to fail at the database anyway
(and because a project may have no reachable warehouse from this machine), a
statement's success is not the signal. The signal is WHERE it was stopped:

  * error contains "this agent is read-only"  -> the tool-layer gate refused it;
    the SQL never reached the connector. This is the property being tested.
  * any other error (connection refused, unknown table, permission denied)
    -> it passed the gate and failed downstream.

So this works with or without a reachable datasource. What it proves is gate
PLACEMENT -- that non-read SQL is stopped before the connector. It does not
prove a full write round-trip; e2e_sql_read_only_mcp.py does that against
sqlite, as does --sqlite-standin below.

Usage:
    <repo>/.venv/bin/python scripts/e2e_sql_read_only_mcp_project.py \
        --project /path/to/project [--datasource NAME] [--sqlite-standin]
Exit: 0 = gate behaves correctly, 1 = it does not
"""

import argparse
import asyncio
import copy
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MISSING = "datus_readonly_probe_missing"

# (label, sql, is_a_read)
CASES = [
    ("SELECT", "SELECT 1", True),
    ("SHOW", "SHOW DATABASES", True),
    ("EXPLAIN", "EXPLAIN SELECT 1", True),
    ("INSERT", f"INSERT INTO {MISSING} (v) VALUES ('x')", False),
    ("UPDATE", f"UPDATE {MISSING} SET v = 'x'", False),
    ("DELETE", f"DELETE FROM {MISSING}", False),
    ("CREATE TABLE (CTAS)", f"CREATE TABLE {MISSING}_2 AS SELECT * FROM {MISSING}", False),
    ("DROP TABLE", f"DROP TABLE {MISSING}", False),
    ("TRUNCATE", f"TRUNCATE TABLE {MISSING}", False),
    ("multi-statement", f"SELECT 1; DROP TABLE {MISSING}", False),
]

GATE_MARKER = "this agent is read-only"


def make_standin_db(dest: Path) -> Path:
    """A throwaway sqlite carrying the probe table, for --sqlite-standin."""
    db = dest / "standin.sqlite"
    con = sqlite3.connect(db)
    con.execute(f"CREATE TABLE {MISSING} (v TEXT)")
    con.execute(f"INSERT INTO {MISSING} (v) VALUES ('seed')")
    con.commit()
    con.close()
    return db


def stage_config(project: Path, dest: Path, *, sql_read_only: bool, standin: str | None = None) -> Path:
    """Copy the project's agent.yml into dest with home rewritten and the flag set.

    ``standin`` replaces the datasource definition with a throwaway sqlite while
    keeping every other part of the project config. Needed when the project's
    real warehouse is unreachable from this machine: DBFuncTool construction
    fails, ``has_db_tools`` goes False, and ``execute_sql`` is never mounted --
    so the gate cannot be exercised at all. With a stand-in the probes really
    execute, which upgrades the check from "gate placement" to a full write
    round-trip. The cost is that the connector dialect is sqlite, not the
    project's; classify_read_only_violation should be checked separately
    against the real dialect.
    """
    raw = yaml.safe_load((project / "conf" / "agent.yml").read_text(encoding="utf-8"))
    agent = copy.deepcopy(raw["agent"])
    agent["home"] = str(dest)  # keep all runtime state out of the project
    # DBFuncTool.__init__ resolves agent_config.active_model() (to size the
    # DataCompressor tokenizer) and raises without one, so execute_sql never
    # mounts. A project whose target lives in .datus/config.yml -- deliberately
    # not loaded here, for isolation -- would fail on that alone. The model has
    # no bearing on the SQL gate, so pin a mock rather than requiring a live API
    # key in a test.
    agent.setdefault("models", {})["__e2e_mock__"] = {
        "type": "openai",
        "api_key": "mock-api-key",
        "model": "mock-model",
        "base_url": "http://localhost:0",
    }
    agent["target"] = "__e2e_mock__"

    if standin:
        agent.setdefault("services", {})["datasources"] = {
            standin: {"type": "sqlite", "uri": str(make_standin_db(dest)), "name": standin, "default": True}
        }
    if sql_read_only:
        agent["sql_read_only"] = True
    else:
        agent.pop("sql_read_only", None)
    cfg = dest / "agent.yml"
    cfg.write_text(yaml.safe_dump({"agent": agent}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return cfg


async def run_matrix(cfg: Path, datasource: str) -> dict[str, tuple[bool, str]]:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "datus.mcp_server", "--datasource", datasource, "--transport", "stdio", "--config", str(cfg)],
        # cwd is the staging dir, never the project: `home: .` is relative, and
        # a project cwd would also pull in its .datus/config.yml override.
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
                    ok, err = payload.get("success") == 1, payload.get("error") or ""
                except (json.JSONDecodeError, AttributeError):
                    ok, err = not res.isError, raw
                results[label] = (ok, (err or "").strip())
    return results


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, type=Path)
    ap.add_argument(
        "--sqlite-standin",
        action="store_true",
        help="swap the datasource for a throwaway sqlite, keeping the rest "
        "of the project config; use when the real warehouse is "
        "unreachable (execute_sql is not mounted without a live "
        "connection, so the gate cannot otherwise be tested)",
    )
    ap.add_argument(
        "--datasource",
        default=None,
        help="defaults to .datus/config.yml default_datasource, else the datasource marked default: true",
    )
    args = ap.parse_args()

    project: Path = args.project.expanduser().resolve()
    agent = yaml.safe_load((project / "conf" / "agent.yml").read_text(encoding="utf-8"))["agent"]
    datasources = agent.get("services", {}).get("datasources", {})

    datasource = args.datasource
    if not datasource:
        override = project / ".datus" / "config.yml"
        if override.exists():
            datasource = (yaml.safe_load(override.read_text(encoding="utf-8")) or {}).get("default_datasource")
    if not datasource:
        datasource = next((n for n, c in datasources.items() if c.get("default")), None)
    if not datasource:
        print("could not determine a datasource; pass --datasource", file=sys.stderr)
        return 1

    print(f"project    : {project}")
    print(f"datasource : {datasource} (type={datasources.get(datasource, {}).get('type')})")
    if args.sqlite_standin:
        print(
            "datasource : SUBSTITUTED with a throwaway sqlite (--sqlite-standin); "
            "rest of the project config is used as-is"
        )
        print(
            f"probes     : non-read statements target `{MISSING}`, which EXISTS in "
            f"the stand-in, so writes really execute when the flag is off\n"
        )
    else:
        print(f"probes     : non-read statements target `{MISSING}`, which must not exist\n")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        on_dir, off_dir = root / "on", root / "off"
        on_dir.mkdir()
        off_dir.mkdir()
        standin = datasource if args.sqlite_standin else None
        print("running matrix with sql_read_only: true  ...")
        on = await run_matrix(stage_config(project, on_dir, sql_read_only=True, standin=standin), datasource)
        print("running matrix with sql_read_only: false ...\n")
        off = await run_matrix(stage_config(project, off_dir, sql_read_only=False, standin=standin), datasource)

    hdr = f"{'statement':<22} {'flag off':<14} {'flag on':<14} verdict"
    print(hdr)
    print("-" * len(hdr))

    failures = []
    for label, _, is_read in CASES:

        def where(entry):
            ok, err = entry
            if ok:
                return "executed"
            return "GATE" if GATE_MARKER in err.lower() else "reached db"

        off_where, on_where = where(off[label]), where(on[label])

        if is_read:
            good = on_where != "GATE"
            verdict = "read not blocked" if good else "READ WRONGLY BLOCKED"
        elif off_where == "GATE":
            good = False
            verdict = "GATE FIRED WITH FLAG OFF"
        elif off[label][1] and off[label][1] == on[label][1]:
            # Identical refusal with the flag off and on -> a statement-shape
            # rule that applies either way (multi-statement input, writable
            # PRAGMA), not the switch. Correct behaviour, but crediting it to
            # the switch would overstate what this change does.
            good = True
            verdict = "blocked either way (not the switch)"
        else:
            good = on_where == "GATE"
            verdict = "stopped by switch, never hit db" if good else "REACHED DB WITH FLAG ON"

        if not good:
            failures.append(label)
        print(f"{label:<22} {off_where:<14} {on_where:<14} {'' if good else '!! '}{verdict}")

    print()
    if failures:
        print(f"FAIL: {', '.join(failures)}")
        for label in failures:
            print(f"  {label}\n    off={off[label]}\n    on ={on[label]}")
        return 1
    print("PASS - agent.sql_read_only stops non-read SQL before the connector")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
