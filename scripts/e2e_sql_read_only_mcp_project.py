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

This is an operator tool, not library code: its verdict table on stdout IS the
deliverable, so it uses `print` rather than the `get_logger` the package
requires. Nothing under `datus/` imports it.
"""

import argparse
import asyncio
import copy
import csv
import io
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple
from uuid import uuid4

import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

MISSING = "datus_readonly_probe_missing"


class Probe(NamedTuple):
    """One statement to run under both flag settings.

    A NamedTuple rather than a bare tuple so the third field is self-describing
    at every use site -- `is_a_read` flips the whole verdict, and a positional
    bool is exactly the field a reader guesses wrong. Not a Pydantic model:
    these are literals built in this file, there is no untrusted input to
    validate, and a standalone operator script should not grow a dependency to
    name three fields.
    """

    label: str
    sql: str
    is_a_read: bool


CASES = [
    Probe("SELECT", "SELECT 1", True),
    Probe("SHOW", "SHOW DATABASES", True),
    Probe("EXPLAIN", "EXPLAIN SELECT 1", True),
    Probe("INSERT", f"INSERT INTO {MISSING} (v) VALUES ('x')", False),
    Probe("UPDATE", f"UPDATE {MISSING} SET v = 'x'", False),
    Probe("DELETE", f"DELETE FROM {MISSING}", False),
    Probe("CREATE TABLE (CTAS)", f"CREATE TABLE {MISSING}_2 AS SELECT * FROM {MISSING}", False),
    Probe("DROP TABLE", f"DROP TABLE {MISSING}", False),
    Probe("TRUNCATE", f"TRUNCATE TABLE {MISSING}", False),
    Probe("multi-statement", f"SELECT 1; DROP TABLE {MISSING}", False),
]

# Substring identifying a refusal that came from the read-only gate specifically,
# as opposed to the statement-shape checks that run regardless. See the same
# constant in e2e_sql_read_only_mcp.py for why neither "read-only" (too loose --
# it also matches the statement-shape message) nor "agent.sql_read_only" (never
# matches -- that wording belongs to the REST route, not the MCP tool) can be
# used here.
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


PROBE_PREFIX = "datus_ro_probe"


def _server_params(cfg: Path, datasource: str) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "datus.mcp_server",
            "--datasource",
            datasource,
            "--transport",
            "stdio",
            "--config",
            str(cfg),
        ],
        # cwd is the staging dir, never the project: `home: .` is relative, and
        # a project cwd would also pull in its .datus/config.yml override.
        cwd=str(cfg.parent),
    )


class McpSession:
    """An MCP session with an execute_sql helper.

    Either spawns its own `datus-mcp --transport stdio` subprocess from a staged
    config, or attaches over streamable HTTP to a server someone else started
    (`endpoint`).
    """

    def __init__(
        self,
        cfg: Path | None,
        datasource: str,
        *,
        dry_run: bool = False,
        endpoint: str | None = None,
    ):
        self._cfg, self._datasource, self._dry_run = cfg, datasource, dry_run
        self._endpoint = endpoint
        self._stack = None
        self._session = None
        self.tools: list[str] = []

    async def __aenter__(self):
        from contextlib import AsyncExitStack

        self._stack = AsyncExitStack()
        if self._endpoint:
            # Attaching to a server we did not launch: its config, including
            # whether sql_read_only is on, is fixed at its startup. That is why
            # --endpoint reports what it observes rather than asserting a
            # flag-on/flag-off contrast it has no way to arrange.
            read, write, _ = await self._stack.enter_async_context(streamablehttp_client(self._endpoint))
        else:
            read, write = await self._stack.enter_async_context(
                stdio_client(_server_params(self._cfg, self._datasource))
            )
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        self.tools = sorted(t.name for t in (await self._session.list_tools()).tools)
        assert "execute_sql" in self.tools, f"execute_sql not exposed; saw {self.tools[:10]}"
        return self

    async def __aexit__(self, *exc):
        await self._stack.__aexit__(*exc)

    async def sql(self, statement: str) -> tuple[bool, str, object]:
        """Return (succeeded, error_text, result)."""
        if self._dry_run:
            print(f"      [dry-run] {statement}")
            return True, "", None
        res = await self._session.call_tool("execute_sql", {"sql": statement})
        raw = "".join(getattr(c, "text", "") for c in res.content)
        try:
            payload = json.loads(raw)
            return payload.get("success") == 1, (payload.get("error") or "").strip(), payload.get("result")
        except (json.JSONDecodeError, AttributeError):
            return not res.isError, raw.strip(), raw


def create_probe_ddl(dialect: str, table: str) -> str:
    """CREATE for a scratch probe table, in the flavour the datasource needs."""
    if (dialect or "").lower() in {"starrocks", "doris"}:
        # PRIMARY KEY so UPDATE/DELETE are permitted; replication_num 1 so a
        # single-BE cluster can satisfy it.
        return (
            f"CREATE TABLE {table} (id INT, v VARCHAR(64)) "
            f"PRIMARY KEY(id) DISTRIBUTED BY HASH(id) BUCKETS 1 "
            f'PROPERTIES ("replication_num" = "1")'
        )
    return f"CREATE TABLE {table} (id INT, v VARCHAR(64))"


def live_cases(dml: str, drop_me: str, ctas: str) -> list[Probe]:
    """Probes for --live-writes, all scoped to this run's own scratch tables."""
    return [
        Probe("SELECT", f"SELECT COUNT(*) FROM {dml}", True),
        Probe("EXPLAIN", f"EXPLAIN SELECT * FROM {dml}", True),
        Probe("INSERT", f"INSERT INTO {dml} (id, v) VALUES (99, 'written')", False),
        Probe("UPDATE", f"UPDATE {dml} SET v = 'mutated' WHERE id = 1", False),
        Probe("DELETE", f"DELETE FROM {dml} WHERE id = 2", False),
        Probe("CREATE TABLE (CTAS)", f"CREATE TABLE {ctas} AS SELECT * FROM {dml}", False),
        Probe("TRUNCATE", f"TRUNCATE TABLE {dml}", False),
        Probe("DROP TABLE", f"DROP TABLE {drop_me}", False),
        Probe("multi-statement", f"SELECT 1; DROP TABLE {dml}", False),
    ]


def scalar_from_result(result: object, column: str) -> int | None:
    """The one integer cell from an `execute_sql` read payload, or None.

    `execute_sql` hands back rows already compressed, so the value is not a
    plain scalar anywhere in the structure -- it is a cell in a CSV string under
    `compressed_data`, with an index column prepended, sitting next to metadata
    keys:

        {"original_rows": 1, "original_columns": ["n"],
         "compressed_data": "index,n\\n0,300", ...}

    Two tempting shortcuts are both wrong on that shape. A regex for the last
    digit run over the serialized payload can pick up digits from a table name
    (the scratch tables embed a hex run id) or from a metadata field if the key
    order changes. Walking the structure for the first int returns
    `original_rows` -- 1 -- and never sees 300 at all. So parse the named column
    out of the CSV, and return None rather than a guess.
    """
    if not isinstance(result, dict):
        return None
    csv_text = result.get("compressed_data")
    if not isinstance(csv_text, str) or not csv_text.strip():
        return None
    try:
        rows = list(csv.DictReader(io.StringIO(csv_text)))
    except csv.Error:
        return None
    if len(rows) != 1:
        return None
    value = (rows[0].get(column) or "").strip()
    return int(value) if value.lstrip("-").isdigit() else None


async def row_count(sess: McpSession, table: str) -> int | None:
    """Rows in `table`, or None if the count could not be read.

    A None must never be treated as "unchanged" -- see the step-5 checks in
    `run_live`, where comparing two unknowns would report success on a
    datasource that was actually mutated.
    """
    ok, _, result = await sess.sql(f"SELECT COUNT(*) AS n FROM {table}")
    if not ok:
        return None
    return scalar_from_result(result, "n")


async def table_exists(sess: McpSession, table: str) -> bool:
    ok, _, _ = await sess.sql(f"SELECT COUNT(*) FROM {table}")
    return ok


async def run_matrix(cfg: Path, datasource: str) -> dict[str, tuple[bool, str]]:
    """Run every CASES probe against a server staged from ``cfg``.

    Goes through ``McpSession`` rather than driving ``stdio_client`` directly:
    the session already owns how the subprocess is spawned (``_server_params``)
    and how an ``execute_sql`` payload is unwrapped, and a second copy of either
    is a copy that can disagree with the one --live-writes uses.
    """
    results: dict[str, tuple[bool, str]] = {}
    async with McpSession(cfg, datasource) as sess:
        for label, sql, _ in CASES:
            ok, err, _ = await sess.sql(sql)
            results[label] = (ok, err)
    return results


async def run_endpoint(endpoint: str) -> int:
    """Probe a server someone else already started.

    Unlike the other modes this cannot arrange a flag-on/flag-off contrast --
    the server's config is fixed at launch -- so it reports what it observes
    rather than asserting a comparison. What it can still establish is where a
    statement was stopped: a refusal carrying the read-only message never
    reached the connector, while any other error means it did.

    Safe to point at a server whose read-only posture you have not confirmed:
    every table-scoped probe targets a table that does not exist, and the DDL
    probe is CTAS from it. The two exceptions are `SET` and `USE`, which name no
    table -- they can only change the server's own session context, which this
    script does not depend on afterwards.
    """
    reads = [
        ("SELECT 1", "SELECT 1"),
        ("SHOW DATABASES", "SHOW DATABASES"),
    ]
    writes = [
        ("INSERT", f"INSERT INTO {MISSING} (v) VALUES ('x')"),
        ("UPDATE", f"UPDATE {MISSING} SET v = 'x'"),
        ("DELETE", f"DELETE FROM {MISSING}"),
        ("CREATE TABLE (CTAS)", f"CREATE TABLE {MISSING}_2 AS SELECT * FROM {MISSING}"),
        ("ALTER TABLE", f"ALTER TABLE {MISSING} ADD COLUMN c INT"),
        ("DROP TABLE", f"DROP TABLE {MISSING}"),
        ("TRUNCATE", f"TRUNCATE TABLE {MISSING}"),
        ("INSERT OVERWRITE", f"INSERT OVERWRITE {MISSING} SELECT 1"),
        ("SUBMIT TASK", f"SUBMIT TASK t AS INSERT INTO {MISSING} SELECT 1"),
        ("SET", "SET is_report_audit_info = true"),
        ("USE", "USE information_schema"),
        ("multi-statement", f"SELECT 1; DROP TABLE {MISSING}"),
    ]

    async with McpSession(None, "", endpoint=endpoint) as sess:
        print(f"tools exposed : {len(sess.tools)}")
        print(f"probes        : non-read statements target `{MISSING}`, which must not exist\n")

        print(f"{'READ probe':<26} outcome")
        print("-" * 74)
        reachable = False
        read_blocked = []
        for label, sql in reads:
            ok, err, _ = await sess.sql(sql)
            gated = GATE_MARKER in err.lower()
            reachable |= ok
            if gated:
                read_blocked.append(label)
            print(f"{label:<26} {'ok' if ok else ('REFUSED BY GATE' if gated else f'db error: {err[:36]}')}")

        print(f"\n{'WRITE / DDL probe':<26} {'refused':<9} stopped by")
        print("-" * 74)
        leaked = []
        gated_count = 0
        for label, sql in writes:
            ok, err, _ = await sess.sql(sql)
            gated = GATE_MARKER in err.lower()
            gated_count += gated
            if ok:
                who = "NOTHING -- IT EXECUTED"
                leaked.append(label)
            elif gated:
                who = "the read-only gate (never reached the connector)"
            else:
                who = f"something else: {err[:40]}"
            print(f"{label:<26} {'no' if ok else 'yes':<9} {'!! ' if ok else ''}{who}")

    print()
    if not reachable:
        print("NOTE: no read succeeded, so the datasource is unreachable from this server.")
        print("      Refusals below the gate are still meaningful (they never got that far),")
        print("      but this run does not show reads working.")
    if leaked:
        print(f"FAIL - reached the warehouse: {', '.join(leaked)}")
        return 1
    if read_blocked:
        print(f"FAIL - the gate wrongly blocked reads: {', '.join(read_blocked)}")
        return 1
    if gated_count == 0:
        # Nothing leaked and no read was wrongly blocked, but no probe was
        # stopped BY THE GATE either -- which is what an unreachable datasource
        # looks like: every write fails downstream for its own reasons. Reporting
        # PASS here would hand a CI job a zero exit status backed by no evidence
        # that the gate is even on.
        print("INCONCLUSIVE - no write probe was stopped by the read-only gate.")
        print("               Either the flag is off on that server, or every probe")
        print("               was refused earlier by other rules; see above.")
        return 1
    print(f"PASS - {gated_count}/{len(writes)} write probes stopped by the read-only gate")
    if gated_count < len(writes):
        print("       (the remainder were refused by other rules; see above)")
    return 0


async def run_live(cfg_on: Path, cfg_off: Path, datasource: str, dialect: str, dry_run: bool) -> int:
    """Full write round-trip against the project's REAL datasource.

    Only for a datasource you are willing to write to. Everything it touches is
    a table it creates itself, named `datus_ro_probe_<run id>_*`, and teardown
    runs in a finally block. If teardown cannot complete, the exact DROP
    statements are printed so nothing is left silently orphaned.

    The decisive assertion is step 5: after the flag-on matrix, the scratch
    tables are byte-for-byte as the flag-off matrix left them. The flag-off run
    has already proven every one of those statements really mutates this
    datasource, so "unchanged" can only mean the gate stopped them.
    """
    run_id = uuid4().hex[:8]
    dml = f"{PROBE_PREFIX}_{run_id}_dml"
    drop_me = f"{PROBE_PREFIX}_{run_id}_drop"
    ctas = f"{PROBE_PREFIX}_{run_id}_ctas"
    scratch = [dml, drop_me, ctas]

    print(f"scratch tables : {', '.join(scratch)}")
    print("               (created by this run, dropped in a finally block)\n")

    async def seed(sess: McpSession) -> None:
        for table in (dml, drop_me):
            ok, err, _ = await sess.sql(create_probe_ddl(dialect, table))
            if not ok and "exist" not in err.lower():
                raise RuntimeError(f"could not create {table}: {err}")
        for i in (1, 2, 3):
            await sess.sql(f"INSERT INTO {dml} (id, v) VALUES ({i}, 'seed{i}')")

    async def drop_all(sess: McpSession) -> list[str]:
        survivors = []
        for table in scratch:
            await sess.sql(f"DROP TABLE {table}")
            if not dry_run and await table_exists(sess, table):
                survivors.append(table)
        return survivors

    cases = live_cases(dml, drop_me, ctas)
    failures: list[str] = []

    try:
        # 1. preflight + setup, flag off
        async with McpSession(cfg_off, datasource, dry_run=dry_run) as sess:
            for table in scratch:
                if not dry_run and await table_exists(sess, table):
                    print(f"ABORT: {table} already exists; refusing to touch it")
                    return 1
            print("1. creating scratch tables (flag off) ...")
            await seed(sess)

            # 2. flag-off matrix -- the negative control: these really execute
            print("2. running probes with sql_read_only: false (writes execute) ...")
            off: dict[str, tuple[bool, str]] = {}
            for label, statement, _ in cases:
                ok, err, _ = await sess.sql(statement)
                off[label] = (ok, err)

            # 3. restore what the flag-off run destroyed, so both runs start level
            print("3. restoring scratch tables ...")
            await sess.sql(f"DROP TABLE {ctas}")
            await sess.sql(f"DROP TABLE {dml}")
            await sess.sql(f"DROP TABLE {drop_me}")
            await seed(sess)
            baseline = await row_count(sess, dml)

        # 4. flag-on matrix
        print("4. running probes with sql_read_only: true  (must all be refused) ...")
        async with McpSession(cfg_on, datasource, dry_run=dry_run) as sess:
            on: dict[str, tuple[bool, str]] = {}
            for label, statement, _ in cases:
                ok, err, _ = await sess.sql(statement)
                on[label] = (ok, err)

            # 5. the decisive check -- reads still work under the flag, so this
            #    integrity verification runs inside the hardened session.
            print("5. verifying the datasource is untouched ...\n")
            after = await row_count(sess, dml)
            drop_survived = await table_exists(sess, drop_me)
            ctas_absent = not await table_exists(sess, ctas)

        if dry_run:
            # Nothing ran, so every probe reports the stub's success and the
            # verdict table would be uniformly, misleadingly red. The statement
            # listing above is the whole point of a dry run.
            print("dry run: statements listed above were NOT executed; no verdicts to report")
            return 0

        hdr = f"{'statement':<22} {'flag off':<14} {'flag on':<14} verdict"
        print(hdr)
        print("-" * len(hdr))
        for label, _, is_read in cases:
            off_ok, off_err = off[label]
            on_ok, on_err = on[label]
            gated = GATE_MARKER in on_err.lower()

            if is_read:
                good = not gated
                verdict = "read not blocked" if good else "READ WRONGLY BLOCKED"
            elif off_err and off_err == on_err:
                good = True
                verdict = "blocked either way (not the switch)"
            elif not off_ok:
                good = not on_ok
                verdict = f"inconclusive - failed with flag off too ({off_err[:40]})"
            else:
                good = (not on_ok) and gated
                verdict = "executed off, refused on" if good else "EXECUTED WITH FLAG ON"

            if not good:
                failures.append(label)
            print(
                f"{label:<22} {'executed' if off_ok else 'failed':<14} "
                f"{'executed' if on_ok else 'refused':<14} {'' if good else '!! '}{verdict}"
            )

        print()
        # An unreadable count is a failure, not a match: if both reads failed,
        # `baseline == after` would compare None with None and report success on
        # a datasource that may well have been mutated.
        counted = baseline is not None and after is not None
        count_text = (
            f"{dml} row count unchanged ({baseline} -> {after})"
            if counted
            else f"{dml} row count could not be read ({baseline} -> {after})"
        )
        checks = [
            (count_text, counted and baseline == after),
            (f"{drop_me} survived the DROP probe", drop_survived),
            (f"{ctas} was never created", ctas_absent),
        ]
        for text, good in checks:
            print(f"  {'ok  ' if good else 'FAIL'} {text}")
            if not good:
                failures.append(text)

    finally:
        print("\n6. dropping scratch tables ...")
        try:
            async with McpSession(cfg_off, datasource, dry_run=dry_run) as sess:
                survivors = await drop_all(sess)
            if survivors:
                print(f"  !! COULD NOT DROP: {', '.join(survivors)}")
                print("  Run these by hand:")
                for table in survivors:
                    print(f"    DROP TABLE {table};")
                failures.append("teardown")
            else:
                print("  ok   all scratch tables removed")
        except Exception as exc:  # noqa: BLE001 - teardown must report, never mask
            print(f"  !! teardown failed: {exc}")
            print("  Run these by hand:")
            for table in scratch:
                print(f"    DROP TABLE {table};")
            failures.append("teardown")

    print()
    if failures:
        print(f"FAIL: {', '.join(failures)}")
        return 1
    print("PASS - agent.sql_read_only blocks real writes against the live datasource")
    return 0


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--endpoint",
        default=None,
        help="probe a server that is ALREADY running, over streamable HTTP "
        "(e.g. http://127.0.0.1:8000/mcp) instead of launching one. Reports what "
        "it observes; it cannot arrange the flag-on/flag-off contrast the other "
        "modes rely on. Makes --project optional.",
    )
    ap.add_argument("--project", required=False, type=Path)
    ap.add_argument(
        "--live-writes",
        action="store_true",
        help="WRITES TO THE PROJECT'S REAL DATASOURCE. Creates its own scratch tables, "
        "runs the probes against them for real, verifies the flag-on run changed "
        "nothing, then drops them in a finally block. Only for a datasource you are "
        "willing to write to.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="with --live-writes, print every statement instead of executing it",
    )
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

    if args.endpoint:
        if args.live_writes or args.sqlite_standin:
            print("--endpoint cannot be combined with the modes that launch a server", file=sys.stderr)
            return 1
        print(f"endpoint      : {args.endpoint} (already running; its config is whatever it was started with)")
        return await run_endpoint(args.endpoint)

    if not args.project:
        print("--project is required unless --endpoint is given", file=sys.stderr)
        return 1
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
    if args.live_writes and args.sqlite_standin:
        print("--live-writes and --sqlite-standin are mutually exclusive", file=sys.stderr)
        return 1
    if args.dry_run and not args.live_writes:
        # Only run_live consults dry_run. Accepting it on the matrix path would
        # execute every probe while the operator believed nothing ran -- and the
        # matrix probes are the ones aimed at a real datasource.
        print("--dry-run only applies to --live-writes; the matrix path always executes", file=sys.stderr)
        return 1
    if args.live_writes:
        print(f"MODE       : LIVE WRITES against the real datasource{' (dry run)' if args.dry_run else ''}\n")
    if args.sqlite_standin:
        print(
            "datasource : SUBSTITUTED with a throwaway sqlite (--sqlite-standin); "
            "rest of the project config is used as-is"
        )
        print(
            f"probes     : non-read statements target `{MISSING}`, which EXISTS in "
            f"the stand-in, so writes really execute when the flag is off\n"
        )
    elif not args.live_writes:
        print(f"probes     : non-read statements target `{MISSING}`, which must not exist\n")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        on_dir, off_dir = root / "on", root / "off"
        on_dir.mkdir()
        off_dir.mkdir()
        standin = datasource if args.sqlite_standin else None
        cfg_on = stage_config(project, on_dir, sql_read_only=True, standin=standin)
        cfg_off = stage_config(project, off_dir, sql_read_only=False, standin=standin)

        if args.live_writes:
            dialect = datasources.get(datasource, {}).get("type", "")
            return await run_live(cfg_on, cfg_off, datasource, dialect, args.dry_run)

        print("running matrix with sql_read_only: true  ...")
        on = await run_matrix(cfg_on, datasource)
        print("running matrix with sql_read_only: false ...\n")
        off = await run_matrix(cfg_off, datasource)

    hdr = f"{'statement':<22} {'flag off':<14} {'flag on':<14} verdict"
    print(hdr)
    print("-" * len(hdr))

    failures = []
    for label, _, is_read in CASES:

        def where(entry: tuple[bool, str]) -> str:
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
