# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Plan a synthetic datasource from DDL alone, before a generator exists.

The ``gen-datasource`` engine prints its whole plan - table roles, the row allocation, column
semantics, the resolved date window, sample generated names, which columns get a business code,
the daily-metric grid - and needs nothing but the DDL to do it. The skill has always said to read
that plan instead of reading the engine.

Reaching it used to cost more than it looks. ``report()`` is a method on ``DDLEngine``, so the
agent had to author ``data/gen.py`` first; the plan was cheap for the engine and expensive for the
model. A measured production run (Datus-saas-dev, trace e2c917f9b380db89efedb580b28e9369) skipped
it, tried to divide an 80,000-row budget across five tables by hand, could not make the total come
out, opened ``ddl_engine.py`` to find the allocator, and spent 36 turns and 19,885 output tokens in
the source without generating a single row.

This module makes the plan reachable in one call, and puts it in the tool schema where a model
that ignores prose still sees it.

The engine lives in the skill bundle (``datus/resources/skills/gen-datasource/scripts``) rather
than in an importable package, so it is loaded by path, once, and lazily - importing it is not
worth doing for every process that mounts the database tools, and a trimmed install that ships no
skills must degrade to a clear message rather than an ImportError at startup.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import threading
from datetime import date, datetime
from pathlib import Path
from types import ModuleType
from typing import Optional

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

#: Guard against a pathological DDL turning a planning call into a long parse.
MAX_DDL_CHARS = 200_000

SKILL_NAME = "gen-datasource"

_engine_lock = threading.Lock()
_engine_module: Optional[ModuleType] = None

#: ``report()`` writes to stdout, and ``redirect_stdout`` swaps ``sys.stdout`` for the whole
#: process. Two planning calls running at once - the agent framework can dispatch tool calls
#: concurrently - would otherwise interleave into each other's buffer, or restore stdout out of
#: order and leave the host's own output pointed at a dead StringIO.
_capture_lock = threading.Lock()


class DatasourcePlanError(Exception):
    """Raised when the plan cannot be produced."""


def skill_scripts_dir() -> Path:
    """Where the packaged engine lives, whether or not it is actually there."""
    import datus

    return Path(datus.__file__).resolve().parent / "resources" / "skills" / SKILL_NAME / "scripts"


def load_engine() -> ModuleType:
    """Import ``ddl_engine`` from the skill bundle, once per process.

    ``ddl_engine`` imports ``genlib`` from its own directory, so that directory has to be on
    ``sys.path`` for the duration of the import. It is removed afterwards: leaving it there would
    let any later ``import genlib`` anywhere in the process resolve to the bundle.
    """
    global _engine_module

    with _engine_lock:
        if _engine_module is not None:
            return _engine_module

        scripts = skill_scripts_dir()
        source = scripts / "ddl_engine.py"
        if not source.exists():
            raise DatasourcePlanError(
                f"The gen-datasource engine is not present in this installation (looked for {source}). "
                f"Planning is unavailable; the skill's own instructions still apply."
            )

        added = str(scripts)
        sys.path.insert(0, added)
        try:
            spec = importlib.util.spec_from_file_location("_datus_gen_datasource_engine", source)
            if spec is None or spec.loader is None:
                raise DatasourcePlanError(f"Could not load the gen-datasource engine from {source}.")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except DatasourcePlanError:
            raise
        except Exception as e:  # a broken bundle must not look like a broken datasource
            raise DatasourcePlanError(f"The gen-datasource engine failed to import: {e}") from e
        finally:
            with contextlib.suppress(ValueError):
                sys.path.remove(added)

        _engine_module = module
        return module


def _parse_end_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except ValueError as e:
        raise DatasourcePlanError(f"end_date must be YYYY-MM-DD, got {value!r}.") from e


def plan_from_ddl(
    ddl: str,
    rows: int = 80_000,
    months: int = 17,
    end_date: Optional[str] = None,
    seed: int = 42,
) -> str:
    """Return what ``DDLEngine(...).report()`` prints for this DDL, generating nothing.

    The profile is deliberately empty. The plan is what the engine infers from the DDL alone, which
    is the thing worth seeing before writing a profile - the skill's order of work is "run report,
    then override only what is wrong".
    """
    if not ddl or not ddl.strip():
        raise DatasourcePlanError("ddl is empty; pass the CREATE TABLE statements to plan.")
    if len(ddl) > MAX_DDL_CHARS:
        raise DatasourcePlanError(f"ddl is {len(ddl)} characters, above the {MAX_DDL_CHARS} limit for planning.")
    if rows < 1:
        raise DatasourcePlanError(f"rows must be positive, got {rows}.")
    if months < 1:
        raise DatasourcePlanError(f"months must be positive, got {months}.")

    engine_module = load_engine()
    resolved_end = _parse_end_date(end_date)  # validate before taking the capture lock
    captured = io.StringIO()
    try:
        # report() writes to stdout: it is normally read back from a subprocess. Here the caller
        # wants the text, and a stray print must not reach the host process's stdout. The lock
        # covers construction as well - the engine prints warnings from __init__ too.
        with _capture_lock, contextlib.redirect_stdout(captured):
            engine = engine_module.DDLEngine(
                ddl,
                rows=rows,
                profile={},
                months=months,
                end_date=resolved_end,
                seed=seed,
            )
            engine.report()
    except DatasourcePlanError:
        raise
    except Exception as e:
        # Almost always a DDL the engine cannot parse. Hand back what it managed to print first -
        # the partial plan usually names the table it choked on.
        partial = captured.getvalue().strip()
        detail = f" Output before the failure:\n{partial}" if partial else ""
        raise DatasourcePlanError(f"The engine could not plan this DDL: {e}.{detail}") from e

    plan = captured.getvalue().strip()
    if not plan:
        raise DatasourcePlanError("The engine produced no plan for this DDL; check that it contains CREATE TABLE.")
    return plan
