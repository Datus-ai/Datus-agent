# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Request-local SQL calculation evidence. No database access or SQL execution.

Query scopes and output dependencies form a graph: shared filters and upstream
calculations are stored once, rather than expanded into every metric formula.
An output is evidence for authoring, not an instruction to create a metric.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlglot import expressions as exp
from sqlglot.optimizer.scope import Scope, traverse_scope

from datus.utils.sql_lineage import extract_comments, preprocess_template, split_statements


@dataclass
class ParsedStatement:
    source: str
    line: int
    sql: str
    tree: exp.Expression | None
    templated: bool = False
    error: str = ""
    input_kind: str = "unknown"
    identity: str = ""


def query_of(tree: exp.Expression) -> tuple[exp.Query | None, exp.Table | None]:
    """Extract the read query from a statement, preserving a DML-level WITH."""
    target = None
    if isinstance(tree, (exp.Insert, exp.Create)):
        target = tree.this.this if isinstance(tree.this, exp.Schema) else tree.this
        query = tree.expression
    else:
        query = tree
    if not isinstance(query, exp.Query):
        return None, target if isinstance(target, exp.Table) else None
    query = query.copy()
    for key in ("with", "with_"):
        if tree is not query and tree.args.get(key) is not None and query.args.get(key) is None:
            query.set(key, tree.args[key].copy())
    return query, target if isinstance(target, exp.Table) else None


class SemanticSourceAnalyzer:
    """Parse each source once; resolve columns again when live schemas arrive."""

    def __init__(
        self, dialect: str | None = None, default_database: str = "", *, source_context: dict[str, str] | None = None
    ):
        self.dialect = dialect or None
        self.default_database = default_database
        self.source_context = dict(source_context or {})
        self._parsed: dict[str, list[ParsedStatement]] = {}
        self._comments: dict[str, list] = {}

    def parse(self, entries: list[dict]) -> list[ParsedStatement]:
        statements = []
        for entry in entries:
            text = entry["sql"]
            digest = hashlib.sha256(text.encode()).hexdigest()
            name = entry["name"]
            if digest not in self._parsed:
                parsed = []
                self._comments[digest] = extract_comments(text, name)
                try:
                    # Do not use the lineage splitter's recovery on broken quotes:
                    # statement boundaries would no longer be trustworthy.
                    sqlglot.tokenize(text, read=self.dialect)
                    pieces = split_statements(text, self.dialect)
                except Exception as exc:
                    pieces = []
                    parsed.append(ParsedStatement(name, 1, text, None, error=str(exc)))
                for raw, line in pieces:
                    normalized, templated = preprocess_template(raw)
                    try:
                        tree = sqlglot.parse_one(normalized, read=self.dialect)
                        parsed.append(ParsedStatement(name, line + 1, raw, tree, templated))
                    except Exception as exc:
                        parsed.append(ParsedStatement(name, line + 1, raw, None, templated, str(exc)))
                self._parsed[digest] = parsed
            # Provenance is per input, even when identical content was cached.
            statements.extend(
                ParsedStatement(
                    name,
                    p.line,
                    p.sql,
                    p.tree,
                    p.templated,
                    p.error,
                    entry.get("input_kind", "unknown"),
                    entry.get("identity", name),
                )
                for p in self._parsed[digest]
            )
        return statements

    def analyze(self, entries: list[dict], schemas: dict[str, list[str]] | None = None) -> dict:
        self.schemas = dict(schemas or {})
        self.queries: dict[str, dict] = {}
        self.outputs: dict[str, dict] = {}
        self.issues: list[dict] = []
        self.statements: list[dict] = []
        self.tables: dict[str, None] = {}
        self._scope_ids: dict[int, str] = {}
        self._scope_outputs: dict[int, list[str]] = {}
        parsed_statements = self.parse(entries)
        statement_ids = self._statement_ids(parsed_statements)
        for parsed, sid in zip(parsed_statements, statement_ids):
            try:
                self._statement(parsed, sid)
            except Exception as exc:
                self._issue(sid, "statement_analysis_failed", source=parsed.source, line=parsed.line, error=str(exc))
                self.statements[-1].update(status="unresolved", outputs=[])
        self._link_producers()
        return {
            "statements": self.statements,
            "queries": list(self.queries.values()),
            "outputs": list(self.outputs.values()),
            "tables": list(self.tables),
            "issues": self.issues,
            "complete": not self.issues,
            "comments": [
                {"source": entry["name"], "line": comment.line, "kind": comment.kind, "text": comment.text}
                for entry in entries
                for comment in self._comments.get(hashlib.sha256(entry["sql"].encode()).hexdigest(), [])
                if comment.kind != "code"
            ],
        }

    def _statement_ids(self, statements: list[ParsedStatement]) -> list[str]:
        """Unrelated additions do not invalidate references; affected evidence does.

        Include referenced schemas and supplied upstream writers, so a changed
        dependency cannot silently change the meaning of an existing output ID.
        """
        tables, targets = [], {}
        for i, statement in enumerate(statements):
            query, target = query_of(statement.tree) if statement.tree is not None else (None, None)
            names = {self._table(t) for t in query.find_all(exp.Table)} if query is not None else set()
            tables.append(names | ({self._table(target)} if target is not None else set()))
            if target is not None:
                targets.setdefault(self._table(target), []).append(i)

        def fingerprint(index):
            visited = set()

            def include(i):
                if i in visited:
                    return
                visited.add(i)
                for table in tables[i]:
                    for writer in targets.get(table, []):
                        if writer != i and (statements[writer].source != statements[i].source or writer < i):
                            include(writer)

            include(index)
            evidence = [
                [s.identity, s.line, s.sql, s.input_kind, {t: self.schemas.get(t) for t in sorted(tables[i])}]
                for i in sorted(visited)
                for s in [statements[i]]
            ]
            return (
                "s_"
                + hashlib.sha256(
                    json.dumps(
                        [self.dialect, self.default_database, self.source_context, evidence], sort_keys=True
                    ).encode()
                ).hexdigest()[:20]
            )

        return [fingerprint(i) for i in range(len(statements))]

    def _sql(self, expression: exp.Expression | None) -> str | None:
        if expression is None:
            return None
        rendered = expression.sql(dialect=self.dialect, comments=False)
        return re.sub(r"__tpl_([A-Za-z_][A-Za-z0-9_]*)__", r"${\1}", rendered)

    def _table(self, table: exp.Table) -> str:
        named = table.copy()
        named.set("alias", None)
        if not named.db and self.default_database:
            named.set("db", exp.to_identifier(self.default_database))
        return named.sql(dialect=self.dialect, comments=False)

    def _issue(self, context: str, reason: str, **detail) -> None:
        issue = {"context": context, "reason": reason, **detail}
        if issue not in self.issues:
            self.issues.append(issue)

    def _schema(self, table: str) -> list[str] | None:
        if table in self.schemas:
            return self.schemas[table]
        return None

    def _statement(self, parsed: ParsedStatement, sid: str) -> None:
        statement: dict[str, Any] = {
            "id": sid,
            "source": parsed.source,
            "input_kind": parsed.input_kind,
            "line": parsed.line,
            "original_sql": parsed.sql,
            "outputs": [],
            "parameters": list(dict.fromkeys(re.findall(r"\$\{([^}]+)\}", parsed.sql))),
            "templated": parsed.templated,
        }
        self.statements.append(statement)
        if parsed.tree is None:
            statement["status"] = "parse_error"
            self._issue(sid, "parse_error", error=parsed.error)
            return
        tree = parsed.tree
        statement["operation"] = tree.key.upper()
        query, target = query_of(tree)
        if target is not None:
            name = self._table(target)
            statement["target_table"] = name
            self.tables[name] = None
        if query is None:
            # Plain DDL supplies useful column order for later INSERTs.
            if isinstance(tree, exp.Create) and isinstance(tree.this, exp.Schema):
                table = tree.this.this
                if isinstance(table, exp.Table):
                    name = self._table(table)
                    self.tables[name] = None
                    self.schemas.setdefault(
                        name, [c.name for c in tree.this.expressions if isinstance(c, exp.ColumnDef)]
                    )
            statement["status"] = "auxiliary"
            if isinstance(tree, (exp.Merge, exp.Update, exp.Delete, exp.Command)):
                self._issue(sid, "unsupported_statement", operation=statement["operation"])
            return
        statement["query_sql"] = self._sql(query)
        try:
            scopes = list(traverse_scope(query))
        except Exception as exc:
            self._issue(sid, "scope_resolution_failed", error=str(exc))
            statement["status"] = "unresolved"
            return
        for i, scope in enumerate(scopes, 1):
            qid = f"{sid}:q{i}"
            self._scope_ids[id(scope)] = qid
            self._scope(scope, qid, sid)
        if not scopes:
            self._issue(sid, "unsupported_query", sql=self._sql(query))
            statement["status"] = "unresolved"
            return
        root = scopes[-1]
        statement["query_id"] = self._scope_ids[id(root)]
        statement["outputs"] = self._scope_outputs[id(root)]
        statement["status"] = "analyzed"
        statement["read_tables"] = list(
            dict.fromkeys(
                self._table(source)
                for scope in scopes
                for source in scope.sources.values()
                if isinstance(source, exp.Table) and isinstance(source.this, exp.Identifier)
            )
        )
        if target is not None:
            declared = tree.this.expressions if isinstance(tree.this, exp.Schema) else []
            names = [c.name for c in declared if isinstance(c, (exp.Identifier, exp.ColumnDef))]
            if not names and isinstance(tree, exp.Create) and all(p.alias_or_name for p in root.expression.selects):
                # CTAS/CREATE VIEW define their column names from the query;
                # the intermediate table need not already exist in the DB.
                names = [self.outputs[oid]["name"] for oid in statement["outputs"]]
                self.schemas[statement["target_table"]] = names
            if not names:
                names = self._schema(statement["target_table"]) or []
            if len(names) == len(statement["outputs"]):
                for name, oid in zip(names, statement["outputs"]):
                    self.outputs[oid]["target_column"] = name
            else:
                self._issue(sid, "target_column_mapping_unresolved", target=statement["target_table"])

    def _scope(self, scope: Scope, qid: str, sid: str) -> None:
        node = scope.expression
        record: dict[str, Any] = {"id": qid, "statement_id": sid, "outputs": []}
        self.queries[qid] = record
        if isinstance(node, exp.SetOperation):
            branches = getattr(scope, "set_operation_scopes", None) or getattr(scope, "union_scopes", [])
            record.update(
                operation=node.key.upper(),
                distinct=node.args.get("distinct") is not False,
                branches=[self._scope_ids[id(s)] for s in branches],
                order_by=self._sql(node.args.get("order")),
                limit=self._sql(node.args.get("limit")),
            )
            widths = [len(self._scope_outputs[id(s)]) for s in branches]
            if len(set(widths)) != 1:
                self._issue(qid, "set_operation_column_count_mismatch", widths=widths)
            for i, first in enumerate(self._scope_outputs[id(branches[0])] if branches else []):
                dependencies = [self._scope_outputs[id(s)][i] for s in branches if i < len(self._scope_outputs[id(s)])]
                self._output(qid, i, self.outputs[first]["name"], None, "set_output", dependencies, [])
            self._scope_outputs[id(scope)] = record["outputs"]
            return
        if not isinstance(node, exp.Select):
            self._issue(qid, "unsupported_scope", expression=node.key)
            self._scope_outputs[id(scope)] = []
            return
        inputs = []
        for alias, (_, source) in scope.selected_sources.items():
            if isinstance(source, exp.Table) and isinstance(source.this, exp.Identifier):
                table = self._table(source)
                self.tables[table] = None
                inputs.append({"alias": alias, "table": table})
            elif isinstance(source, Scope):
                inputs.append({"alias": alias, "query_id": self._scope_ids.get(id(source))})
                if id(source) not in self._scope_ids:
                    self._issue(qid, "query_source_unresolved", alias=alias)
            else:
                self._issue(qid, "unsupported_relation", alias=alias)
        record.update(
            operation="SELECT",
            inputs=inputs,
            distinct=self._sql(node.args.get("distinct")),
            filters={
                key: self._sql(node.args.get(key)) for key in ("where", "having", "qualify") if node.args.get(key)
            },
            joins=[
                {
                    "relation": join.this.alias_or_name,
                    "join_type": " ".join(str(join.args.get(k) or "") for k in ("method", "side", "kind")).strip()
                    or "INNER",
                    "condition": self._sql(join.args.get("on")),
                    "using": [self._sql(c) for c in join.args.get("using") or []],
                    "cardinality": "unverified",
                }
                for join in node.args.get("joins") or []
            ],
            order_by=self._sql(node.args.get("order")),
            limit=self._sql(node.args.get("limit")),
            named_windows=[self._sql(window) for window in node.args.get("windows") or []],
            subqueries=[self._scope_ids[id(child)] for child in scope.subquery_scopes],
            group_by=[],
            group_by_outputs=[],
            group_by_output_sets=[],
        )
        projections = self._expand_stars(scope, qid)
        for i, projection in enumerate(projections):
            expr = projection.this if isinstance(projection, exp.Alias) else projection
            # Expanded stars create detached Column nodes. Walk the expression
            # itself, stopping at nested queries, instead of relying on parents.
            own_nodes = [n for n in expr.walk(prune=lambda n: isinstance(n, exp.Query)) if not isinstance(n, exp.Query)]
            windows = [n for n in own_nodes if isinstance(n, exp.Window)]
            kind = (
                "window"
                if windows
                else "aggregate"
                if any(isinstance(n, exp.AggFunc) for n in own_nodes)
                else "passthrough"
                if isinstance(expr, exp.Column)
                else "expression"
            )
            dependencies, columns = [], []
            for column in (n for n in own_nodes if isinstance(n, exp.Column) and not n.is_star):
                deps, origins = self._resolve(scope, column, qid)
                dependencies.extend(deps)
                columns.extend(origins)
            for child in scope.subquery_scopes:
                if any(n is child.expression for n in expr.walk()):
                    dependencies.extend(self._scope_outputs.get(id(child), []))
            name = scope.outer_columns[i] if i < len(scope.outer_columns) else projection.alias_or_name
            oid = self._output(qid, i, name or f"column_{i + 1}", self._sql(expr), kind, dependencies, columns)
            if windows:
                self.outputs[oid]["windows"] = [self._sql(window) for window in windows]
        self._scope_outputs[id(scope)] = record["outputs"]
        group = node.args.get("group")
        for key in group.expressions if group else []:
            resolved = key
            if isinstance(key, exp.Literal) and key.is_int and 1 <= int(key.this) <= len(projections):
                resolved = projections[int(key.this) - 1]
            elif isinstance(key, exp.Column) and not key.table:
                matches = [p for p in projections if isinstance(p, exp.Alias) and p.alias == key.name]
                if len(matches) == 1:
                    resolved = matches[0]
            resolved = resolved.this if isinstance(resolved, exp.Alias) else resolved
            record["group_by"].append(self._sql(resolved))
            matches = [oid for oid in record["outputs"] if self.outputs[oid]["expression"] == self._sql(resolved)]
            record["group_by_outputs"].extend(matches)
            record["group_by_output_sets"].append(matches)
            if not matches:
                self._issue(qid, "group_key_not_projected", expression=self._sql(resolved))
        if group and any(group.args.get(k) for k in ("grouping_sets", "cube", "rollup", "all")):
            record["grouping"] = self._sql(group)
            self._issue(qid, "grouping_requires_interpretation", expression=self._sql(group))

        # Row populations can depend on projected calculations even when the
        # selected value is COUNT(*). Keep those dependencies alongside the
        # predicates, rather than losing their definitions during detail reads.
        context_dependencies, context_columns = [], []
        for clause_name in ("where", "having", "qualify", "joins", "order"):
            clauses = node.args.get(clause_name) or []
            for clause in clauses if isinstance(clauses, list) else [clauses]:
                for column in clause.find_all(exp.Column):
                    if column.find_ancestor(exp.Query) is not node or column.is_star:
                        continue
                    aliases = [
                        oid
                        for oid in record["outputs"]
                        if not column.table and self.outputs[oid]["name"] == column.name
                    ]
                    if clause_name in ("having", "qualify", "order") and len(aliases) == 1:
                        context_dependencies.extend(aliases)
                    else:
                        deps, origins = self._resolve(scope, column, qid)
                        context_dependencies.extend(deps)
                        context_columns.extend(origins)
        record["dependencies"] = list(dict.fromkeys(context_dependencies))
        record["source_columns"] = [dict(items) for items in dict.fromkeys(tuple(c.items()) for c in context_columns)]

    def _expand_stars(self, scope: Scope, qid: str) -> list[exp.Expression]:
        projections = []
        for projection in scope.expression.selects:
            if not projection.is_star:
                projections.append(projection)
                continue
            star = projection.this if isinstance(projection, exp.Column) else projection
            if any(star.args.get(key) for key in ("except", "except_", "replace", "rename")):
                self._issue(qid, "star_expansion_unresolved", expression=self._sql(projection))
                continue
            if isinstance(projection, exp.Column):
                sources = [(projection.table, scope.sources.get(projection.table))]
            else:
                sources = [(alias, source) for alias, (_, source) in scope.selected_sources.items()]
            for alias, source in sources:
                if isinstance(source, Scope):
                    names = [self.outputs[oid]["name"] for oid in self._scope_outputs.get(id(source), [])]
                elif isinstance(source, exp.Table):
                    names = self._schema(self._table(source)) or []
                else:
                    names = []
                if not names:
                    self._issue(qid, "star_expansion_unresolved", expression=self._sql(projection), alias=alias)
                projections.extend(exp.column(name, table=alias) for name in names)
        return projections

    def _resolve(self, scope: Scope, column: exp.Column, qid: str) -> tuple[list[str], list[dict]]:
        if column.table:
            candidates = [(column.table, scope.sources.get(column.table))]
        else:
            candidates = []
            unknown = []
            for alias, (_, source) in scope.selected_sources.items():
                if isinstance(source, Scope):
                    names = [self.outputs[oid]["name"] for oid in self._scope_outputs.get(id(source), [])]
                elif isinstance(source, exp.Table):
                    names = self._schema(self._table(source))
                else:
                    names = None
                if names is None:
                    unknown.append((alias, source))
                elif column.name in names:
                    candidates.append((alias, source))
            candidates.extend(unknown)
            if not candidates and len(scope.selected_sources) == 1:
                alias, (_, source) = next(iter(scope.selected_sources.items()))
                candidates = [(alias, source)]
        if len(candidates) != 1 or candidates[0][1] is None:
            self._issue(qid, "column_origin_unresolved", expression=self._sql(column))
            return [], []
        _, source = candidates[0]
        if isinstance(source, exp.Table):
            table = self._table(source)
            names = self._schema(table)
            if names is not None and column.name not in names:
                self._issue(qid, "source_column_missing", table=table, column=column.name)
            return [], [{"table": table, "column": column.name}]
        if isinstance(source, Scope):
            matches = [
                oid for oid in self._scope_outputs.get(id(source), []) if self.outputs[oid]["name"] == column.name
            ]
            if len(matches) == 1:
                return matches, []
        self._issue(qid, "column_origin_unresolved", expression=self._sql(column))
        return [], []

    def _output(self, qid, index, name, expression, kind, dependencies, columns) -> str:
        oid = f"{qid}:o{index + 1}"
        self.outputs[oid] = {
            "id": oid,
            "query_id": qid,
            "name": name,
            "expression": expression,
            "kind": kind,
            "dependencies": list(dict.fromkeys(dependencies)),
            "source_columns": [dict(items) for items in dict.fromkeys(tuple(c.items()) for c in columns)],
        }
        self.queries[qid]["outputs"].append(oid)
        return oid

    def _link_producers(self) -> None:
        producers: dict[tuple[str, str], list[str]] = {}
        statements = {s["id"]: s for s in self.statements}
        statement_order = {sid: i for i, sid in enumerate(statements)}
        for stmt in self.statements:
            if not stmt.get("target_table"):
                continue
            for oid in stmt["outputs"]:
                output = self.outputs[oid]
                if output.get("target_column"):
                    producers.setdefault((stmt["target_table"], output["target_column"]), []).append(oid)
        # Query dependencies describe row populations; output dependencies
        # describe values. Resolve both across supplied physical-table writers.
        for oid, output in [*self.outputs.items(), *self.queries.items()]:
            for column in output.get("source_columns", []):
                matches = [
                    candidate
                    for candidate in producers.get((column["table"], column["column"]), [])
                    if candidate.split(":", 1)[0] != oid.split(":", 1)[0]
                    and (
                        statements[candidate.split(":", 1)[0]]["source"] != statements[oid.split(":", 1)[0]]["source"]
                        or statement_order[candidate.split(":", 1)[0]] < statement_order[oid.split(":", 1)[0]]
                    )
                ]
                if len(matches) == 1:
                    output["dependencies"].append(matches[0])
                elif len(matches) > 1:
                    self._issue(oid, "multiple_upstream_writers", candidates=matches)
        for oid in self.outputs:
            self.outputs[oid]["calculation_queries"] = self._calculation_queries(oid)

    def _calculation_queries(self, oid: str) -> list[str]:
        visited, active, calculations = set(), set(), []

        def visit(current):
            if current in active:
                self._issue(oid, "cyclic_output_dependency", dependency=current)
                return
            if current in visited:
                return
            visited.add(current)
            active.add(current)
            output = self.outputs[current]
            if output["kind"] in {"aggregate", "window"}:
                calculations.append(output["query_id"])
            for dependency in output["dependencies"]:
                visit(dependency)
            active.remove(current)

        visit(oid)
        return list(dict.fromkeys(calculations))
