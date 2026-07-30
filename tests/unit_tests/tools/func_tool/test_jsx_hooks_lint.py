# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for the "early return before hooks" scanner.

The scanner runs inside ``validate_render`` and *fails* the subagent's terminal
action, so a false positive blocks a perfectly good artifact and sends the model
editing correct code. The bulk of this file is therefore negative cases — code
that must never be flagged — grouped by the tokenizer hazard they exercise:
nested function frames, JSX text, strings/templates/regex, comments, and
property keys named ``return``.
"""

from __future__ import annotations

import pytest

from datus.tools.func_tool._jsx_hooks_lint import find_hook_order_issues


def names(source: str) -> list[str]:
    return [issue.hook_name for issue in find_hook_order_issues(source)]


# --------------------------------------------------------------------------- #
# Nested function frames — a `return` belongs to its own function              #
# --------------------------------------------------------------------------- #


class TestNestedFunctionFrames:
    def test_inner_arrow_early_return_does_not_leak_to_the_component(self):
        # The exact shape that must never be misjudged: an inner arrow returns
        # early, then the *component* goes on to call its hooks.
        source = """
export default function App() {
  const f = () => { if (xx) return; };

  useXXHooks();
  const [a, setA] = useState();

  return <View />;
}
"""
        assert find_hook_order_issues(source) == []

    def test_use_effect_callback_early_return(self):
        source = """
export default function App() {
  const { data } = useQuerySql('queries/sales');

  useEffect(() => {
    if (!data) return;
    track(data);
  }, [data]);

  const totals = useMemo(() => sum(data), [data]);
  return <div>{totals}</div>;
}
"""
        assert find_hook_order_issues(source) == []

    def test_use_callback_with_guard_then_more_hooks(self):
        source = """
export default function App() {
  const onPick = useCallback((row) => {
    if (!row) {
      return null;
    }
    return row.id;
  }, []);

  const [sel, setSel] = useState(null);
  return <Table onPick={onPick} sel={sel} />;
}
"""
        assert find_hook_order_issues(source) == []

    def test_map_callback_before_hooks(self):
        source = """
export default function App({ data }) {
  const rows = data.map((d) => {
    if (!d) return null;
    return d.value;
  });

  const [page, setPage] = useState(0);
  return <List rows={rows} page={page} />;
}
"""
        assert find_hook_order_issues(source) == []

    def test_helper_function_declared_above_the_component(self):
        source = """
function formatValue(v) {
  if (v == null) return '-';
  return String(v);
}

export default function App() {
  const [a] = useState();
  return <div>{formatValue(a)}</div>;
}
"""
        assert find_hook_order_issues(source) == []

    def test_nested_component_defined_after_the_outer_return(self):
        source = """
export default function App() {
  const [a] = useState();
  if (!a) return null;

  function Inner() {
    const [b] = useState();
    return <div>{b}</div>;
  }

  return <Inner />;
}
"""
        assert find_hook_order_issues(source) == []

    def test_arrow_inside_jsx_attribute_before_a_later_hook(self):
        source = """
export default function App() {
  const el = <button onClick={() => { if (!ready) return; go(); }}>Go</button>;
  const [ready, setReady] = useState(false);
  return el;
}
"""
        assert find_hook_order_issues(source) == []

    def test_object_method_shorthand_with_early_return(self):
        source = """
export default function App() {
  const api = {
    load(x) {
      if (!x) return null;
      return fetchIt(x);
    },
  };

  const [a] = useState();
  return <div>{api.load(a)}</div>;
}
"""
        assert find_hook_order_issues(source) == []

    def test_iife_with_early_return(self):
        source = """
export default function App() {
  const cfg = (function () {
    if (!window.cfg) return {};
    return window.cfg;
  })();

  const [a] = useState(cfg);
  return <div>{a}</div>;
}
"""
        assert find_hook_order_issues(source) == []

    def test_async_arrow_with_early_return(self):
        source = """
export default function App() {
  const load = async () => {
    if (!id) return;
    await go(id);
  };

  const [a] = useState();
  return <div onClick={load}>{a}</div>;
}
"""
        assert find_hook_order_issues(source) == []


# --------------------------------------------------------------------------- #
# Blocks share the enclosing function frame                                    #
# --------------------------------------------------------------------------- #


class TestBlocksAreNotFrames:
    @pytest.mark.parametrize(
        "guard",
        [
            "if (loading) return <Spinner />;",
            "if (loading) {\n    return <Spinner />;\n  }",
            "if (a) {\n    if (b) {\n      return null;\n    }\n  }",
            "for (const x of xs) {\n    if (x.bad) return null;\n  }",
            "while (n--) {\n    if (n < 0) return null;\n  }",
            "try {\n    return risky();\n  } catch (e) {\n    log(e);\n  }",
            "switch (mode) {\n    case 'a':\n      return <A />;\n  }",
        ],
    )
    def test_guard_in_a_block_still_belongs_to_the_component(self, guard: str):
        source = f"""
export default function App() {{
  {guard}
  const totals = useMemo(() => 1, []);
  return <div>{{totals}}</div>;
}}
"""
        assert names(source) == ["useMemo"]

    def test_block_without_return_is_fine(self):
        source = """
export default function App() {
  if (debug) {
    console.log('mounting');
  }
  const [a] = useState();
  return <div>{a}</div>;
}
"""
        assert find_hook_order_issues(source) == []

    def test_loop_with_continue_is_fine(self):
        source = """
export default function App({ xs }) {
  const out = [];
  for (const x of xs) {
    if (!x) continue;
    out.push(x);
  }
  const [a] = useState();
  return <div>{a}{out.length}</div>;
}
"""
        assert find_hook_order_issues(source) == []


# --------------------------------------------------------------------------- #
# JSX text, strings, templates, regex, comments                                #
# --------------------------------------------------------------------------- #


class TestHooksInsideTheReturnedExpression:
    """A hook written into the returned JSX runs on every render that reaches
    the return, so it is legal — the anchor must not swallow its own statement.
    """

    def test_hook_in_the_only_return(self):
        source = """
export default function App() {
  return <div ref={useRef(null)}><Child /></div>;
}
"""
        assert find_hook_order_issues(source) == []

    def test_context_provider_value(self):
        source = """
export default function App({ a, x }) {
  return (
    <Ctx.Provider value={useMemo(() => a * x, [a, x])}>
      <Child />
    </Ctx.Provider>
  );
}
"""
        assert find_hook_order_issues(source) == []

    def test_hook_in_a_guards_own_returned_expression(self):
        # A conditional hook, which is a different violation family and out of
        # this scanner's declared scope — but it must not be misreported as an
        # ordering problem either.
        source = """
export default function App({ x }) {
  if (x) return <A data={useMemo(() => 1, [])} />;
  return <B />;
}
"""
        assert find_hook_order_issues(source) == []

    def test_hook_after_the_return_statement_is_still_caught(self):
        source = """
export default function App({ x }) {
  if (!x) return <Empty />;
  const y = useMemo(() => 1, []);
  return <div>{y}</div>;
}
"""
        assert names(source) == ["useMemo"]

    def test_braced_guard_without_a_semicolon_closes_at_the_block(self):
        source = """
export default function App({ x }) {
  if (!x) {
    return null
  }
  const y = useMemo(() => 1, []);
  return <div>{y}</div>;
}
"""
        assert names(source) == ["useMemo"]

    def test_arrow_inside_the_returned_jsx_does_not_end_the_statement(self):
        source = """
export default function App({ items }) {
  return (
    <List
      onPick={(i) => { track(i); }}
      footer={<Foot ref={useRef(null)} />}
    />
  );
}
"""
        assert find_hook_order_issues(source) == []


class TestTokenizerHazards:
    def test_return_as_jsx_text_before_hooks(self):
        source = """
export default function App() {
  const hint = <p>press return to continue</p>;
  const [a] = useState();
  return <div>{hint}{a}</div>;
}
"""
        assert find_hook_order_issues(source) == []

    def test_apostrophe_in_jsx_text_before_hooks(self):
        source = """
export default function App() {
  const hint = <p>Don't return home yet</p>;
  const [a] = useState();
  return <div>{hint}{a}</div>;
}
"""
        assert find_hook_order_issues(source) == []

    def test_return_inside_a_string_literal(self):
        source = """
export default function App() {
  const msg = 'if empty return null';
  const other = "return early";
  const [a] = useState();
  return <div>{msg}{other}{a}</div>;
}
"""
        assert find_hook_order_issues(source) == []

    def test_return_inside_a_template_literal(self):
        source = """
export default function App({ n }) {
  const msg = `return ${n} rows`;
  const [a] = useState();
  return <div>{msg}{a}</div>;
}
"""
        assert find_hook_order_issues(source) == []

    def test_template_literal_with_nested_arrow_returning_early(self):
        source = """
export default function App({ xs }) {
  const msg = `total ${xs.map((x) => { if (!x) return 0; return x.v; }).length}`;
  const [a] = useState();
  return <div>{msg}{a}</div>;
}
"""
        assert find_hook_order_issues(source) == []

    def test_template_url_does_not_swallow_the_rest_of_the_file(self):
        # `//` inside a template body is text, not a comment. Reading it as one
        # ate the closing backtick and bailed the whole scan, so the real
        # violation below went unreported.
        source = """
export default function App({ id }) {
  const url = `https://api.example.com/items/${id}`;
  if (!id) return null;
  const y = useMemo(() => fetch(url), [url]);
  return <div>{y}</div>;
}
"""
        assert names(source) == ["useMemo"]

    def test_block_comment_opener_in_a_template_is_text(self):
        source = """
export default function App({ id }) {
  const label = `rate /* per unit ${id}`;
  if (!id) return null;
  const y = useMemo(() => 1, []);
  return <div>{y}{label}</div>;
}
"""
        assert names(source) == ["useMemo"]

    def test_comments_still_work_inside_a_template_expression(self):
        # Inside `${ }` we are back in JS, where a comment is a comment.
        source = """
export default function App({ id }) {
  const label = `id: ${/* keep */ id}`;
  const [a] = useState();
  return <div>{label}{a}</div>;
}
"""
        assert find_hook_order_issues(source) == []

    def test_return_inside_line_and_block_comments(self):
        source = """
export default function App() {
  // if (loading) return <Spinner />;
  /* legacy:
     if (!data) return null;
  */
  const [a] = useState();
  return <div>{a}</div>;
}
"""
        assert find_hook_order_issues(source) == []

    def test_regex_literal_containing_return(self):
        source = """
export default function App() {
  const re = /return\\s+null/g;
  const [a] = useState();
  return <div>{re.source}{a}</div>;
}
"""
        assert find_hook_order_issues(source) == []

    def test_self_closing_jsx_with_expression_attribute_before_hooks(self):
        source = """
export default function App({ x }) {
  const el = <Foo bar={x} baz={{ a: 1 }} />;
  const [a] = useState();
  return <div>{el}{a}</div>;
}
"""
        assert find_hook_order_issues(source) == []

    def test_division_is_not_read_as_a_regex(self):
        source = """
export default function App({ total, count }) {
  const avg = total / count / 2;
  const [a] = useState();
  return <div>{avg}{a}</div>;
}
"""
        assert find_hook_order_issues(source) == []

    def test_property_key_named_return(self):
        source = """
export default function App() {
  const codes = { return: 1, exit: 2 };
  const [a] = useState();
  return <div>{codes.return}{a}</div>;
}
"""
        assert find_hook_order_issues(source) == []

    def test_member_access_named_return(self):
        source = """
export default function App({ gen }) {
  gen.return(undefined);
  const [a] = useState();
  return <div>{a}</div>;
}
"""
        assert find_hook_order_issues(source) == []


# --------------------------------------------------------------------------- #
# Real violations                                                              #
# --------------------------------------------------------------------------- #


class TestViolations:
    def test_classic_loading_guard_above_a_hook(self):
        source = """
export default function App() {
  const { data, loading } = useQuerySql('queries/sales');
  if (loading) return <Spinner />;
  const totals = useMemo(() => sum(data), [data]);
  return <div>{totals}</div>;
}
"""
        issues = find_hook_order_issues(source)
        assert [i.hook_name for i in issues] == ["useMemo"]
        assert issues[0].return_line == 4
        assert issues[0].hook_line == 5

    def test_braced_guard_above_a_hook(self):
        source = """
export default function App({ data }) {
  if (!data) {
    return null;
  }
  const [sel, setSel] = useState(null);
  return <div>{sel}</div>;
}
"""
        assert names(source) == ["useState"]

    def test_namespaced_hook_after_a_return(self):
        source = """
export default function App({ data }) {
  if (!data) return null;
  const [sel] = React.useState(null);
  return <div>{sel}</div>;
}
"""
        assert names(source) == ["useState"]

    def test_every_hook_after_the_return_is_reported(self):
        source = """
export default function App({ data }) {
  if (!data) return null;
  const [a] = useState();
  const b = useMemo(() => 1, []);
  const c = useRef(null);
  return <div>{a}{b}{c}</div>;
}
"""
        assert names(source) == ["useState", "useMemo", "useRef"]

    def test_only_the_first_return_is_used_as_the_anchor(self):
        source = """
export default function App({ data }) {
  if (!data) return null;
  if (data.empty) return <Empty />;
  const [a] = useState();
  return <div>{a}</div>;
}
"""
        issues = find_hook_order_issues(source)
        assert len(issues) == 1
        assert issues[0].return_line == 3

    def test_violation_inside_a_custom_hook(self):
        source = """
export function useRows(data) {
  if (!data) return [];
  const [rows, setRows] = useState([]);
  return rows;
}
"""
        assert names(source) == ["useState"]

    def test_violation_in_a_nested_component_only(self):
        source = """
export default function App() {
  const [a] = useState();

  function Inner({ data }) {
    if (!data) return null;
    const [b] = useState();
    return <div>{b}</div>;
  }

  return <Inner data={a} />;
}
"""
        assert names(source) == ["useState"]

    def test_hook_called_inline_inside_jsx_after_a_guard(self):
        # Taken from a real generated report: the guard sits ~60 lines above a
        # `ref={useRef(null)}` buried in the returned JSX, so nothing about the
        # two lines looks related when read in isolation.
        source = """
export default function Trend({ rows }) {
  const data = useMemo(() => rows.map(N), [rows]);
  if (rows.length === 0) return null;

  return (
    <Section title="Trend">
      <div ref={useRef(null)} style={{ marginTop: 32 }}>
        <Chart data={data} />
      </div>
    </Section>
  );
}
"""
        assert names(source) == ["useRef"]

    def test_two_components_each_reported(self):
        source = """
export function A({ x }) {
  if (!x) return null;
  const [a] = useState();
  return <div>{a}</div>;
}

export default function B({ y }) {
  if (!y) return null;
  const [b] = useMemo(() => [1], []);
  return <div>{b}</div>;
}
"""
        assert names(source) == ["useState", "useMemo"]


# --------------------------------------------------------------------------- #
# Bail-out behaviour — never guess                                             #
# --------------------------------------------------------------------------- #


class TestBailOut:
    @pytest.mark.parametrize(
        "source",
        [
            "",
            "const a = 1;\n",
            "export default function App() { return <div />; }\n",
        ],
    )
    def test_benign_sources_report_nothing(self, source: str):
        assert find_hook_order_issues(source) == []

    def test_unbalanced_braces_bail_out(self):
        source = """
export default function App({ data }) {
  if (!data) return null;
  const [a] = useState();
"""
        assert find_hook_order_issues(source) == []

    def test_stray_closing_brace_bails_out(self):
        source = """
}
export default function App({ data }) {
  if (!data) return null;
  const [a] = useState();
  return <div>{a}</div>;
}
"""
        assert find_hook_order_issues(source) == []

    def test_unterminated_block_comment_bails_out(self):
        source = """
export default function App({ data }) {
  if (!data) return null;
  const [a] = useState();
  /* trailing
"""
        assert find_hook_order_issues(source) == []

    def test_oversized_source_is_skipped(self):
        source = "export default function App(){ if(!x) return null; const [a]=useState(); }\n"
        padding = "// " + ("x" * 600 * 1024) + "\n"
        assert find_hook_order_issues(padding + source) == []

    @pytest.mark.parametrize(
        "guard",
        [
            # `else` is not in the statement-start character set that keeps JSX
            # text from being read as code.
            "if (x) foo(); else return null;",
            # No semicolon and no braces: the return statement never closes, so
            # later hooks are not attributed to it.
            "if (!x) return null",
        ],
    )
    def test_documented_misses_stay_silent(self, guard: str):
        source = f"""
export default function App({{ x }}) {{
  {guard}
  const y = useMemo(() => 1, []);
  return <div>{{y}}</div>;
}}
"""
        # Recall traded away on purpose — see the module docstring. Pinned so a
        # future change to either rule is a deliberate decision, not a surprise.
        assert find_hook_order_issues(source) == []

    def test_use_prefixed_non_hook_identifiers_are_ignored(self):
        source = """
export default function App({ data }) {
  if (!data) return null;
  const u = user(data);
  const v = used(data);
  const w = useful;
  return <div>{u}{v}{w}</div>;
}
"""
        assert find_hook_order_issues(source) == []
