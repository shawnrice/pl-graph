# GQL conformance corpus

One set of GQL cases, run as an engine regression snapshot. Each `*.jsonl` file holds
cases extracted from a lenke-core behavioral test file. The runner (`../gql_corpus.rs`)
loads each case's fixture into lenke-engine and asserts its result multiset matches the
recorded outcome in `snapshots.jsonl`. That snapshot was captured while lenke-core still
existed and the differential was green, so each recorded outcome equals core's
spec-anchored answer. lenke-core has since been deleted; the live byte-identity contract
is now upheld by the TS engine (`@lenke/core`) fuzzers, and this corpus guards against
engine regressions from the frozen baseline.

## Case format — one JSON object per line

```json
{
  "name": "short_snake_name",
  "fixture": "<core-dialect NDJSON>",
  "query": "MATCH ... RETURN ...",
  "ordered": false
}
```

- **name** — short identifier (source test fn name + a suffix if the test has several queries).
- **fixture** — the graph as **core-dialect** NDJSON, one object per line joined with `\n`
  (i.e. `{"type":"node","id":"a","labels":["P"],"properties":{...}}` and
  `{"type":"edge","from":"a","to":"b","labels":["R"],"properties":{...}}`). Copy the exact
  lines the source test passes to `graph_of([...])`/`ndjson::decode(...)`. If the test uses
  `modern()` / `modern_gql()`, set `"fixture":"@modern"` (a built-in).
- **query** — the exact GQL string the test runs (`rows(g, Q)` / `q(g, Q)` / `qp(...)` without params).
- **ordered** — `true` **only** if the query has `ORDER BY` (result compared position-for-position);
  otherwise omit it or `false` (multiset comparison).

## What to INCLUDE

- Read queries `MATCH ... RETURN <scalars>` (names, numbers, counts, bools, strings).
- Single-statement write-and-return where the RETURN is scalar (`CREATE (n:X {..}) RETURN n.k`).
- Queries the test expects to ERROR — include them; the snapshot records the rejection and the runner checks the engine still rejects.
- A test with several queries → one case per query (same fixture, `name` suffixed `_1`, `_2`, …).

## What to SKIP (do not emit a case)

- **Parameterized** queries (`qp(g, Q, params)` / `$name` in the query) — the runner passes no params.
- **Multi-statement** tests (a write statement, then a _separate_ read) — only one `query` per case.
- Queries returning **whole nodes/edges/paths/maps** (`RETURN n`, `RETURN p` for a path) — their
  representation is not comparable across engines. Only scalar RETURNs.
- Tests whose fixture or expected can't be read off directly (custom Rust logic, loops building data).

When unsure, skip — a missing case is fine, a wrong one is noise. The runner reports every
case that diverges from its snapshot, so genuine engine regressions surface for review.

## Verify

```
cargo test -p lenke-engine --test gql_corpus -- --nocapture
```

After an INTENDED behavior change, regenerate the frozen baseline (and review the diff —
an unexplained change there is a regression):

```
CORPUS_SNAPSHOT=1 cargo test -p lenke-engine --test gql_corpus
```

## Excluded: parser recursion-depth cases

Four hardening cases (`h_deep_nested_{parens,not,label_negation,lists}`) feed
thousands of nested tokens to assert a clean syntax error. The engine's
recursive-descent parser has no recursion-depth guard, so it **overflows the stack**
(uncatchable) instead of erroring. They are omitted from the corpus (they would abort
the run) and tracked as a known gap: the engine parser needs depth limits.
