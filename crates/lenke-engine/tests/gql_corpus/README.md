# GQL conformance corpus

One set of GQL cases, run against **both** engines. Each `*.jsonl` file holds cases
extracted from a lenke-core behavioral test file. The runner (`../gql_corpus.rs`)
loads each case's fixture into lenke-core (the reference) **and** lenke-engine, runs
the query on both, and asserts the engine's result multiset matches core's. Core's
own inline tests still pin core to the spec; this extends the same query surface to
the engine.

## Case format — one JSON object per line

```json
{"name":"short_snake_name","fixture":"<core-dialect NDJSON>","query":"MATCH ... RETURN ...","ordered":false}
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
- Queries the test expects to ERROR — include them; the runner checks error-parity (both must reject).
- A test with several queries → one case per query (same fixture, `name` suffixed `_1`, `_2`, …).

## What to SKIP (do not emit a case)

- **Parameterized** queries (`qp(g, Q, params)` / `$name` in the query) — the runner passes no params.
- **Multi-statement** tests (a write statement, then a *separate* read) — only one `query` per case.
- Queries returning **whole nodes/edges/paths/maps** (`RETURN n`, `RETURN p` for a path) — their
  representation is not comparable across engines. Only scalar RETURNs.
- Tests whose fixture or expected can't be read off directly (custom Rust logic, loops building data).

When unsure, skip — a missing case is fine, a wrong one is noise. The runner reports every
engine≠core mismatch, so genuine engine gaps surface for review.

## Verify

```
cargo test --release --manifest-path crates/lenke-engine/Cargo.toml --test gql_corpus -- --nocapture
```
