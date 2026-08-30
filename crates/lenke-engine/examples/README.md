# Benchmarks

The `lenke-engine` bench corpus. **Look here before writing a new one** — the
suite is indexed by *question*, and the answer to yours is very likely already a
case in one of the group binaries.

Two kinds of entry point:

- **Themed group binaries** (`ingest_bench`, `query_bench`, …) — each groups a
  subsystem's questions as selectable *cases*. This is the consolidation: one
  binary per subsystem instead of one per question.
- **`bench_all`** — runs every group in one process. This is the **regression
  sweep**: run it to catch a slowdown in a subsystem you did not think you
  touched. The per-group binaries are for iterating on one subsystem.

A handful of focused probes stay as their own binaries because they answer one
sharp question with a bespoke fixture (see the bottom).

## Running

```
# the whole corpus, for a regression sweep
cargo run --release --manifest-path crates/lenke-engine/Cargo.toml --example bench_all

# one subsystem
cargo run --release --manifest-path crates/lenke-engine/Cargo.toml --example query_bench

# one case (a substring matched against each case's `group/case` label);
# works on any binary, including bench_all
cargo run --release ... --example query_bench -- seeded
cargo run --release ... --example bench_all   -- index
```

Environment: `BENCH_REPS=<n>` (samples per case, min is reported; default 7),
`BENCH_N=<n>` (primary sweep size; default 200k). Native only — these use
`std::time::Instant` and cannot run under wasm. For the pure-TS / wasm / FFI
comparison, use `cd packages/native && bun run bench` instead.

## By question

| If you are asking…                                                  | Run                                |
| ------------------------------------------------------------------- | ---------------------------------- |
| Where does NDJSON decode time go?                                   | `ingest_bench -- phases`           |
| How close is decode to the raw-scan ceiling?                        | `ingest_bench -- ceiling`          |
| What does parallel decode / encode buy?                             | `ingest_bench -- threads` / `encode` |
| How many resident bytes does an element cost?                       | `ingest_bench -- mem`              |
| What does a GQL / Gremlin query shape cost?                         | `query_bench -- gql` / `gremlin`   |
| Which counts still enumerate vs shortcut?                           | `query_bench -- counts`            |
| What does a row cost, by what is returned?                          | `query_bench -- perrow`            |
| What does turning query text into a plan cost?                      | `query_bench -- plan`              |
| Does an indexed key seek beat a scan?                               | `query_bench -- seeded`            |
| _(coming)_ Should adjacency storage change, and what do writes pay? | `storage_bench`                    |
| _(coming)_ How far is WHERE from a columnar kernel?                 | `storage_bench -- eval`            |
| _(coming)_ Does the type / property / temporal index seed a seek?   | `index_bench`                      |
| _(coming)_ What do the graph algorithms / neighborAggregate / CALL cost? | `algo_bench`                  |
| _(coming)_ What do map / record / temporal-column values cost?      | `value_bench`                      |
| _(coming)_ How do shapes scale with size? CDC? AML / HRIS shapes?   | `scale_bench`                      |

The _(coming)_ rows are groups being ported from the retired `lenke-core` corpus;
they land in follow-up commits. The moot ones are **not** coming back: the
core-vs-engine (`cross_engine_shortcuts`) and Gremlin-arm-vs-IR audits
(`arm_audit`, `migration_arm_price_audit`, …) priced migrations that are now
complete, and query-parallelism benches (`bench_parallel_query_speedup`) target
an evaluator the engine does not have.

## The focused probes (stay their own binary)

Each answers one sharp question with a bespoke fixture, and opens with that
question in its module header — read the header before touching it.

- `expand_bench` — what a type-filtered `expand` pays to scan a node's whole
  adjacency and filter by edge type (scales with degree × type spread).
- `interval_bench` — what an "as of T" bitemporal query pays to post-filter all
  of a node's edges by validity interval, vs an interval-index seek.
- `spelling_probe` — that equivalent query spellings optimize to the SAME plan
  and so cost the same (a plan mismatch is the real signal; time is the backstop).

## Before trusting a number

The hard-won rules live in the repo `CLAUDE.md` ("Benchmarks: look before you
build") and are enforced by the shared harness: min-of-N (never a mean),
`black_box` every result, a dep-free deterministic RNG, and **sweep the size**
across the 200k–1M cache transition rather than trusting one point. Record a
rejected optimization with its numbers next to the code it would have changed —
several have been re-attempted for want of that note.
