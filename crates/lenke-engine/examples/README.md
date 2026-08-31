# Benchmarks

The `lenke-engine` bench corpus. **Look here before writing a new one** — the
suite is indexed by _question_, and the answer to yours is very likely already a
case in one of the group binaries.

Two kinds of entry point:

- **Themed group binaries** (`ingest_bench`, `query_bench`, …) — each groups a
  subsystem's questions as selectable _cases_. This is the consolidation: one
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

| If you are asking…                                                    | Run                                  |
| --------------------------------------------------------------------- | ------------------------------------ |
| Where does NDJSON decode time go?                                     | `ingest_bench -- phases`             |
| How close is decode to the raw-scan ceiling?                          | `ingest_bench -- ceiling`            |
| What does parallel decode / encode buy?                               | `ingest_bench -- threads` / `encode` |
| How many resident bytes does an element cost?                         | `ingest_bench -- mem`                |
| What does a GQL / Gremlin query shape cost?                           | `query_bench -- gql` / `gremlin`     |
| Which counts still enumerate vs shortcut?                             | `query_bench -- counts`              |
| What does a row cost, by what is returned?                            | `query_bench -- perrow`              |
| What does turning query text into a plan cost?                        | `query_bench -- plan`                |
| Does an indexed key seek beat a scan?                                 | `query_bench -- seeded`              |
| What does a bulk write (SET) pay?                                     | `storage_bench -- write`             |
| What does a second label on every edge cost?                          | `storage_bench -- multilabel`        |
| What do the graph algorithms cost, and how do they parallelize?       | `algo_bench -- run` / `parallel`     |
| What does neighborAggregate / a CALL cost?                            | `algo_bench -- neighboragg` / `call` |
| What does a map/record property cost vs flat scalars (stored, codec)? | `value_bench -- maps` / `codec`      |
| How do shapes scale across the cache transition?                      | `scale_bench -- sweep`               |
| What does content-derived CDC scope extraction cost per write?        | `scale_bench -- cdc`                 |

**Deferred** (need crate-private access or a large bespoke fixture, so not in a
group binary yet): the eval-vs-columnar floor (`eval_vec` is crate-private — it
lives as an ignored test), temporal-column cost (`temporal_bench`, needs host
`Temporal` construction), and the AML / HRIS domain-shaped workloads. **Not
coming back** — the core-vs-engine (`cross_engine_shortcuts`) and Gremlin-arm-vs-IR
audits (`arm_audit`, `migration_arm_price_audit`, …) priced migrations that are
complete, and query-parallelism benches (`bench_parallel_query_speedup`) target an
evaluator the engine does not have.

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
