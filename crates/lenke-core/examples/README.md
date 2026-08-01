# Benchmarks

Every one of these runs with `cargo run --release --example <name>`. There are
also a few benchmarks that need crate-private access and live as `#[ignore]`d
tests instead; those are listed at the bottom.

**Look here before writing a new one.** The suite is broader than it looks from
the file names — `storage_probe` is the adjacency-layout question, and
`eval_vs_columnar` is the predicate-evaluation question, neither of which is
obvious without opening them.

## By question

| If you are asking…                                                | Run                                                      |
| ----------------------------------------------------------------- | -------------------------------------------------------- |
| Is a query shape slow? Which of the four perf levers moved?       | `perf_bench`                                             |
| How do query shapes scale with graph size?                        | `scale_bench`                                            |
| What does an individual GQL query shape cost?                     | `gql_bench`                                              |
| Same, for Gremlin traversals                                      | `gremlin_bench`                                          |
| **Should adjacency storage change, and what would writes pay?**   | **`storage_probe`**                                      |
| How far is the WHERE path from a hand-written columnar kernel?    | `eval_vs_columnar`                                       |
| Does the property index actually seed a seek? (GQL / Gremlin)     | `edge_type_index_bench`, `gremlin_index_bench`           |
| What do the graph algorithms cost?                                | `algo_bench`, `neighbor_aggregate_bench`                 |
| What does `CALL` add over calling an algorithm directly?          | `call_bench`                                             |
| What do map/record properties cost — stored, and through a codec? | `map_bench`, `map_codec_bench`                           |
| What do temporal columns cost?                                    | `temporal_bench`                                         |
| What do path selectors and per-hop predicates cost?               | `path_selector_bench`                                    |
| What does a record-typed constraint cost to declare?              | `record_debox_bench`                                     |
| What does CDC scope extraction cost per write?                    | `cdc_extract_bench`                                      |
| How much memory does a graph of N vertices take?                  | `mem_probe`                                              |
| **Where does NDJSON ingest time go, and what is the ceiling?**    | **`ingest_phase_bench`**, plus `ingest_throughput` below |
| Are the count fast paths correct? (not a benchmark)               | `count_check`                                            |

## Benchmarks that live as ignored tests

They need crate-private access — the JSON parser, the GQL evaluator — so they
cannot be examples. Run with:

```
cargo test --release <name> -- --ignored --nocapture
```

| Name                                    | Question                                                                                                                                 |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `ingest_throughput_against_the_ceiling` | How close is decode to what the machine can do? Sweeps 10k / 200k / 1M, and covers edge locality and edge ids. `INGEST_N=…` to override. |
| `query_row_cost`                        | What does a query pay PER ROW, by column type and by `RETURN n`? `ROWS=…` to override.                                                   |
| `bench_parallel_query_speedup`          | How does query parallelism scale? Needs `--features parallel-query`; vary `RAYON_NUM_THREADS`.                                           |
| `bench_aml_shapes`                      | An AML-shaped workload — layering and structuring patterns over a transaction network.                                                   |
| `bench_hris_shapes`                     | An HRIS-shaped workload — an org hierarchy with `REPORTS_TO`.                                                                            |
| `bench_temporal_index`                  | Bitemporal index bake-off over an SCD-2 org.                                                                                             |
| `bench_allen_relations`                 | All thirteen Allen relations over a batch of edge versions.                                                                              |
| `bench_var_length_matcher`              | The whole var-length matcher surface.                                                                                                    |

## Things worth knowing before trusting a number

These are all lessons this suite has taught the hard way, each after a wrong
conclusion was drawn and committed.

- **Sweep the size.** A workload that fits in cache answers a different question
  than one that does not; the transition sits between 200k and 1M elements. A
  faster hash function measured −5% at 200k and nothing at 1M.
- **Match the graph shape to the claim.** One edge per node is sparse, and every
  per-edge cost scales with the edge:node ratio while per-node costs do not. A
  change to the edge path measured flat at 1:1 and −5% at 5:1.
- **Check the fixture's degree.** An adjacency change that only helps low-degree
  vertices was measured against a degree-4 fixture, where it can only ever lose.
- **Give edges ids.** `encode` emits them, so every reloaded snapshot has them;
  omitting them skips the external-id bookkeeping entirely.
- **Match the sample count on both sides**, and prefer min or p25 over the mean.
  Several conclusions here were single-run against single-run and did not
  survive repetition.
- **Know the noise floor before believing a small delta.** The `whole node` row
  of `query_row_cost` has ranged 72–161 ms for the same binary, because it runs
  after other shapes and inherits their allocator state. Anything under ~10%
  needs its own isolated harness.

## Measuring wasm and the pure-TS engine

Nothing here reaches either. These examples compile for
`wasm32-unknown-unknown` but cannot RUN there — `Instant::now()` panics (that
target has no clock), there is no stdout, and there is no runner, so a green
`cargo build --target wasm32-unknown-unknown --example …` proves nothing. And the
pure-TS engine is a separate implementation no Rust benchmark can touch at all.

Both are reachable from one JS harness, through the same workloads:

```
cd packages/native
bun run bench                        # ts, ffi and wasm side by side
BENCH_ENGINES=ts,wasm bun run bench
BENCH_N=1000000 bun run bench        # bigger workload
```

It covers ingest, all five codecs both directions, and representative query
shapes. As of writing, at 20k nodes, relative to pure-TS:

|                              | ffi        | wasm       |
| ---------------------------- | ---------- | ---------- |
| decode ndjson (nodes)        | 0.49x      | 0.62x      |
| decode ndjson (5 edges/node) | 0.24x      | 0.36x      |
| encode (five codecs)         | 0.39-0.63x | 0.56-0.87x |
| query: count / group         | 0.33-0.35x | 0.51-0.53x |
| query: 1-hop traversal       | 0.23x      | 0.34x      |

So the Rust core is 1.5-4x faster than pure-TS depending on the workload, and
wasm gives up perhaps a third of that advantage. The gap is widest exactly where
this repo has been optimizing — edge-heavy ingest and traversal — which is worth
remembering: those wins do not reach pure-TS users.

Part of the decode gap is threads rather than codegen: decode defaults to the
parallel path and only ffi has any, so wasm and TS both run it serially.

### Usage-shaped workloads

`bun run bench` is the ingest shape: load a big document, run one query over all
of it. Serving is the opposite — many small operations against a warm graph, and
often reads and writes INTERLEAVED, which is the pattern that has historically
broken things here (a write invalidates the read-side snapshot, so alternating
them can repack the adjacency every cycle).

```
cd packages/native
bun run bench:usage
BENCH_OPS=20000 BENCH_GRAPH=50000 bun run bench:usage
```

Point lookup, permission check, 2-hop recommendation, keyed dedup, property
update, append, and three interleaved read/write shapes — drawn from the
applications this engine has been exercised against. Reported as operations per
second on all three engines.

Every workload runs TWICE, without indexes and with — the `(-)` and `(+)`
columns. The difference is not marginal: a point lookup goes from 922 to 112k
ops/sec on the TS engine and 3.5k to 173k on the Rust one, because a
`WHERE u.name = $n` with no index is a full scan that reads exactly like a
lookup. Benchmarking only one column is how a scan gets mistaken for a lookup.

Both directions carry signal. Reads gain a seek; **writes pay maintenance** —
appending a node carrying two indexed keys costs 249k -> 228k ops/sec on ffi and
214k -> 167k on wasm. (Watch for the trap: an append whose properties are all
UNindexed pays nothing, so that row looked like indexes were free on writes until
the fixture was changed to carry one.)

**It found a 60x planner cliff.** These two are semantically identical:

```
MATCH (u:User)-[:MEMBER_OF]->(t:Team) WHERE u.name = $n RETURN count(*)   2.2k ops/s
MATCH (u:User {name: $n})-[:MEMBER_OF]->(t:Team) RETURN count(*)        136.8k ops/s
```

On the Rust engine a WHERE-form anchor followed by a traversal stops seeding
from the index and falls back to a scan; the inline form keeps the seek. Alone,
both forms seed fine — it is the traversal that loses it. The pure-TS engine
seeds both, so on that one shape pure-TS is ~240x FASTER than native. Both forms
are kept as separate rows so the cliff cannot regress unseen.

Also visible: **interleaving a write with a traversal** costs about 2.5x the
traversal alone, which is the read-snapshot invalidation showing up where it was
predicted to; and **100 updates inside a transaction** run at roughly half the
per-write rate of the same updates auto-committing, which is the undo log and
deferred checks.

## Known gaps

- **The FFI boundary** — marshalling, param encoding, result decoding — is only
  ever measured end-to-end from JS, never in isolation.
- **Transactions** are not covered from JS. `perf_bench` shows `set_prop` inside
  a transaction at ~2.5x the cost of outside one on the Rust core; the TS
  equivalent is unmeasured.
