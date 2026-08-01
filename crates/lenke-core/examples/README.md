# Benchmarks

Every one of these runs with `cargo run --release --example <name>`. There are
also a few benchmarks that need crate-private access and live as `#[ignore]`d
tests instead; those are listed at the bottom.

**Look here before writing a new one.** The suite is broader than it looks from
the file names — `storage_probe` is the adjacency-layout question, and
`eval_vs_columnar` is the predicate-evaluation question, neither of which is
obvious without opening them.

## By question

| If you are asking… | Run |
| --- | --- |
| Is a query shape slow? Which of the four perf levers moved? | `perf_bench` |
| How do query shapes scale with graph size? | `scale_bench` |
| What does an individual GQL query shape cost? | `gql_bench` |
| Same, for Gremlin traversals | `gremlin_bench` |
| **Should adjacency storage change, and what would writes pay?** | **`storage_probe`** |
| How far is the WHERE path from a hand-written columnar kernel? | `eval_vs_columnar` |
| Does the property index actually seed a seek? (GQL / Gremlin) | `edge_type_index_bench`, `gremlin_index_bench` |
| What do the graph algorithms cost? | `algo_bench`, `neighbor_aggregate_bench` |
| What does `CALL` add over calling an algorithm directly? | `call_bench` |
| What do map/record properties cost — stored, and through a codec? | `map_bench`, `map_codec_bench` |
| What do temporal columns cost? | `temporal_bench` |
| What do path selectors and per-hop predicates cost? | `path_selector_bench` |
| What does a record-typed constraint cost to declare? | `record_debox_bench` |
| What does CDC scope extraction cost per write? | `cdc_extract_bench` |
| How much memory does a graph of N vertices take? | `mem_probe` |
| **Where does NDJSON ingest time go, and what is the ceiling?** | **`ingest_phase_bench`**, plus `ingest_throughput` below |
| Are the count fast paths correct? (not a benchmark) | `count_check` |

## Benchmarks that live as ignored tests

They need crate-private access — the JSON parser, the GQL evaluator — so they
cannot be examples. Run with:

```
cargo test --release <name> -- --ignored --nocapture
```

| Name | Question |
| --- | --- |
| `ingest_throughput_against_the_ceiling` | How close is decode to what the machine can do? Sweeps 10k / 200k / 1M, and covers edge locality and edge ids. `INGEST_N=…` to override. |
| `query_row_cost` | What does a query pay PER ROW, by column type and by `RETURN n`? `ROWS=…` to override. |
| `bench_parallel_query_speedup` | How does query parallelism scale? Needs `--features parallel-query`; vary `RAYON_NUM_THREADS`. |
| `bench_aml_shapes` | An AML-shaped workload — layering and structuring patterns over a transaction network. |
| `bench_hris_shapes` | An HRIS-shaped workload — an org hierarchy with `REPORTS_TO`. |
| `bench_temporal_index` | Bitemporal index bake-off over an SCD-2 org. |
| `bench_allen_relations` | All thirteen Allen relations over a batch of edge versions. |
| `bench_var_length_matcher` | The whole var-length matcher surface. |

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

## Known gaps

- **Codec throughput beyond NDJSON.** `map_codec_bench` covers map properties;
  encode/decode for pg-json, graphson, pg-text and csv has only ever been
  measured from throwaway scripts on the TS side.
- **The TS engine has no benchmarks at all.** Every number here is the Rust core.
  Pure-TS users get none of these results, and nothing would catch a regression
  there.
- **The FFI boundary** — marshalling, param encoding, result decoding — is only
  ever measured end-to-end from JS, never in isolation.
