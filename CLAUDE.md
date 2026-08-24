# Working in this repo

## Benchmarks: look before you build

The engine has a few native Rust benchmark examples under
`crates/lenke-engine/examples/`. Read their module headers before writing a new
one — each opens with the exact question it answers:

- `expand_bench` — the adjacency / edge-type question: what does a type-filtered
  `expand` pay to scan a node's whole adjacency and filter by edge type? It scales
  with degree and with how many types the degree spans (a selective type over a
  high-degree, many-type node is where an index could win; a degree-4 single-type
  fixture is where it can only lose).
- `interval_bench` — the bitemporal question: what does an "as of T" query pay to
  expand all of a node's edges and post-filter by validity interval, vs. an
  interval-index seek?
- `spelling_probe` — equivalent-spelling plan + perf coverage; see the section
  below.

Writing a new benchmark feels like progress and reading three file headers feels
like overhead. It is the other way round — re-deriving a question one of these
already answers is the expensive path. Add a new example only when the question
genuinely is not covered.

(A larger benchmark corpus, indexed by a README, lived in the retired `lenke-core`
crate and went with it when it was deleted; rebuilding the parts that still matter
against `lenke-engine` is open work.)

### Three engines, two harnesses

Rust examples measure the **native** build only. They compile for
`wasm32-unknown-unknown` but cannot run there — `Instant::now()` panics (no
clock), no stdout, no runner — and they cannot reach the pure-TS engine at all.

For **wasm** or **pure-TS**, or to compare engines:

```
cd packages/native && bun run bench          # ts, ffi and wasm side by side
BENCH_ENGINES=ts,wasm bun run bench
BENCH_N=1000000 bun run bench
```

Roughly: the Rust core is 1.5-4x pure-TS depending on workload, and wasm gives
up about a third of that. The gap is widest on edge-heavy ingest and traversal —
which is where most optimization work lands, so those wins do not reach pure-TS
users. Check both when changing anything shared.

`bun run bench:usage` is the serving counterpart: small operations against a warm
graph, including interleaved read/write. Bulk throughput and per-operation cost
are different questions and a change can help one while hurting the other —
interleaving a write with a traversal already costs ~2.5x the traversal alone,
because a write invalidates the read-side snapshot.

### Before trusting a number

Each of these is here because a wrong conclusion was drawn and committed first.

- **Sweep the size.** Cache-resident and not are different questions; the
  transition is between 200k and 1M elements. A faster hash measured −5% at 200k
  and nothing at 1M.
- **Match the fixture to the claim.** One edge per node is sparse — per-edge costs
  scale with the edge:node ratio, per-node costs do not. A change measured flat at
  1:1 and −5% at 5:1. Likewise degree: an adjacency change that only helps
  low-degree vertices was judged against a degree-4 fixture, where it can only
  lose.
- **Give edges ids.** `encode` emits them, so every reloaded snapshot has them.
  Omitting them skips the external-id path entirely.
- **Match sample counts on both sides**, and prefer min or p25 over the mean.
  Several conclusions here were single-run against single-run and did not survive
  repetition.
- **Know the noise floor.** Some rows range 2x for the same binary. Anything under
  ~10% needs its own isolated harness, and "obviously correct so it must be
  faster" is not evidence — several such changes measured neutral or worse.

Record rejected optimizations with their numbers, next to the code they would
have changed. Several have been re-attempted otherwise.

### Equivalent spellings must cost the same

Every index-seeding bug found in the old engine had one shape: the planner
recognized one spelling of a predicate and scanned for another that meant exactly
the same thing — `$x = u.k` vs `u.k = $x`, `k = $a OR k = $b` vs `k IN [$a, $b]`, a
clause `WHERE` vs an inline `{k: $x}`, `5 <= u.n` vs `u.n >= 5`. Each cost 100-300x
and each returned the correct answer, so no correctness test could catch it.

This engine lowers BOTH GQL and Gremlin to one `ir::Plan` and runs one `exec`, so
speed is a property of the optimized plan, not the surface syntax — two spellings
that optimize to the same plan cannot differ. The `spelling_probe` example checks
that claim directly: for each group of queries that should be equivalent it prints
the canonicalized optimized plan and the measured time, and flags any group whose
members disagree on either. A plan mismatch is the real signal; the time is the
backstop. When adding a predicate form to the planner, add its spellings there.

## Gates

`bun run lint` and `cargo clippy --all-targets -- -D warnings` are separate from
`bun run fmt` (oxfmt) and `cargo fmt`. Run each as **its own command** and check
its exit code — piping clippy into `grep -c` and chaining with `&&` takes the
exit status from `grep`, which succeeds when it finds errors. That has let broken
lint through twice.

Byte-identity between the TS and Rust engines is a hard invariant. Any change to
storage, ordering or codecs needs the fuzzers, not just the unit tests:

```
cd packages/native && FUZZ_SEED=<n> bun test src/codec-fuzz.test.ts \
  src/differential-fuzz.test.ts src/write-fuzz.test.ts src/injection-fuzz.test.ts
```

`backend-parity-fuzz` compares the wasm build against the FFI build, and
**neither `bun run build` nor `cargo build` rebuilds the wasm** — only
`cd packages/native && bun run build:wasm` does. A stale `lenke_engine.wasm`
turns that fuzzer into a comparison between your change and an old copy of
itself: it reports failures you did not cause and passes you did not earn.
Found with the artifact two days behind the `.so`, after chasing a `max()`
"divergence" that was the previous build. Rebuild it before trusting that
suite, and check the timestamps if it says something surprising:

```
stat -c '%y %n' crates/lenke-engine/target/release/liblenke_engine.so \
  crates/lenke-engine/target/wasm32-unknown-unknown/release/lenke_engine.wasm
```
