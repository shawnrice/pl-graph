# Temporal column de-boxing

> **Historical (lenke-core).** The design and results below describe the de-boxing as
> implemented in the retired `lenke-core` crate (`graph.rs`, `Val::Temporal`, the
> `temporal_bench` example). The same change shipped in `lenke-engine` as a per-type
> `Column::Temporal` over `Value::Temporal`; only the file/symbol names differ there.

Status: **shipped** (both phases, all 6 types; full native + TS/FFI gate green,
byte-identical). Overflow-policy prerequisite shipped.

## Implementation

One `Column::Temporal { data: TemporalCol, present: BitSet }` variant wraps a
per-type SoA inner enum (`TemporalCol` in `graph.rs`) — so each exhaustive `match
col` gains a single arm while per-type width is preserved inside (Date = `Vec<i32>`,
Duration = four parallel `Vec`s). Threaded through `value_kind`/`empty_col_for(_kind)`/
`col_set`/`value_id`/`push_absent`/`element_len`/`is_present`/`to_mixed`/`remove_value`
(graph.rs), the NDJSON encoder, and the scalar prop-read (eval.rs). `TemporalKind` +
`Temporal::kind()` classify a value to its column kind; a key mixing temporal
sub-kinds (or temporal + other) promotes to `Mixed` like any type disagreement.

Phase (b) is a **vectorized gather** (`gather_temporal` → `VVec::Gen` of
`Val::Temporal`, chained after `gather_num`/`gather_str` at the Prop sites): a temporal
property reconstructs from the packed arrays in a tight loop instead of the per-row
`Binding`+`eval` dispatch. Phase (c) adds **typed comparators** for filter and min/max
(`temporal_cmp_vec` / `temporal_minmax`) — see Results → Phase (c). ORDER BY's typed
sort is the one remaining piece (see Remaining opportunity).

## Context

Temporal property values (`DATE`, `TIME`, `LOCAL DATETIME`, `ZONED TIME`,
`ZONED DATETIME`, `DURATION`) are currently stored in `Column::Mixed`
(`Vec<Option<Value>>`, ~40 B/slot) — there is no typed temporal column, so every
temporal read is a two-hop discriminant walk (`Option` → `Value` → `Temporal`) and
every filter/sort/aggregate falls to the scalar path (the vectorized scan only
engages typed `Column::Num`/`Str`).

A `Value::Temporal(Temporal)` is inline (no heap): `Temporal` is 40 B, pinned by its
largest variant `Duration` (3×i64 + u32). So a single unified `Column::Temporal
{ Vec<Temporal> }` would save ~0 memory — **per-type packed columns are the only
thing that recovers memory**, because only they can use each type's true width.

## Decision

De-box all six fixed-width temporal types into per-type packed `Column` variants,
mirroring `Column::Num`. **Measured** packed width per slot vs the 40 B Mixed slot
(SoA, `size_of` sum; `temporal_bench` `vertex_prop_bytes`) — this is the **memory**
axis, distinct from speed (see Results):

| Type             | Packed layout            | B/slot | vs Mixed |
| ---------------- | ------------------------ | ------ | -------- |
| `DATE`           | `Vec<i32>`               | 4      | 9.70×    |
| `TIME`           | `Vec<u32>,Vec<u32>`      | 8      | 4.92×    |
| `ZONED TIME`     | `secs,nanos,offset`      | 10     | 3.95×    |
| `LOCAL DATETIME` | `Vec<i64>,Vec<u32>`      | 12     | 3.30×    |
| `ZONED DATETIME` | `secs,nanos,offset`      | 14     | 2.83×    |
| `DURATION`       | `months,days,secs,nanos` | 28     | 1.42×    |

(SoA packs tighter than an AoS struct would — `DURATION` is 28 B, not a padded 32;
`ZONED DATETIME` 14, not 16 — so the multi-field ratios beat the back-of-envelope
estimates.) `DATE` at `i32` (4 B, Arrow `Date32`-aligned) is why the overflow decision
kept the i32 wall — the packed date column is the payoff.

Wire form is unchanged: `value_id` reconstructs `Value::Temporal(...)`, so codecs,
Arrow egress, FFI, and byte-identity with the (non-columnar) TS engine are untouched.
A key that mixes temporal subtypes (or a temporal + non-temporal) promotes to `Mixed`
exactly like `Num`/`Str` do today.

### Two phases

- **(a) storage** — new `Column` variants + `value_kind`/`empty_col_for`/`col_set`/
  `value_id`/`push_absent`/`element_len`/`is_present`/`to_mixed`/`remove_value` arms.
  Buys the memory win + discriminant-free reads. Near-zero risk (wire form via
  `value_id`).
- **(b) vectorized** — engage the vectorized scan for temporal props. Shipped as a
  gather (`gather_temporal` → `VVec::Gen`), replacing per-row `eval` dispatch with a
  tight reconstruct loop. A fully-typed integer-loop comparator (Duration SoA
  early-resolve on `months`) is the further, deferred step (see Results → Remaining
  opportunity).

Tiered expectation held: the gather (project 2×, others ~1.4–1.5×) is the low-risk
majority; the typed-compare ceiling (esp. `DURATION` sorts) is what's left on the table.

## Baseline (before de-boxing)

`cargo run --release --example temporal_bench` — 200k Person vertices, one column of
each type, all in `Column::Mixed` / scalar eval. Build reported **~219 B/vertex across
6 cols (~36 B/col)**, empirically confirming the ~40 B Mixed slot.

| type      | filter>p count | order by | project | min/max |
| --------- | -------------- | -------- | ------- | ------- |
| date      | 13.4 ms        | 37.7 ms  | 4.3 ms  | 21.0 ms |
| time      | 14.3 ms        | 37.2 ms  | 4.4 ms  | 21.1 ms |
| datetime  | 11.7 ms        | 37.1 ms  | 4.2 ms  | 22.3 ms |
| ztime     | 14.5 ms        | 36.3 ms  | 4.3 ms  | 21.9 ms |
| zdatetime | 13.2 ms        | 37.1 ms  | 4.3 ms  | 25.0 ms |
| duration  | 14.4 ms        | 38.4 ms  | 6.3 ms  | 21.3 ms |

Flat across types (all share the Mixed scalar path) — the differentiation is what
de-boxing should produce. `project` is expected to move least (it rebuilds a 40 B
`Value` per row regardless); `filter`/`order`/`min-max` are the vectorization targets.

## Results (200k vertices, same harness)

Each cell is the **mean of all six type rows** for that op/phase (they cluster tightly
— the gather is kind-agnostic), baseline → phase (a) storage → phase (b) gather:

| op             | baseline | phase (a) | phase (b) | total |
| -------------- | -------- | --------- | --------- | ----- |
| filter>p count | 13.57 ms | 11.37 ms  | 9.61 ms   | 1.41× |
| order by       | 37.28 ms | 31.59 ms  | 25.99 ms  | 1.43× |
| project        | 4.63 ms  | 4.06 ms   | 2.18 ms   | 2.13× |
| min/max        | 22.10 ms | 19.86 ms  | 14.36 ms  | 1.54× |

Per-type spread is small except two outliers: **duration** is consistently slowest
(baseline project 6.3 ms — its lexicographic compare is why it heads the deferred
typed-compare list), and **zdatetime** min/max ran high at baseline (25.0 ms). Raw
per-type numbers come straight from `temporal_bench`.

**Speed and memory are different axes** — the memory ratio is per-type (Date 9.70×,
Duration 1.42×; table above), but speed is roughly uniform (~1.4–2.1×) because the
shipped phase-(b) gather reconstructs a `Val::Temporal` per row regardless of packed
width. The 10×-fewer-bytes of a Date column would only translate to speed in a
cache-bound _integer-loop_ scan — the deferred typed-compare, and even there speed is
bounded by more than byte count. `project` doubling reflects the gather removing the
per-row eval dispatch (its dominant cost), not the memory ratio.

### Phase (c) — typed comparator (shipped for filter + min/max)

Two isolated interceptions in the vectorized scan, each reusing the **canonical**
comparator so they're byte-identical by construction (no re-derived ordering):

- `temporal_cmp_vec` — `<temporal col> <op> <temporal scalar>` (literal or `$param`,
  either operand order) via `compare_vals`. Skips the per-row `Binding`+`Env`+expr-tree
  dispatch. **filter: ~9.5 → ~1.7 ms (~5.5×; ~7× vs baseline).**
- `temporal_minmax` — global `min`/`max` fold via `Temporal::cmp_total` (first-seen on
  ties, identical to the scalar `fold_extreme`); previously bailed to the scalar
  accumulator. **min/max: ~14 → ~1.3 ms (~11×; ~16× vs baseline).**

Both use `compare_vals`/`cmp_total` on reconstructed Copy temporals — correctness is
inherited, not reimplemented (covers duration-unordered → UNKNOWN, cross-kind → UNKNOWN,
three-valued nulls). Regression test: `vectorized_temporal_filter_matches_canonical_compare`.

Updated op scoreboard (avg across types, baseline → final):

| op             | baseline | final   | speedup |
| -------------- | -------- | ------- | ------- |
| filter>p count | 13.57 ms | ~1.7 ms | ~7×     |
| min/max        | 22.10 ms | ~1.3 ms | ~16×    |
| project        | 4.63 ms  | ~2.1 ms | ~2.2×   |
| order by       | 37.28 ms | ~26 ms  | ~1.4×   |

### Phase (c) — ORDER BY typed sort (single-key fast path, shipped)

A **single** temporal ORDER BY key now sorts `Vec<Option<Temporal>>` via
`Temporal::cmp_total` directly (`temporal_sort_key` + `temporal_compare_sort`),
skipping the `Val` wrapper + dispatch. Multi-key / non-temporal / mixed keys fall
through to the **unchanged** generic `Vec<Val>` sort — so the shared comparator is
untouched and only the narrow new branch carries risk (guarded by
`fuzz_temporal_order_by_vec_eq_scalar`, which runs every temporal ORDER BY through
both engines). **order by: ~26 → ~20 ms (~1.3×; ~1.9× vs baseline).**

### Phase (c) — ORDER BY dense sort key (all instant kinds)

The typed comparator closed _dispatch_ but not _cache density_ (`Option<Temporal>` is
40 B/key). A **dense packed key** — one monotonic `i128` per row
([`TemporalCol::monotonic_key`], each instant kind packs its components highest-field
first; the signed zoned `offset` biased to unsigned) — sorts a flat integer array like
a numeric column. `Duration`'s 4-component lexicographic order doesn't reduce to one
integer, so it stays on the typed comparator.

The win is **workload-dependent**, which is why it's kept: measured on a full-sort
alone it's marginal (~5%, because materializing 200k `Val::Temporal` **outputs**
co-dominates — inherent, the result _is_ temporal values), but on **top-k**
(`ORDER BY ts LIMIT n` — output is tiny, comparisons are the whole cost) it's a real
win. A database serves both, so both matter.

| workload         | typed comparator | dense `i128` key |
| ---------------- | ---------------- | ---------------- |
| full-sort (200k) | ~20 ms           | ~19 ms (~5%)     |
| top-k 20         | ~2.4 ms          | ~1.9 ms (~1.25×) |

Design note: `i128` is uniform across the instant kinds (`DateTime`/`Zoned*` _need_ it —
`secs` is a full `i64`). A `Date`/`Time`-only `i64` key measured faster on top-k (~1.36
vs ~1.82 ms, since `i128` compares two words) but only covers two kinds and needs a
second packing width + cascade level — not worth the split; uniform `i128` wins on
simplicity. The packing's monotonicity is an invariant the compiler can't enforce (it
depends on each struct's field ranges + derived `Ord`); `fuzz_temporal_order_by_vec_eq_scalar`
is what guards it — it caught nothing here, including the zoned offset bias.

### Final op scoreboard (avg across types, baseline → final)

| op              | baseline | final   | speedup |
| --------------- | -------- | ------- | ------- |
| filter>p count  | 13.57 ms | ~1.7 ms | ~7×     |
| min/max         | 22.10 ms | ~1.3 ms | ~16×    |
| project         | 4.63 ms  | ~2.1 ms | ~2.2×   |
| order by (full) | 37.28 ms | ~19 ms  | ~2×     |
| order by top-k  | —        | ~1.9 ms | (dense) |

## Prerequisite shipped

Date/datetime range overflow (`instant ± huge duration` past the i32 day range) now
raises `E_DATA_EXCEPTION` in both engines (native `FAULT_DATE_OVERFLOW`), superseding
the old D4 → null — same policy as duration overflow and division by zero. The i32
date wall is retained (Arrow `Date32`-aligned; covers all real dates), which is what
keeps the packed date column at 4 B.
