# GQL corpus — completed burn-down and permanent deferrals

The GQL conformance corpus (the `*.jsonl` files in this directory) is a regression
snapshot: each case runs on `lenke-engine` and its result is compared to the frozen
outcome recorded in `snapshots.jsonl` — see [`README.md`](./README.md) for how that
snapshot is captured and regenerated. This file records the end state of the
burn-down that grew the engine's GQL surface up to the snapshot.

**Every buildable feature gap is cleared.** The corpus sits at its principled floor:
the remaining divergences (~40 cases) are INTENTIONAL value-contract choices,
baselined on purpose — not gaps to close. They are the same deliberate policies the
engine documents elsewhere (one f64 numeric model, strict `CAST`, first-class null,
the cross-type comparison rules) applied where core's behavior was looser. By group:

- **f64 numeric model** — `num_string_overflow`, oversized integer literals,
  `distinct_nan`: one numeric type (f64); a value outside it is rejected, never
  silently widened.
- **Strict CAST** — casting bool / list / int-with-null throws where core coerced.
- **Range / quantifier bounds** — `range_bounded` caps; `zero_bound_3` (core
  collapses `{0,0}` to `{0,1}`; the engine's ISO-correct answer is 4).
- **Reserved words** — a reserved word used as an identifier is rejected
  (`m_reserved_word`).
- **Closed RECORD schemas** — `is_typed_closed`, `inline_constraint`: a record typed
  against a closed schema rejects extra or mismatched fields.
- **Hardening arithmetic** — `bool*num`, `str+num`, oversized-int, overflow-exponent:
  a type error, not a coercion.
- **Temporal** — `sum` / `avg` over temporal (and mixed) reject; `date_part` is
  strict; `temporal_duration` ordering (core lacked duration ordering — the engine's
  comparison is the correct one).
- **Faulting aggregates / CALL config** — `faulting_aggregate`, `call_config_*_error`:
  surface the fault instead of returning a partial result.

Because these are intentional, the corpus baseline is these cases, not zero. A NEW
divergence outside this set is a regression; regenerate the snapshot only for an
intended behavior change (see the README).

## Known non-corpus deferral

One shape has no corpus case and is deferred: an EMPTY inner repetition (`{0,n}`)
inside a nested subpath group — core's epsilon-cycle closure. Nothing in the suite
exercises it (the nested shapes all use an inner `min >= 1`).

## Byte-identity

The live contract — that `lenke-engine`, its wasm build, and the pure-TS
`@lenke/core` engine agree bit-for-bit — is upheld by the TS-side differential /
codec / write / injection fuzzers under `packages/native/src/`, with the frozen
`snapshots.jsonl` here as the GQL regression baseline.
