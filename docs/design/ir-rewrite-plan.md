# Plan: one logical IR, a rewrite-rule optimizer, and one value contract

This is the forward plan for three of the four directions from the architecture
review — **#1 one logical IR + rewrite optimizer**, **#2 a physical-operator
layer with lineage as a planned property**, and **#4 one value-semantics
contract**. It deliberately excludes **#3** (whether pure-TS must exist); that is
a product decision, not an engineering one, and none of the work here depends on
its answer.

It is the successor to [`query-ir.md`](./query-ir.md), which is the running LOG
of the IR experiment — what was tried, measured, kept, and reverted. Read that
for the evidence; read this for the sequence. Where the two disagree, the log is
the record of fact and this is the intent.

## The thesis, in one paragraph

Today the engine has two languages (GQL, Gremlin), two execution models (a
row/`Trav` interpreter `apply`, and a columnar path), and optimizations written
as **shape recognizers on surface syntax** — so the same idea gets implemented
per language and per spelling, and a query worded differently silently misses
the fast path. The target is: both languages lower to **one logical IR**;
optimizations are **rewrite rules on that IR** (written once, firing on every
spelling in both languages); a **physical planner** chooses row vs columnar
operators from a **lineage property** the analysis already computes; and value
semantics (order, equality, coercion, null) live in **one contract** every layer
consults. Storage is not touched.

## What this is NOT

- **Not a greenfield rewrite.** It is a strangler-fig migration of a live,
  shipping system. Every phase leaves the engine correct, byte-identical across
  Rust/TS, and no slower. There is no "big bang" cutover.
- **Not an attempt to delete `apply`.** The row/column split is irreducible:
  path/tags/sack need per-traverser identity, and the fast columnar ops
  (`walk_count`, `reach_back`, degree products, bitset dedup) work by destroying
  it. `apply` is reframed as _the row physical-operator set_, not deleted. See
  [`query-ir.md`](./query-ir.md) "The larger version, and why not yet".
- **Not a promise of a smaller codebase.** It removes _accidental_ complexity
  (the fastpath ladder, per-language optimization asymmetry, scattered value
  policy). The row/column and Rust/TS dualities remain unless #3 retires pure-TS.
  Honest expected outcome: **the same capability with far fewer
  "fixed-one-engine-forgot-the-other / one-spelling-missed-the-rung" bugs**, and
  a structure where coverage is legible.

## Non-negotiable constraints (hold on every commit)

1. **Byte-identity Rust↔TS.** The six differential fuzzers stay green throughout.
   No phase may land with a known divergence.
2. **No perf regression** past the noise floor, measured against `main` with the
   existing harnesses (`perf_bench`, `exists_probe`, `gremlin_bench`, the
   `*_audit` ignored tests). A rewrite rule that is slower than the fastpath it
   replaces does not land (this already happened once — `order().by(k)`).
3. **Shippable at every phase boundary.** Each phase is independently
   revertible. Old and new paths coexist behind a switch until the new one is
   proven, then the old one is deleted in a named commit.
4. **Equivalence is proved, not assumed.** A rewrite rule is a _desugaring_
   (meaning-preserving) or it does not land. The `{2,2}`-vs-two-segments trap
   (`query-ir.md` "Desugaring vs mistranslation") is the standing example: right
   on every fixture anyone would think to write, wrong on a self-loop.
   `equivalent_spellings_cost_the_same` gets a new group per rule.

## What already exists to build on

This is a consolidation, not an invention. The pieces are present and partly
wired:

| Target concept                 | What exists today                                                       |
| ------------------------------ | ----------------------------------------------------------------------- |
| Logical IR                     | GQL's plan (`CQuery`/`CLinear`/`CClause`/`CExpr`) — the seed            |
| Front-end lowering             | `gremlin::to_gql::tail` + `pattern::compile` (Gremlin → GQL, partial)   |
| Lineage as a plan property     | `gremlin::analysis` (`needs_path`/`reads_tags`); GQL's `refs_slot`      |
| Shared access path             | `crate::seek` (`adj`, `walk_count`, `reach_back`, `Frontier`)           |
| Columnar value model           | `crate::value::Col`, `Col::from_property` (shared GQL/Gremlin)          |
| Value semantics (partial)      | `crate::value` (`cmp_total`, `group_key_bits`) — scattered, not central |
| The rewrite rules, hand-rolled | the `try_*` fastpaths in `gql/eval/fastpath.rs` + `gremlin/exec.rs`     |
| Safety net                     | six differential fuzzers; the `*_audit` ignored tests                   |

## Target architecture (the end state)

```
GQL text ──▶ GQL parser ──┐
                          ├──▶  LOGICAL IR  ──▶  rewrite optimizer  ──▶  physical planner  ──▶  execution
Gremlin text ▶ Grem parser┘     (language-        (rules; fires on        (picks row vs             over storage
                                 agnostic          any spelling,           column per operator      (seek, Col,
                                 algebra)          both languages)         from LINEAGE property)    columnar store)
                                     │                                            │
                                     └──────────── VALUE CONTRACT ────────────────┘
                                          (order / equality / coercion / null —
                                           consulted by IR eval, operators, storage)
```

- **Front-ends** are thin: parse + lower to the logical IR. Nothing downstream
  branches on which language a plan came from. (Per-language _contracts_ — e.g.
  GQL's NaN-as-no-value in predicates vs Gremlin's — are encoded as plan
  attributes, not as forks in the executor.)
- **Logical IR** is relational-with-graph-extensions: scan, expand/hop, filter,
  project, aggregate/group, join, order, page, plus the graph-native ops
  (var-length, shortest-path, the algorithm calls). It carries a **lineage
  requirement** (does anything downstream read path/tags/sack).
- **Optimizer** is a fixed set of rewrite rules applied to fixpoint (start
  simple: a single ordered pass; add a cost model only if a rule needs one). Each
  rule is meaning-preserving and independently testable.
- **Physical planner** lowers each logical operator to a physical one, choosing
  row vs column by the lineage requirement — the decision `needs_path` gates
  today, promoted from a retrofit to a first-class step.
- **Physical operators**: a row set (what `apply` is) and a column set (what the
  columnar path is), behind one interface so coverage is enumerable.

## The phases

Each phase lists **entry**, **exit**, **verification**, and **rollback**. Phases
are ordered so each de-risks the next; 1 and 4a can proceed in parallel.

### Phase 0 — Instrument and freeze the contract (no behavior change)

The measurement and safety scaffolding, so every later phase can prove itself.

- Stand up a **plan-diff harness**: for a corpus of queries (both languages),
  dump the logical IR before/after a rewrite and assert the _result_ is
  unchanged and the _plan_ changed as intended. This is the desugaring proof
  obligation made mechanical.
- Extend `equivalent_spellings_cost_the_same` into the standing gate for every
  rule added later.
- Snapshot the current `perf_bench`/`exists_probe`/`gremlin_bench` numbers as the
  regression baseline in-repo.

**Exit:** the harness exists and is green on today's engine. No behavior change.
**Rollback:** trivial (test-only).

### Phase 1 — One value contract (#4)

Lowest risk, everything else depends on it, and it is where the subtle
divergences hide (NaN order, walk-vs-trail, coercion).

- Create `value::contract` (or promote within `crate::value`): the single home
  for **total order** (`cmp_total`), **equality** (`structuralEq` / `val_key`),
  **coercion** (bool↔num, cross-type → UNKNOWN vs fault), and **null policy**
  (null-as-value, NaN-as-no-value-in-predicates).
- Replace every scattered restatement — the NaN handling in `gql/eval`,
  aggregation, sort, and `seek` — with a call into the contract. This is a pure
  refactor: same behavior, one source.
- Mirror the contract in TS (`packages/*/src`) as the same single module.

**Exit:** grep finds NaN/equality/coercion policy in one place per engine; all
fuzzers green. **Verification:** differential fuzzers (this is exactly what they
exist for); no perf change (it is a refactor). **Rollback:** per-callsite,
independently.

### Phase 2 — Promote GQL's plan to THE logical IR, and make Gremlin lower into it (#1a)

This is `query-ir.md` steps 0–2, sequenced.

- **2a. Restructure `run_collect` so the Gremlin↔pattern boundary is a handoff,
  not a cliff.** Consume the longest lowerable _prefix_ into a frontier, hand the
  rest to the row set. Must start from the universe, not a seed (the recorded
  failed attempt started from `index_seed`, which declines a bare `V()` — see
  `query-ir.md` "The plan"). This converts every future rule from "helps queries
  that lower end-to-end" to "helps every query containing the shape".
- **2b. Unify the two fallbacks' binding model.** A `Binding` is a `Trav`
  without traverser state; Gremlin `as()` tags compile to slots the way GQL
  variables do. One row-at-a-time driver; the only per-language part is
  step/clause interpretation. (Lineage stays a `Trav` concern — see Phase 4.)
- **2c. The join layer.** GQL's multi-clause matcher and Gremlin's `match()` are
  the same operation; share it. This is the layer that must be shared before
  _either_ per-language matcher can shrink.

**Exit:** both languages produce logical-IR plans for the shapes that lower;
the row driver is one implementation. **Verification:** fuzzers + full perf
suite per sub-phase; the `route_audit`/`arm_shadow_audit` tests track coverage.
**Rollback:** each sub-phase behind `LOWERING_OFF`-style switches already in the
codebase.

### Phase 3 — Rewrite-rule optimizer: turn fastpaths into rules (#1b)

The payoff phase. Each hand-rolled fastpath becomes a rule on the logical IR,
and its shape-recognizer version is deleted once the rule subsumes it.

- Build the rule-application driver (ordered pass to fixpoint; no cost model
  until a rule demands one).
- Port, one at a time, each as a rule:
  - **count-without-enumerate** ← `try_walk_count` (aggregate pushdown)
  - **selective EXISTS backward** ← `semi_join_back` (semi-join reordering)
  - **grouped count = degree product** ← `try_grouped_walk_count`
  - **distinct = bitmap / frontier collapse** ← `distinct_expand_count`
  - **comma-join = product** ← `try_count_comma_join`
  - **count-of-groups** ← `try_count_groups`
  - … and the survivors in `fastpath.rs`.
- Each port: rule lands, plan-diff harness proves equivalence, perf ≥ the
  fastpath it replaces, THEN the fastpath is deleted in the same commit. A rule
  that is slower does not land (the `order().by(k)` precedent).
- The ladder — the reason "the same idea was implemented eight times" — dies
  here, because a rule fires on the normalized IR regardless of spelling or
  language.

**Exit:** `run_part`'s `try_*` ladder is gone; the rules are the optimizer.
**Verification:** `equivalent_spellings_cost_the_same` (a group per rule),
`migration_arm_price_audit`, full perf suite, fuzzers. **Rollback:** a rule can
be disabled without disabling the others.

### Phase 4 — Physical-operator layer with lineage as a planned property (#2)

Reframe, not rewrite. Make the row/column choice a planner decision instead of a
runtime "decline to the total interpreter".

- Define the physical-operator interface; register the existing row set
  (`apply`'s arms) and column set (the terminals/`Col` path) as implementations.
- Promote `analysis`'s `needs_path`/`reads_tags` (plus a `needs_sack`) to a
  **lineage requirement** on the logical plan, computed once. The planner picks
  the column physical op when lineage is not required and one exists, else the
  row op.
- Retire the `MIGRATE_OFF`/`PATTERN_OFF`/`LOWERING_OFF` decline scaffolding in
  favor of explicit planner selection (keep the test switches that A/B, drop the
  ones that model "fell through").
- Coverage becomes enumerable: "which logical ops have only a row physical impl"
  is a query over the registry, not an archaeology dig through decline paths.

**Exit:** row-vs-column is chosen by the planner from the lineage property;
`apply` is _the row operator set_, documented as such. **Verification:** fuzzers;
the arm audits become physical-op coverage reports. **Rollback:** the planner
can be pinned to always-row (which is the total set) as a safety valve.

### Phase 5 — Delete the scaffolding

- Remove dead decline paths, superseded switches, and any fastpath comments now
  encoded as rules.
- Update `query-ir.md` and the memory notes to reflect the landed architecture.

## Sequencing and parallelism

```
Phase 0  ─── everything depends on it
   │
   ├── Phase 1 (value contract) ────────────┐   (independent; can run in parallel with 2)
   │                                         │
   └── Phase 2 (logical IR / lowering) ──▶ Phase 3 (rewrite rules) ──▶ Phase 4 (physical layer) ──▶ Phase 5
```

Phase 1 and Phase 2 are independent and can proceed together. Phase 3 needs the
logical IR total enough (Phase 2) to host the rules. Phase 4 needs the rules
(Phase 3) settled so the physical selection is stable.

## Risks and how each is bounded

- **A rewrite is a mistranslation.** The single biggest risk (`{2,2}` trap).
  Bounded by: plan-diff harness proving result-equivalence, the differential
  fuzzers, and `equivalent_spellings_cost_the_same`. A rule lands only with a
  proof obligation discharged.
- **Perf regression from generality.** The fastpaths are fast because they are
  specialized. Bounded by: perf ≥ fastpath is a landing gate; the fastpath is
  deleted only after the rule matches its number.
- **Scope creep into #3 / a real rewrite.** Bounded by: storage untouched,
  `apply` reframed not deleted, phases independently shippable. If a phase stops
  paying, stop — the engine is correct at every boundary.
- **The row/column reality reasserting itself.** We are not fighting it; Phase 4
  formalizes it. The failure mode is pretending it can be unified — explicitly a
  non-goal.

## What success looks like (measurable)

- Fastpath ladder count → rewrite-rule count (target: the `try_*` family in
  `run_part` is gone; N composable rules remain).
- The bug class "optimization applied to one engine / one spelling only" is
  structurally impossible (rules fire on the normalized IR).
- Value policy: one module per engine; grep proves it.
- `apply` is documented as the row physical-operator set with enumerable
  coverage, not a fallback interpreter.
- Every phase: fuzzers green, perf ≥ baseline.

## Open decisions for the human

1. **#3 timing.** This plan is independent of the pure-TS question, but if #3
   retires pure-TS, Phase 1's "mirror in TS" and the differential-fuzzer tax
   both shrink dramatically. Worth deciding before Phase 1 if it is on the table.
2. **Optimizer ambition.** Start with an ordered fixpoint pass (proposed). A
   cost-based optimizer (Cascades-style) is a larger commitment — defer until a
   rule genuinely needs cardinality estimates to choose (e.g. seed-side
   selection, which today is a heuristic).
3. **How aggressively to delete.** Phase 5 can be conservative (leave dead
   switches as documented no-ops) or thorough. Recommend thorough, but it is the
   riskiest for churn.
