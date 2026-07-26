# GQL extensions & the sigil convention

Status: **implemented** — the unique-constraint primitive and `_MERGE` (node +
edge form, dialect flag, WHERE-gate) ship on both engines, byte-identical, with a
cross-engine differential and an iso-strict conformance gate. Only multi-hop
compound `_MERGE` (v2) remains deferred. This is the authoritative spec for how
lenke adds capabilities ISO GQL does not define, and the first such capability,
`_MERGE` (keyed upsert). Written to survive context compaction — read it top to
bottom before touching the lexer/parser/eval.

---

## 0. Why extensions exist at all

ISO/IEC 39075 GQL v1 (ratified 2024) is a _first edition_. What it specifies well
is the **declarative read surface** — pattern matching, projection, aggregation,
the type system. What it barely touches is the **write / schema / transaction**
layer: no `MERGE`/upsert, no constraint DDL, transactions largely
implementation-defined, no temporal types. (Transactions are now implemented at
the core as an engine-neutral host API — `graph.transaction(fn)` / `graph.tx()`,
matching how Gremlin exposes them — see `docs/design/r-tx.md`. GQL's ISO
`START TRANSACTION`/`COMMIT`/`ROLLBACK` keywords are a planned thin veneer over
those primitives, not invented syntax.) Every graph vendor (Neo4j, Ultipa,
TigerGraph, SQL Server) fills those gaps with **mutually incompatible,
non-conformant extensions** (see the research summarized in `docs/dogfood/` and
the memory `iso-gql-reference`). When the whole field is extending in different
directions, the standard is not yet carrying that weight.

lenke's stance, which drives everything below:

- **GQL is a query _surface_, not the foundation.** The foundation is the core
  engine (the Rust store, byte-identical dual execution, codecs, indexes). GQL —
  and Gremlin — are friendly front doors over it.
- **The advanced / write / schema layer belongs to the core + the host
  language.** Because lenke is an _embedded, in-process_ library, the escape
  hatch when the query language runs out is just the host language operating on
  fast native primitives — no APOC-style bolt-on with an impedance boundary. So
  we do **not** chase query-language completeness; we invest in fast primitives
  and an ergonomic host surface, and add query-language sugar only where it earns
  its keep.
- **When we do add non-ISO syntax, it is marked, gated, and minimal** — the sigil
  convention below.

---

## 1. The sigil convention (the general rule)

Every keyword or construct that has **no standalone meaning in ISO GQL** wears a
leading-underscore sigil: `_MERGE`, `_ON_CREATE`, `_ON_UPDATE`,
`_ON_UPDATE_NOTHING`, and every extension after them. Constructs that **are**
valid standalone ISO GQL stay **bare**.

**The test:** _standalone-valid as ISO GQL → bare; only meaningful inside the
extension → sigil._

- A pattern `(a:User {email:$e})` is valid after `MATCH`/`INSERT` → **bare**.
- A `SET x.p = v` assignment is a valid GQL clause → **bare**.
- A `WHERE <pred>` predicate is valid GQL → **bare**.
- `_ON_UPDATE` / `_ON_CREATE` have no meaning outside `_MERGE` → **sigil**.
- `_MERGE` itself is not an ISO verb → **sigil**.

**The marked container owns the non-standard semantics.** A bare sub-construct
never independently asserts a non-standard meaning; it inherits its context from
the sigil'd container that encloses it. This is already how standard GQL behaves —
the same pattern `(a:User {…})` means "find" after `MATCH` and "make" after
`INSERT`; the _clause_ decides. Inside `_MERGE`, the pattern's treatment is
match-or-create, but the `_MERGE` sigil is what signals that. So the invariant
holds: **no bare token ever carries lenke-specific semantics.**

### Why this shape

- **Collision-proof by construction.** ISO keywords are bare words; `_MERGE` is a
  different token. We can _never_ collide with a future GQL edition. Direct
  precedent: the C standard reserves every identifier beginning with an
  underscore (+ uppercase / another underscore) for "the implementation" — a
  sigil-marked namespace set aside for non-standard extensions precisely so they
  can't clash with the standard. (It is also, in spirit, APOC's `apoc.*`
  namespace — but inline, not a procedure-call detour.)
- **Self-documenting at the call site.** A reader sees `_MERGE` and knows it is
  non-portable without consulting a doc. The friction is the honest signal — read
  it like `unsafe` in Rust; the mild ugliness discourages casual dependence on
  extensions, which is the right incentive.
- **Migratable.** If ISO later standardizes create-or-match, we add the _bare_
  form as the conformant one and keep `_MERGE` as a deprecated alias. No
  ambiguity — they are literally distinct tokens.

### What this must NOT do

- **Never add sigil words to the ISO `RESERVED` set** (`packages/gql/src/lexer.ts`
  `RESERVED`, `crates/lenke-core/src/gql/lexer.rs` `RESERVED_WORDS`). That set is
  "verbatim from the standard so the list can't drift" — keep it pristine.
- **Extensions are contextual keywords.** `_MERGE` is recognized only in
  statement-leading position; `_ON_*` only inside a `_MERGE`. Elsewhere a
  `_`-leading word remains an ordinary identifier, so extensions can _never_
  shrink the conformant identifier namespace (and never worsen the reserved-word
  footgun that breaks bare labels).

### The dialect flag

The GQL parse entry takes a dialect option: `'lenke'` (default, extensions
permitted) or `'iso-strict'` (any sigil/extension construct is a parse error).

- Users who want portable GQL get a lint by parsing under `iso-strict`.
- **The differential/conformance harness runs under `iso-strict`**, which
  _proves_ the ISO surface is self-contained — an extension can never leak into
  the conformance corpus by accident. This is the testable version of Ultipa
  keeping its extensions out of its conformance feature tables.

The sigil marks intent for the human; the flag is the switch for tooling. They
compose — `iso-strict` rejects a construct even though it is clearly marked.

---

## 2. `_MERGE` — keyed upsert (the first extension)

### 2.1 Scope (v1): single upserted element with matched context

`_MERGE` upserts **exactly one element** — one node, or one edge. Any _other_
element in the pattern is a **keyed match that locates** the upsert target; it is
not created. Concretely:

- **Node upsert:** `_MERGE (p:Presence {sessionId:$s, x:$x, y:$y})` — upserts the
  Presence node.
- **Edge upsert:** `_MERGE (u:User {id:$u})-[m:MEMBER]->(g:Group {id:$g})` — the
  endpoints `u` and `g` are matched by key; the **edge** `m` is the upserted
  element.

**Deliberately deferred to v2: multi-create compound patterns** (where any
element in the path may be created). That case needs per-element dispositions and
leans on multi-anchor index-seed planning (`R-SEED` in `docs/dogfood/ROADMAP.md`).
It is out of scope because the performance, predictability, and — especially — the
_conflict ergonomics_ of multi-element create are exactly where Cypher's MERGE
becomes a foot-cannon. v1 stays bounded, footgun-free, and covers the two use
cases that motivated the feature (presence, authz ensure-tuple).

### 2.2 Key vs payload

- **Key** = the property(ies) named by a **declared unique constraint** on the
  element's label (see §3). The conflict target is **inferred** from
  `pattern ∩ declared unique constraints`. If a label carries multiple unique
  constraints and the pattern touches more than one, that is **ambiguous → error**;
  the user must disambiguate (explicit conflict-target syntax, TBD, only needed
  for the multi-constraint case).
- **Payload** = every other inline property in the element's pattern.

A `_MERGE` on a label with **no** applicable unique constraint is a **compile
error** — we cannot define "the key." (This is the deliberate Pattern-B choice;
see §2.6. It is what makes the key/payload split unambiguous and the upsert
concurrency-safe.)

### 2.3 Dispositions

**Create path** (element absent): insert the pattern (key + payload), then apply
`_ON_CREATE SET …` birth-only extras.

**Update path** (element present): **exactly one** disposition, and an explicit
one **replaces** the default entirely (SQL-faithful — the `DO UPDATE SET` clause
_is_ the whole conflict action):

| Update disposition           | Meaning                                    | SQL analog                          |
| ---------------------------- | ------------------------------------------ | ----------------------------------- |
| _(default, bare)_            | clobber payload → pattern's declared state | `DO UPDATE SET <every payload col>` |
| `_ON_UPDATE SET … [WHERE p]` | replaces default; do exactly this          | `DO UPDATE SET … WHERE p`           |
| `_ON_UPDATE_NOTHING`         | the empty disposition; leave untouched     | `DO NOTHING`                        |

`_ON_UPDATE_NOTHING` and `_ON_UPDATE SET …` are mutually exclusive (you cannot
both do-nothing and update). Because an explicit `_ON_UPDATE` _replaces_ the
default, `_ON_UPDATE_NOTHING` falls out for free as "an update that sets nothing"
— no special-casing.

Note the (documentable) asymmetry, same shape as SQL: **omitting** the update
disposition means _clobber_; `_ON_UPDATE_NOTHING` means _nothing_. "Say nothing"
≠ "say do-nothing." (SQL: omitting `ON CONFLICT` errors; you opt into a
disposition.)

### 2.4 Property matrix

| Where you put a prop | Written on create | Written on update                         |
| -------------------- | ----------------- | ----------------------------------------- |
| pattern payload      | yes               | yes — **only under the default clobber**  |
| `_ON_CREATE SET`     | yes               | no                                        |
| `_ON_UPDATE SET`     | no                | yes (and it is the _whole_ update action) |

One-liner: _the pattern is identity + desired state; `_ON_CREATE` adds birth-only
fields; the update path is one disposition, defaulting to "make the payload match
the pattern."_

### 2.5 WHERE-gated conditional update (last-write-wins / optimistic concurrency)

```
_MERGE (d:Doc {id: $id})
  _ON_UPDATE SET d.body = $body, d.version = $v WHERE d.version < $v
```

"Only overwrite if the incoming version is newer; otherwise leave it." If the
predicate is false it is a **no-op, not an error** (SQL semantics). `WHERE` stays
bare: it is a standard predicate doing a standard filtering job; the marked
`_ON_UPDATE` container owns the "this gates the upsert" meaning.

### 2.6 Worked example (full form)

```
_MERGE (u:User {email: $e, name: $n})
  _ON_CREATE SET u.created = $now
  _ON_UPDATE SET u.lastSeen = $now
```

- `email` is the unique-constraint key → conflict target.
- `name` is payload.
- Absent → create User{email,name}, set created.
- Present → run the explicit update (set lastSeen); because the explicit
  `_ON_UPDATE` _replaces_ the default, `name` is **not** re-clobbered here.

Seed-if-absent (`ON CONFLICT DO NOTHING` with a rich insert):

```
_MERGE (u:User {email: $e, name: $n})
  _ON_CREATE SET u.created = $now
  _ON_UPDATE_NOTHING
```

Presence (bare clobber is exactly right; ensure-exists would freeze the cursor):

```
_MERGE (p:Presence {sessionId: $s, x: $x, y: $y})
```

### 2.7 Documented divergences

- **vs Cypher MERGE.** We are _not_ whole-pattern match-or-create; v1 upserts a
  single element with matched context, element-keyed, which avoids Cypher's
  duplicate-node footgun by construction. We **clobber payload by default**
  (Cypher never clobbers — its bare `MERGE` treats all inline props as match
  key). We use `_ON_UPDATE` (data-op framing), not `_ON_MATCH` (traversal
  framing). We **require** a unique constraint (Cypher's MERGE works without one
  — and races into duplicates under concurrency, which its own docs warn about).
- **vs SQL upsert.** We **clobber by default** (SQL's minimal `DO NOTHING` leaves
  the row; SQL makes you list every column in `DO UPDATE SET`). Otherwise
  aligned: conflict target, `WHERE`-gated update, `DO NOTHING`.
- **null interaction.** A payload `null` **stores** null (present, not absent),
  consistent with lenke's first-class-null policy (memory `null-first-class-policy`,
  `nan-comparison-policy` is the sibling precedent). Removal is still explicit
  (`REMOVE` / `.properties(k).drop()`), never via upserting null.

---

## 3. The unique-constraint primitive

`createUniqueConstraint(label, keys)` — a **programmatic host API**, mirroring the
existing `createVertexIndex` / `createEdgeIndex` seam (memory
`native-property-indexes`). Both engines, byte-identical.

**Why programmatic, not GQL DDL.** A host API is **not** GQL, so it can _never_
collide with a future GQL constraint-DDL edition — the strongest forward-compat
guarantee. (ISO has pre-reserved `constraint`/`unique` "for future use", which
makes guessing their eventual DDL syntax the _riskiest_ place to extend, not the
safest.) When ISO ships constraint DDL, that is purely additive and can call the
same underlying primitive. It also matches how indexes are already declared, so
it is the low-surprise seam.

**Semantics.**

- A unique constraint declares that `(label, keys)` identifies at most one
  element. It is **backed by an index** for seek performance (the key lookup in
  `_MERGE` and the enforcement check both seek, not scan).
- A plain `INSERT` that violates a unique constraint **errors** (SQL-family
  default; coded error, both engines identical).
- `_MERGE` _uses_ the constraint to locate the existing element and reconcile —
  it is the constraint-aware write that does not error on a "conflict" but
  applies the disposition instead.

---

## 4. Byte-identical requirement

Both engines (`@lenke/gql` + `crates/lenke-core`) implement `_MERGE` and the
constraint identically. The differential harness
(`packages/native/src/gql-functions-conformance.test.ts` and siblings) must cover
upsert cases and assert byte-identical results across TS and Rust. The
conformance corpus runs under `iso-strict` (§1) to prove the ISO surface stays
pure. New behavioral tests live in both `packages/gql/src/*.test.ts` and
`crates/lenke-core/src/gql/*.rs`, each carrying the divergence comments from §2.7.

---

## 5. Build slices (see task list / ROADMAP `R-MERGE`)

1. Unique-constraint primitive — Rust core (storage + index-backing +
   INSERT-time enforcement).
2. Unique-constraint primitive — TS engine parity.
3. Dialect flag + contextual extension-keyword lexing (both engines).
4. `_MERGE` parser + AST: node form (both engines).
5. `_MERGE` executor: node upsert, dispositions, clobber/replace/nothing (both).
6. `_MERGE` parser + executor: edge form with matched endpoints (both).
7. `WHERE`-gated conditional update (both).
8. Differential harness cases + `iso-strict` conformance gate.
9. Docs: `packages/gql/README.md` (supersede the "there is no MERGE" section),
   constraint API docs, divergence notes; this file kept current.

Per slice: edit → `cargo test` / `bun test` the touched area → commit locally
(**never push** — standing constraint). Full gate `bun run check` + `bun run
build` before calling it done.

## 6. Extension functions — temporal components (`_year` … `_second`)

Status: **implemented** (2026-07-25). The first _function_-shaped extensions (the
sigil convention was born on statement keywords like `_MERGE`; it applies to
functions too).

`_year(x)`, `_month(x)`, `_day(x)`, `_hour(x)`, `_minute(x)`, `_second(x)` extract
a calendar/clock component from a temporal value. They wear the sigil because
**date-part extraction is not in the ISO GQL function catalogue** — verified
against the 39075 Feature-ID taxonomy (as reproduced in Neo4j's `docs-cypher`
`appendix/gql-conformance/*.adoc`): it is neither a mandatory value function nor a
catalogued optional feature. Other vendors each solve it differently and
non-portably (Ultipa `year()`, Spanner SQL `EXTRACT`, Fabric omits it, Cypher a
`.year` accessor) — precisely the "no single conformant form" situation the sigil
exists to mark. So the bare `year(...)` stays an unknown function; only `_year(…)`
resolves.

Semantics (both engines byte-identical): `_year`/`_month`/`_day` require a
date-bearing temporal (`DATE`/`LOCAL DATETIME`/`ZONED DATETIME`);
`_hour`/`_minute`/`_second` require a time-bearing one (`LOCAL TIME`/`LOCAL
DATETIME`/`ZONED *`). A zoned value is read in its **own stored offset** (local
wall clock). A string is **not coerced** — it throws `E_INVALID_VALUE` (wrap with
`date()`/`local_datetime()`/`local_time()`); a temporal lacking the requested
component, or a duration, throws likewise; `null` → `null`. **Migration:** if a
future GQL edition standardizes component extraction, add the bare conformant name
then and keep `_year` as a deprecated alias (per §1's migration rule).
