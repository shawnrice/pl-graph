# lenke ISO GQL — feature checklist

Where lenke's **GQL engine** stands against the ISO/IEC 39075:2024 feature set.
Two lenses are combined: the **optional Feature-ID taxonomy** (the _query_
surface) is transcribed from Neo4j's reproduction of 39075 (see
[references.md](./references.md) §2); the **statement / program surface** (the
_structural_ layer — transactions, sessions, catalog & schema DDL) is derived by
walking the [TuGraph ANTLR grammar](https://github.com/TuGraph-family/gql-grammar)
of the spec, which covers what the Cypher-centric pages do not. In both cases
lenke's status was determined **empirically** — each feature was exercised with a
probe query (parse + execute), not inferred from the presence of a reserved word
(the reserved-word list is verbatim from the spec and says nothing about
implementation).

Last verified: **2026-07-26** (`@lenke/gql` portable engine; native is
byte-identical).

**Legend:** ✅ supported · 🟡 partial · ❌ not yet · 🔷 lenke extension (non-ISO,
sigil'd) · ➖ excluded by design · ❓ not yet verified

> **Caveats.** (1) The Feature-ID taxonomy is a faithful _secondary_ source
> (Neo4j, a GQL co-author); the spec _text_ + Feature-ID Annex are paywalled,
> though the [grammar BNF is free](./references.md#1-the-standard-itself). (2)
> Feature IDs `GF08`/`GF09` couldn't be resolved from any free source, and the
> optional table covers only the ~75 IDs Neo4j documents (the ID gaps are real
> spec features no free source names). (3) Every mandatory production in Neo4j's
> list has now been probed; the whole language _syntax_ surface has been walked
> from the grammar.

---

## Gaps (prioritized)

Everything lenke does **not** do, pulled out of the tables below and sorted by
whether it's worth closing. Tiers 1–2 are real gaps; Tier 3 is deliberate.

### Tier 1 — Mandatory-conformance gaps

**No open mandatory gaps.** Of the three the audit found, one (`IS TYPED`) is now
shipped; the other two are a deliberate decline, not a to-do:

- **`normalize()` + `IS [NOT] NORMALIZED`** — the Unicode string-normalization
  feature (NFC/NFD/NFKC/NFKD). **Declined** (2026-07 conformance audit): a
  byte-identical implementation needs a Unicode-normalization table as a Rust
  dependency **plus** a matching TS copy, and its only use — canonicalizing two
  encodings of the same glyph — is niche and better handled at ingest. A
  conscious zero-dependency-vs-strict-conformance tradeoff on this one mandatory
  function; would only return as an opt-in if a real workload needs it. So lenke
  is knowingly, minimally non-conformant here.

The one genuinely-open item — **`IS TYPED` — is now closed** (2026-07-26): the
`x IS [NOT] TYPED <type> [NOT NULL]` value-type predicate ships on both engines,
byte-identical. Null conforms to any nullable type (`NOT NULL` excludes it);
numeric split is boundary-inferred (INTEGER = whole-valued number, since lenke has
one f64 type). The `::` alias and parameterized types (`LIST<T>`, `VARCHAR(n)`)
are deferred. So the only remaining mandatory-set gap is the deliberately-declined
Unicode normalization above.

### Tier 2 — Genuine optional-feature gaps (real, not by design)

Ordered roughly cheap→involved.

| Gap                                                               | Notes                                                                                                                                                                                                                                                                                                                                |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Multi-`MATCH` `EXISTS { … }` (GQ22)                               | Single-`MATCH` `EXISTS` works.                                                                                                                                                                                                                                                                                                       |
| Map / record values (GV45) — record VALUE ✅ shipped              | The ISO record VALUE ships end-to-end both engines (GQL + Gremlin): `{k:v}` construct, `.field`/`['k']` access, structural equality, total-order ORDER BY/DISTINCT, SET, stored map persistence + a dotted-path index (`n.meta.city` seek, 140x vs scan). Only the record TYPE (`RECORD {a::INT}`) is deferred with the type system. |
| `DIFFERENT EDGES` / `REPEATABLE ELEMENTS` match modes (G002/G003) | Graph-pattern match modes (distinct from the path modes, which work).                                                                                                                                                                                                                                                                |

### Tier 3 — Excluded by design (NOT gaps to close)

Deliberate consequences of lenke being an embedded, single-graph,
schemaless-by-default engine. Documented so they aren't mistaken for gaps.

- **Multi-graph `USE`, catalog DDL** (`CREATE/DROP GRAPH`, `CREATE SCHEMA`) — graph lifecycle is the host API.
- **Schema / graph-type DDL** (`CREATE GRAPH TYPE`, node/edge type specs, typed graphs) — typed schemas are host-side (`defineNode`) + in-engine constraints.
- **Session management** (`SESSION SET/RESET/CLOSE`) — no multi-session catalog.
- **`INT64` / integer subtypes** (GV12) — single f64 numeric model.
- **`BYTES`/`BINARY`/`VARBINARY`, sized `VARCHAR(n)`** — out of value model.
- **`BETWEEN`** — not ISO GQL at all (absent from the grammar); correctly rejected.
- **List indexing `[i]` / slicing `[i..j]`** — not ISO GQL (no subscript production in the grammar; it only has list _literals_ `[a,b]`). lenke already ships bare `[i]` as a Cypher-style convenience; `[i..j]` slicing would be _new_ non-ISO surface, so it's **not a conformance gap** — a convenience-feature decision, and one in tension with the sigil convention (a bare Cypher-ism).

---

## Baseline (mandatory) features

lenke supports the core mandatory read/write surface: `MATCH`, `INSERT`, `SET`,
`REMOVE`, `DELETE`, `RETURN`, `ORDER BY` + `LIMIT`, `CASE`, aggregates
(`count/sum/avg/min/max`), comparison / `null` / `EXISTS` predicates, property
reference, numeric value functions (`char_length`), and character string
functions (`left/lower/right/trim/upper`).

**Mandatory gaps** (found by walking every `<production>` in Neo4j's
`supported-mandatory` list and probing lenke — see the [Gaps](#gaps-prioritized)
section):

| Mandatory production          | Gap                   | Detail                                                                                                                                                    |
| ----------------------------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `<character string function>` | `normalize()`         | **Declined** — Unicode normalization; needs a normalization-table dep, niche, handle at ingest. See [Gaps → Tier 1](#tier-1--mandatory-conformance-gaps). |
| `<normalized predicate>`      | `IS [NOT] NORMALIZED` | **Declined** — same Unicode-normalization feature as `normalize()`.                                                                                       |
| `<value type predicate>`      | `IS TYPED`            | **Closed 2026-07-26** — `x IS [NOT] TYPED <type> [NOT NULL]` (also optional GA06). `::` alias + parameterized types deferred.                             |

Everything else in the mandatory list is supported (INSERT/SET/REMOVE/DELETE,
MATCH/OPTIONAL MATCH, RETURN/FINISH, ORDER BY/SKIP/OFFSET/LIMIT, UNION/UNION ALL,
comparison/null/EXISTS predicates, CASE/nullIf/coalesce, avg/count/max/min/sum,
char_length/character_length, `||`, left/lower/right/trim/upper).

---

## Optional features

### Paths, patterns & path search

| ID   | Feature                                     | lenke | Notes                                                                                                                                                                                                 |
| ---- | ------------------------------------------- | :---: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| G004 | Path variables (`p = …`)                    |  ✅   | Bound `Path` value, both engines byte-identical.                                                                                                                                                      |
| G005 | Path search prefix                          |  ✅   |                                                                                                                                                                                                       |
| G010 | Explicit `WALK`                             |  ✅   |                                                                                                                                                                                                       |
| G011 | Path mode `TRAIL`                           |  ✅   | lenke default.                                                                                                                                                                                        |
| G013 | Path mode `ACYCLIC`                         |  ✅   | `SIMPLE` also supported.                                                                                                                                                                              |
| G016 | Any path search (`ANY`)                     |  ✅   |                                                                                                                                                                                                       |
| G017 | All shortest (`ALL SHORTEST`)               |  ✅   |                                                                                                                                                                                                       |
| G018 | Any shortest (`ANY SHORTEST`)               |  ✅   |                                                                                                                                                                                                       |
| G019 | Counted shortest (`SHORTEST k`)             |  ✅   |                                                                                                                                                                                                       |
| G020 | Counted shortest group (`SHORTEST k GROUP`) |  ✅   |                                                                                                                                                                                                       |
| G035 | Quantified paths (`{n,m}`)                  |  ✅   |                                                                                                                                                                                                       |
| G036 | Quantified edges                            |  ✅   |                                                                                                                                                                                                       |
| G043 | Complete full edge patterns                 |  ✅   | `-[e:R WHERE …]->` per-hop predicate supported.                                                                                                                                                       |
| G060 | Bounded graph pattern quantifier            |  ✅   |                                                                                                                                                                                                       |
| G061 | Unbounded graph pattern quantifier          |  ✅   | `*` / `+`.                                                                                                                                                                                            |
| G074 | Label expressions: wildcard label           |  ✅   | `(:%)`.                                                                                                                                                                                               |
| G002 | Different-edges match mode                  |  ❌   | `MATCH DIFFERENT EDGES …` rejected.                                                                                                                                                                   |
| G003 | Explicit `REPEATABLE ELEMENTS`              |  ❌   | Rejected.                                                                                                                                                                                             |
| G050 | Parenthesized path pattern `WHERE`          |  ✅   | `((a)-[e]->(b) WHERE a.age < b.age)` — subpath WHERE spanning both endpoints; distinct from the clause WHERE, composes with it. Quantified subpath (`( … )+`) + subpath variable deferred (rejected). |
| G051 | Parenthesized path non-local predicate      |  ✅   | Non-local (multi-element) predicate inside the subpath — covered by the same parenthesized-subpath WHERE.                                                                                             |

### Built-in functions

| ID   | Feature                            | lenke | Notes                                                                                                                                                               |
| ---- | ---------------------------------- | :---: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GF01 | Enhanced numeric functions         |  ✅   | `abs/floor/ceil/sqrt/…`                                                                                                                                             |
| GF02 | Trigonometric functions            |  ✅   |                                                                                                                                                                     |
| GF03 | Logarithmic functions              |  ✅   |                                                                                                                                                                     |
| GF04 | Enhanced path functions            |  ✅   | `path_length`, `nodes`, `edges`, `elements`.                                                                                                                        |
| GF05 | Multi-character trim functions     |  ✅   | `ltrim`/`rtrim`/`btrim`, incl. the 2-arg character-set form (`btrim('xxhi','x')`→`hi`) — previously silently ignored the char arg.                                  |
| GF06 | Explicit `TRIM` function           |  ✅   |                                                                                                                                                                     |
| GF07 | Temporal duration functions        |  ✅   | `duration_between`.                                                                                                                                                 |
| GF10 | Advanced aggregates: general set   |  ✅   | `stddev_pop`/`stddev_samp`.                                                                                                                                         |
| GF11 | Advanced aggregates: binary set    |  ✅   | `percentile_cont`/`percentile_disc`.                                                                                                                                |
| GA05 | Cast specification (`CAST`)        |  ✅   |                                                                                                                                                                     |
| G100 | `ELEMENT_ID` function              |  ✅   | `element_id(x)`.                                                                                                                                                    |
| —    | `labels()` / `type()`              |  ❓   | **Not verified — see the note below.** Neither appears to be an ISO function at all; both are Cypher inheritances lenke ships.                                      |
| GA06 | Value type predicates (`IS TYPED`) |  ✅   | `x IS [NOT] TYPED <type> [NOT NULL]`, both engines. Null conforms to nullable types; INTEGER = whole number (f64 model). `::` alias + parameterized types deferred. |

> **`labels()` and `type()` are UNVERIFIED and we do not currently believe they
> are conformant.** No free source we have consulted shows either as an ISO GQL
> function. The standard's label facilities appear to be the `IS LABELED`
> predicate (G111) and pattern label expressions; its only element-level function
> appears to be `ELEMENT_ID` (G100). Two secondary derivations agree that
> `labels` has no production — the
> [opengql grammar](https://github.com/opengql/grammar) and Microsoft Fabric's
> Annex D conformance table — but **neither is the spec text**, and per the
> reliability lesson in [references.md](./references.md) a secondary source is
> not a primary reproduction. `element_type()` likewise appears nowhere.
>
> lenke ships both as **Cypher inheritances**, with `labels()` widened to accept
> an edge (returning its type set) to match the two vendors that ship it —
> Spanner's `LABELS(GRAPH_ELEMENT) -> ARRAY<STRING>` and Fabric's
> `labels(node_or_edge)`. That is a defensible vendor-consensus choice, not a
> conformance claim.
>
> **To verify** (deliberately not scheduled): resolve `labels`/`type`/
> `element_type` against a primary reproduction of 39075 — the Neo4j `.adoc`
> sources this checklist already uses for Feature IDs, or the spec text itself.
> If ISO does define them, re-check the argument domain: an ISO `labels()` that
> is node-only would make lenke's edge form an extension, which by the repo's own
> convention would need the leading-underscore sigil.

### Query composition & clauses

| ID   | Feature                                                    | lenke | Notes                                                                                                                                                                                                                                                                                                                                                                   |
| ---- | ---------------------------------------------------------- | :---: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GQ03 | Composite query: `UNION`                                   |  ✅   | `UNION`/`EXCEPT`/`INTERSECT`.                                                                                                                                                                                                                                                                                                                                           |
| GQ08 | `FILTER` statement                                         |  ✅   |                                                                                                                                                                                                                                                                                                                                                                         |
| GQ09 | `LET` statement                                            |  ✅   |                                                                                                                                                                                                                                                                                                                                                                         |
| GQ12 | `OFFSET` clause                                            |  ✅   | `OFFSET`/`SKIP`. Both ISO positions — see GQ13.                                                                                                                                                                                                                                                                                                                         |
| GQ13 | `ORDER BY` + page: `LIMIT`                                 |  ✅   | Both ISO positions ship: trailing a RETURN (`primitiveResultStatement`) **and** as a statement of its own before it (`primitiveQueryStatement`), e.g. `MATCH … ORDER BY x LIMIT 10 RETURN …`. In statement position the sort/slice applies to the binding table, so the RETURN only projects the survivors. `SKIP` is the Cypher synonym and does not open a statement. |
| GQ14 | Complex expressions in sort keys                           |  ✅   |                                                                                                                                                                                                                                                                                                                                                                         |
| GQ16 | Pre-projection aliases in sort keys                        |  ✅   |                                                                                                                                                                                                                                                                                                                                                                         |
| GQ20 | Advanced linear composition (`NEXT`)                       |  ✅   |                                                                                                                                                                                                                                                                                                                                                                         |
| GA07 | Ordering by discarded binding variables                    |  ✅   |                                                                                                                                                                                                                                                                                                                                                                         |
| GQ22 | `EXISTS` predicate: multiple `MATCH`                       |  ❌   | Single-`MATCH` `EXISTS { … }` works; multi-`MATCH` rejected.                                                                                                                                                                                                                                                                                                            |
| GQ01 | `USE` graph clause                                         |  ➖   | Single-graph embedded engine by design.                                                                                                                                                                                                                                                                                                                                 |
| GP01 | Inline procedure (`CALL { … }`)                            |  ✅   |                                                                                                                                                                                                                                                                                                                                                                         |
| GP03 | Inline procedure, nested variable scope (`CALL (x) { … }`) |  ✅   | Correlated lateral join.                                                                                                                                                                                                                                                                                                                                                |
| GP04 | Named procedure calls (`CALL name() YIELD`)                |  ✅   | The home for graph algorithms.                                                                                                                                                                                                                                                                                                                                          |

### Updates

| ID   | Feature                         | lenke | Notes                     |
| ---- | ------------------------------- | :---: | ------------------------- |
| GD01 | Updatable graphs                |  ✅   |                           |
| GD02 | Graph label set changes         |  ✅   | `SET n:L` / `REMOVE n:L`. |
| GD04 | `DELETE` with simple expression |  ✅   |                           |
| GE07 | Boolean `XOR`                   |  ✅   |                           |

### Value types

| ID        | Feature                                       | lenke | Notes                                                                                                                                                                                                                                                                                                                                                        |
| --------- | --------------------------------------------- | :---: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| GV39      | Temporal: date, local datetime, local time    |  ✅   |                                                                                                                                                                                                                                                                                                                                                              |
| GV40      | Temporal: zoned datetime, zoned time          |  ✅   | Numeric offset (no named zones).                                                                                                                                                                                                                                                                                                                             |
| GV41      | Duration types                                |  ✅   |                                                                                                                                                                                                                                                                                                                                                              |
| GV50      | List value types                              |  ✅   |                                                                                                                                                                                                                                                                                                                                                              |
| GV55      | Path value types                              |  ✅   |                                                                                                                                                                                                                                                                                                                                                              |
| GV70      | Null type (`null`)                            |  ✅   | null is a first-class stored value (deliberate divergence).                                                                                                                                                                                                                                                                                                  |
| GV45      | Record type (open `RECORD` / map)             |  🟡   | Record VALUE shipped both engines: `{k:v}` constructor, `.field`/`['k']` access, structural equality, total-order ORDER BY/DISTINCT, `SET n.x = {…}`. Native substrate stores maps + a dotted-path index (`n.meta.city` seek). Remaining: TS-core stored-map persistence + Gremlin. The record TYPE (`RECORD {a::INT}`) stays deferred with the type system. |
| GV12      | 64-bit signed integer (`INT64`)               |  ➖   | One numeric type (f64); bigint rejected `E_INVALID_VALUE` (see `numeric-model-f64`).                                                                                                                                                                                                                                                                         |
| GV23/GV24 | Float type-name synonyms (`DOUBLE`/`FLOAT64`) |  ✅   | `CAST(x AS DOUBLE)` accepted; the underlying numeric type is f64.                                                                                                                                                                                                                                                                                            |
| GV66      | Open dynamic unions                           |  ❓   | Type-system feature; not verified.                                                                                                                                                                                                                                                                                                                           |
| GV67      | Closed dynamic unions                         |  ❓   | Not verified.                                                                                                                                                                                                                                                                                                                                                |
| GV71      | Empty type (`NOTHING`)                        |  ❓   | Not verified.                                                                                                                                                                                                                                                                                                                                                |

### Literals, identifiers, comments

| ID   | Feature                        | lenke | Notes                                             |
| ---- | ------------------------------ | :---: | ------------------------------------------------- |
| GG01 | Graph with open graph type     |  ✅   | Schemaless — the baseline conformance graph type. |
| GL01 | Hexadecimal literals           |  ✅   | `0xFF`.                                           |
| GL02 | Octal literals                 |  ✅   | `0o17`.                                           |
| GB03 | Double-solidus comments (`//`) |  ✅   |                                                   |
| GB01 | Long identifiers               |  ✅   | (assumed; no length cap encountered).             |

---

## Statement & program surface (structural)

This layer is **grammar-derived** — it comes from walking the ISO GQL statement
taxonomy in the [TuGraph ANTLR grammar](https://github.com/TuGraph-family/gql-grammar)
(`gqlProgram → programActivity → session/transaction activity → procedureBody`),
not from Neo4j's Cypher-centric feature pages (which barely cover it). Status was
probed against the engine + confirmed in the parser.

| Area                                                                         | GQL surface                                                                                   | lenke | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | :---: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Transactions**                                                             | `START TRANSACTION [READ ONLY \| READ WRITE]`, `COMMIT [WORK]`, `ROLLBACK [WORK]`             |  ✅   | Shipped both engines as **session commands** — the graph _is_ the ISO session, so tx state persists across `query()` calls (per-statement auto-frames nest inside). `READ ONLY` rejects writes. Also a host API (`graph.transaction(fn)` / `graph.tx()`). The single-program combined form (`START TRANSACTION <stmts> … COMMIT` in one query, grammar's `transactionActivity`) is **not** parsed — issue the commands separately. Deferred: MVCC / savepoints / true nesting. |
| **Session management**                                                       | `SESSION SET SCHEMA/GRAPH/TIME ZONE/PARAMETER`, `SESSION RESET`, `SESSION CLOSE`              |  ➖   | Embedded engine — no multi-session catalog concept. The graph is an _implicit_ session (see transactions); the explicit `SESSION …` commands aren't parsed.                                                                                                                                                                                                                                                                                                                    |
| **Catalog DDL**                                                              | `CREATE`/`DROP GRAPH`, `CREATE`/`DROP SCHEMA`                                                 |  ➖   | Single-graph, embedded — graph lifecycle is the **host API** (`new Graph()` / `createEmptyGraph`), not in-query DDL.                                                                                                                                                                                                                                                                                                                                                           |
| **Graph-type / schema DDL**                                                  | `CREATE GRAPH TYPE …`, node/edge type specifications, typed/closed graphs (`OF <graph type>`) |  ➖   | Schemaless by default (open graph type = GG01 ✅). Typed schemas are **host-side** (`defineNode` + Standard Schema) and in-engine constraints (`createUniqueConstraint`/`createValidator`/`createInvariant`) — deliberately _not_ GQL DDL (the write/schema layer belongs to the core + host; see `docs/design/gql-extensions.md`).                                                                                                                                            |
| **`USE` / focused statements**                                               | `USE <graph> …`                                                                               |  ➖   | Single-graph → no graph selection.                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| **Procedure-body binding-variable defs**                                     | `PROPERTY GRAPH g = …`, `BINDING TABLE t = …`, `VALUE x = …` (before the statement block)     |  ❌   | Not supported. The `LET` statement covers value binding within a linear query; there is no first-class **binding-table** value.                                                                                                                                                                                                                                                                                                                                                |
| **`BYTES`/`BINARY`/`VARBINARY` types, sized `VARCHAR(n)`, integer subtypes** | The wider `predefinedType` set                                                                |  ➖   | lenke's value model is f64 / string / bool / temporal / list / path / null (see the Value-types table + `numeric-model-f64`); byte-string types, length-parameterised strings, and integer subtypes are out of model.                                                                                                                                                                                                                                                          |

---

## Expression, predicate & clause surface (grammar-derived)

Fine-grained coverage the Feature-ID tables don't capture, walked from the
grammar's `expression`/`expressionPredicate`/`functionCall` and clause
productions and probed against the engine.

### Predicates

| Predicate                                  | lenke | Notes                                                                                                                                              |
| ------------------------------------------ | :---: | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Comparison `= <> < > <= >=`                |  ✅   |                                                                                                                                                    |
| `IS [NOT] NULL`                            |  ✅   | Also the idiom for property existence.                                                                                                             |
| `IN` (list membership)                     |  ✅   |                                                                                                                                                    |
| Labeled `n:L` / `IS [NOT] LABELED`         |  ✅   |                                                                                                                                                    |
| Boolean test `IS [NOT] TRUE/FALSE/UNKNOWN` |  ✅   |                                                                                                                                                    |
| `IS [NOT] DIRECTED`                        |  ✅   | Every lenke edge is directed → true; null/non-edge → null.                                                                                         |
| `IS [NOT] SOURCE/DESTINATION OF`           |  ✅   | `node IS SOURCE/DESTINATION OF edge` — reads the edge's stored endpoints.                                                                          |
| `IS [NOT] NORMALIZED`                      |  ❌   |                                                                                                                                                    |
| `ALL_DIFFERENT(…)` / `SAME(…)`             |  ✅   | Element-identity predicates (≥2 operands); three-valued on null.                                                                                   |
| `PROPERTY_EXISTS(n, k)`                    |  ✅   | Presence test — distinguishes an absent key from a stored null (`n.k IS NOT NULL` cannot, since null is first-class). Both engines byte-identical. |
| `BETWEEN`                                  |  ➖   | Not an ISO GQL predicate (absent from the grammar); correctly rejected.                                                                            |

### Expressions & operators

| Construct                                        | lenke | Notes                                                                                                                     |
| ------------------------------------------------ | :---: | ------------------------------------------------------------------------------------------------------------------------- |
| `NOT` / `AND` / `OR` / `XOR`                     |  ✅   | (`XOR` = GE07.)                                                                                                           |
| `!` unary-not operator                           |  ✅   | Tight-binding (harder than the loose `NOT` keyword).                                                                      |
| Unary `+` / `-`                                  |  ✅   |                                                                                                                           |
| Concatenation `\|\|`                             |  ✅   | Strings, lists, paths.                                                                                                    |
| Arithmetic `+ - * /`, `%`/`mod`, `^`/`power`     |  ✅   | (`^` ≤1 ULP vs `power` on some inputs — see gql README.)                                                                  |
| `CASE` (simple + searched), `NULLIF`, `COALESCE` |  ✅   |                                                                                                                           |
| Property `.`, list index `[i]`                   |  ✅   |                                                                                                                           |
| List slice `[i..j]`                              |  ❌   | No slicing.                                                                                                               |
| `LET … IN … END` (inline let-expression)         |  ✅   | Scoped locals (later binding sees earlier); binding RHS ends at the structural `IN` (parenthesize a bare `IN` predicate). |
| `VALUE { subquery }` (scalar subquery)           |  ✅   | Correlated; 0 rows→NULL, 1→value, >1 non-agg→cardinality error, aggregate RETURN folds the group.                         |

### Scalar / string functions (detail)

| Function                                           | lenke | Notes                                                              |
| -------------------------------------------------- | :---: | ------------------------------------------------------------------ |
| `char_length`/`character_length`                   |  ✅   |                                                                    |
| `octet_length`/`byte_length`                       |  ✅   |                                                                    |
| `ceil`/`ceiling`                                   |  ✅   | Synonyms.                                                          |
| `upper`/`lower` (fold), `left`/`right` (substring) |  ✅   |                                                                    |
| `trim(x)` / `btrim`/`ltrim`/`rtrim`                |  ✅   |                                                                    |
| `TRIM(LEADING\|TRAILING\|BOTH … FROM x)`           |  ✅   | SQL trim-specification form; desugars to trim/ltrim/rtrim.         |
| `normalize(x)`                                     |  ❌   | **Mandatory-feature gap** (part of `<character string function>`). |

### Clauses & projection

| Clause                                  | lenke | Notes                                                                                                                                                                            |
| --------------------------------------- | :---: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OPTIONAL MATCH`                        |  ✅   |                                                                                                                                                                                  |
| `RETURN DISTINCT`, `count(DISTINCT …)`  |  ✅   |                                                                                                                                                                                  |
| `ORDER BY … ASC/DESC NULLS FIRST/LAST`  |  ✅   |                                                                                                                                                                                  |
| `DETACH DELETE`                         |  ✅   |                                                                                                                                                                                  |
| Implicit grouping (non-aggregated keys) |  ✅   | Cypher-style — the default grouping model.                                                                                                                                       |
| Explicit `GROUP BY` clause              |  ✅   | ISO (on the RETURN statement). Drives grouping — incl. `GROUP BY` a non-returned key, and no-aggregate `GROUP BY` (= DISTINCT).                                                  |
| `HAVING` clause                         |  ✅   | On the ISO `SELECT` statement (not `RETURN`, per spec). Post-aggregation group filter; references group keys + aggregates (incl. aggregates not in the SELECT list).             |
| `SELECT` statement                      |  ✅   | `SELECT [DISTINCT] {* \| items} FROM [<graph>] MATCH … [WHERE] [GROUP BY] [HAVING] [ORDER BY] [OFFSET] [LIMIT]` — SQL-shaped, desugars to MATCH + RETURN. Single implicit graph. |

---

## lenke extensions (non-ISO — sigil'd 🔷)

Deliberately outside the standard; each wears the leading-underscore sigil
(`docs/design/gql-extensions.md`) so it never collides with a future GQL edition
and is self-documenting as non-portable. The `iso-strict` dialect rejects all of
these.

| Construct                                                   | What                          | Why an extension                                                                                                      |
| ----------------------------------------------------------- | ----------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `_MERGE` (+ `_ON_CREATE`/`_ON_UPDATE`/`_ON_UPDATE_NOTHING`) | Keyed upsert                  | GQL has no `MERGE`/upsert.                                                                                            |
| `_year`/`_month`/`_day`/`_hour`/`_minute`/`_second`         | Temporal component extraction | Not in the GQL function catalogue (verified against the taxonomy — not mandatory, not a catalogued optional feature). |
| `createUniqueConstraint` (host API)                         | Constraint DDL                | GQL barely specifies schema/constraint DDL.                                                                           |

---

## Summary

**Query surface** (the parts GQL specifies well): lenke implements the great
majority — read / pattern-matching / path-search / function / composition,
predicates, `CASE`/`COALESCE`/`NULLIF`, `OPTIONAL MATCH`, `DISTINCT`, `ORDER BY …
NULLS FIRST/LAST`, `DETACH DELETE`, the full scalar/aggregate function set. The
genuine gaps found (deep grammar walk): `IS NORMALIZED` (declined — see
Tier 1), multi-`MATCH` `EXISTS` (GQ22),
map/record
values (GV45), the `DIFFERENT
EDGES`/`REPEATABLE ELEMENTS` match modes (G002/G003), and — by design —
multi-graph `USE` (GQ01) and
`INT64` (GV12). The remaining **mandatory** gap is the deliberately-declined
Unicode normalization (`normalize()` + `IS NORMALIZED`, zero-dep tradeoff); with
`IS TYPED` shipped, there are **no open mandatory gaps**. See
[Gaps](#gaps-prioritized).

**Statement/program surface** (grammar-derived): **transactions are supported**
(`START TRANSACTION`/`COMMIT`/`ROLLBACK` as session commands, both engines).
Session management, catalog DDL (`CREATE GRAPH`), and schema/graph-type DDL
(`CREATE GRAPH TYPE`) are **deliberately excluded** — lenke is an embedded,
single-graph, schemaless-by-default engine, so graph lifecycle and typed schemas
live in the host API, not in GQL DDL. This is a design stance (see
`docs/design/gql-extensions.md`), not a gap to close.

**So: is this the whole of 39075?** The _query surface_ is thoroughly mapped and
the _statement/program surface_ is now covered structurally. What remains only
partially seen is the **full optional Feature-ID catalogue**: this checklist
covers the ~75 IDs Neo4j documents, but the ID gaps (e.g. `G006–G009`,
`G021–G034`, `G037–G042`) are real spec features no free source enumerates — a
complete ID-by-ID scorecard needs the paywalled Annex.

This is a living document — re-run the probes (and a proper per-production
mandatory audit) when the GQL surface changes.
