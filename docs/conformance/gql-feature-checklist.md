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

Last verified: **2026-07-25** (`@lenke/gql` portable engine; native is
byte-identical).

**Legend:** ✅ supported · 🟡 partial · ❌ not yet · 🔷 lenke extension (non-ISO,
sigil'd) · ➖ excluded by design · ❓ not yet verified

> **Caveats.** (1) The taxonomy is a faithful _secondary_ source (Neo4j, a GQL
> co-author), not the paywalled spec text. (2) Feature IDs `GF08`/`GF09` could not
> be resolved from any free source. (3) The **mandatory** layer below is a
> summary + spot-checks, not a full per-production audit — treat baseline
> conformance as "substantially yes, with the noted gaps," pending a deeper pass.

---

## Baseline (mandatory) features

lenke supports the core mandatory read/write surface: `MATCH`, `INSERT`, `SET`,
`REMOVE`, `DELETE`, `RETURN`, `ORDER BY` + `LIMIT`, `CASE`, aggregates
(`count/sum/avg/min/max`), comparison / `null` / `EXISTS` predicates, property
reference, numeric value functions (`char_length`), and character string
functions (`left/lower/right/trim/upper`).

**Known mandatory gaps (honest):**

| Gap                              | Detail                                                                                                                                                                              |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🟡 `normalize()` string function | The mandatory `<character string function>` set includes `normalize()`; lenke implements the rest (`left/lower/right/trim/upper`) but **not** `normalize()` → `E_UNKNOWN_FUNCTION`. |
| ❓ full per-production audit     | The mandatory table maps GQL grammar productions to features; a rigorous production-by-production conformance pass has not been run.                                                |

---

## Optional features

### Paths, patterns & path search

| ID   | Feature                                     | lenke | Notes                                                                              |
| ---- | ------------------------------------------- | :---: | ---------------------------------------------------------------------------------- |
| G004 | Path variables (`p = …`)                    |  ✅   | Bound `Path` value, both engines byte-identical.                                   |
| G005 | Path search prefix                          |  ✅   |                                                                                    |
| G010 | Explicit `WALK`                             |  ✅   |                                                                                    |
| G011 | Path mode `TRAIL`                           |  ✅   | lenke default.                                                                     |
| G013 | Path mode `ACYCLIC`                         |  ✅   | `SIMPLE` also supported.                                                           |
| G016 | Any path search (`ANY`)                     |  ✅   |                                                                                    |
| G017 | All shortest (`ALL SHORTEST`)               |  ✅   |                                                                                    |
| G018 | Any shortest (`ANY SHORTEST`)               |  ✅   |                                                                                    |
| G019 | Counted shortest (`SHORTEST k`)             |  ✅   |                                                                                    |
| G020 | Counted shortest group (`SHORTEST k GROUP`) |  ✅   |                                                                                    |
| G035 | Quantified paths (`{n,m}`)                  |  ✅   |                                                                                    |
| G036 | Quantified edges                            |  ✅   |                                                                                    |
| G043 | Complete full edge patterns                 |  ✅   | `-[e:R WHERE …]->` per-hop predicate supported.                                    |
| G060 | Bounded graph pattern quantifier            |  ✅   |                                                                                    |
| G061 | Unbounded graph pattern quantifier          |  ✅   | `*` / `+`.                                                                         |
| G074 | Label expressions: wildcard label           |  ✅   | `(:%)`.                                                                            |
| G002 | Different-edges match mode                  |  ❌   | `MATCH DIFFERENT EDGES …` rejected.                                                |
| G003 | Explicit `REPEATABLE ELEMENTS`              |  ❌   | Rejected.                                                                          |
| G050 | Parenthesized path pattern `WHERE`          |  ❌   | `((a)-->(b) WHERE …)` rejected (per-_edge_ `WHERE` works; per-_subpath_ does not). |
| G051 | Parenthesized path non-local predicate      |  ❓   | Not separately verified; expected ❌ alongside G050.                               |

### Built-in functions

| ID   | Feature                                   | lenke | Notes                                        |
| ---- | ----------------------------------------- | :---: | -------------------------------------------- |
| GF01 | Enhanced numeric functions                |  ✅   | `abs/floor/ceil/sqrt/…`                      |
| GF02 | Trigonometric functions                   |  ✅   |                                              |
| GF03 | Logarithmic functions                     |  ✅   |                                              |
| GF04 | Enhanced path functions                   |  ✅   | `path_length`, `nodes`, `edges`, `elements`. |
| GF05 | Multi-character trim functions            |  ✅   | `ltrim`/`rtrim`/`btrim`.                     |
| GF06 | Explicit `TRIM` function                  |  ✅   |                                              |
| GF07 | Temporal duration functions               |  ✅   | `duration_between`.                          |
| GF10 | Advanced aggregates: general set          |  ✅   | `stddev_pop`/`stddev_samp`.                  |
| GF11 | Advanced aggregates: binary set           |  ✅   | `percentile_cont`/`percentile_disc`.         |
| GA05 | Cast specification (`CAST`)               |  ✅   |                                              |
| G100 | `ELEMENT_ID` function                     |  ✅   | `element_id(x)`.                             |
| GA06 | Value type predicates (`IS TYPED` / `::`) |  ❌   | Rejected.                                    |

### Query composition & clauses

| ID   | Feature                                                    | lenke | Notes                                                        |
| ---- | ---------------------------------------------------------- | :---: | ------------------------------------------------------------ |
| GQ03 | Composite query: `UNION`                                   |  ✅   | `UNION`/`EXCEPT`/`INTERSECT`.                                |
| GQ08 | `FILTER` statement                                         |  ✅   |                                                              |
| GQ09 | `LET` statement                                            |  ✅   |                                                              |
| GQ12 | `OFFSET` clause                                            |  ✅   | `OFFSET`/`SKIP`.                                             |
| GQ13 | `ORDER BY` + page: `LIMIT`                                 |  ✅   |                                                              |
| GQ14 | Complex expressions in sort keys                           |  ✅   |                                                              |
| GQ16 | Pre-projection aliases in sort keys                        |  ✅   |                                                              |
| GQ20 | Advanced linear composition (`NEXT`)                       |  ✅   |                                                              |
| GA07 | Ordering by discarded binding variables                    |  ✅   |                                                              |
| GQ22 | `EXISTS` predicate: multiple `MATCH`                       |  ❌   | Single-`MATCH` `EXISTS { … }` works; multi-`MATCH` rejected. |
| GQ01 | `USE` graph clause                                         |  ➖   | Single-graph embedded engine by design.                      |
| GP01 | Inline procedure (`CALL { … }`)                            |  ✅   |                                                              |
| GP03 | Inline procedure, nested variable scope (`CALL (x) { … }`) |  ✅   | Correlated lateral join.                                     |
| GP04 | Named procedure calls (`CALL name() YIELD`)                |  ✅   | The home for graph algorithms.                               |

### Updates

| ID   | Feature                         | lenke | Notes                     |
| ---- | ------------------------------- | :---: | ------------------------- |
| GD01 | Updatable graphs                |  ✅   |                           |
| GD02 | Graph label set changes         |  ✅   | `SET n:L` / `REMOVE n:L`. |
| GD04 | `DELETE` with simple expression |  ✅   |                           |
| GE07 | Boolean `XOR`                   |  ✅   |                           |

### Value types

| ID        | Feature                                       | lenke | Notes                                                                                |
| --------- | --------------------------------------------- | :---: | ------------------------------------------------------------------------------------ |
| GV39      | Temporal: date, local datetime, local time    |  ✅   |                                                                                      |
| GV40      | Temporal: zoned datetime, zoned time          |  ✅   | Numeric offset (no named zones).                                                     |
| GV41      | Duration types                                |  ✅   |                                                                                      |
| GV50      | List value types                              |  ✅   |                                                                                      |
| GV55      | Path value types                              |  ✅   |                                                                                      |
| GV70      | Null type (`null`)                            |  ✅   | null is a first-class stored value (deliberate divergence).                          |
| GV45      | Record type (open `RECORD` / map)             |  ❌   | Map/record literal `{a:1}` rejected; no first-class map value.                       |
| GV12      | 64-bit signed integer (`INT64`)               |  ➖   | One numeric type (f64); bigint rejected `E_INVALID_VALUE` (see `numeric-model-f64`). |
| GV23/GV24 | Float type-name synonyms (`DOUBLE`/`FLOAT64`) |  ✅   | `CAST(x AS DOUBLE)` accepted; the underlying numeric type is f64.                    |
| GV66      | Open dynamic unions                           |  ❓   | Type-system feature; not verified.                                                   |
| GV67      | Closed dynamic unions                         |  ❓   | Not verified.                                                                        |
| GV71      | Empty type (`NOTHING`)                        |  ❓   | Not verified.                                                                        |

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
| **Graph-type / schema DDL**                                                  | `CREATE GRAPH TYPE …`, node/edge type specifications, typed/closed graphs (`OF <graph type>`) |  ➖   | Schemaless by default (open graph type = GG01 ✅). Typed schemas are **host-side** (`defineNode` + Standard Schema, R-TYPED) and in-engine constraints (`createUniqueConstraint`/`createValidator`/`createInvariant`, R-CONSTRAINTS) — deliberately _not_ GQL DDL (the write/schema layer belongs to the core + host; see `docs/design/gql-extensions.md`).                                                                                                                    |
| **`USE` / focused statements**                                               | `USE <graph> …`                                                                               |  ➖   | Single-graph → no graph selection.                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| **Procedure-body binding-variable defs**                                     | `PROPERTY GRAPH g = …`, `BINDING TABLE t = …`, `VALUE x = …` (before the statement block)     |  ❌   | Not supported. The `LET` statement covers value binding within a linear query; there is no first-class **binding-table** value.                                                                                                                                                                                                                                                                                                                                                |
| **`BYTES`/`BINARY`/`VARBINARY` types, sized `VARCHAR(n)`, integer subtypes** | The wider `predefinedType` set                                                                |  ➖   | lenke's value model is f64 / string / bool / temporal / list / path / null (see the Value-types table + `numeric-model-f64`); byte-string types, length-parameterised strings, and integer subtypes are out of model.                                                                                                                                                                                                                                                          |

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
majority — read / pattern-matching / path-search / function / composition — and
is missing or excludes a small set: `IS TYPED` predicates (GA06), multi-`MATCH`
`EXISTS` (GQ22), map/record values (GV45), parenthesized-subpath `WHERE` (G050),
the `DIFFERENT EDGES`/`REPEATABLE ELEMENTS` match modes (G002/G003), and — by
design — multi-graph `USE` (GQ01) and `INT64` (GV12). The one concrete
**mandatory** gap found is `normalize()`.

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
