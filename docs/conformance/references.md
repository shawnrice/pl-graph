# Graph-query conformance — reference sources

A curated, annotated bibliography for reasoning about **ISO GQL** (and, where
relevant, **openCypher / TinkerPop Gremlin**) conformance. Assembled while
deciding how lenke should implement features conformantly. The companion
[gql-feature-checklist.md](./gql-feature-checklist.md) applies these sources to
lenke's actual surface.

> **Reliability lesson (read first).** The ISO standard itself is paywalled, so
> most work here leans on _reproductions_. AI/search summaries of these pages
> **confabulate specifics** — during this research a summary asserted "GF04 =
> datetime functions"; the primary source shows GF04 is _Enhanced path
> functions_ (`path_length`). **Always resolve a Feature ID against a primary
> reproduction (the Neo4j `.adoc` source), never a summary.**

---

## 1. The standard itself

- **ISO/IEC 39075:2024 — Information technology — Database languages — GQL** _(text
  paywalled)_. <https://www.iso.org/standard/76120.html> · free browse (front
  matter/ToC only): <https://www.iso.org/obp/ui/en/#!iso:std:76120:en>
  The authoritative text. Conformance is defined in **subclause 24.2**:
  a system conforms by supporting the data model + the **mandatory** features;
  **optional** features each carry a Feature ID (letter(s)+digits, e.g. `G035`,
  `GF07`, `GV39`). Mandatory features have **no** ID and are cited by subclause.
  The prose + the **Feature-ID Annex** are behind the paywall.

- **ISO GQL grammar (BNF) — FREE, authoritative.** ISO publishes the machine-readable
  grammar as a "digital artifact," no paywall:
  <https://standards.iso.org/iso-iec/39075/ed-1/en/ISO_IEC_39075(en).bnf.txt>
  (~78 KB, `<GQL-program>` downward). This is the **primary** source for the full
  _syntax_ surface — better than any vendor reproduction for "what productions
  exist." It does **not** contain the Feature-ID taxonomy (that's in the paywalled
  Annex), so pair it with §2 for IDs. The [TuGraph ANTLR grammar](#3-independent-vendor-implementations-cross-check-divergence)
  is a convenient navigable rendering of the same grammar.

## 2. Best FREE reproductions of the feature taxonomy

- **Neo4j Cypher Manual — GQL conformance appendix.** _The single most useful
  free source for Feature IDs._ Neo4j co-authored GQL and enumerates real 39075
  Feature IDs mapped to concrete capabilities. Open-source (`neo4j/docs-cypher`),
  so the raw AsciiDoc is fetchable when the rendered pages 403 a scraper:
  - Supported mandatory: <https://neo4j.com/docs/cypher-manual/current/appendix/gql-conformance/supported-mandatory/>
  - Currently unsupported mandatory: <https://neo4j.com/docs/cypher-manual/current/appendix/gql-conformance/unsupported-mandatory/>
  - **Supported optional** (the Feature-ID table): <https://neo4j.com/docs/cypher-manual/current/appendix/gql-conformance/supported-optional/>
  - Optional features w/ analogous Cypher: <https://neo4j.com/docs/cypher-manual/current/appendix/gql-conformance/analogous-cypher/>
  - Additional Cypher features (Cypher, **not** in GQL): <https://neo4j.com/docs/cypher-manual/current/appendix/gql-conformance/additional-cypher/>
  - Raw source: <https://github.com/neo4j/docs-cypher> → `modules/ROOT/pages/appendix/gql-conformance/*.adoc`
    (fetch via `gh api -H "Accept: application/vnd.github.raw" repos/neo4j/docs-cypher/contents/<path>?ref=dev`).

- **Ultipa GQL documentation.** _Best free reference for function / operator /
  `CAST` semantics_ (per-function behavior, examples). Note: Ultipa lists its own
  extensions as GQL synonyms (e.g. it offered `relationships` as a synonym for
  `edges`), so treat its _naming_ claims as vendor-flavoured; the _semantics_ are
  reliable. <https://www.ultipa.com/docs/gql/> — datetime fns:
  <https://www.ultipa.com/docs/gql/datetime-functions> · conformance model:
  <https://www.ultipa.com/docs/gql/gql-conformance>

## 3. Independent vendor implementations (cross-check divergence)

Comparing implementations reveals what is _mandated_ vs _implementation-defined_.
Where they disagree, the feature is not carrying a single conformant form.

- **Google Spanner Graph — GQL.** Function reference + ISO-standards statement.
  Uses SQL-style `EXTRACT`, `EDGES()`/`NODES()`/`PATH_LENGTH()`.
  <https://docs.cloud.google.com/spanner/docs/reference/standard-sql/graph-gql-functions>
  · <https://docs.cloud.google.com/spanner/docs/graph/iso-standards>
- **Microsoft Fabric — GQL (graph).** Expressions/functions + language guide.
  Minimal temporal surface (`zoned_datetime()` only; extracts year via integer
  math), uses `edges()`/`nodes()`/`path_length()`.
  <https://learn.microsoft.com/en-us/fabric/graph/gql-expressions> ·
  <https://learn.microsoft.com/en-us/fabric/graph/gql-language-guide>
- **TuGraph — `gql-grammar`.** An ANTLR4 grammar for ISO/IEC 39075 — useful for
  checking _syntax_ shapes. <https://github.com/TuGraph-family/gql-grammar>
- **Neo4j Cypher** (openCypher) — the largest deployed near-GQL dialect; its
  divergences (`.year` accessor, `relationships()`, `date.truncate()`) mark what
  is _Cypher-only, not GQL_.

## 4. Academic / semantics

- **"A Researcher's Digest of GQL"** — Francis, Gheerbrant, Guagliardo, Libkin,
  Marsault, Martens, Murlak, Peterfreund, Rogova, Vrgoč. ICDT 2023. The
  authoritative free treatment of GQL/SQL-PGQ pattern-matching semantics.
  <https://drops.dagstuhl.de/storage/00lipics/lipics-vol255-icdt2023/LIPIcs.ICDT.2023.1/LIPIcs.ICDT.2023.1.pdf>
- **"GQL and SQL/PGQ: Theoretical Models and Expressive Power."**
  <https://arxiv.org/html/2409.01102>
- **GQL standards working group** portal: <https://www.gqlstandards.org/>

## 5. Adjacent standards (for the non-GQL engines)

- **openCypher** — <https://opencypher.org/> (Cypher's open spec; lenke's GQL
  engine deliberately builds GQL, not Cypher-isms).
- **Apache TinkerPop / Gremlin** — <https://tinkerpop.apache.org/docs/current/reference/>
  (the reference for lenke's Gremlin engine; date-part extraction, for instance,
  is not a TinkerPop concept).
- **PGQL 2.x** (Oracle; a SQL/PGQ ancestor) — <https://pgql-lang.org/spec/2.1/>

---

## How lenke uses these

- **Feature-ID status** → Neo4j `supported-optional.adoc` (§2) is the spine of the
  [checklist](./gql-feature-checklist.md).
- **Function semantics / "is X conformant?"** → Ultipa (§2) for behavior, then
  cross-checked against Spanner + Fabric (§3). If the three diverge, the feature
  is implementation-defined and — if we add it — wears the sigil (see
  `docs/design/gql-extensions.md`).
- **Pattern-matching edge cases** → the Researcher's Digest (§4).
- Related engine-internal notes: `docs/design/gql-extensions.md` (the sigil
  convention), the memory `iso-gql-reference`.
