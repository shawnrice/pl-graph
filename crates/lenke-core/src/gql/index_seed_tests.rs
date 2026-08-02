//! Property-index SEEDING tests: equality / range predicates served from a sorted
//! secondary index, WHERE-derived seed hints, and smaller-side seed selection.
//!
//! Behavioral-parity port of `packages/gql/src/index-seed.test.ts`. Result parity
//! is the contract — the TS-internal plan structures are deliberately not
//! asserted, only that the same rows come back.
//! have no Rust equivalent and are treated as unsupported.

use super::eval::Params;
use super::parse;
use crate::graph::{Graph, Value};

// ---------------------------------------------------------------------------
// Fixture — TinkerPop "Modern" graph.  Identical to tests.rs `modern()`.
// ---------------------------------------------------------------------------

/// The shared TinkerPop "Modern" fixture (see `crate::fixtures`).
fn modern() -> Graph {
    crate::fixtures::modern_gql()
}

// ---------------------------------------------------------------------------
// Helpers — copied from tests.rs
// ---------------------------------------------------------------------------

fn n(x: f64) -> Value {
    Value::Num(x)
}
fn s(x: &str) -> Value {
    Value::Str(x.into())
}

/// Run a query and return (columns, rows).
fn q(g: &mut Graph, query: &str) -> (Vec<String>, Vec<Vec<Value>>) {
    let parsed = parse(query).unwrap_or_else(|e| panic!("parse error for `{query}`: {e}"));
    let rs = parsed
        .execute(g, &Params::new())
        .unwrap_or_else(|e| panic!("exec error for `{query}`: {e}"));
    (rs.cols.clone(), rs.rows().map(|r| r.to_vec()).collect())
}

fn rows(g: &mut Graph, query: &str) -> Vec<Vec<Value>> {
    q(g, query).1
}

/// A simple total ordering on `Value` for test sorting (mirrors JS `.sort()`
/// on primitive scalars).  Null < Bool < Num < Str; Lists not needed here.
fn cmp_val(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;
    match (a, b) {
        (Value::Null, Value::Null) => Equal,
        (Value::Null, _) => Less,
        (_, Value::Null) => Greater,
        (Value::Bool(x), Value::Bool(y)) => x.cmp(y),
        (Value::Bool(_), _) => Less,
        (_, Value::Bool(_)) => Greater,
        (Value::Num(x), Value::Num(y)) => x.partial_cmp(y).unwrap_or(Equal),
        (Value::Num(_), _) => Less,
        (_, Value::Num(_)) => Greater,
        (Value::Str(x), Value::Str(y)) => x.as_ref().cmp(y.as_ref()),
        _ => Equal,
    }
}

/// Sort a column from the result rows (mirrors the TS `sorted` helper).
fn sorted_col(mut rows: Vec<Vec<Value>>, col_idx: usize) -> Vec<Value> {
    let mut vals: Vec<Value> = rows.iter_mut().map(|r| r.swap_remove(col_idx)).collect();
    vals.sort_by(cmp_val);
    vals
}

/// Run a query, grab one column by name, and sort it.
fn sorted(g: &mut Graph, query: &str, col: &str) -> Vec<Value> {
    let (cols, r) = q(g, query);
    let idx = cols.iter().position(|c| c == col).unwrap_or_else(|| {
        panic!("column `{col}` not found in {cols:?}");
    });
    sorted_col(r, idx)
}

// ===========================================================================
// describe('GQL property-index seeding', ...)
// ===========================================================================

/// TS: 'an equality property constraint returns the same rows whether or not
/// name is indexed'
#[test]
fn idx_equality_constraint_same_rows_with_or_without_index() {
    let q_str = "MATCH (p:Person {name: 'marko'})-[:KNOWS]->(b) RETURN b.name";

    let plain_rows = {
        let mut g = modern();
        sorted(&mut g, q_str, "b.name")
    };
    let indexed_rows = {
        let mut g = modern();
        g.create_vertex_index("name");
        sorted(&mut g, q_str, "b.name")
    };

    assert_eq!(indexed_rows, plain_rows);
    assert_eq!(indexed_rows, vec![s("josh"), s("vadas")]);
}

/// TS: 'the label constraint still excludes a same-named wrong-label vertex'
#[test]
fn idx_label_constraint_excludes_wrong_label() {
    let mut g = modern();
    g.create_vertex_index("name");

    // lop is Software; seeding from the name bucket must still honor :Person.
    let r1 = rows(&mut g, "MATCH (p:Person {name: 'lop'}) RETURN p.name");
    assert!(r1.is_empty());

    let r2 = rows(&mut g, "MATCH (s:Software {name: 'lop'}) RETURN s.name");
    assert_eq!(r2, vec![vec![s("lop")]]);
}

/// TS: 'a non-indexed key still works via the scan fallback'
#[test]
fn idx_non_indexed_key_scan_fallback() {
    let mut g = modern();
    g.create_vertex_index("name"); // age is NOT indexed
    let result = sorted(&mut g, "MATCH (p:Person {age: 32}) RETURN p.name", "p.name");
    assert_eq!(result, vec![s("josh")]);
}

/// TS: 'seeding reflects live mutations'
#[test]
fn idx_seeding_reflects_live_mutations() {
    let mut g = modern();
    g.create_vertex_index("name");
    // Add a second vertex with name='marko' (age=50); the index must pick it up.
    g.add_vertex(
        &["Person".to_string()],
        vec![
            ("name".to_string(), s("marko")),
            ("age".to_string(), n(50.0)),
        ],
    );
    let result = sorted(
        &mut g,
        "MATCH (p:Person {name: 'marko'}) RETURN p.age",
        "p.age",
    );
    assert_eq!(result, vec![n(29.0), n(50.0)]);
}

/// TS: 'an empty bucket yields no rows'
#[test]
fn idx_empty_bucket_yields_no_rows() {
    let mut g = modern();
    g.create_vertex_index("name");
    let r = rows(&mut g, "MATCH (p:Person {name: 'nobody'}) RETURN p.name");
    assert!(r.is_empty());
}

// ===========================================================================
// describe('GQL WHERE-derived seed hints', ...)
// Ages: marko=29, vadas=27, josh=32, peter=35
// ===========================================================================

/// Helper: run query on both plain and indexed graph, return both result sets.
fn both(query_str: &str) -> (Vec<Vec<Value>>, Vec<Vec<Value>>) {
    let plain = {
        let mut g = modern();
        rows(&mut g, query_str)
    };
    let indexed = {
        let mut g = modern();
        g.create_vertex_index("name");
        g.create_vertex_index("age");
        rows(&mut g, query_str)
    };
    (plain, indexed)
}

/// Helper: sorted column from rows by index 0.
fn sort_rows(mut r: Vec<Vec<Value>>) -> Vec<Value> {
    let mut vals: Vec<Value> = r.iter_mut().map(|row| row[0].clone()).collect();
    vals.sort_by(cmp_val);
    vals
}

/// TS: 'WHERE equality seeds and matches the scan'
#[test]
fn idx_where_equality_seeds_and_matches_scan() {
    let (plain, indexed) = both("MATCH (p:Person) WHERE p.name = 'marko' RETURN p.age");
    assert_eq!(indexed, plain);
    assert_eq!(indexed, vec![vec![n(29.0)]]);
}

/// TS: 'WHERE range seeds and matches the scan'
#[test]
fn idx_where_range_seeds_and_matches_scan() {
    let (plain, indexed) = both("MATCH (p:Person) WHERE p.age > 30 RETURN p.name");
    assert_eq!(sort_rows(indexed.clone()), sort_rows(plain));
    assert_eq!(sort_rows(indexed), vec![s("josh"), s("peter")]);
}

/// TS: 'a two-sided WHERE range works (each bound is a sound conjunct)'
#[test]
fn idx_two_sided_where_range_works() {
    let (plain, indexed) = both("MATCH (p:Person) WHERE p.age >= 29 AND p.age < 35 RETURN p.name");
    assert_eq!(sort_rows(indexed.clone()), sort_rows(plain));
    assert_eq!(sort_rows(indexed), vec![s("josh"), s("marko")]);
}

/// TS: 'flipped comparison (const on the left) seeds too'
#[test]
fn idx_flipped_comparison_seeds() {
    let (plain, indexed) = both("MATCH (p:Person) WHERE 30 < p.age RETURN p.name");
    assert_eq!(sort_rows(indexed.clone()), sort_rows(plain));
    assert_eq!(sort_rows(indexed), vec![s("josh"), s("peter")]);
}

/// TS: 'WHERE IN seeds from a union and matches the scan'
#[test]
fn idx_where_in_seeds_and_matches_scan() {
    let (plain, indexed) = both("MATCH (p:Person) WHERE p.name IN ['marko', 'josh'] RETURN p.age");
    assert_eq!(sort_rows(indexed.clone()), sort_rows(plain));
    assert_eq!(sort_rows(indexed), vec![n(29.0), n(32.0)]);
}

/// TS: 'an OR predicate is NOT seeded (would miss a branch)'
#[test]
fn idx_or_predicate_not_seeded_still_correct() {
    let (plain, indexed) =
        both("MATCH (p:Person) WHERE p.name = 'marko' OR p.age > 30 RETURN p.name");
    assert_eq!(sort_rows(indexed.clone()), sort_rows(plain));
    assert_eq!(sort_rows(indexed), vec![s("josh"), s("marko"), s("peter")]);
}

/// TS: 'inline node WHERE seeds the start node'
#[test]
fn idx_inline_node_where_seeds_start_node() {
    let (plain, indexed) = both("MATCH (p:Person WHERE p.age > 30) RETURN p.name");
    assert_eq!(sort_rows(indexed.clone()), sort_rows(plain));
    assert_eq!(sort_rows(indexed), vec![s("josh"), s("peter")]);
}

/// TS: 'WHERE seeding still honors the rest of the pattern'
#[test]
fn idx_where_seeding_honors_rest_of_pattern() {
    let (plain, indexed) =
        both("MATCH (a:Person)-[:KNOWS]->(b) WHERE a.name = 'marko' RETURN b.name");
    assert_eq!(sort_rows(indexed.clone()), sort_rows(plain));
    assert_eq!(sort_rows(indexed), vec![s("josh"), s("vadas")]);
}

/// TS: 'multiple seekable conjuncts seed from the most selective one'
#[test]
fn idx_multiple_seekable_conjuncts_most_selective() {
    let (plain, indexed) =
        both("MATCH (p:Person) WHERE p.age > 28 AND p.name = 'josh' RETURN p.name");
    assert_eq!(sort_rows(indexed.clone()), sort_rows(plain));
    assert_eq!(sort_rows(indexed), vec![s("josh")]);
}

/// TS: 'an element-map equality and a WHERE range together still match the scan'
#[test]
fn idx_element_map_equality_and_where_range_match_scan() {
    let (plain, indexed) = both("MATCH (p:Person {name: 'marko'}) WHERE p.age < 30 RETURN p.age");
    assert_eq!(sort_rows(indexed.clone()), sort_rows(plain));
    assert_eq!(sort_rows(indexed), vec![n(29.0)]);
}

// ===========================================================================
// describe('GQL smaller-side seed selection', ...)
// ===========================================================================

/// TS: 'seeds from the selective far end and walks back (results match the scan)'
#[test]
fn idx_seeds_from_selective_far_end_walks_back() {
    let q_str = "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE b.name = 'josh' RETURN a.name";

    let plain_rows = {
        let mut g = modern();
        sorted(&mut g, q_str, "a.name")
    };
    let indexed_rows = {
        let mut g = modern();
        g.create_vertex_index("name");
        sorted(&mut g, q_str, "a.name")
    };

    assert_eq!(indexed_rows, plain_rows);
    assert_eq!(indexed_rows, vec![s("marko")]); // marko KNOWS josh
}

/// TS: 'far-end element-map constraint also drives the seed side'
#[test]
fn idx_far_end_element_map_drives_seed_side() {
    let q_str = "MATCH (a:Person)-[:KNOWS]->(b:Person {name: 'vadas'}) RETURN a.name";

    let plain_rows = {
        let mut g = modern();
        sorted(&mut g, q_str, "a.name")
    };
    let indexed_rows = {
        let mut g = modern();
        g.create_vertex_index("name");
        sorted(&mut g, q_str, "a.name")
    };

    assert_eq!(indexed_rows, plain_rows);
    assert_eq!(indexed_rows, vec![s("marko")]); // marko KNOWS vadas
}

/// TS: 'a variable-length segment keeps its orientation and still matches'
#[test]
fn idx_var_length_segment_keeps_orientation() {
    let q_str = "MATCH (a:Person)-[:KNOWS]->{1,2}(b:Person) WHERE b.name = 'josh' RETURN a.name";

    let plain_rows = {
        let mut g = modern();
        sorted(&mut g, q_str, "a.name")
    };
    let indexed_rows = {
        let mut g = modern();
        g.create_vertex_index("name");
        sorted(&mut g, q_str, "a.name")
    };

    assert_eq!(indexed_rows, plain_rows);
}

/// TS: 'an unlabeled start seeds the indexed far end instead of a full scan'
#[test]
fn idx_unlabeled_start_seeds_indexed_far_end() {
    let q_str = "MATCH (a)-[:KNOWS]->(b:Person) WHERE b.name = 'josh' RETURN a.name";

    let plain_rows = {
        let mut g = modern();
        sorted(&mut g, q_str, "a.name")
    };
    let indexed_rows = {
        let mut g = modern();
        g.create_vertex_index("name");
        sorted(&mut g, q_str, "a.name")
    };

    assert_eq!(indexed_rows, plain_rows);
    assert_eq!(indexed_rows, vec![s("marko")]);
}

/// TS: 'multi-hop pattern seeds from the selective end either way'
/// (Previously skipped: the two-hop `(a)->(b)->(c)` pattern panicked in
/// build_scan; fixed by tracking which slots are bound per segment.)
#[test]
fn idx_multi_hop_seeds_from_selective_end() {
    let q_str = "MATCH (a:Person {name: 'marko'})-[:KNOWS]->(b)-[:CREATED]->(c) RETURN c.name";

    let plain_rows = {
        let mut g = modern();
        sorted(&mut g, q_str, "c.name")
    };
    let indexed_rows = {
        let mut g = modern();
        g.create_vertex_index("name");
        sorted(&mut g, q_str, "c.name")
    };

    assert_eq!(indexed_rows, plain_rows);
    assert_eq!(indexed_rows, vec![s("lop"), s("ripple")]);
}

// ===========================================================================
// Multi-anchor comma patterns (index-seed planning)
//
// A comma-joined MATCH `(a {..}), (b {..})` is a nested-loop cross-join; before
// multi-anchor index-seed planning it bailed out of every vectorized (seek-capable) path and full-scanned
// *every* anchor — an O(n) footgun on a large graph. These lock in that each
// anchor now seeds from its property index (inline props AND WHERE conjuncts),
// byte-identical to the scan fallback, and that unseedable predicates still fall
// back correctly.
// ===========================================================================

/// Both comma anchors carry an indexed inline `{name: ...}`: each seeds
/// independently, and the cross-join is identical to the scan.
#[test]
fn idx_multi_anchor_inline_both_seed() {
    let q_str =
        "MATCH (a:Person {name: 'marko'}), (b:Software {name: 'lop'}) RETURN a.name, b.name";
    let (plain, indexed) = both(q_str);
    assert_eq!(indexed, plain);
    assert_eq!(indexed, vec![vec![s("marko"), s("lop")]]);
}

/// The C4 shape: `WHERE a.k=$x AND b.k=$y` across comma patterns. The AND-chain
/// splits so each anchor seeds on *its own* conjunct (slot-filtered), not the
/// other's — parity with the scan is the proof it stays sound.
#[test]
fn idx_multi_anchor_where_both_seed() {
    let q_str =
        "MATCH (a:Person), (b:Software) WHERE a.name = 'marko' AND b.name = 'lop' RETURN a.name, b.name";
    let (plain, indexed) = both(q_str);
    assert_eq!(indexed, plain);
    assert_eq!(indexed, vec![vec![s("marko"), s("lop")]]);
}

/// A three-anchor cross-join still seeds every anchor.
#[test]
fn idx_three_anchor_all_seed() {
    let q_str = "MATCH (a:Person {name: 'marko'}), (b:Person {name: 'josh'}), (c:Software {name: 'ripple'}) RETURN a.name, b.name, c.name";
    let (plain, indexed) = both(q_str);
    assert_eq!(indexed, plain);
    assert_eq!(indexed, vec![vec![s("marko"), s("josh"), s("ripple")]]);
}

/// A var-to-var WHERE (`a.name = b.name`) is NOT a literal hint, so neither
/// anchor may seed on it — both scan, and the (empty) result is unchanged.
/// Guards against the AND-split wrongly seeding one side from the other's slot.
#[test]
fn idx_multi_anchor_var_to_var_where_not_seeded() {
    let q_str = "MATCH (a:Person), (b:Software) WHERE a.name = b.name RETURN a.name, b.name";
    let (plain, indexed) = both(q_str);
    assert_eq!(indexed, plain);
    // No Person shares a name with any Software vertex.
    assert!(indexed.is_empty());
}

/// One anchor seeds inline; the other is unconstrained (full label scan). The
/// seeded side must not disturb the scanned side's rows.
#[test]
fn idx_multi_anchor_mixed_seed_and_scan() {
    let q_str = "MATCH (a:Person {name: 'marko'}), (b:Software) RETURN a.name, b.name";
    let (plain, indexed) = both(q_str);
    assert_eq!(indexed, plain);
    assert_eq!(
        indexed,
        vec![vec![s("marko"), s("lop")], vec![s("marko"), s("ripple")]]
    );
}

// ---------------------------------------------------------------------------
// A clause WHERE anchors the seek, not only an inline `{k: v}`.
//
// `scan_start_seed` never received the surrounding clause's WHERE, so
// `(u:P {k:$x})-[:R]->(y)` seeked while the identical
// `(u:P)-[:R]->(y) WHERE u.k = $x` scanned the whole label bucket — 60x on a 5k
// graph, on the form people actually write. Both seeked correctly when the node
// stood ALONE, which is what made it easy to miss.
// ---------------------------------------------------------------------------

/// A star: one hub with `n` spokes, every spoke carrying a distinct `k`.
fn star(n: usize) -> Graph {
    let mut lines =
        vec![r#"{"type":"node","id":"hub","labels":["H"],"properties":{}}"#.to_string()];

    for i in 0..n {
        lines.push(format!(
            r#"{{"type":"node","id":"s{i}","labels":["P"],"properties":{{"k":"key{i}"}}}}"#
        ));
        lines.push(format!(
            r#"{{"type":"edge","id":"e{i}","labels":["R"],"from":"s{i}","to":"hub","properties":{{}}}}"#
        ));
    }

    crate::ndjson::decode(&lines.join("\n")).expect("fixture decodes")
}

#[test]
fn clause_where_anchors_a_traversal_seek() {
    let mut g = star(200);

    g.create_vertex_index("k");

    // Both forms must find the one spoke, and the WHERE form is the one that
    // used to fall back to a scan.
    let inline = rows(&mut g, "MATCH (u:P {k: 'key7'})-[:R]->(h:H) RETURN u.k");
    let where_ = rows(
        &mut g,
        "MATCH (u:P)-[:R]->(h:H) WHERE u.k = 'key7' RETURN u.k",
    );

    assert_eq!(inline, vec![vec![s("key7")]]);
    assert_eq!(
        where_, inline,
        "the WHERE form must return what the inline form does"
    );
}

#[test]
fn clause_where_seek_agrees_with_the_unindexed_scan() {
    let mut plain = star(200);
    let mut indexed = star(200);

    indexed.create_vertex_index("k");

    for q in [
        "MATCH (u:P)-[:R]->(h:H) WHERE u.k = 'key3' RETURN u.k",
        "MATCH (u:P)-[:R]->(h:H) WHERE u.k = 'missing' RETURN u.k",
        // A conjunction: only the indexed conjunct can seed, the rest filters.
        "MATCH (u:P)-[:R]->(h:H) WHERE u.k = 'key9' AND h.id IS NULL RETURN u.k",
        // A DISJUNCTION must not seed — seeding one arm would drop the other's
        // matches, so this is the case where a wrong hint shows up as wrong rows.
        "MATCH (u:P)-[:R]->(h:H) WHERE u.k = 'key1' OR u.k = 'key2' RETURN u.k",
    ] {
        let mut a = rows(&mut plain, q);
        let mut b = rows(&mut indexed, q);

        a.sort_by(|x, y| cmp_val(&x[0], &y[0]));
        b.sort_by(|x, y| cmp_val(&x[0], &y[0]));

        assert_eq!(a, b, "index changed the answer for `{q}`");
    }
}

#[test]
fn clause_where_seek_survives_a_negated_or_absent_key() {
    let mut plain = star(50);
    let mut indexed = star(50);

    indexed.create_vertex_index("k");

    for q in [
        // `<>` is not seekable; it must still return every non-match.
        "MATCH (u:P)-[:R]->(h:H) WHERE u.k <> 'key1' RETURN count(*) AS c",
        // A key the index does not cover at all.
        "MATCH (u:P)-[:R]->(h:H) WHERE u.id = 's4' RETURN count(*) AS c",
    ] {
        assert_eq!(rows(&mut plain, q), rows(&mut indexed, q), "for `{q}`");
    }
}

// ---------------------------------------------------------------------------
// `key IN [...]` seeds a union of point seeks.
//
// There was no `In` arm in the hint at all, so an IN-list scanned even on an
// indexed key — 220 statements/sec against 147k for a single `=` over twenty
// values on a 20k graph. It also made the natural batching fix for a hot write
// loop (one IN statement instead of N statements) slower than the loop it
// replaced whenever an index existed.
// ---------------------------------------------------------------------------

#[test]
fn in_list_seeds_and_matches_the_scan() {
    let mut plain = star(200);
    let mut indexed = star(200);

    indexed.create_vertex_index("k");

    for q in [
        "MATCH (u:P) WHERE u.k IN ['key3', 'key7'] RETURN u.k",
        "MATCH (u:P)-[:R]->(h:H) WHERE u.k IN ['key3', 'key7'] RETURN u.k",
        // Nothing in the list exists.
        "MATCH (u:P) WHERE u.k IN ['nope', 'also-nope'] RETURN u.k",
        // Mixed hit and miss.
        "MATCH (u:P) WHERE u.k IN ['key1', 'nope'] RETURN u.k",
    ] {
        let mut a = rows(&mut plain, q);
        let mut b = rows(&mut indexed, q);

        a.sort_by(|x, y| cmp_val(&x[0], &y[0]));
        b.sort_by(|x, y| cmp_val(&x[0], &y[0]));

        assert_eq!(a, b, "index changed the answer for `{q}`");
    }
}

#[test]
fn in_list_with_a_repeated_value_does_not_duplicate_rows() {
    // A union of point seeks visits the same element once per occurrence, so
    // without dedup a repeated list value returns the element twice — duplicate
    // ROWS, a wrong answer rather than wasted work.
    let mut indexed = star(50);

    indexed.create_vertex_index("k");

    assert_eq!(
        rows(
            &mut indexed,
            "MATCH (u:P) WHERE u.k IN ['key1', 'key1', 'key2'] RETURN u.k"
        )
        .len(),
        2
    );
}

#[test]
fn an_empty_in_list_matches_nothing() {
    let mut plain = star(50);
    let mut indexed = star(50);

    indexed.create_vertex_index("k");

    let q = "MATCH (u:P) WHERE u.k IN [] RETURN u.k";

    assert!(rows(&mut indexed, q).is_empty());
    assert_eq!(rows(&mut plain, q), rows(&mut indexed, q));
}

#[test]
fn not_in_is_not_seekable_and_still_correct() {
    // The matches of a NOT IN are everything the list does not name, which no
    // point seek enumerates. It must fall back to the scan rather than seed from
    // the list and return its complement.
    let mut plain = star(50);
    let mut indexed = star(50);

    indexed.create_vertex_index("k");

    let q = "MATCH (u:P) WHERE NOT (u.k IN ['key1', 'key2']) RETURN count(*) AS c";

    assert_eq!(rows(&mut indexed, q), vec![vec![n(48.0)]]);
    assert_eq!(rows(&mut plain, q), rows(&mut indexed, q));
}

// ---------------------------------------------------------------------------
// Spellings that mean the same thing seek the same way.
//
// The hint recognized one spelling of each predicate and scanned for the others,
// so a semantically identical query cost 100-300x depending on how it was
// written. Two more of that family, found by probing every predicate form
// against an indexed graph and comparing rates:
//
//   `$x = u.k`               107x slower than `u.k = $x`  (only `left` inspected)
//   `u.k = $a OR u.k = $b`   220x slower than the same IN-list
// ---------------------------------------------------------------------------

#[test]
fn a_reversed_comparison_seeks_like_the_forward_one() {
    let mut plain = star(200);
    let mut indexed = star(200);

    indexed.create_vertex_index("k");

    for q in [
        "MATCH (u:P) WHERE 'key5' = u.k RETURN u.k",
        "MATCH (u:P)-[:R]->(h:H) WHERE 'key5' = u.k RETURN u.k",
    ] {
        assert_eq!(rows(&mut plain, q), rows(&mut indexed, q), "for `{q}`");
    }
}

#[test]
fn a_reversed_range_keeps_its_direction() {
    // The operands flip, so the operator must too: `'key5' < u.k` means
    // `u.k > 'key5'`. Getting this backwards returns the complement — a wrong
    // answer that a rate check would never notice.
    let mut plain = star(20);
    let mut indexed = star(20);

    indexed.create_vertex_index("k");

    for q in [
        "MATCH (u:P) WHERE 'key5' < u.k RETURN count(*) AS c",
        "MATCH (u:P) WHERE 'key5' >= u.k RETURN count(*) AS c",
        "MATCH (u:P) WHERE u.k > 'key5' RETURN count(*) AS c",
    ] {
        assert_eq!(rows(&mut plain, q), rows(&mut indexed, q), "for `{q}`");
    }

    // And the two spellings agree with each other, not just with the scan.
    assert_eq!(
        rows(
            &mut indexed,
            "MATCH (u:P) WHERE 'key5' < u.k RETURN count(*) AS c"
        ),
        rows(
            &mut indexed,
            "MATCH (u:P) WHERE u.k > 'key5' RETURN count(*) AS c"
        ),
    );
}

#[test]
fn a_disjunction_seeds_only_when_every_branch_can() {
    // The union of the branches' candidates IS the candidate set, so one
    // unseekable branch means its matches are absent from the union — missing
    // rows, not slow ones. `other` is deliberately not indexed.
    let mut plain = star(200);
    let mut indexed = star(200);

    indexed.create_vertex_index("k");

    for q in [
        // Both branches indexed → seeks.
        "MATCH (u:P) WHERE u.k = 'key3' OR u.k = 'key7' RETURN u.k",
        // One branch on an UNindexed key → must fall back and still find both.
        "MATCH (u:P) WHERE u.k = 'key3' OR u.id = 's7' RETURN u.k",
        // Overlapping branches must not double-count.
        "MATCH (u:P) WHERE u.k = 'key3' OR u.k = 'key3' RETURN u.k",
        // Nested.
        "MATCH (u:P) WHERE (u.k = 'key1' OR u.k = 'key2') OR u.k = 'key3' RETURN u.k",
    ] {
        let mut a = rows(&mut plain, q);
        let mut b = rows(&mut indexed, q);

        a.sort_by(|x, y| cmp_val(&x[0], &y[0]));
        b.sort_by(|x, y| cmp_val(&x[0], &y[0]));

        assert_eq!(a, b, "index changed the answer for `{q}`");
    }
}

#[test]
fn a_disjunction_across_two_variables_is_not_seeded() {
    // Branches on DIFFERENT variables identify different elements, so unioning
    // their candidates is meaningless. Must agree with the scan regardless.
    let mut plain = star(30);
    let mut indexed = star(30);

    indexed.create_vertex_index("k");

    let q = "MATCH (a:P), (b:P) WHERE a.k = 'key1' OR b.k = 'key2' RETURN count(*) AS c";

    assert_eq!(rows(&mut plain, q), rows(&mut indexed, q));
}

// ---------------------------------------------------------------------------
// SEMANTIC EQUIVALENCE: spellings that mean the same thing must also COST the
// same.
//
// Every seeding gap found so far had the same shape — the hint recognized one
// spelling of a predicate and scanned for another that meant exactly the same
// thing. `u.k IN [$a]` vs `u.k = $a`, `$x = u.k` vs `u.k = $x`,
// `u.k = $a OR u.k = $b` vs the IN-list, a clause WHERE vs an inline `{k: $x}`.
// Each was 100-300x, and each was found by hand.
//
// This generalizes the hunt: each group below is a set of queries that must
// return the same rows AND run within a factor of each other. Correctness alone
// would not have caught any of them — every one returned the right answer.
//
// The bound is deliberately loose (`MAX_RATIO`). A missed seek is two to three
// orders of magnitude; anything a slack factor could hide is not the class of
// bug this exists for, and a tight bound on wall-clock in a test suite is a
// flake generator.
// ---------------------------------------------------------------------------

/// How much slower the worst spelling in a group may be than the best.
const MAX_RATIO: f64 = 12.0;

/// A graph big enough that a scan and a seek are unmistakably different.
fn equiv_graph() -> Graph {
    let mut lines: Vec<String> = Vec::new();

    for i in 0..20_000 {
        lines.push(format!(
            r#"{{"type":"node","id":"u{i}","labels":["P"],"properties":{{"k":"key{i:06}","n":{i},"m":{{"city":"c{i}"}}}}}}"#
        ));
    }
    for i in 0..20_000 {
        lines.push(format!(
            r#"{{"type":"edge","id":"e{i}","labels":["R"],"from":"u{i}","to":"u{}","properties":{{"w":{i}}}}}"#,
            (i + 1) % 20_000
        ));
    }

    let mut g = crate::ndjson::decode(&lines.join("\n")).expect("fixture decodes");

    g.create_vertex_index("k");
    g.create_vertex_index("n");
    g.create_vertex_index("m.city");
    g.create_edge_index("w");

    g
}

/// Median-of-5 seconds for one execution of `q`.
fn equiv_time(g: &mut Graph, q: &str) -> f64 {
    let plan = parse(q).unwrap_or_else(|e| panic!("parse error for `{q}`: {e}"));
    let mut best = f64::MAX;

    for _ in 0..5 {
        let t = std::time::Instant::now();
        let rs = plan
            .execute(g, &Params::new())
            .unwrap_or_else(|e| panic!("exec error for `{q}`: {e}"));
        let secs = t.elapsed().as_secs_f64();

        std::hint::black_box(rs.rows().count());
        if secs < best {
            best = secs;
        }
    }

    best
}

#[test]
#[ignore = "timing-sensitive; run with --ignored --nocapture"]
fn equivalent_spellings_cost_the_same() {
    // Each group: every query must return identical rows and run within
    // MAX_RATIO of the fastest member.
    let groups: &[(&str, &[&str])] = &[
        (
            "point equality",
            &[
                "MATCH (u:P) WHERE u.k = 'key000005' RETURN count(*) AS c",
                "MATCH (u:P) WHERE 'key000005' = u.k RETURN count(*) AS c",
                "MATCH (u:P {k: 'key000005'}) RETURN count(*) AS c",
                "MATCH (u:P) WHERE u.k IN ['key000005'] RETURN count(*) AS c",
            ],
        ),
        (
            "two values",
            &[
                "MATCH (u:P) WHERE u.k IN ['key000005', 'key000009'] RETURN count(*) AS c",
                "MATCH (u:P) WHERE u.k = 'key000005' OR u.k = 'key000009' RETURN count(*) AS c",
                "MATCH (u:P) WHERE 'key000005' = u.k OR 'key000009' = u.k RETURN count(*) AS c",
            ],
        ),
        (
            "equality through a traversal",
            &[
                "MATCH (u:P)-[:R]->(x) WHERE u.k = 'key000005' RETURN count(*) AS c",
                "MATCH (u:P {k: 'key000005'})-[:R]->(x) RETURN count(*) AS c",
                "MATCH (u:P)-[:R]->(x) WHERE 'key000005' = u.k RETURN count(*) AS c",
                "MATCH (u:P)-[:R]->(x) WHERE u.k IN ['key000005'] RETURN count(*) AS c",
            ],
        ),
        (
            "numeric range",
            &[
                "MATCH (u:P) WHERE u.n >= 5 AND u.n <= 9 RETURN count(*) AS c",
                "MATCH (u:P) WHERE 5 <= u.n AND 9 >= u.n RETURN count(*) AS c",
                "MATCH (u:P) WHERE u.n >= 5 AND 9 >= u.n RETURN count(*) AS c",
            ],
        ),
        (
            "dotted path",
            &[
                "MATCH (u:P) WHERE u.m.city = 'c5' RETURN count(*) AS c",
                "MATCH (u:P) WHERE 'c5' = u.m.city RETURN count(*) AS c",
                "MATCH (u:P) WHERE u.m.city IN ['c5'] RETURN count(*) AS c",
            ],
        ),
        (
            "edge property",
            &[
                "MATCH ()-[e:R]->() WHERE e.w = 5 RETURN count(*) AS c",
                "MATCH ()-[e:R]->() WHERE 5 = e.w RETURN count(*) AS c",
                "MATCH ()-[e:R]->() WHERE e.w IN [5] RETURN count(*) AS c",
                "MATCH ()-[e:R {w: 5}]->() RETURN count(*) AS c",
            ],
        ),
        (
            "conjunction with a non-seekable extra",
            &[
                "MATCH (u:P) WHERE u.k = 'key000005' AND u.n >= 0 RETURN count(*) AS c",
                "MATCH (u:P) WHERE u.n >= 0 AND u.k = 'key000005' RETURN count(*) AS c",
            ],
        ),
        // Which comma pattern is written first must not decide which one drives
        // the join. Before `pick_pattern`, the anchored-last spelling enumerated
        // the unanchored pattern as the outer loop — measured 121,336x apart at
        // 300k vertices, and unbounded (see docs/design/query-ir.md).
        (
            "pattern order, anchored by inline props",
            &[
                "MATCH (u:P {k: 'key000005'})-[:R]->(x), (x)-[:R]->(y) RETURN count(*) AS c",
                "MATCH (x)-[:R]->(y), (u:P {k: 'key000005'})-[:R]->(x) RETURN count(*) AS c",
            ],
        ),
        // A comma join that CHAINS is the same query as one path. Before
        // `fuse_chain` the comma spelling stayed on the scalar join: 134x for two
        // patterns, 347x for three.
        (
            "chained comma vs one path",
            &[
                "MATCH (u:P)-[:R]->(x)-[:R]->(y) WHERE u.k = 'key000005' RETURN count(*) AS c",
                "MATCH (u:P)-[:R]->(x), (x)-[:R]->(y) WHERE u.k = 'key000005' RETURN count(*) AS c",
            ],
        ),
        (
            "pattern order, anchored by a clause WHERE",
            &[
                "MATCH (u:P)-[:R]->(x), (x)-[:R]->(y) WHERE u.k = 'key000005' RETURN count(*) AS c",
                "MATCH (x)-[:R]->(y), (u:P)-[:R]->(x) WHERE u.k = 'key000005' RETURN count(*) AS c",
            ],
        ),
    ];
    let mut g = equiv_graph();
    let mut failures: Vec<String> = Vec::new();

    for (name, queries) in groups {
        // Same rows, first of all — a "fast" spelling that answers differently
        // is not an equivalent spelling.
        let expect = rows(&mut g, queries[0]);

        for q in &queries[1..] {
            let got = rows(&mut g, q);

            assert_eq!(
                &got, &expect,
                "[{name}] `{q}` disagreed with `{}`",
                queries[0]
            );
        }

        let times: Vec<f64> = queries.iter().map(|q| equiv_time(&mut g, q)).collect();
        let fastest = times.iter().copied().fold(f64::MAX, f64::min);

        for (q, t) in queries.iter().zip(&times) {
            let ratio = t / fastest;

            println!("  {ratio:>6.1}x  [{name}] {q}");
            if ratio > MAX_RATIO {
                failures.push(format!(
                    "[{name}] {ratio:.0}x slower than the best spelling in its group:\n    {q}"
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "\nspellings that mean the same thing but do not cost the same:\n\n{}\n",
        failures.join("\n\n")
    );
}

// ---------------------------------------------------------------------------
// Self-joins. A pattern naming the same variable twice is an EQUALITY against
// what the first occurrence bound; the frontier enforces it per row rather than
// refusing the pattern.
// ---------------------------------------------------------------------------

fn cyc() -> Graph {
    crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["P"],"properties":{"n":1}}"#,
            r#"{"type":"node","id":"b","labels":["P"],"properties":{"n":2}}"#,
            r#"{"type":"node","id":"c","labels":["P"],"properties":{"n":3}}"#,
            r#"{"type":"edge","id":"e1","labels":["R"],"from":"a","to":"b","properties":{}}"#,
            r#"{"type":"edge","id":"e2","labels":["R"],"from":"b","to":"a","properties":{}}"#,
            r#"{"type":"edge","id":"e3","labels":["R"],"from":"b","to":"c","properties":{}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes")
}

#[test]
fn a_repeated_variable_closes_the_cycle() {
    let mut g = cyc();

    // Only a<->b is a 2-cycle; b->c does not come back. Free endpoints would
    // give 3, so a wrong equality shows up as a row count.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a:P)-[:R]->(b:P)-[:R]->(a) RETURN count(*) AS c"
        ),
        vec![vec![n(2.0)]]
    );
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a:P)-[:R]->(b:P)-[:R]->(c:P) RETURN count(*) AS c"
        ),
        vec![vec![n(3.0)]]
    );
}

#[test]
fn a_repeated_variable_projects_the_bound_element() {
    let mut g = cyc();
    let mut got = rows(&mut g, "MATCH (a:P)-[:R]->(b:P)-[:R]->(a) RETURN a.n AS x");

    got.sort_by(|l, r| cmp_val(&l[0], &r[0]));
    // Both a and b start a 2-cycle; `a` must project the element it bound.
    assert_eq!(got, vec![vec![n(1.0)], vec![n(2.0)]]);
}

#[test]
fn a_repeated_variable_composes_with_a_filter() {
    let mut g = cyc();

    assert_eq!(
        rows(
            &mut g,
            "MATCH (a:P)-[:R]->(b:P)-[:R]->(a) WHERE a.n = 1 RETURN b.n AS x"
        ),
        vec![vec![n(2.0)]]
    );
}

// ---------------------------------------------------------------------------
// Var-length walks run vectorized. The walk itself is `reachable_each` — the
// same one the scalar matcher drives — so bounds, path MODE and the zero-length
// case cannot drift; what is new is the frontier fanning out around it.
// ---------------------------------------------------------------------------

fn chain() -> Graph {
    crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"1","labels":["P"],"properties":{"k":"a"}}"#,
            r#"{"type":"node","id":"2","labels":["P"],"properties":{"k":"b"}}"#,
            r#"{"type":"node","id":"3","labels":["P"],"properties":{"k":"c"}}"#,
            r#"{"type":"edge","id":"e1","labels":["R"],"from":"1","to":"2","properties":{}}"#,
            r#"{"type":"edge","id":"e2","labels":["R"],"from":"2","to":"3","properties":{}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes")
}

fn ks(g: &mut Graph, q: &str) -> Vec<String> {
    let mut out: Vec<String> = rows(g, q)
        .into_iter()
        .map(|r| match &r[0] {
            Value::Str(s) => s.to_string(),
            other => format!("{other:?}"),
        })
        .collect();

    out.sort();
    out
}

#[test]
fn a_var_length_walk_respects_its_bounds() {
    let mut g = chain();

    assert_eq!(
        ks(&mut g, "MATCH (a:P {k:'a'})-[:R]->{1,1}(b) RETURN b.k AS x"),
        vec!["b"]
    );
    assert_eq!(
        ks(&mut g, "MATCH (a:P {k:'a'})-[:R]->{1,2}(b) RETURN b.k AS x"),
        vec!["b", "c"]
    );
    assert_eq!(
        ks(&mut g, "MATCH (a:P {k:'a'})-[:R]->{2,3}(b) RETURN b.k AS x"),
        vec!["c"]
    );
    assert!(ks(&mut g, "MATCH (a:P {k:'a'})-[:R]->{3,4}(b) RETURN b.k AS x").is_empty());
}

#[test]
fn a_star_walk_includes_the_zero_length_path() {
    let mut g = chain();

    // Zero hops is the start itself, so `b` binds to `a`.
    assert_eq!(
        ks(&mut g, "MATCH (a:P {k:'a'})-[:R]->*(b) RETURN b.k AS x"),
        vec!["a", "b", "c"]
    );
}

#[test]
fn a_var_length_landing_still_applies_its_constraints() {
    let mut g = chain();

    // The label and inline constraint on the LANDING node are checked at every
    // depth, not just the last — the walk reaches b and c, the filter keeps one.
    assert_eq!(
        ks(
            &mut g,
            "MATCH (a:P {k:'a'})-[:R]->{1,2}(b:P {k:'c'}) RETURN b.k AS x"
        ),
        vec!["c"]
    );
    assert!(ks(
        &mut g,
        "MATCH (a:P {k:'a'})-[:R]->{1,2}(b:Nope) RETURN b.k AS x"
    )
    .is_empty());
}

#[test]
fn a_var_length_walk_composes_with_a_following_hop() {
    let mut g = chain();

    // The frontier must carry the walk's landings into the next segment.
    assert_eq!(
        ks(
            &mut g,
            "MATCH (a:P {k:'a'})-[:R]->{1,1}(b)-[:R]->(c) RETURN c.k AS x"
        ),
        vec!["c"]
    );
}
