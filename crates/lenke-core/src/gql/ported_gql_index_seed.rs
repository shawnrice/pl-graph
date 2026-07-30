//! Ported GQL index-seeding tests — a faithful behavioral-parity port of the
//! TypeScript `packages/gql/src/index-seed.test.ts` spec.
//!
//! Covers property-index seeding (equality / range `has`-style predicates served
//! from a sorted secondary index), WHERE-derived seed hints, and smaller-side
//! seed selection. Result parity is the contract; TS-internal plan structures
//! have no Rust equivalent and are treated as unsupported.

use super::eval::Params;
use super::parse;
use crate::graph::{Graph, Value};
use crate::ndjson;

// ---------------------------------------------------------------------------
// Fixture — TinkerPop "Modern" graph.  Identical to tests.rs `modern()`.
// ---------------------------------------------------------------------------

fn modern() -> Graph {
    let lines = [
        r#"{"type":"node","id":"marko","labels":["Person"],"properties":{"name":"marko","age":29}}"#,
        r#"{"type":"node","id":"vadas","labels":["Person"],"properties":{"name":"vadas","age":27}}"#,
        r#"{"type":"node","id":"josh","labels":["Person"],"properties":{"name":"josh","age":32}}"#,
        r#"{"type":"node","id":"peter","labels":["Person"],"properties":{"name":"peter","age":35}}"#,
        r#"{"type":"node","id":"lop","labels":["Software"],"properties":{"name":"lop","lang":"java"}}"#,
        r#"{"type":"node","id":"ripple","labels":["Software"],"properties":{"name":"ripple","lang":"java"}}"#,
        r#"{"type":"edge","from":"marko","to":"vadas","labels":["KNOWS"],"properties":{"weight":0.5}}"#,
        r#"{"type":"edge","from":"marko","to":"josh","labels":["KNOWS"],"properties":{"weight":1.0}}"#,
        r#"{"type":"edge","from":"marko","to":"lop","labels":["CREATED"],"properties":{"weight":0.4}}"#,
        r#"{"type":"edge","from":"josh","to":"ripple","labels":["CREATED"],"properties":{"weight":1.0}}"#,
        r#"{"type":"edge","from":"josh","to":"lop","labels":["CREATED"],"properties":{"weight":0.4}}"#,
        r#"{"type":"edge","from":"peter","to":"lop","labels":["CREATED"],"properties":{"weight":0.2}}"#,
    ];
    ndjson::decode(&lines.join("\n")).unwrap()
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
