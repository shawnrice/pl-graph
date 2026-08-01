//! The shared access path, tested against the graph directly — no query
//! language involved, which is the point: these are the semantics both front
//! ends inherit rather than re-derive.

use super::{ElementSeek, Operand, SeekOp};
use crate::graph::{Graph, IdxKey};

/// 200 vertices with an indexed `k` (string) and `n` (numeric), and 200 edges
/// with an indexed `w`. `dup` is deliberately low-cardinality so "pick the
/// smallest candidate set" has something to get wrong.
fn indexed() -> Graph {
    let mut lines = Vec::new();

    for i in 0..200 {
        lines.push(format!(
            r#"{{"type":"node","id":"u{i}","labels":["P"],"properties":{{"k":"key{i:03}","n":{i},"dup":{}}}}}"#,
            i % 2
        ));
        lines.push(format!(
            r#"{{"type":"edge","id":"e{i}","labels":["R"],"from":"u{i}","to":"u{}","properties":{{"w":{i}}}}}"#,
            (i + 1) % 200
        ));
    }

    let mut g = crate::ndjson::decode(&lines.join("\n")).expect("fixture decodes");

    g.create_vertex_index("k");
    g.create_vertex_index("n");
    g.create_vertex_index("dup");
    g.create_edge_index("w");
    g
}

/// No parameters bound — every `Param` is unresolvable.
fn no_params(_: usize) -> Option<IdxKey> {
    None
}

fn s(v: &str) -> Operand {
    Operand::Lit(IdxKey::Str(v.into()))
}

fn n(v: f64) -> Operand {
    Operand::Lit(IdxKey::Num(v))
}

fn sorted(mut ids: Vec<u32>) -> Vec<u32> {
    ids.sort_unstable();
    ids
}

#[test]
fn nothing_recognized_means_scan() {
    let g = indexed();

    assert!(ElementSeek::node().is_empty());
    assert!(ElementSeek::node().resolve(&g, &no_params).is_none());
}

#[test]
fn a_point_equality_seeks_one_element() {
    let g = indexed();
    let mut seek = ElementSeek::node();

    seek.push("k".into(), SeekOp::Eq, s("key005"));

    assert_eq!(seek.resolve(&g, &no_params).map(|ids| ids.len()), Some(1));
}

#[test]
fn an_unindexed_key_does_not_seek() {
    let g = indexed();
    let mut seek = ElementSeek::node();

    seek.push("nope".into(), SeekOp::Eq, s("x"));

    assert!(seek.resolve(&g, &no_params).is_none());
}

#[test]
fn two_bounds_on_one_key_become_a_single_range() {
    let g = indexed();
    let mut seek = ElementSeek::node();

    // `n >= 5 AND n <= 9` is five elements, not "everything >= 5".
    seek.push("n".into(), SeekOp::Ge, n(5.0));
    seek.push("n".into(), SeekOp::Le, n(9.0));

    assert_eq!(seek.resolve(&g, &no_params).map(|ids| ids.len()), Some(5));
}

#[test]
fn a_flipped_operator_means_the_same_range() {
    let g = indexed();
    let mut forward = ElementSeek::node();
    let mut reversed = ElementSeek::node();

    forward.push("n".into(), SeekOp::Ge, n(5.0));
    forward.push("n".into(), SeekOp::Le, n(9.0));
    // `5 <= u.n AND 9 >= u.n`, as a front end would normalize it.
    reversed.push("n".into(), SeekOp::Le.flipped(), n(5.0));
    reversed.push("n".into(), SeekOp::Ge.flipped(), n(9.0));

    assert_eq!(
        sorted(reversed.resolve(&g, &no_params).unwrap()),
        sorted(forward.resolve(&g, &no_params).unwrap())
    );
}

#[test]
fn the_smallest_candidate_set_wins() {
    let g = indexed();
    let mut seek = ElementSeek::node();

    // `dup = 0` matches 100 elements, `k = 'key004'` matches one. Both are
    // necessary conditions, so either is a valid superset — but seeding from
    // `dup` and filtering is 100x the work. Order is reversed from the answer
    // on purpose: taking the FIRST usable conjunct is the bug this prevents.
    seek.push("dup".into(), SeekOp::Eq, n(0.0));
    seek.push("k".into(), SeekOp::Eq, s("key004"));

    assert_eq!(seek.resolve(&g, &no_params).map(|ids| ids.len()), Some(1));
}

#[test]
fn an_unseekable_conjunct_does_not_block_a_seekable_one() {
    let g = indexed();
    let mut seek = ElementSeek::node();

    // A conjunct on an unindexed key just cannot narrow anything; the indexed
    // one still seeds and the caller re-verifies the rest.
    seek.push("nope".into(), SeekOp::Eq, s("x"));
    seek.push("k".into(), SeekOp::Eq, s("key004"));

    assert_eq!(seek.resolve(&g, &no_params).map(|ids| ids.len()), Some(1));
}

#[test]
fn a_disjunction_unions_its_branches() {
    let g = indexed();
    let mut seek = ElementSeek::node();

    seek.push_any_of("k".into(), vec![s("key001"), s("key002"), s("key003")]);

    assert_eq!(seek.resolve(&g, &no_params).map(|ids| ids.len()), Some(3));
}

#[test]
fn a_repeated_value_in_a_disjunction_yields_one_row() {
    let g = indexed();
    let mut seek = ElementSeek::node();

    // The seed is a candidate LIST, so a duplicate would become a duplicate row.
    seek.push_any_of("k".into(), vec![s("key001"), s("key001")]);

    assert_eq!(seek.resolve(&g, &no_params).map(|ids| ids.len()), Some(1));
}

#[test]
fn a_singleton_disjunction_is_a_point_equality() {
    let g = indexed();
    let mut one_of = ElementSeek::node();
    let mut eq = ElementSeek::node();

    // `k IN ['key004']` must not take a different path from `k = 'key004'`.
    one_of.push_any_of("k".into(), vec![s("key004")]);
    eq.push("k".into(), SeekOp::Eq, s("key004"));

    assert_eq!(one_of.resolve(&g, &no_params), eq.resolve(&g, &no_params));
}

#[test]
fn an_empty_disjunction_matches_nothing_rather_than_scanning() {
    let g = indexed();
    let mut seek = ElementSeek::node();

    // `k IN []` is not "no constraint" — it is "no rows". Returning None here
    // would scan and then let the caller's filter produce the right answer
    // slowly; returning an empty seed is both correct and free.
    seek.push_any_of("k".into(), Vec::new());

    assert_eq!(seek.resolve(&g, &no_params), Some(Vec::new()));
}

#[test]
fn one_unseekable_branch_makes_the_whole_disjunction_unseekable() {
    let g = indexed();
    let mut seek = ElementSeek::node();

    // Unlike a conjunct, a missing branch LOSES rows: the union would no longer
    // be a superset. The param is unbound here, standing in for any value the
    // index cannot answer.
    seek.push_any_of("k".into(), vec![s("key001"), Operand::Param(0)]);

    assert!(seek.resolve(&g, &no_params).is_none());
}

#[test]
fn a_bound_param_seeks_exactly_like_a_literal() {
    let g = indexed();
    let mut by_param = ElementSeek::node();
    let mut by_lit = ElementSeek::node();

    by_param.push("k".into(), SeekOp::Eq, Operand::Param(0));
    by_lit.push("k".into(), SeekOp::Eq, s("key007"));

    let bound = |slot: usize| (slot == 0).then(|| IdxKey::Str("key007".into()));

    assert_eq!(by_param.resolve(&g, &bound), by_lit.resolve(&g, &no_params));
}

#[test]
fn an_edge_seek_reads_the_edge_index() {
    let g = indexed();
    let mut on_edge = ElementSeek::edge();
    let mut on_node = ElementSeek::node();

    on_edge.push("w".into(), SeekOp::Eq, n(5.0));
    // The same key name against the VERTEX index, which has no `w`.
    on_node.push("w".into(), SeekOp::Eq, n(5.0));

    assert_eq!(
        on_edge.resolve(&g, &no_params).map(|ids| ids.len()),
        Some(1)
    );
    assert!(on_node.resolve(&g, &no_params).is_none());
}

#[test]
fn a_contradiction_seeks_nothing_without_scanning() {
    let g = indexed();
    let mut seek = ElementSeek::node();

    seek.push("n".into(), SeekOp::Ge, n(9.0));
    seek.push("n".into(), SeekOp::Le, n(5.0));

    assert_eq!(seek.resolve(&g, &no_params), Some(Vec::new()));
}
