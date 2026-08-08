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

#[test]
fn an_undirected_expansion_follows_the_self_loop_rule() {
    let g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["P"],"properties":{}}"#,
            r#"{"type":"node","id":"b","labels":["P"],"properties":{}}"#,
            r#"{"type":"edge","id":"s","labels":["R"],"from":"a","to":"a","properties":{}}"#,
            r#"{"type":"edge","id":"e","labels":["R"],"from":"a","to":"b","properties":{}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");
    let a = vec![0u32];

    use super::{expand, expand_count, Dir, SelfLoops};

    // `a` has one self-loop and one ordinary edge. Undirected, TinkerPop reaches
    // the loop from each side (2 + 1 = 3); GQL matches the edge once (1 + 1 = 2).
    assert_eq!(expand_count(&g, &a, Dir::Both, &[], SelfLoops::Twice), 3);
    assert_eq!(expand_count(&g, &a, Dir::Both, &[], SelfLoops::Once), 2);
    assert_eq!(
        expand(&g, &a, Dir::Both, &[], SelfLoops::Twice).len(),
        expand_count(&g, &a, Dir::Both, &[], SelfLoops::Twice)
    );

    // A DIRECTED walk keeps the loop under either rule — it is only reachable
    // from one side, so there is nothing to double-count.
    for rule in [SelfLoops::Once, SelfLoops::Twice] {
        assert_eq!(expand_count(&g, &a, Dir::Out, &[], rule), 2);
        assert_eq!(expand_count(&g, &a, Dir::In, &[], rule), 1);
    }
}

/// A graph whose adjacency makes the two DISTINCT-count spellings disagree if
/// the bitmap is wrong: parallel edges to the same target (the duplicate a
/// dedup must collapse), a self-loop (which `SelfLoops` treats differently per
/// direction), and a vertex reachable from two different sources.
fn distinct_fixture() -> Graph {
    crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["V"],"properties":{}}"#,
            r#"{"type":"node","id":"b","labels":["V"],"properties":{}}"#,
            r#"{"type":"node","id":"c","labels":["V"],"properties":{}}"#,
            r#"{"type":"node","id":"d","labels":["V"],"properties":{}}"#,
            // Two parallel a→b edges: two walks, ONE distinct endpoint.
            r#"{"type":"edge","id":"p1","from":"a","to":"b","labels":["R"],"properties":{}}"#,
            r#"{"type":"edge","id":"p2","from":"a","to":"b","labels":["R"],"properties":{}}"#,
            r#"{"type":"edge","id":"p3","from":"a","to":"c","labels":["R"],"properties":{}}"#,
            // `c` reaches `b` too, so `b` is reachable two ways.
            r#"{"type":"edge","id":"p4","from":"c","to":"b","labels":["R"],"properties":{}}"#,
            r#"{"type":"edge","id":"p5","from":"d","to":"d","labels":["R"],"properties":{}}"#,
            r#"{"type":"edge","id":"p6","from":"a","to":"d","labels":["S"],"properties":{}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes")
}

/// The bitmap distinct-count must equal deduplicating the materialized
/// expansion — the spelling it replaced — for every direction and type filter.
#[test]
fn a_distinct_expansion_counts_what_deduplicating_it_would() {
    use super::{Dir, SelfLoops};
    let g = distinct_fixture();
    let all: Vec<u32> = g.vertex_indices().collect();
    let r = g.etype.get("R").expect("R exists");
    let st = g.etype.get("S").expect("S exists");
    let repeated: Vec<u32> = vec![all[0], all[0], all[0]];
    let empty: Vec<u32> = Vec::new();

    for (name, src, dir, etypes, loops) in [
        (
            "out R, all sources",
            &all,
            Dir::Out,
            vec![r],
            SelfLoops::Once,
        ),
        ("in R, all sources", &all, Dir::In, vec![r], SelfLoops::Once),
        // `Both` with `SelfLoops::Once` drops the loop on the IN side, which the
        // bitmap has to honour the same way the counting form does.
        (
            "both R, all sources",
            &all,
            Dir::Both,
            vec![r],
            SelfLoops::Once,
        ),
        ("both R, twice", &all, Dir::Both, vec![r], SelfLoops::Twice),
        ("out, any type", &all, Dir::Out, vec![], SelfLoops::Once),
        ("out S only", &all, Dir::Out, vec![st], SelfLoops::Once),
        // A repeated source: the same neighbour arrives twice and still counts
        // once.
        (
            "repeated source",
            &repeated,
            Dir::Out,
            vec![r],
            SelfLoops::Once,
        ),
        ("empty source", &empty, Dir::Out, vec![r], SelfLoops::Once),
    ] {
        let bitmap = super::distinct_expand_count(&g, src, dir, &etypes, loops);
        let mut materialized = super::expand(&g, src, dir, &etypes, loops);
        materialized.sort_unstable();
        materialized.dedup();
        assert_eq!(bitmap, materialized.len(), "{name}");
    }
}

/// Parallel edges are the case a dedup exists for: `a` has two edges to `b` and
/// one to `c`, so three walks and TWO distinct endpoints.
#[test]
fn parallel_edges_are_one_distinct_endpoint() {
    use super::{Dir, SelfLoops};
    let g = distinct_fixture();
    let r = g.etype.get("R").expect("R exists");
    let a = g.vertex_indices().next().expect("a is the first vertex");
    assert_eq!(
        super::expand_count(&g, &[a], Dir::Out, &[r], SelfLoops::Once),
        3
    );
    assert_eq!(
        super::distinct_expand_count(&g, &[a], Dir::Out, &[r], SelfLoops::Once),
        2
    );
}

/// Collapsing the intermediate frontier must not change a DISTINCT walk count.
///
/// The fixture reaches `b` two ways, so the midpoint set and the midpoint
/// MULTISET differ — which is the whole premise: a set of endpoints depends on
/// which midpoints were reached, not on how often. The counting form must be
/// left alone, and is checked here too so the optimization cannot leak into it.
#[test]
fn collapsing_the_midpoints_preserves_a_distinct_walk_count() {
    use super::{Dir, SelfLoops};
    let g = distinct_fixture();
    let all: Vec<u32> = g.vertex_indices().collect();
    let r = g.etype.get("R").expect("R exists");

    for hops in [
        vec![(Dir::Out, Some(vec![r]))],
        vec![(Dir::Out, Some(vec![r])), (Dir::Out, Some(vec![r]))],
        vec![(Dir::In, Some(vec![r])), (Dir::Out, Some(vec![r]))],
        vec![
            (Dir::Out, Some(vec![r])),
            (Dir::Out, Some(vec![r])),
            (Dir::Out, Some(vec![r])),
        ],
        vec![(Dir::Both, Some(vec![r])), (Dir::Both, Some(vec![r]))],
    ] {
        // Reference: expand every hop with multiplicity, then deduplicate the
        // final endpoints — the definition the fast path has to preserve.
        let mut cur: Vec<u32> = all.clone();
        for (dir, et) in &hops {
            cur = super::expand(
                &g,
                &cur,
                *dir,
                et.as_deref().unwrap_or(&[]),
                SelfLoops::Once,
            );
        }
        cur.sort_unstable();
        let counted = cur.len();
        cur.dedup();

        assert_eq!(
            super::walk_count(&g, &all, &hops, SelfLoops::Once, true),
            cur.len(),
            "distinct count for {hops:?}"
        );
        assert_eq!(
            super::walk_count(&g, &all, &hops, SelfLoops::Once, false),
            counted,
            "the counting form must keep every arrival for {hops:?}"
        );
    }
}
