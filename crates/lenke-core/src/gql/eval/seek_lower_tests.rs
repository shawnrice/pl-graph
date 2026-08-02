//! Semantically identical predicates must lower to the SAME `ElementSeek`.
//!
//! This is a stronger claim than the timing test in `gql/index_seed_tests.rs`,
//! and it is the actual goal. `equivalent_spellings_cost_the_same` can only
//! observe that two spellings ended up equally fast — it cannot tell "both
//! seek" from "both scan", and it says nothing at all about a spelling nobody
//! thought to add to it. Comparing the lowered structure is decidable: either
//! the two collapsed to one shape or they did not.
//!
//! Every group below contains a spelling that once cost 60-220x more than its
//! neighbours while returning the identical answer.

use super::super::plan::CQuery;
use super::seek_lower::{element_seek, inline_of};
use super::{resolve_ctx, Val};
use crate::graph::Graph;
use crate::seek::ElementSeek;

/// Indexed on `k`, `n` and `m.city`; edges indexed on `w`.
fn indexed() -> Graph {
    let mut lines = Vec::new();

    for i in 0..50 {
        lines.push(format!(
            r#"{{"type":"node","id":"u{i}","labels":["P"],"properties":{{"k":"key{i:03}","n":{i},"m":{{"city":"c{i}"}}}}}}"#
        ));
        lines.push(format!(
            r#"{{"type":"edge","id":"e{i}","labels":["R"],"from":"u{i}","to":"u{}","properties":{{"w":{i}}}}}"#,
            (i + 1) % 50
        ));
    }

    let mut g = crate::ndjson::decode(&lines.join("\n")).expect("fixture decodes");

    g.create_vertex_index("k");
    g.create_vertex_index("n");
    g.create_vertex_index("m.city");
    g.create_edge_index("w");
    g
}

/// The `ElementSeek` a query's first pattern lowers to.
///
/// Deliberately goes through the real parser and lowerer: a test that built
/// `CExpr` by hand would prove the lowering agrees with itself.
fn seek_of(g: &Graph, query: &str, edge: bool) -> ElementSeek {
    let plan: CQuery = super::prepare_plan(query).expect("prepares");
    let params: Vec<Val> = vec![Val::Str("key005".into()), Val::Str("key009".into())];
    let ctx = resolve_ctx(g, &plan, &params);
    let (path, where_) = first_pattern(&plan);
    let node = if edge { None } else { Some(&path.start) };
    let inline = node.map_or_else(Vec::new, |n| inline_of(n));
    let slot = if edge {
        path.segments.first().and_then(|s| s.rel.var_slot)
    } else {
        path.start.var_slot
    };
    let rel_inline: Vec<(&str, &super::super::plan::CExpr)> = if edge {
        path.segments
            .first()
            .map(|s| {
                s.rel
                    .props
                    .iter()
                    .map(|p| (p.key.as_str(), &p.value))
                    .collect()
            })
            .unwrap_or_default()
    } else {
        inline
    };

    element_seek(where_, &rel_inline, g, &ctx, slot, edge)
}

/// The first `MATCH` pattern and its clause `WHERE`.
fn first_pattern(
    plan: &CQuery,
) -> (
    &super::super::plan::CPath,
    Option<&super::super::plan::CExpr>,
) {
    for part in &plan.parts {
        for clause in &part.clauses {
            if let super::super::plan::CClause::Match {
                patterns, where_, ..
            } = clause
            {
                return (&patterns[0], where_.as_ref());
            }
        }
    }

    panic!("query has no MATCH");
}

/// Assert every spelling in a group lowers to one structure.
fn same_shape(name: &str, edge: bool, queries: &[&str]) {
    let g = indexed();
    let first = seek_of(&g, queries[0], edge);

    for q in &queries[1..] {
        assert_eq!(
            seek_of(&g, q, edge),
            first,
            "[{name}]\n  `{q}`\ndid not collapse to the same shape as\n  `{}`",
            queries[0]
        );
    }

    assert!(!first.is_empty(), "[{name}] nothing was recognized at all");
}

#[test]
fn point_equality_spellings_collapse() {
    same_shape(
        "point equality",
        false,
        &[
            "MATCH (u:P) WHERE u.k = $a RETURN count(*) AS c",
            "MATCH (u:P) WHERE $a = u.k RETURN count(*) AS c",
            "MATCH (u:P {k: $a}) RETURN count(*) AS c",
            "MATCH (u:P) WHERE u.k IN [$a] RETURN count(*) AS c",
        ],
    );
}

#[test]
fn a_disjunction_and_an_in_list_collapse() {
    same_shape(
        "two values",
        false,
        &[
            "MATCH (u:P) WHERE u.k IN [$a, $b] RETURN count(*) AS c",
            "MATCH (u:P) WHERE u.k = $a OR u.k = $b RETURN count(*) AS c",
            "MATCH (u:P) WHERE $a = u.k OR $b = u.k RETURN count(*) AS c",
        ],
    );
}

#[test]
fn a_nested_disjunction_flattens() {
    same_shape(
        "flattened OR",
        false,
        &[
            "MATCH (u:P) WHERE u.k = $a OR u.k = $b OR u.k = 'key013' RETURN count(*) AS c",
            "MATCH (u:P) WHERE (u.k = $a OR u.k = $b) OR u.k = 'key013' RETURN count(*) AS c",
            "MATCH (u:P) WHERE u.k = $a OR (u.k = $b OR u.k = 'key013') RETURN count(*) AS c",
        ],
    );
}

#[test]
fn reversed_range_bounds_collapse() {
    same_shape(
        "numeric range",
        false,
        &[
            "MATCH (u:P) WHERE u.n >= 5 AND u.n <= 9 RETURN count(*) AS c",
            "MATCH (u:P) WHERE 5 <= u.n AND 9 >= u.n RETURN count(*) AS c",
            "MATCH (u:P) WHERE u.n >= 5 AND 9 >= u.n RETURN count(*) AS c",
        ],
    );
}

#[test]
fn a_traversal_does_not_change_the_anchor_shape() {
    let g = indexed();

    // The clause WHERE before a traversal was a 60x gap: the anchor's seek is
    // the same predicate whether or not a pattern hangs off it.
    assert_eq!(
        seek_of(
            &g,
            "MATCH (u:P)-[:R]->(x) WHERE u.k = $a RETURN count(*) AS c",
            false
        ),
        seek_of(&g, "MATCH (u:P) WHERE u.k = $a RETURN count(*) AS c", false)
    );
}

#[test]
fn dotted_path_spellings_collapse() {
    same_shape(
        "dotted path",
        false,
        &[
            "MATCH (u:P) WHERE u.m.city = 'c5' RETURN count(*) AS c",
            "MATCH (u:P) WHERE 'c5' = u.m.city RETURN count(*) AS c",
            "MATCH (u:P) WHERE u.m.city IN ['c5'] RETURN count(*) AS c",
        ],
    );
}

#[test]
fn edge_property_spellings_collapse() {
    same_shape(
        "edge property",
        true,
        &[
            "MATCH ()-[e:R]->() WHERE e.w = 5 RETURN count(*) AS c",
            "MATCH ()-[e:R]->() WHERE 5 = e.w RETURN count(*) AS c",
            "MATCH ()-[e:R]->() WHERE e.w IN [5] RETURN count(*) AS c",
            "MATCH ()-[e:R {w: 5}]->() RETURN count(*) AS c",
        ],
    );
}

#[test]
fn predicates_that_must_not_seek_recognize_nothing() {
    let g = indexed();

    // These are not gaps. Each one's matches are the COMPLEMENT of something a
    // point or range seek enumerates, so lowering them to a seek would drop
    // rows — the one failure mode a normalization pass makes likely.
    for q in [
        "MATCH (u:P) WHERE u.k <> $a RETURN count(*) AS c",
        "MATCH (u:P) WHERE u.k IS NULL RETURN count(*) AS c",
        "MATCH (u:P) WHERE NOT u.k = $a RETURN count(*) AS c",
        "MATCH (u:P) WHERE u.k IN [$a, $b] OR u.n <> 3 RETURN count(*) AS c",
    ] {
        assert!(
            seek_of(&g, q, false).is_empty(),
            "`{q}` must not lower to a seek"
        );
    }
}

#[test]
fn one_unseekable_branch_abandons_the_whole_disjunction() {
    let g = indexed();

    // `nope` is unindexed, so its branch cannot seek. Unlike a conjunct, that
    // has to abandon the union: seeding from the other branch alone would
    // silently drop every row the unseekable branch matched.
    assert!(seek_of(
        &g,
        "MATCH (u:P) WHERE u.k = $a OR u.nope = 'x' RETURN count(*) AS c",
        false
    )
    .is_empty());
}

#[test]
fn an_unseekable_conjunct_leaves_the_seekable_one_alone() {
    let g = indexed();

    // A conjunct is the opposite case: every conjunct is a necessary condition,
    // so ignoring one still leaves a valid candidate superset.
    assert_eq!(
        seek_of(
            &g,
            "MATCH (u:P) WHERE u.k = $a AND u.nope = 'x' RETURN count(*) AS c",
            false
        ),
        seek_of(&g, "MATCH (u:P) WHERE u.k = $a RETURN count(*) AS c", false)
    );
}

#[test]
fn conjunct_order_does_not_change_the_shape() {
    let g = indexed();

    assert_eq!(
        seek_of(
            &g,
            "MATCH (u:P) WHERE u.n >= 0 AND u.k = $a RETURN count(*) AS c",
            false
        )
        .resolve(&indexed(), &|_: usize| None)
        .map(|ids| ids.len()),
        seek_of(
            &g,
            "MATCH (u:P) WHERE u.k = $a AND u.n >= 0 RETURN count(*) AS c",
            false
        )
        .resolve(&indexed(), &|_: usize| None)
        .map(|ids| ids.len())
    );
}
