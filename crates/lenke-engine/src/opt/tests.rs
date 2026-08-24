use super::*;
use crate::exec::{run, Rows};
use crate::ir::{CompareOp, Dir, Expr, PathMode, Plan};
use crate::store::{Builder, Store};
use crate::value::Value;
use std::sync::Arc;

fn s(x: &str) -> Value {
    Value::Str(Arc::from(x))
}
fn n(x: f64) -> Value {
    Value::Num(x)
}
fn prop(slot: usize, key: &str) -> Expr {
    Expr::Prop {
        slot,
        key: key.to_string(),
    }
}
fn cmp(op: CompareOp, l: Expr, r: Expr) -> Expr {
    Expr::Compare {
        op,
        left: Box::new(l),
        right: Box::new(r),
    }
}

fn social() -> Store {
    let mut b = Builder::default();
    let a = b.node(&["Person"], &[("name", s("alice")), ("age", n(30.0))]);
    let bob = b.node(&["Person"], &[("name", s("bob")), ("age", n(25.0))]);
    let c = b.node(&["Person"], &[("name", s("carol")), ("age", n(40.0))]);
    b.edge(a, bob, "KNOWS");
    b.edge(a, c, "KNOWS");
    b.edge(bob, c, "KNOWS");
    b.build()
}

fn bag(rows: &Rows) -> Vec<String> {
    let mut out: Vec<String> = rows
        .rows
        .iter()
        .map(|r| r.iter().map(|v| format!("{v:?};")).collect::<String>())
        .collect();
    out.sort();
    out
}

/// Optimizing must never change the answer — the core invariant.
fn assert_rows_preserved(plan: &Plan, store: &Store) -> Plan {
    let before = bag(&run(plan, store));
    let opt = optimize(plan.clone());
    assert_eq!(before, bag(&run(&opt, store)), "optimize changed the rows");
    opt
}

/// `Scan(label) + Filter(prop = lit)` seeds to `IndexSeek` — for BOTH
/// spellings, which must land on the SAME seek target (the
/// equivalent-spellings-cost-the-same invariant) and preserve the rows.
#[test]
fn scan_filter_eq_seeds_index_both_spellings() {
    let store = social();
    let plan_of = |pred| {
        Plan::Scan {
            label: Some("Person".into()),
        }
        .filter(pred)
        .project(vec![("name".into(), prop(0, "name"))])
    };
    let a = plan_of(cmp(CompareOp::Eq, prop(0, "name"), Expr::Lit(s("alice"))));
    let b = plan_of(cmp(CompareOp::Eq, Expr::Lit(s("alice")), prop(0, "name")));

    let oa = assert_rows_preserved(&a, &store);
    let ob = assert_rows_preserved(&b, &store);

    // Both spellings become Project{ IndexSeek{Person, name, "alice"} }.
    let target = |p: &Plan| -> (String, String, Value) {
        let Plan::Project { input, .. } = p else {
            panic!("expected Project, got {p:?}")
        };
        let Plan::IndexSeek { label, key, value } = input.as_ref() else {
            panic!("expected IndexSeek under Project, got {input:?}")
        };
        (label.clone(), key.clone(), value.clone())
    };
    let (la, ka, va) = target(&oa);
    let (lb, kb, vb) = target(&ob);
    assert_eq!((la.as_str(), ka.as_str()), ("Person", "name"));
    assert_eq!((la, ka), (lb, kb)); // identical target for both spellings
    assert!(matches!(&va, Value::Str(x) if &**x == "alice"));
    assert!(matches!(&vb, Value::Str(x) if &**x == "alice"));
    assert_eq!(bag(&run(&oa, &store)), vec!["Str(\"alice\");"]);
}

/// A dotted record-field equality `n.meta.city = 'NYC'` seeds a dotted
/// `IndexSeek` — both spellings land the SAME target — and gives the right
/// rows both WITH an index (index_lookup) and WITHOUT (the path-resolving scan
/// fallback).
#[test]
fn dotted_field_eq_seeds_index_both_spellings() {
    use crate::value::make_record;
    use std::sync::Arc;
    let build = || {
        let mut b = Builder::default();
        b.node(
            &["Person"],
            &[
                ("name", s("alice")),
                ("meta", make_record(vec![(Arc::from("city"), s("NYC"))])),
            ],
        );
        b.node(
            &["Person"],
            &[
                ("name", s("bob")),
                ("meta", make_record(vec![(Arc::from("city"), s("LA"))])),
            ],
        );
        b.build()
    };
    let field = Expr::Field {
        base: Box::new(prop(0, "meta")),
        key: "city".into(),
    };
    let plan_of = |pred| {
        Plan::Scan {
            label: Some("Person".into()),
        }
        .filter(pred)
        .project(vec![("name".into(), prop(0, "name"))])
    };
    let a = plan_of(cmp(CompareOp::Eq, field.clone(), Expr::Lit(s("NYC"))));
    let b = plan_of(cmp(CompareOp::Eq, Expr::Lit(s("NYC")), field.clone()));

    let mut store = build();
    store.create_index("meta.city");
    let oa = assert_rows_preserved(&a, &store);
    let ob = assert_rows_preserved(&b, &store);

    let target = |p: &Plan| -> (String, Value) {
        let Plan::Project { input, .. } = p else {
            panic!("expected Project, got {p:?}")
        };
        let Plan::IndexSeek { key, value, .. } = input.as_ref() else {
            panic!("expected a (dotted) IndexSeek, got {input:?}")
        };
        (key.clone(), value.clone())
    };
    let (ka, va) = target(&oa);
    let (kb, _) = target(&ob);
    assert_eq!(ka, "meta.city");
    assert_eq!(ka, kb); // both spellings, same dotted target
    assert!(matches!(&va, Value::Str(x) if &**x == "NYC"));
    assert_eq!(bag(&run(&oa, &store)), vec!["Str(\"alice\");"]);

    // Same seed, NO index: the scan fallback resolves the path and matches.
    let no_index = build();
    let oc = optimize(a.clone());
    assert!(
        matches!(&oc, Plan::Project { input, .. } if matches!(input.as_ref(), Plan::IndexSeek { .. }))
    );
    assert_eq!(bag(&run(&oc, &no_index)), vec!["Str(\"alice\");"]);
}

/// A range filter is NOT seeded (that is D2); an unlabelled scan cannot seek.
/// An unlabelled scan cannot seek (IndexSeek/RangeSeek need a label), so its
/// filter is preserved. (A labelled range filter now seeds — see the range
/// seed test.)
#[test]
fn unlabelled_scan_not_seeded() {
    let store = social();
    let unlabelled = Plan::Scan { label: None }
        .filter(cmp(CompareOp::Eq, prop(0, "name"), Expr::Lit(s("alice"))))
        .project(vec![("name".into(), prop(0, "name"))]);
    assert!(plan_contains_filter(&assert_rows_preserved(
        &unlabelled,
        &store
    )));
}

/// A range filter over a labelled scan seeds to a `RangeSeek`, for BOTH
/// spellings (`age > 28` and `28 < age`), which land the SAME target and
/// preserve rows (equivalent-spellings for ranges).
#[test]
fn range_filter_seeds_both_spellings() {
    // A RangeSeek is only planned when a range index can serve it (else the
    // vectorized Filter path is faster than the seek's scan-and-box fallback),
    // so this fixture builds one on `age`.
    let mut store = social();
    store.create_range_index("age");
    let plan_of = |pred| {
        Plan::Scan {
            label: Some("Person".into()),
        }
        .filter(pred)
        .project(vec![("name".into(), prop(0, "name"))])
    };
    // age > 28  and  28 < age  are the same predicate, different spelling.
    let a = plan_of(cmp(CompareOp::Gt, prop(0, "age"), Expr::Lit(n(28.0))));
    let b = plan_of(cmp(CompareOp::Lt, Expr::Lit(n(28.0)), prop(0, "age")));
    // Optimize through the indexed path (both spellings must normalize to the
    // same RangeSeek) and confirm the rows are unchanged from the raw plan.
    let oa = optimize_indexed(a.clone(), &store);
    let ob = optimize_indexed(b.clone(), &store);
    assert_eq!(bag(&run(&a, &store)), bag(&run(&oa, &store)));
    assert_eq!(bag(&run(&b, &store)), bag(&run(&ob, &store)));

    let target = |p: &Plan| -> (String, CompareOp, Value) {
        let Plan::Project { input, .. } = p else {
            panic!("expected Project, got {p:?}")
        };
        let Plan::RangeSeek { key, op, value, .. } = input.as_ref() else {
            panic!("expected RangeSeek under Project, got {input:?}")
        };
        (key.clone(), *op, value.clone())
    };
    let (ka, opa, va) = target(&oa);
    let (kb, opb, vb) = target(&ob);
    assert_eq!(ka, "age");
    assert_eq!(opa, CompareOp::Gt); // both normalize to prop > 28
    assert_eq!((ka, opa), (kb, opb));
    assert!(matches!(va, Value::Num(x) if x == 28.0));
    assert!(matches!(vb, Value::Num(x) if x == 28.0));
    // answer: alice(30), carol(40)
    let mut got = bag(&run(&oa, &store));
    got.sort();
    assert_eq!(got, vec!["Str(\"alice\");", "Str(\"carol\");"]);
}

#[test]
fn pushdown_below_expand() {
    let store = social();
    // Filter on slot 0 (the source) sits above the Expand; it should move
    // below it.
    let plan = Plan::Scan {
        label: Some("Person".into()),
    }
    .expand(0, Dir::Out, &["KNOWS".to_string()])
    .filter(cmp(CompareOp::Eq, prop(0, "name"), Expr::Lit(s("alice"))));
    let opt = assert_rows_preserved(&plan, &store);
    // Shape: Expand{ input: <pushed-down predicate> }. The pushed filter over
    // Scan(label) then seeds to an IndexSeek — either form proves the
    // predicate moved below the Expand.
    match opt {
        Plan::Expand { input, .. } => {
            assert!(
                matches!(*input, Plan::Filter { .. } | Plan::IndexSeek { .. }),
                "predicate now below expand (as Filter or IndexSeek)"
            );
        }
        other => panic!("expected Expand at top, got {other:?}"),
    }
}

#[test]
fn no_pushdown_when_predicate_reads_the_expanded_slot() {
    let store = social();
    // Filter on slot 1 (the expanded neighbour) cannot move below the Expand.
    let plan = Plan::Scan {
        label: Some("Person".into()),
    }
    .expand(0, Dir::Out, &["KNOWS".to_string()])
    .filter(cmp(CompareOp::Ge, prop(1, "age"), Expr::Lit(n(40.0))));
    let opt = assert_rows_preserved(&plan, &store);
    // Shape unchanged: Filter still on top of Expand.
    match opt {
        Plan::Filter { input, .. } => {
            assert!(
                matches!(*input, Plan::Expand { .. }),
                "filter stays above expand"
            );
        }
        other => panic!("expected Filter at top, got {other:?}"),
    }
}

#[test]
fn varlen_split_pushdown_source_below_target_above() {
    let store = social();
    // `a.name = 'alice' AND b.age >= 40` over a var-length hop: the source
    // conjunct (slot 0) pushes below the VarLength; the target conjunct (slot 1,
    // the appended endpoint) stays above. Rows must be unchanged.
    let plan = Plan::Scan {
        label: Some("Person".into()),
    }
    .var_length(0, Dir::Out, &["KNOWS".to_string()], 1, 2, PathMode::Trail)
    .filter(Expr::And(
        Box::new(cmp(CompareOp::Eq, prop(0, "name"), Expr::Lit(s("alice")))),
        Box::new(cmp(CompareOp::Ge, prop(1, "age"), Expr::Lit(n(40.0)))),
    ));
    let opt = assert_rows_preserved(&plan, &store);
    // Shape: Filter{target} over VarLength{ input: <pushed source> }.
    match opt {
        Plan::Filter { input, pred } => {
            assert!(!refs_below(&pred, 1), "the residual reads the target slot");
            let Plan::VarLength { input: vin, .. } = *input else {
                panic!("expected VarLength under the residual filter");
            };
            assert!(
                matches!(*vin, Plan::Filter { .. } | Plan::IndexSeek { .. }),
                "the source predicate moved below the VarLength"
            );
        }
        other => panic!("expected a residual Filter on top, got {other:?}"),
    }
}

#[test]
fn shortest_path_source_filter_pushes_down() {
    let store = social();
    // A pure source filter over ShortestPath pushes fully below it (no residual).
    let plan = Plan::Scan {
        label: Some("Person".into()),
    }
    .shortest_path(
        0,
        Dir::Out,
        &["KNOWS".to_string()],
        1,
        None,
        crate::ir::ShortestSelector::Any,
        None,
    )
    .filter(cmp(CompareOp::Eq, prop(0, "name"), Expr::Lit(s("alice"))));
    let opt = assert_rows_preserved(&plan, &store);
    match opt {
        Plan::ShortestPath { input, .. } => assert!(
            matches!(*input, Plan::Filter { .. } | Plan::IndexSeek { .. }),
            "source predicate now below the ShortestPath"
        ),
        other => panic!("expected ShortestPath at top, got {other:?}"),
    }
}

#[test]
fn adjacent_filters_merge() {
    let store = social();
    // Unlabelled scan so seeding (which needs a label) does not fire — this
    // isolates the merge rule. social() is all-Person, so the rows match.
    let plan = Plan::Scan { label: None }
        .filter(cmp(CompareOp::Ge, prop(0, "age"), Expr::Lit(n(28.0))))
        .filter(cmp(CompareOp::Le, prop(0, "age"), Expr::Lit(n(35.0))));
    let opt = assert_rows_preserved(&plan, &store);
    // And the answer: only alice(30) is in [28,35].
    assert_eq!(run(&opt, &store).rows.len(), 1);
    // Shape: one Filter (an And) over the Scan.
    match &opt {
        Plan::Filter { input, pred } => {
            assert!(matches!(pred, Expr::And(..)), "merged into an AND");
            assert!(
                matches!(**input, Plan::Scan { .. }),
                "single filter over scan"
            );
        }
        other => panic!("expected a single Filter, got {other:?}"),
    }
}

#[test]
fn driver_reaches_fixpoint_merge_then_pushdown() {
    let store = social();
    // Two filters (slot 0) above an Expand: the driver must MERGE them and
    // then PUSH the merged filter below the Expand — two rules, to a fixpoint.
    // Unlabelled scan so seeding does not fire, isolating merge + pushdown.
    let plan = Plan::Scan { label: None }
        .expand(0, Dir::Out, &["KNOWS".to_string()])
        .filter(cmp(CompareOp::Le, prop(0, "age"), Expr::Lit(n(100.0))))
        .filter(cmp(CompareOp::Ge, prop(0, "age"), Expr::Lit(n(0.0))));
    let opt = assert_rows_preserved(&plan, &store);
    match opt {
        Plan::Expand { input, .. } => match *input {
            Plan::Filter { input, pred } => {
                assert!(matches!(pred, Expr::And(..)), "the two filters merged");
                assert!(matches!(*input, Plan::Scan { .. }));
            }
            other => panic!("expected merged Filter below Expand, got {other:?}"),
        },
        other => panic!("expected Expand at top, got {other:?}"),
    }
}

#[test]
fn pushdown_into_join_left() {
    let store = social();
    // A filter on a left-side slot pushes into the Join's left input.
    // Unlabelled left scan so the pushed filter stays a Filter (not seeded),
    // which is what this test checks.
    let left = Plan::Scan { label: None }.expand(0, Dir::Out, &["KNOWS".to_string()]);
    let right = Plan::Scan {
        label: Some("Person".into()),
    }
    .expand(0, Dir::Out, &["KNOWS".to_string()]);
    // left width is 2 (a, b); a filter on slot 0 (left's a) pushes left.
    let plan = Plan::join(left, right, vec![(0, 0)]).filter(cmp(
        CompareOp::Ge,
        prop(0, "age"),
        Expr::Lit(n(30.0)),
    ));
    let opt = assert_rows_preserved(&plan, &store);
    match opt {
        Plan::Join { left, .. } => {
            // The left input now begins with a Filter somewhere it was pushed
            // to (top of the left subtree, above or below its own expand).
            let has_filter = plan_contains_filter(&left);
            assert!(has_filter, "filter pushed into the left side");
        }
        other => panic!("expected Join at top, got {other:?}"),
    }
}

fn plan_contains_filter(p: &Plan) -> bool {
    match p {
        Plan::Filter { .. } => true,
        Plan::PathRecord { input, .. }
        | Plan::Expand { input, .. }
        | Plan::Unwind { input, .. }
        | Plan::OptionalExpand { input, .. }
        | Plan::IntervalExpand { input, .. }
        | Plan::VarLength { input, .. }
        | Plan::RepeatGroup { input, .. }
        | Plan::NestedGroup { input, .. }
        | Plan::ShortestPath { input, .. }
        | Plan::Aggregate { input, .. }
        | Plan::OrderPage { input, .. }
        | Plan::Project { input, .. }
        | Plan::Update { input, .. }
        | Plan::CallInline { input, .. }
        | Plan::Distinct { input }
        | Plan::DistinctBy { input, .. }
        | Plan::Tail { input, .. }
        | Plan::Sample { input, .. }
        | Plan::Branch { input, .. }
        | Plan::Reconverge { input, .. }
        | Plan::NullPadIfEmpty { input, .. }
        | Plan::OptionalScan { input, .. }
        | Plan::GroupToMap { input }
        | Plan::AlgoAnnotate { input, .. }
        | Plan::Tree { input, .. }
        | Plan::MapSlot { input, .. }
        | Plan::EdgeVertex { input, .. }
        | Plan::Enumerate { input, .. }
        | Plan::Subgraph { input, .. }
        | Plan::ShortestPathEnum { input, .. }
        | Plan::SortLocal { input, .. } => plan_contains_filter(input),
        Plan::Join { left, right, .. } | Plan::Union { left, right, .. } => {
            plan_contains_filter(left) || plan_contains_filter(right)
        }
        Plan::PerElementBranch {
            input, cond, arms, ..
        } => {
            plan_contains_filter(input)
                || cond.as_deref().is_some_and(plan_contains_filter)
                || arms.iter().any(plan_contains_filter)
        }
        Plan::InsertReturn { tail, .. } => plan_contains_filter(tail),
        Plan::UpdateReturn { input, tail, .. } => {
            plan_contains_filter(input) || plan_contains_filter(tail)
        }
        Plan::Scan { .. }
        | Plan::NodeSeed { .. }
        | Plan::EdgeScan
        | Plan::EdgeSeed { .. }
        | Plan::Row
        | Plan::IndexSeek { .. }
        | Plan::RangeSeek { .. }
        | Plan::Insert { .. }
        | Plan::InsertFrom { .. }
        | Plan::Merge { .. }
        | Plan::MergeEdge { .. }
        | Plan::AddEdge { .. }
        | Plan::CallProcedure { .. }
        | Plan::TxControl { .. } => false,
    }
}

/// Optimizing a plan from the GQL front-end preserves rows — the rules fire
/// on either language's output because the IR is neutral.
#[test]
fn optimizes_a_parsed_gql_plan() {
    let store = social();
    let plan = crate::gql::parse(
        "MATCH (a:Person)-[:KNOWS]->(b) WHERE a.name = 'alice' RETURN b.name AS b",
    )
    .unwrap();
    // The WHERE (slot 0) should end up pushed below the Expand.
    let _ = assert_rows_preserved(&plan, &store);
}

/// `count(<bound element>)` over a pure chain canonicalizes to argument-free
/// `count(*)` (same plan → same O(1) fast path), while `count(<property>)` and
/// `count(DISTINCT …)` are left alone. The spelling-perf-cliff guard.
#[test]
fn count_of_bound_element_canonicalizes_to_count_star() {
    let store = social();
    let plan_str = |q: &str| format!("{:?}", optimize(crate::gql::parse(q).unwrap()));

    // count(n) == count(*) and count(b) == count(*) (1-hop), same optimized plan.
    assert_eq!(
        plan_str("MATCH (n:Person) RETURN count(n) AS c"),
        plan_str("MATCH (n:Person) RETURN count(*) AS c"),
    );
    assert_eq!(
        plan_str("MATCH (a:Person)-[:KNOWS]->(b) RETURN count(b) AS c"),
        plan_str("MATCH (a:Person)-[:KNOWS]->(b) RETURN count(*) AS c"),
    );
    // A property count and a DISTINCT element count are NOT count(*).
    assert_ne!(
        plan_str("MATCH (n:Person) RETURN count(n.age) AS c"),
        plan_str("MATCH (n:Person) RETURN count(*) AS c"),
    );
    assert_ne!(
        plan_str("MATCH (n:Person) RETURN count(DISTINCT n) AS c"),
        plan_str("MATCH (n:Person) RETURN count(*) AS c"),
    );

    // And every one of these still returns the SAME rows before/after optimizing.
    for q in [
        "MATCH (n:Person) RETURN count(n) AS c",
        "MATCH (a:Person)-[:KNOWS]->(b) RETURN count(b) AS c",
        "MATCH (n:Person) RETURN count(n.age) AS c",
    ] {
        assert_rows_preserved(&crate::gql::parse(q).unwrap(), &store);
    }
}

/// A multi-predicate `WHERE a = x AND b = y` seeds ONE conjunct into a seek and
/// keeps the rest as a residual filter — so it costs the same as the inline
/// `(n:L {a: x, b: y})` twin (which already seeded), not a full-scan conjunction.
/// The `count(*)`-shaped fixture is the one that showed the 34x gap.
#[test]
fn multi_predicate_where_seeds_a_conjunct() {
    let store = social();
    let seeded = |q: &str| {
        let p = optimize(crate::gql::parse(q).unwrap());
        format!("{p:?}").contains("IndexSeek")
    };
    // Both AND and the inline map seed; a lone equality already did.
    assert!(seeded(
        "MATCH (n:Person) WHERE n.name = 'alice' AND n.age = 30 RETURN count(*) AS c"
    ));
    assert!(seeded(
        "MATCH (n:Person {name: 'alice', age: 30}) RETURN count(*) AS c"
    ));
    // The two spellings optimize to the SAME plan (same seek + residual).
    assert_eq!(
        format!(
            "{:?}",
            optimize(
                crate::gql::parse("MATCH (n:Person {name: 'alice', age: 30}) RETURN n.age AS a")
                    .unwrap()
            )
        ),
        format!(
            "{:?}",
            optimize(
                crate::gql::parse(
                    "MATCH (n:Person) WHERE n.name = 'alice' AND n.age = 30 RETURN n.age AS a"
                )
                .unwrap()
            )
        ),
    );
    // And the rewrite never changes the rows (a 3-conjunct case too).
    for q in [
            "MATCH (n:Person) WHERE n.name = 'alice' AND n.age = 30 RETURN n.name AS x",
            "MATCH (n:Person) WHERE n.age >= 20 AND n.age <= 40 AND n.name = 'carol' RETURN n.name AS x",
        ] {
            assert_rows_preserved(&crate::gql::parse(q).unwrap(), &store);
        }
}

/// The multi-predicate seed is INDEX-AWARE: it seeds the conjunct backed by a
/// real index (so the seek reads the index, not a scan), falling back to the
/// first seekable conjunct only when nothing is indexed. The store-less
/// `optimize` stays blind (unchanged behavior).
#[test]
fn multi_predicate_seed_prefers_the_indexed_conjunct() {
    let fresh = || {
        let mut b = Builder::default();
        for i in 0..40u32 {
            b.node(
                &["Person"],
                &[
                    ("dept", s(if i % 2 == 0 { "eng" } else { "sales" })),
                    ("age", n(f64::from(i % 10))),
                ],
            );
        }
        b.build()
    };
    let q = "MATCH (n:Person) WHERE n.dept = 'eng' AND n.age = 4 RETURN n.age AS a";
    let plan = crate::gql::parse(q).unwrap();
    let seeded_key = |store: &Store| -> Option<String> {
        fn find(p: &Plan) -> Option<String> {
            match p {
                Plan::IndexSeek { key, .. } => Some(key.clone()),
                Plan::Filter { input, .. }
                | Plan::Project { input, .. }
                | Plan::Aggregate { input, .. } => find(input),
                _ => None,
            }
        }
        find(&optimize_indexed(plan.clone(), store))
    };

    // No index → first seekable conjunct (dept).
    let mut plain = fresh();
    assert_eq!(seeded_key(&plain).as_deref(), Some("dept"));
    // Index on age → the seed switches to age (the indexed conjunct).
    plain.create_index("age");
    assert_eq!(seeded_key(&plain).as_deref(), Some("age"));
    // Index on dept instead → seeds dept.
    let mut dept_idx = fresh();
    dept_idx.create_index("dept");
    assert_eq!(seeded_key(&dept_idx).as_deref(), Some("dept"));
    // The store-less path is unchanged (blind): seeds the first conjunct.
    assert!(format!("{:?}", optimize(plan)).contains("key: \"dept\""));
}

/// Gremlin `order().by(k)` + `range(lo, hi)` lowers to two stacked OrderPages;
/// merging the page into the sort must NOT change which rows come back — the
/// delicate case is a TIE at the page boundary, where the surviving rows depend
/// on tie-breaking. A fixture with many equal keys stresses exactly that.
#[test]
fn stacked_orderpage_merge_preserves_rows_under_ties() {
    let mut b = Builder::default();
    // 12 nodes, keys in {0,1,2} — heavy ties, so a top-k boundary lands inside a
    // tie group and the merged vs stacked forms must agree on which rows win.
    for i in 0..12u32 {
        b.node(
            &["P"],
            &[("k", n(f64::from(i % 3))), ("id", n(f64::from(i)))],
        );
    }
    let store = b.build();
    // Gremlin forms that produce stacked OrderPages (sort, then page).
    for q in [
        "g.V().hasLabel('P').order().by('k', desc).range(0, 2).values('id')",
        "g.V().hasLabel('P').order().by('k').range(1, 4).values('id')",
        "g.V().hasLabel('P').order().by('k', desc).range(2, 5).values('id')",
    ] {
        let plan = crate::gremlin::parse(q).unwrap();
        // Sanity: the UNoptimized plan really is two stacked OrderPages. `values`
        // now skips absent properties, so a `Filter` (PropertyExists) sits between
        // the Project and the OrderPages — descend through it.
        let below_project = match &plan {
            Plan::Project { input, .. } => match input.as_ref() {
                Plan::Filter { input, .. } => input.as_ref(),
                other => other,
            },
            other => other,
        };
        assert!(
            matches!(below_project, Plan::OrderPage { input: inner, keys, .. }
                    if keys.is_empty() && matches!(inner.as_ref(), Plan::OrderPage { .. })),
            "expected stacked OrderPages for `{q}`"
        );
        assert_rows_preserved(&plan, &store);
    }
}
