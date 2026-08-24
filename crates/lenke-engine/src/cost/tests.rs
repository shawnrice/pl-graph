use super::*;
use crate::ir::{Dir, PathMode};
use crate::store::{Builder, Store};
use crate::value::Value;

const CITIES: &[&str] = &["oslo", "bergen", "troms", "moss"];

fn fixture(n: u32, deg: u32) -> Store {
    let mut b = Builder::default();
    for i in 0..n {
        let city = Value::Str(CITIES[(i % 4) as usize].into());
        if i % 3 == 0 {
            b.node(&["Person", "VIP"], &[("city", city)]);
        } else {
            b.node(&["Person"], &[("city", city)]);
        }
    }
    for i in 0..n {
        for d in 0..deg {
            b.edge(i, (i * 7 + d * 13 + 1) % n, "R");
        }
    }
    b.build()
}

fn scan(label: &str) -> Plan {
    Plan::Scan {
        label: Some(label.into()),
    }
}

#[test]
fn base_scans_are_exact() {
    let st = fixture(3000, 4);
    let all = estimate(&scan("Person"), &st);
    assert!(all.exact);
    assert_eq!(all.rows as u32, 3000);
    // Every third node is a VIP.
    let vip = estimate(&scan("VIP"), &st);
    assert!(vip.exact);
    assert_eq!(vip.rows as u32, st.nodes_with_label("VIP").len() as u32);
}

#[test]
fn expand_and_varlength_scale_by_fanout() {
    let (n, deg) = (3000u32, 4u32);
    let st = fixture(n, deg);
    // avg_degree = edges/nodes = deg. A 1-hop ≈ nodes × deg.
    let one = estimate(&scan("Person").expand(0, Dir::Out, &["R".to_string()]), &st);
    assert!(!one.exact);
    assert!((one.rows - f64::from(n * deg)).abs() < 1.0, "1-hop ≈ n*deg");
    // VarLength {1,2}: n × (deg + deg²).
    let two = estimate(
        &scan("Person").var_length(0, Dir::Out, &["R".to_string()], 1, 2, PathMode::Trail),
        &st,
    );
    let expected = f64::from(n) * (f64::from(deg) + f64::from(deg * deg));
    assert!((two.rows - expected).abs() < 2.0, "varlen ≈ n*(d+d²)");
    // The deep intermediate (n×(d+d²) = 60k) routes to bounded memory under a
    // budget below it.
    let budget = Budget {
        materialize_rows: 50_000.0,
    };
    assert!(prefer_bounded_memory(
        &scan("Person").var_length(0, Dir::Out, &["R".to_string()], 1, 2, PathMode::Trail),
        &st,
        &budget
    ));
}

#[test]
fn grouped_aggregate_uses_exact_dict_distinct() {
    let st = fixture(3000, 4);
    // `city` has 4 distinct values → dict-encoded → exact group count.
    assert_eq!(st.distinct_count("city"), Some(4));
    let plan = scan("Person").aggregate(vec![("k".into(), prop(0, "city"))], vec![agg_count()]);
    let g = estimate(&plan, &st);
    assert_eq!(g.rows as u32, 4, "group count = dict distinct");
}

#[test]
fn scalar_aggregate_is_one_row() {
    let st = fixture(1000, 4);
    let plan = scan("Person").aggregate(Vec::new(), vec![agg_count()]);
    let c = estimate(&plan, &st);
    assert!(c.exact);
    assert_eq!(c.rows as u32, 1);
}

#[test]
fn limit_caps_the_estimate() {
    let st = fixture(3000, 4);
    let plan = scan("Person")
        .expand(0, Dir::Out, &["R".to_string()])
        .order_page(Vec::new(), None, Some(50));
    assert_eq!(estimate(&plan, &st).rows as u32, 50);
}

// --- tiny plan-builder helpers ---
fn prop(slot: usize, key: &str) -> Expr {
    Expr::Prop {
        slot,
        key: key.into(),
    }
}
fn agg_count() -> crate::ir::Agg {
    crate::ir::Agg {
        func: crate::ir::AggFn::Count,
        arg: None,
        distinct: false,
        name: "c".into(),
        frac: None,
        null_on_empty: false,
        numeric_only: false,
    }
}
