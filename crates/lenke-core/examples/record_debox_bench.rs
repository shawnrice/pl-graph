//! Record-typed constraints, step 2: what declaring a `RECORD` type constraint
//! buys. The
//! same graph, read boxed (a `Value::Map` per vertex in a `Mixed` column) vs
//! de-boxed (the constraint scatters each field into a typed sub-column —
//! `Column::Record`). We measure the three things the de-boxing is supposed to
//! improve: memory, a whole-map read, and a single-field read (`n.meta.city`).
//! Run:
//!   cargo run --release --example record_debox_bench

use std::time::Instant;

use lenke_core::gql::eval::Params;
use lenke_core::gql::prepare;
use lenke_core::graph::{Builder, Graph, NodeRec, Value};

const N: usize = 50_000;

fn map(pairs: &[(&str, Value)]) -> Value {
    Value::Map(
        pairs
            .iter()
            .map(|(k, v)| ((*k).into(), v.clone()))
            .collect(),
    )
}

/// Every vertex carries `meta = {a,b,c,d,city,tier}` — a 6-field record.
fn build() -> Graph {
    let mut b = Builder::default();
    for i in 0..N {
        b.nodes.push(NodeRec::owned(
            format!("p{i}"),
            vec!["P".to_string()],
            vec![(
                "meta".to_string(),
                map(&[
                    ("a", Value::Num((i % 100) as f64)),
                    ("b", Value::Num((i % 7) as f64)),
                    ("c", Value::Bool(i % 2 == 0)),
                    ("d", Value::Str(format!("s{}", i % 20).into())),
                    ("city", Value::Str(format!("c{}", i % 50).into())),
                    ("tier", Value::Num((i % 5) as f64)),
                ]),
            )],
        ));
    }
    b.finalize()
}

const RECORD_TYPE: &str =
    "record{a::number,b::number,c::boolean,d::string,city::string,tier::number}";

fn bench(g: &mut Graph, q: &str, iters: u32) -> f64 {
    let plan = prepare(q).unwrap();
    let params = Params::new();
    let _ = plan.execute(g, &params).unwrap();
    let t = Instant::now();
    for _ in 0..iters {
        let _ = plan.execute(g, &params).unwrap();
    }
    t.elapsed().as_secs_f64() * 1e6 / iters as f64
}

fn run(g: &mut Graph, label: &str) {
    let whole = bench(g, "MATCH (n:P) RETURN n.meta AS m", 20);
    let field = bench(g, "MATCH (n:P) RETURN n.meta.city AS c", 20);
    let filt = bench(
        g,
        "MATCH (n:P) WHERE n.meta.city = 'c0' RETURN n.meta.tier AS t",
        20,
    );
    let (heap, _) = g.vertex_prop_bytes("meta").unwrap();
    println!(
        "{label:<9} whole-map {whole:>8.0} us   field {field:>8.0} us   nested-WHERE {filt:>8.0} us   meta heap {:>6.2} MB",
        heap as f64 / 1e6,
    );
}

fn main() {
    let mut g = build();
    eprintln!("built {N} vertices, 6-field `meta` record each\n");

    // Boxed baseline: the column is a `Mixed` of `Value::Map`.
    run(&mut g, "boxed");

    // Declaring the constraint de-boxes `meta` into six typed sub-columns.
    g.create_type_constraint("P", "meta", RECORD_TYPE).unwrap();
    run(&mut g, "de-boxed");

    // A truer memory picture: `vertex_prop_bytes` reports only the boxed slot, not
    // the per-vertex heap `Vec` of pairs it points at. The real boxed footprint
    // includes ~6 × (Arc<str> + Value) per vertex on top of the slot.
    let slot = std::mem::size_of::<Option<Value>>();
    let pair = std::mem::size_of::<(std::sync::Arc<str>, Value)>();
    let boxed_true = N * (slot + 6 * pair);
    eprintln!(
        "\nboxed real footprint ≈ {:.2} MB (slot {} B + 6 × {} B pair per vertex)",
        boxed_true as f64 / 1e6,
        slot,
        pair,
    );
}
