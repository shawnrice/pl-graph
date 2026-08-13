//! Native core-vs-engine probe for the SHAPES THE ENGINE TRAILS on (from the
//! cross-engine FFI/wasm comparison): grouped aggregation, var-length repeat, and
//! label / edge-hop materialization. The FFI harness (`packages/native`) found the
//! gaps; this isolates them natively — no FFI, no JSON on the engine side — so a
//! profiler sees only the engine's own execution and iteration is fast.
//!
//!   cargo run --release --example hotpath_probe                 # all shapes
//!   PROBE=groupcount cargo run --release --example hotpath_probe # one, many reps
//!
//! Core is timed MATERIALIZED (`results_to_json`) to match the harness — else an
//! element-returning shape measures work core merely deferred behind a lazy handle.
//! Under `perf`, set PROBE to a single shape and raise PROBE_REPS.

use std::hint::black_box;
use std::time::Instant;

const CITIES: &[&str] = &["oslo", "bergen", "trondheim", "tromso", "stavanger"];

fn fixture_ndjson(n: u32, deg: u32) -> (String, String) {
    let mut es = String::new();
    let mut cs = String::new();
    for i in 0..n {
        let labels = if i % 3 == 0 {
            r#"["Person","VIP"]"#
        } else {
            r#"["Person"]"#
        };
        let props = format!(
            r#""name":"n{i}","age":{},"score":{},"city":"{}""#,
            i % 100,
            i.wrapping_mul(7) % 1000,
            CITIES[(i % CITIES.len() as u32) as usize],
        );
        es.push_str(&format!(
            r#"{{"id":"{i}","labels":{labels},"props":{{{props}}}}}"#
        ));
        es.push('\n');
        cs.push_str(&format!(
            r#"{{"type":"node","id":"{i}","labels":{labels},"properties":{{{props}}}}}"#
        ));
        cs.push('\n');
    }
    let mut e = 0u64;
    let mut push_edge = |es: &mut String, cs: &mut String, from: u32, to: u32, ty: &str| {
        let w = e % 1000;
        es.push_str(&format!(
            r#"{{"id":"e{e}","from":"{from}","to":"{to}","type":"{ty}","props":{{"w":{w}}}}}"#
        ));
        es.push('\n');
        cs.push_str(&format!(
            r#"{{"type":"edge","id":"e{e}","labels":["{ty}"],"from":"{from}","to":"{to}","properties":{{"w":{w}}}}}"#
        ));
        cs.push('\n');
        e += 1;
    };
    for i in 0..n {
        for d in 0..deg {
            let to = (i
                .wrapping_mul(7)
                .wrapping_add(d.wrapping_mul(13))
                .wrapping_add(1))
                % n;
            push_edge(&mut es, &mut cs, i, to, "R");
        }
        if i % 5 == 0 {
            push_edge(
                &mut es,
                &mut cs,
                i,
                (i.wrapping_mul(3).wrapping_add(2)) % n,
                "F",
            );
        }
    }
    (es, cs)
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn time_engine(store: &lenke_engine::store::Store, q: &str, reps: usize) -> (f64, usize) {
    let plan = lenke_engine::gremlin::parse(q).expect("engine parse");
    let plan = lenke_engine::opt::optimize_indexed(plan, store);
    let mut best = f64::MAX;
    let mut rows = 0;
    for _ in 0..reps {
        let t = Instant::now();
        let out = lenke_engine::exec::run(&plan, store);
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        rows = out.rows.len();
        black_box(&out);
    }
    (best, rows)
}

fn time_core(graph: &mut lenke_core::graph::Graph, q: &str, reps: usize) -> (f64, usize) {
    let t = lenke_core::gremlin::parse(q).expect("core parse");
    let mut best = f64::MAX;
    let mut rows = 0;
    for _ in 0..reps {
        let ti = Instant::now();
        let rs = lenke_core::gremlin::try_run(graph, &t).expect("core exec");
        let s = lenke_core::gremlin::exec::results_to_json(graph, &rs);
        black_box(&s);
        best = best.min(ti.elapsed().as_secs_f64() * 1e3);
        rows = rs.len();
    }
    (best, rows)
}

fn main() {
    let n = env_u32("PROBE_N", 100_000);
    let deg = env_u32("PROBE_DEG", 4);
    let reps = env_u32("PROBE_REPS", 15) as usize;
    let only = std::env::var("PROBE").ok();

    let shapes: &[(&str, &str)] = &[
        ("groupcount", "g.V().groupCount().by('city')"),
        ("repeat", "g.V().repeat(__.out()).times(2).count()"),
        (
            "repeat-dedup",
            "g.V().repeat(__.both()).times(2).dedup().count()",
        ),
        ("id-label", "g.V().label()"),
        (
            "gql-hop",
            "g.V().hasLabel('Person').outE('R').inV().count()",
        ),
        ("values-hop", "g.V().out().values('name')"),
    ];

    eprintln!("building fixture: {n} vertices, out-degree {deg}…");
    let (es, cs) = fixture_ndjson(n, deg);
    let store = lenke_engine::ndjson::from_ndjson(&es).expect("engine fixture");
    let mut graph = lenke_core::ndjson::decode(&cs).expect("core fixture");
    eprintln!(
        "graph: {} vertices, {} edges\n",
        store.node_count(),
        store.edge_count()
    );

    println!(
        "{:>8}  {:>10}  {:>10}  {:>7}  shape",
        "ratio", "core ms", "engine ms", "rows"
    );
    println!("{}", "-".repeat(70));
    for (tag, q) in shapes {
        if let Some(o) = &only {
            if o != tag {
                continue;
            }
        }
        // Warm both, then time.
        let _ = time_core(&mut graph, q, 1);
        let _ = time_engine(&store, q, 1);
        let (c, cr) = time_core(&mut graph, q, reps);
        let (e, er) = time_engine(&store, q, reps);
        let flag = if c / e >= 1.0 { ' ' } else { '!' };
        println!(
            "{flag}{:>6.2}x  {c:>10.3}  {e:>10.3}  {er:>7}  {tag}{}",
            c / e,
            if cr == er {
                String::new()
            } else {
                format!("  (core rows={cr}!)")
            }
        );
    }
}
