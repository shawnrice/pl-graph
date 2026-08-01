//! Codec throughput for map/record properties. Two graphs of the same size —
//! one with a wide `meta` RECORD per vertex, one with the same fields as flat
//! SCALAR columns — isolate the map overhead in encode + decode across the
//! structured codecs. Run:
//!   cargo run --release --example map_codec_bench

use std::time::Instant;

use lenke_core::codec::{deserialize, serialize};
use lenke_core::graph::{Builder, Graph, NodeRec, Value};

const N: usize = 50_000;

fn map(pairs: Vec<(&str, Value)>) -> Value {
    Value::Map(pairs.into_iter().map(|(k, v)| (k.into(), v)).collect())
}

/// Each vertex carries `meta = {a,b,c,d,city,tier}` — a 6-field record.
fn build_maps() -> Graph {
    let mut b = Builder::default();
    for i in 0..N {
        b.nodes.push(NodeRec::owned(
            format!("p{i}"),
            vec!["P".to_string()],
            vec![(
                "meta".to_string(),
                map(vec![
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

/// The same six fields as flat scalar columns (the SoA-column fast path).
fn build_scalars() -> Graph {
    let mut b = Builder::default();
    for i in 0..N {
        b.nodes.push(NodeRec::owned(
            format!("p{i}"),
            vec!["P".to_string()],
            vec![
                ("a".to_string(), Value::Num((i % 100) as f64)),
                ("b".to_string(), Value::Num((i % 7) as f64)),
                ("c".to_string(), Value::Bool(i % 2 == 0)),
                ("d".to_string(), Value::Str(format!("s{}", i % 20).into())),
                (
                    "city".to_string(),
                    Value::Str(format!("c{}", i % 50).into()),
                ),
                ("tier".to_string(), Value::Num((i % 5) as f64)),
            ],
        ));
    }
    b.finalize()
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn main() {
    let maps = build_maps();
    let scalars = build_scalars();
    eprintln!("graphs: {N} vertices each (record vs flat scalar), 6 fields\n");

    // pg-text/csv reject maps, so only the structured codecs carry them.
    let formats = ["ndjson", "pg-json", "graphson"];

    println!(
        "{:<10} {:>12} {:>12} {:>10}   {:>12} {:>12} {:>10}",
        "codec", "map enc ms", "sca enc ms", "enc x", "map dec ms", "sca dec ms", "dec x"
    );
    println!("{}", "-".repeat(84));

    for fmt in formats {
        // Encode (median of 5).
        let (mut me, mut se) = (f64::MAX, f64::MAX);
        let (mut map_blob, mut sca_blob) = (String::new(), String::new());
        for _ in 0..5 {
            let t = Instant::now();
            map_blob = serialize(&maps, fmt).unwrap();
            me = me.min(ms(t));
            let t = Instant::now();
            sca_blob = serialize(&scalars, fmt).unwrap();
            se = se.min(ms(t));
        }
        // Decode (median of 5).
        let (mut md, mut sd) = (f64::MAX, f64::MAX);
        for _ in 0..5 {
            let t = Instant::now();
            let _ = deserialize(&map_blob, fmt).unwrap();
            md = md.min(ms(t));
            let t = Instant::now();
            let _ = deserialize(&sca_blob, fmt).unwrap();
            sd = sd.min(ms(t));
        }
        println!(
            "{fmt:<10} {me:>12.1} {se:>12.1} {:>9.1}x   {md:>12.1} {sd:>12.1} {:>9.1}x",
            me / se,
            md / sd
        );
        eprintln!(
            "  ({fmt}) blob sizes: map {:.1} MB, scalar {:.1} MB",
            map_blob.len() as f64 / 1e6,
            sca_blob.len() as f64 / 1e6
        );
    }
}
