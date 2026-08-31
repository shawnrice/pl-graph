//! Value-type cost — what a map/record property costs against the same fields as
//! flat scalar columns, stored and through the NDJSON codec. Consolidates the
//! retired `map_bench` and `map_codec_bench`.
//!
//! (Temporal-column cost — `temporal_bench` — needs host-side `Temporal`
//! construction; it is deferred rather than faked here. See examples/README.md.)
//!
//! Cases (filter with `-- <name>`):
//!   maps   — a scalar query (age filter) on a flat vs a map-carrying graph (the
//!            regression baseline: a map must not slow the scalar columns), and a
//!            dotted-path map-field query (`p.meta.city`).
//!   codec  — NDJSON decode + encode of a map graph vs the same fields flat: the
//!            map overhead in the codec.

use crate::harness::{best_ms, section, Cfg};
use lenke_engine::ndjson::{from_ndjson, to_ndjson};
use lenke_engine::store::Store;

const CITIES: [&str; 4] = ["NYC", "LA", "SF", "CHI"];

fn flat_ndjson(n: usize) -> String {
    (0..n)
        .map(|i| {
            format!(
                r#"{{"type":"node","id":"v{i}","labels":["P"],"properties":{{"age":{},"city":"{}","tier":{}}}}}"#,
                i % 100,
                CITIES[i % 4],
                i % 3
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn map_ndjson(n: usize) -> String {
    (0..n)
        .map(|i| {
            format!(
                r#"{{"type":"node","id":"v{i}","labels":["P"],"properties":{{"age":{},"meta":{{"city":"{}","tier":{}}}}}}}"#,
                i % 100,
                CITIES[i % 4],
                i % 3
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn time_gql(q: &str, store: &Store, reps: usize) -> Result<f64, String> {
    let plan = lenke_engine::opt::optimize_indexed(lenke_engine::gql::parse(q)?, store);
    Ok(best_ms(reps, || lenke_engine::exec::run(&plan, store)) * 1e3)
}

pub fn run(cfg: &Cfg) {
    let n = cfg.scale.unwrap_or(200_000).min(200_000);

    if cfg.want("value/maps") {
        section("value/maps (scalar regression + map-field access)");
        let flat = from_ndjson(&flat_ndjson(n)).expect("decodes");
        let maps = from_ndjson(&map_ndjson(n)).expect("decodes");
        let scalar = "MATCH (p:P) WHERE p.age > 50 RETURN count(*) AS c";
        let field = "MATCH (p:P) WHERE p.meta.city = 'NYC' RETURN count(*) AS c";
        println!("{:30} {:>10}", "query", "us");
        for (name, q, store) in [
            ("scalar age (flat)", scalar, &flat),
            ("scalar age (map graph)", scalar, &maps),
            ("map field meta.city", field, &maps),
        ] {
            match time_gql(q, store, cfg.reps) {
                Ok(us) => println!("{name:30} {us:>10.1}"),
                Err(e) => println!("{name:30} {:>10}  ({})", "n/a", e.trim()),
            }
        }
    }

    if cfg.want("value/codec") {
        section("value/codec (NDJSON round-trip: flat vs map)");
        println!(
            "{:22} {:>10} {:>10} {:>8}",
            "graph", "decode_ms", "encode_ms", "MiB"
        );
        for (name, text) in [
            ("flat scalars", flat_ndjson(n)),
            ("meta record", map_ndjson(n)),
        ] {
            let mib = text.len() as f64 / (1024.0 * 1024.0);
            let decode_ms = best_ms(cfg.reps, || from_ndjson(&text).expect("decodes"));
            let store = from_ndjson(&text).expect("decodes");
            let encode_ms = best_ms(cfg.reps, || to_ndjson(&store));
            println!("{name:22} {decode_ms:>10.2} {encode_ms:>10.2} {mib:>8.1}");
        }
    }
}
