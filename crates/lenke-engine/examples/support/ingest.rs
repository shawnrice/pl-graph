//! Ingest & footprint questions — where NDJSON decode time goes, how close it is
//! to the machine, what parallel decode/encode buys, and how much memory a graph
//! of N elements costs. Consolidates the retired `ingest_phase_bench`,
//! `ingest_throughput` and `mem_probe`, plus the parallel-serialization payoff.
//!
//! Cases (filter with `-- <name>`):
//!   phases  — total decode vs the record-materialization proxy vs teardown/free.
//!             The build phase dominates; the proxy is an UPPER bound on what
//!             borrowing every id/label/key/value from the input could recover
//!             (already partly taken — see the note in the original bench).
//!   ceiling — decode vs a raw byte scan of the same text: the distance from what
//!             the machine can do just touching the bytes.
//!   threads — from_ndjson_threads at 1/2/4/8: the parallel-parse speedup.
//!   encode  — to_ndjson vs to_ndjson_threads at 1/2/4/8: the parallel-encode
//!             speedup, and round-trip MiB/s.
//!   mem     — resident bytes per element for nodes-only and 5-edges/node graphs.

use crate::harness::{best_ms, rss_bytes, section, social_ndjson, Cfg, DEFAULT_NODES};
use lenke_engine::ndjson::{from_ndjson, from_ndjson_threads, to_ndjson, to_ndjson_threads};
use std::time::Instant;

type Record = (String, Vec<String>, Vec<(String, String)>);

pub fn run(cfg: &Cfg) {
    // Decode is linear, so no cap is needed — a large BENCH_N just decodes more.
    let base = cfg.scale.unwrap_or(DEFAULT_NODES);

    if cfg.want("ingest/phases") {
        section("ingest/phases");
        println!(
            "{:26} {:>7}  {:>10} {:>12} {:>10} {:>10}",
            "phases", "MiB", "decode_ms", "recmat_ms", "teardn_ms", "free_ms"
        );
        for (nodes, props, deg, label) in [
            (base, 1, 0, "nodes x1 prop"),
            (base, 4, 0, "nodes x4 props"),
            (base, 2, 5, "nodes+5edges x2 props"),
        ] {
            let text = social_ndjson(nodes, props, deg);
            let mib = text.len() as f64 / (1024.0 * 1024.0);

            let decode_ms = best_ms(cfg.reps, || from_ndjson(&text).expect("decodes"));

            // Stand the intermediate records up on their own (an owned String per
            // id/label/key/value) so their construction+teardown is an upper bound
            // on the record-materialization slice of the build phase.
            let recmat_ms = best_ms(cfg.reps, || {
                let recs: Vec<Record> = (0..nodes)
                    .map(|i| {
                        (
                            format!("v{i}"),
                            vec!["Person".to_string()],
                            (0..props)
                                .map(|k| (format!("k{k}"), format!("v{i}_{k}")))
                                .collect(),
                        )
                    })
                    .collect();
                recs
            });

            let g = from_ndjson(&text).expect("decodes");
            let t = Instant::now();
            drop(g);
            let free_ms = t.elapsed().as_secs_f64() * 1e3;
            // teardown of the proxy records, measured separately.
            let recs: Vec<Record> = (0..nodes)
                .map(|i| (format!("v{i}"), vec!["Person".to_string()], Vec::new()))
                .collect();
            let t = Instant::now();
            drop(recs);
            let teardn_ms = t.elapsed().as_secs_f64() * 1e3;

            println!(
                "{label:26} {mib:7.1}  {decode_ms:>10.2} {recmat_ms:>12.2} {teardn_ms:>10.2} {free_ms:>10.2}"
            );
        }
    }

    if cfg.want("ingest/ceiling") {
        let text = social_ndjson(base, 2, 5);
        let mib = text.len() as f64 / (1024.0 * 1024.0);
        let scan_ms = best_ms(cfg.reps, || {
            let bytes = text.as_bytes();
            let mut sum = 0u64;
            let mut lines = 0u64;
            for &b in bytes {
                sum = sum.wrapping_add(u64::from(b));
                lines += u64::from(b == b'\n');
            }
            (sum, lines)
        });
        let decode_ms = best_ms(cfg.reps, || from_ndjson(&text).expect("decodes"));
        section("ingest/ceiling (5 edges/node)");
        println!(
            "{:>7} {:>10} {:>10} {:>8}",
            "MiB", "rawscan_ms", "decode_ms", "ratio"
        );
        println!(
            "{mib:>7.1} {scan_ms:>10.2} {decode_ms:>10.2} {:>8.1}",
            decode_ms / scan_ms
        );
    }

    if cfg.want("ingest/threads") {
        let text = social_ndjson(base, 2, 5);
        let mib = text.len() as f64 / (1024.0 * 1024.0);
        section("ingest/threads (parallel decode)");
        println!(
            "{:>8} {:>10} {:>9} {:>10}",
            "threads", "decode_ms", "speedup", "MiB/s"
        );
        let mut serial = f64::NAN;
        for t in [1u32, 2, 4, 8] {
            let ms = best_ms(cfg.reps, || from_ndjson_threads(&text, t).expect("decodes"));
            if t == 1 {
                serial = ms;
            }
            println!(
                "{t:>8} {ms:>10.2} {:>9.2} {:>10.1}",
                serial / ms,
                mib / (ms / 1e3)
            );
        }
    }

    if cfg.want("ingest/encode") {
        let text = social_ndjson(base, 2, 5);
        let store = from_ndjson(&text).expect("decodes");
        drop(text);
        section("ingest/encode (parallel encode)");
        println!(
            "{:>8} {:>10} {:>9} {:>10}",
            "threads", "encode_ms", "speedup", "MiB/s"
        );
        let one = to_ndjson(&store);
        let mib = one.len() as f64 / (1024.0 * 1024.0);
        drop(one);
        let serial = best_ms(cfg.reps, || to_ndjson(&store));
        println!(
            "{:>8} {serial:>10.2} {:>9.2} {:>10.1}",
            1,
            1.0,
            mib / (serial / 1e3)
        );
        for t in [2u32, 4, 8] {
            let ms = best_ms(cfg.reps, || to_ndjson_threads(&store, t));
            println!(
                "{t:>8} {ms:>10.2} {:>9.2} {:>10.1}",
                serial / ms,
                mib / (ms / 1e3)
            );
        }
    }

    if cfg.want("ingest/mem") {
        section("ingest/mem (resident footprint)");
        if rss_bytes().is_none() {
            println!("(VmRSS unavailable — Linux only)");
        } else {
            println!("{:26} {:>12} {:>14}", "graph", "rss_MiB", "bytes/elem");
            for (nodes, deg, label) in [(base, 0, "nodes only"), (base, 5, "nodes + 5 edges/node")]
            {
                let text = social_ndjson(nodes, 2, deg);
                let before = rss_bytes().unwrap_or(0);
                let g = from_ndjson(&text).expect("decodes");
                drop(text);
                let after = rss_bytes().unwrap_or(before);
                let elems = (nodes + nodes * deg) as f64;
                let used = after.saturating_sub(before) as f64;
                println!(
                    "{label:26} {:>12.1} {:>14.1}",
                    used / (1024.0 * 1024.0),
                    used / elems
                );
                std::hint::black_box(&g);
                drop(g);
            }
        }
    }
}
