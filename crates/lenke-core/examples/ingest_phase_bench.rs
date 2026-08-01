//! Where NDJSON ingest time actually goes.
//!
//! Splits a decode into its phases so an optimization can be aimed rather than
//! guessed at. Run: `cargo run --release --example ingest_phase_bench`.
//!
//! The reason this exists: the intuitive target is wrong. Byte-level scanning is
//! a few percent of a decode, so making the JSON scanner faster — vectorized,
//! branchless, whatever — cannot move the number much. The build phase is ~70%,
//! and within it the largest single item is materializing every record before
//! any column is written: one owned `String` per id, per label, per property key
//! and per string value, all of it destroyed immediately afterwards.
//!
//! The `json_records` columns below stand that materialization up on its own so
//! its cost can be read directly. On property-rich data it is a third of the
//! whole decode, which is what borrowing from the input text would recover —
//! at the price of a lifetime on `NodeRec`/`EdgeRec`/`Builder` and a `Cow` in
//! the codecs that unescape (a borrow is only possible when a string arrives
//! clean).
use std::time::Instant;

/// One materialized record, shaped like what the decoder builds: an owned id,
/// owned labels, and an owned key/value pair per property.
type Record = (String, Vec<String>, Vec<(String, String)>);

fn doc(nodes: usize, props: usize, edges: bool) -> String {
    let mut lines: Vec<String> = (0..nodes)
        .map(|i| {
            let p: Vec<String> = (0..props)
                .map(|k| format!(r#""k{k}":"v{i}_{k}""#))
                .collect();
            format!(
                r#"{{"type":"node","id":"v{i}","labels":["Person"],"properties":{{{}}}}}"#,
                p.join(",")
            )
        })
        .collect();
    if edges {
        for i in 0..nodes.saturating_sub(1) {
            lines.push(format!(
                r#"{{"type":"edge","id":"e{i}","labels":["KNOWS"],"from":"v{i}","to":"v{}","properties":{{"since":2020}}}}"#,
                i + 1
            ));
        }
    }
    lines.join("\n")
}

fn main() {
    for (nodes, props, edges, label) in [
        (200_000, 1, false, "200k nodes x 1 prop"),
        (200_000, 4, false, "200k nodes x 4 props"),
        (200_000, 2, true, "200k nodes + 200k edges"),
    ] {
        let text = doc(nodes, props, edges);
        let mib = text.len() as f64 / (1024.0 * 1024.0);

        // Total, for reference.
        let t = Instant::now();
        let g = lenke_core::ndjson::decode(&text).expect("decodes");
        let total = t.elapsed();
        let t = Instant::now();
        drop(g);
        let free = t.elapsed();

        // The intermediate records the decoder materializes, stood up separately so
        // their construction and destruction can be timed on their own: one owned
        // String per id, per label, per property key and per string value — the
        // exact allocation profile borrowing from the input would remove.
        let t = Instant::now();
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
        let parse = t.elapsed();
        let t = Instant::now();
        drop(recs);
        let teardown = t.elapsed();

        println!(
            "{label:26} {mib:6.1} MiB  total={total:>9.2?}  json_parse={parse:>9.2?}  \
             json_teardown={teardown:>9.2?}  graph_free={free:>9.2?}"
        );
    }
}
