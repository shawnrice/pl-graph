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
//!
//! Read that number with care. A cheaper version of the same idea has already
//! been tried and REJECTED: `Value::Str` is an `Arc<str>` by the time it reaches
//! the column writer, which then interned it through `Dict::intern(&str)` and so
//! allocated and copied the string a second time. Adopting the existing `Arc`
//! instead removes an allocation and a memcpy per distinct value — and measured
//! neutral above 40-byte values and 6% WORSE at 8 bytes (500k x 8B: 486 -> 518
//! ms). Short-lived allocations are served from a hot free list and die in
//! order; keeping them alive for the graph's lifetime costs more in locality
//! than the copy ever cost.
//!
//! So "fewer allocations" does not automatically mean "faster" here, and the
//! 15-36% above — measured by standing records up with `format!`, which also
//! formats — is an upper bound rather than an expected gain.
//!
//! The borrow was then done (`Json<'a>` holds `Cow<'a, str>`, records pass it
//! through) and delivered 13-24%.
//!
//! What is LEFT after it, and why it is not worth taking: string VALUES still
//! allocate an `Arc` each, because `Value::Str` is `Arc<str>`. Measured
//! directly — 400k `Arc<str>::from` plus their drop costs 11.5 ms at 8-byte
//! values, 13.6 ms at 14 bytes, 16.6 ms at 40 bytes. Against a 220-275 ms
//! decode of the same shape that is **~5%**, not the 13% the pre-borrow numbers
//! suggested: most of that 13% was the id/label/key copying, which the borrow
//! already removed. (The repeated-string-vs-number gap was 36 ms before the
//! borrow and is 7.6 ms after it.)
//!
//! A third idea, also tried and REJECTED: `element_props` and `node_labels`
//! build a `Vec` per element, for every codec on both sides, which looked like
//! obvious per-element overhead. Both ways of removing it were measured on
//! pg-json encode at 400k elements, 15 reps (min / p25 / median ms):
//!
//!     Vec per element (today)   85.3 / 85.7 / 89.9
//!     lazy iterator             84.0 / 85.0 / 88.4
//!     caller-supplied scratch   83.7 / 84.5 / 87.7
//!
//! ~1-2%, against an estimate of 15% — the `Vec` costs a couple of nanoseconds
//! per element, not fifty. A small short-lived allocation comes off a hot free
//! list and goes straight back; the estimate assumed a malloc costs what a
//! malloc costs in isolation. Threading a scratch buffer through five codecs, or
//! splitting the helper in two so single-pass callers can take an iterator while
//! csv (which needs a slice for its two passes) keeps the `Vec`, is not worth
//! that. Note this is the SAME shape of error as the `Arc` adoption above:
//! counting allocations is not measuring time.
//!
//! Buying that ~5% means putting a lifetime on `Value` — the core stored and
//! result type, carried through query evaluation, Gremlin, the algorithms and
//! the FFI — and growing it from a 16-byte `Arc<str>` to a 24-byte `Cow`. That
//! is a far wider blast radius than the parser change for a fifth of the
//! return. Not worth it; measure again if `Value` is being reworked anyway.
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
