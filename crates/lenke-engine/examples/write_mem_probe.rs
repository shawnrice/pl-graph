//! Writes + memory: how does the engine compare to core on INGEST throughput and
//! resident memory for the same graph? Complements the read-side benches. Native only.
//!   cargo run --release --manifest-path crates/lenke-engine/Cargo.toml \
//!     --example write_mem_probe

use lenke_engine::store::Builder;
use lenke_engine::value::Value;
use std::time::Instant;

const N: u32 = 500_000;
const DEG: u32 = 4;

/// Resident set size in bytes (Linux `/proc/self/statm`, field 2 × page size).
fn rss() -> u64 {
    let s = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
    let pages: u64 = s
        .split_whitespace()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    pages * 4096
}

/// The graph in core's NDJSON dialect and the engine's, for the same logical model.
fn dialects() -> (String, String) {
    let (mut core, mut eng) = (String::new(), String::new());
    for i in 0..N {
        core.push_str(&format!(
            r#"{{"type":"node","id":"n{i}","labels":["Person"],"properties":{{"age":{},"name":"name{i}"}}}}"#,
            i % 100
        ));
        core.push('\n');
        eng.push_str(&format!(
            r#"{{"id":"n{i}","labels":["Person"],"props":{{"age":{},"name":"name{i}"}}}}"#,
            i % 100
        ));
        eng.push('\n');
    }
    let mut e = 0u64;
    for i in 0..N {
        for d in 0..DEG {
            let to = (i
                .wrapping_mul(7)
                .wrapping_add(d.wrapping_mul(13))
                .wrapping_add(1))
                % N;
            core.push_str(&format!(
                r#"{{"type":"edge","id":"e{e}","labels":["R"],"from":"n{i}","to":"n{to}"}}"#
            ));
            core.push('\n');
            eng.push_str(&format!(
                r#"{{"id":"e{e}","from":"n{i}","to":"n{to}","type":"R","props":{{}}}}"#
            ));
            eng.push('\n');
            e += 1;
        }
    }
    (core, eng)
}

fn main() {
    let elems = u64::from(N) + u64::from(N) * u64::from(DEG);
    println!(
        "fixture: {N} nodes + {} edges = {elems} elements\n",
        u64::from(N) * u64::from(DEG)
    );
    let (core_nd, eng_nd) = dialects();

    // --- INGEST throughput (NDJSON decode, each engine's own dialect) ---
    let base = rss();
    let t = Instant::now();
    let cg = lenke_core::ndjson::decode(&core_nd).expect("core");
    let core_ms = t.elapsed().as_secs_f64() * 1e3;
    let core_rss = rss().saturating_sub(base);
    let core_nodes = cg.vertex_count();
    drop(cg);

    let base2 = rss();
    let t = Instant::now();
    let st = lenke_engine::ndjson::from_ndjson(&eng_nd).expect("engine");
    let eng_ms = t.elapsed().as_secs_f64() * 1e3;
    let eng_rss = rss().saturating_sub(base2);
    let eng_nodes = st.node_count();
    drop(st);

    println!("INGEST (ndjson decode):");
    println!(
        "  core   {core_ms:8.1} ms   ({:.1} M elem/s)   nodes={core_nodes}",
        elems as f64 / core_ms / 1e3
    );
    println!(
        "  engine {eng_ms:8.1} ms   ({:.1} M elem/s)   nodes={eng_nodes}",
        elems as f64 / eng_ms / 1e3
    );
    println!(
        "  ratio  {:.2}x (core_ms/eng_ms; >1 = engine faster)\n",
        core_ms / eng_ms
    );

    println!("MEMORY (RSS delta building the graph):");
    println!(
        "  core   {:6.1} MB   ({:.1} B/elem)",
        core_rss as f64 / 1e6,
        core_rss as f64 / elems as f64
    );
    println!(
        "  engine {:6.1} MB   ({:.1} B/elem)",
        eng_rss as f64 / 1e6,
        eng_rss as f64 / elems as f64
    );
    println!(
        "  ratio  {:.2}x (core/engine; >1 = engine smaller)\n",
        core_rss as f64 / eng_rss.max(1) as f64
    );

    // --- Per-op WRITE throughput (Builder = the engine's bulk-insert path) ---
    let t = Instant::now();
    let mut b = Builder::default();
    for i in 0..N {
        b.node(&["Person"], &[("age", Value::Num(f64::from(i % 100)))]);
    }
    for i in 0..N {
        for d in 0..DEG {
            b.edge(
                i,
                (i.wrapping_mul(7).wrapping_add(d).wrapping_add(1)) % N,
                "R",
            );
        }
    }
    let st = b.build();
    let build_ms = t.elapsed().as_secs_f64() * 1e3;
    println!(
        "ENGINE Builder (bulk insert + build): {build_ms:.1} ms ({:.1} M elem/s), nodes={}",
        elems as f64 / build_ms / 1e3,
        st.node_count()
    );
}
