//! Head-to-head: `lenke-engine` (the from-scratch engine, with its optimizer) vs
//! `lenke-core` (the reference engine) on identical data and identical GQL.
//!
//! Both engines get the SAME logical graph — built from one deterministic model,
//! fed to lenke-engine via its `Builder` and to lenke-core via NDJSON — and the
//! SAME query text (a shape may carry a separate `core` spelling only where the
//! two dialects differ, e.g. CALL procedure names are snake_case here / camelCase
//! there). Each shape is timed min-of-`REPS` (release) on both; the ratio is
//! core_ms / engine_ms (>1 means the new engine is faster).
//!
//! Shapes that fail to parse/execute on EITHER engine are SKIPped with the reason
//! rather than aborting the run — so the candidate list can be broad and the
//! harness reports exactly which surface each engine covers.
//!
//! Native only. Run (size/degree via env):
//!   cargo run --release --manifest-path crates/lenke-engine/Cargo.toml \
//!     --example vs_core_bench
//!   BENCH_N=1000000 BENCH_DEG=8 cargo run --release ... --example vs_core_bench
//!   BENCH_FILTER=call cargo run --release ... --example vs_core_bench   # substring match
//!
//! Threading: the engine is SINGLE-THREADED (it targets wasm); core uses rayon. The
//! only shape core leads on default hardware is `call/labelprop`, and that is core
//! spreading a slower serial algorithm across cores, not a better algorithm. For the
//! fair algorithm-to-algorithm comparison (and the one that reflects the wasm
//! target) pin core to one core — there the engine wins labelprop ~6x:
//!   RAYON_NUM_THREADS=1 cargo run --release ... --example vs_core_bench

use lenke_core::gql::eval::Params as CoreParams;
use lenke_engine::store::{Builder, Store};
use lenke_engine::value::Value;
use std::time::Instant;

/// One benchmarked query. `e` runs on lenke-engine; `c` runs on lenke-core.
/// They are identical unless the dialects genuinely differ.
struct Shape {
    e: &'static str,
    c: &'static str,
    tag: &'static str,
}

/// Same text on both engines.
const fn same(tag: &'static str, q: &'static str) -> Shape {
    Shape { e: q, c: q, tag }
}
fn engine_fixture(n: u32, deg: u32) -> Store {
    let mut b = Builder::default();
    for i in 0..n {
        // Every third node also carries a `VIP` label (multi-label filters).
        if i % 3 == 0 {
            b.node(
                &["Person", "VIP"],
                &[
                    ("name", Value::Str(format!("n{i}").into())),
                    ("age", Value::Num(f64::from(i % 100))),
                ],
            );
        } else {
            b.node(
                &["Person"],
                &[
                    ("name", Value::Str(format!("n{i}").into())),
                    ("age", Value::Num(f64::from(i % 100))),
                ],
            );
        }
    }
    for i in 0..n {
        for d in 0..deg {
            b.edge(
                i,
                (i.wrapping_mul(7)
                    .wrapping_add(d.wrapping_mul(13))
                    .wrapping_add(1))
                    % n,
                "R",
            );
        }
    }
    let mut st = b.build();
    // Weight every edge so the edge property `w` forms a full raw column.
    for eid in st.all_edges() {
        st.set_edge_prop(eid, "w", Value::Num(f64::from(eid % 1000)));
    }
    // A hash index on `name` (what a user filtering by name would build) — lets the
    // anchor flip and point lookups seek instead of scan.
    st.create_index("name");
    // A range index on `name` too — a user who runs `name < 'n15'` builds one, and
    // core's `create_vertex_index` is already a sorted structure that serves ranges,
    // so this is the parity fixture for the range shapes (not an extra advantage).
    st.create_range_index("name");
    st
}

fn core_fixture(n: u32, deg: u32) -> lenke_core::graph::Graph {
    // The same graph in lenke-core's NDJSON dialect.
    let mut s = String::new();
    for i in 0..n {
        let labels = if i % 3 == 0 {
            r#"["Person","VIP"]"#
        } else {
            r#"["Person"]"#
        };
        s.push_str(&format!(
            r#"{{"type":"node","id":"{i}","labels":{labels},"properties":{{"name":"n{i}","age":{}}}}}"#,
            i % 100
        ));
        s.push('\n');
    }
    let mut e = 0u64;
    for i in 0..n {
        for d in 0..deg {
            let to = (i
                .wrapping_mul(7)
                .wrapping_add(d.wrapping_mul(13))
                .wrapping_add(1))
                % n;
            s.push_str(&format!(
                r#"{{"type":"edge","id":"e{e}","labels":["R"],"from":"{i}","to":"{to}","properties":{{"w":{}}}}}"#,
                e % 1000
            ));
            s.push('\n');
            e += 1;
        }
    }
    let mut g = lenke_core::ndjson::decode(&s).expect("core load");
    g.create_vertex_index("name"); // same index both engines get
    g
}

fn time_engine(store: &Store, q: &str, reps: usize) -> Result<(f64, usize), String> {
    let parsed = lenke_engine::gql::parse(q).map_err(|e| format!("parse: {e}"))?;
    let plan = lenke_engine::opt::optimize_indexed(parsed, store);
    let mut best = f64::MAX;
    let mut rows = 0;
    for _ in 0..reps {
        let t = Instant::now();
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            lenke_engine::exec::run(&plan, store)
        }))
        .map_err(|_| "engine panic".to_string())?;
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        rows = out.rows.len();
    }
    Ok((best, rows))
}

fn time_core(
    graph: &mut lenke_core::graph::Graph,
    q: &str,
    reps: usize,
) -> Result<(f64, usize), String> {
    let prepared = lenke_core::gql::prepare(q).map_err(|e| format!("parse: {e}"))?;
    let params = CoreParams::new();
    let mut best = f64::MAX;
    let mut rows = 0;
    for _ in 0..reps {
        let t = Instant::now();
        let rs = prepared
            .execute(graph, &params)
            .map_err(|e| format!("exec: {e}"))?;
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        rows = rs.nrows;
    }
    Ok((best, rows))
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn shapes() -> Vec<Shape> {
    vec![
        // --- scans / filters ---
        same("scan/gt", "MATCH (p:Person) WHERE p.age > 90 RETURN p.name AS name"),
        same("scan/range-and", "MATCH (p:Person) WHERE p.age >= 10 AND p.age < 20 RETURN p.name AS name"),
        same("scan/eq-num", "MATCH (p:Person) WHERE p.age = 42 RETURN p.name AS name"),
        same("scan/or", "MATCH (p:Person) WHERE p.age < 5 OR p.age > 95 RETURN p.name AS name"),
        same("scan/in", "MATCH (p:Person) WHERE p.age IN [1, 2, 3, 4, 5] RETURN p.name AS name"),
        same("scan/eq-str", "MATCH (p:Person) WHERE p.name = 'n12345' RETURN p.age AS age"),
        same("scan/lt-str", "MATCH (p:Person) WHERE p.name < 'n15' RETURN p.age AS age"),
        same("scan/ne-count", "MATCH (p:Person) WHERE p.name <> 'n0' RETURN count(*) AS c"),
        same("scan/label2", "MATCH (p:VIP) RETURN count(*) AS c"),
        same("scan/not", "MATCH (p:Person) WHERE NOT p.age > 50 RETURN count(*) AS c"),
        // --- string predicates / functions ---
        same("str/starts", "MATCH (p:Person) WHERE p.name STARTS WITH 'n1' RETURN count(*) AS c"),
        same("str/ends", "MATCH (p:Person) WHERE p.name ENDS WITH '9' RETURN count(*) AS c"),
        same("str/contains", "MATCH (p:Person) WHERE p.name CONTAINS '234' RETURN count(*) AS c"),
        same("str/upper", "MATCH (p:Person) RETURN upper(p.name) AS u"),
        same("str/substring", "MATCH (p:Person) RETURN substring(p.name, 0, 2) AS s"),
        same("str/size", "MATCH (p:Person) RETURN char_length(p.name) AS l, count(*) AS c"),
        // --- projections / expressions ---
        same("proj/2col", "MATCH (p:Person) RETURN p.name AS name, p.age AS age"),
        same("proj/arith", "MATCH (p:Person) RETURN p.age + 1 AS a"),
        same("proj/arith2", "MATCH (p:Person) RETURN p.age * 2 + 1 AS a"),
        same("proj/case", "MATCH (p:Person) RETURN CASE WHEN p.age > 50 THEN 1 ELSE 0 END AS g, count(*) AS c"),
        same("proj/coalesce", "MATCH (p:Person) RETURN coalesce(p.age, 0) AS a"),
        // --- aggregations ---
        same("agg/count", "MATCH (p:Person) RETURN count(*) AS c"),
        same("agg/avg", "MATCH (p:Person) RETURN avg(p.age) AS a"),
        same("agg/multi", "MATCH (p:Person) RETURN sum(p.age) AS s, min(p.age) AS mn, max(p.age) AS mx"),
        same("agg/grouped", "MATCH (p:Person) RETURN p.age AS age, count(*) AS c"),
        same("agg/filtered-count", "MATCH (p:Person) WHERE p.age > 50 RETURN count(*) AS c"),
        same("agg/distinct-count", "MATCH (p:Person) RETURN count(DISTINCT p.age) AS c"),
        same("agg/group-str", "MATCH (p:Person) RETURN substring(p.name,0,2) AS k, count(*) AS c"),
        // --- WITH / pipeline ---
        same("with/filter", "MATCH (p:Person) WITH p.age AS a WHERE a > 50 RETURN count(*) AS c"),
        same("with/agg", "MATCH (p:Person) WITH p.age AS a, count(*) AS c WHERE c > 100 RETURN count(*) AS g"),
        // --- ordering / paging / distinct ---
        same("ord/num", "MATCH (p:Person) RETURN p.age AS age ORDER BY age"),
        same("ord/num-limit", "MATCH (p:Person) RETURN p.age AS age ORDER BY age DESC LIMIT 10"),
        same("ord/str-limit", "MATCH (p:Person) RETURN p.name AS name ORDER BY name LIMIT 100"),
        same("ord/skip-limit", "MATCH (p:Person) RETURN p.age AS age ORDER BY age SKIP 1000 LIMIT 50"),
        same("dist/num", "MATCH (p:Person) RETURN DISTINCT p.age AS age"),
        same("dist/multi", "MATCH (p:Person) RETURN DISTINCT p.age AS age, substring(p.name,0,1) AS pre"),
        // --- traversal ---
        same("trav/1hop-count", "MATCH (a:Person)-[:R]->(b) RETURN count(*) AS c"),
        same("trav/1hop-proj", "MATCH (a:Person)-[:R]->(b) RETURN b.name AS who"),
        same("trav/1hop-filter", "MATCH (a:Person)-[:R]->(b) WHERE b.age > 90 RETURN b.name AS who"),
        same("trav/1hop-group", "MATCH (a:Person)-[:R]->(b) RETURN b.age AS age, count(*) AS c"),
        same("trav/2hop-count", "MATCH (a:Person)-[:R]->()-[:R]->(c) RETURN count(*) AS c"),
        same("trav/3hop-count", "MATCH (a:Person)-[:R]->()-[:R]->()-[:R]->(d) RETURN count(*) AS c"),
        same("trav/in-count", "MATCH (a:Person)<-[:R]-(b) RETURN count(*) AS c"),
        same("trav/dist-target", "MATCH (a:Person)-[:R]->(b) RETURN DISTINCT b.age AS age"),
        // --- edge-property filters ---
        same("edge/wfilter", "MATCH (a:Person)-[r:R]->(b) WHERE r.w > 500 RETURN count(*) AS c"),
        same("edge/wproj", "MATCH (a:Person)-[r:R]->(b) WHERE r.w < 10 RETURN r.w AS w, b.name AS who"),
        // --- multi-pattern join ---
        same("join/tri", "MATCH (a:Person)-[:R]->(b), (b)-[:R]->(c) RETURN count(*) AS c"),
        // --- var-length paths ---
        same("var/1to2", "MATCH (a:Person)-[:R]->{1,2}(b) RETURN count(*) AS c"),
        same("var/1to3", "MATCH (a:Person)-[:R]->{1,3}(b) RETURN count(*) AS c"),
        // --- cardinality-driven anchor flip: selective indexed filter on the TARGET ---
        same("flip/target", "MATCH (a:Person)-[:R]->(b) WHERE b.name = 'n99' RETURN a.name AS s, b.name AS t"),
        // --- AML-shaped: fixed multi-hop chains with per-hop edge predicates ---
        same("aml/struct3", "MATCH (a:Person)-[e1:R]->(b)-[e2:R]->(c)-[e3:R]->(d) WHERE e1.w > e2.w AND e2.w > e3.w RETURN a.name AS s, d.name AS t LIMIT 5000"),
        same("aml/chain5", "MATCH (a:Person)-[:R]->(b)-[:R]->(c)-[:R]->(d)-[:R]->(e) WHERE a.name = 'n5' RETURN count(*) AS c"),
        same("aml/reach6", "MATCH (a:Person)-[:R]->{1,6}(b) WHERE a.name = 'n5' RETURN count(*) AS c"),
        // --- OPTIONAL MATCH ---
        same("opt/expand", "MATCH (a:Person) WHERE a.age < 5 OPTIONAL MATCH (a)-[:R]->(b) RETURN count(b) AS c"),
        // --- bounded reachability (source-filter pushdown below the traversal) ---
        same("path/reach", "MATCH (a:Person)-[:R]->{1,3}(b) WHERE a.age = 1 RETURN count(DISTINCT b) AS c"),
        same("path/reach-src", "MATCH (a:Person)-[:R]->{1,4}(b) WHERE a.name = 'n5' RETURN count(*) AS c"),
        // --- EXISTS / subquery ---
        same("exists/hop", "MATCH (a:Person) WHERE EXISTS { (a)-[:R]->(b) WHERE b.age > 90 } RETURN count(*) AS c"),
        same("call/inline", "MATCH (a:Person) WHERE a.age < 3 CALL (a) { MATCH (a)-[:R]->(b) RETURN b.age AS ba } RETURN count(*) AS c"),
        // --- UNION ---
        same("union/all", "MATCH (p:Person) WHERE p.age < 5 RETURN p.name AS n UNION ALL MATCH (p:Person) WHERE p.age > 95 RETURN p.name AS n"),
        same("union/dedup", "MATCH (p:Person) WHERE p.age < 5 RETURN p.age AS a UNION MATCH (p:Person) WHERE p.age < 10 RETURN p.age AS a"),
        // --- list / misc functions ---
        same("fn/minmax-str", "MATCH (p:Person) RETURN min(p.name) AS lo, max(p.name) AS hi"),
        same("fn/abs-round", "MATCH (p:Person) RETURN sum(abs(p.age - 50)) AS s"),
        // --- OPTIONAL projecting the optional var ---
        same("opt/proj", "MATCH (a:Person) WHERE a.age < 5 OPTIONAL MATCH (a)-[:R]->(b) RETURN a.name AS a, b.age AS ba"),
        // --- CALL algorithms (snake_case on both engines) ---
        same("call/degree", "CALL degree() YIELD degree RETURN sum(degree) AS s"),
        same("call/wcc", "CALL connected_components() YIELD componentId RETURN count(DISTINCT componentId) AS c"),
        same("call/scc", "CALL strongly_connected_components() YIELD componentId RETURN count(DISTINCT componentId) AS c"),
        same("call/pagerank", "CALL pagerank() YIELD score RETURN count(*) AS c"),
        same("call/labelprop", "CALL label_propagation() YIELD label RETURN count(DISTINCT label) AS c"),
        same("call/oncycle", "CALL on_cycle() YIELD onCycle RETURN count(*) AS c"),
    ]
}

fn main() {
    const REPS: usize = 5;
    let n = env_u32("BENCH_N", 200_000);
    let deg = env_u32("BENCH_DEG", 4);
    let filter = std::env::var("BENCH_FILTER").unwrap_or_default();
    eprintln!(
        "fixture: {n} Person nodes (1/3 also VIP), degree {deg} ({} R edges, weighted)",
        u64::from(n) * u64::from(deg)
    );
    // The engine is single-threaded (it must run on wasm). Core uses rayon, so the
    // only shape it "wins" — call/labelprop — is core spreading a SLOWER serial
    // algorithm across cores; pin core with RAYON_NUM_THREADS=1 for the fair
    // algorithm-to-algorithm comparison (there the engine wins labelprop ~6x). Print
    // the pool size so a labelprop ratio is never read without knowing which it is.
    eprintln!(
        "core rayon threads: {} (set RAYON_NUM_THREADS=1 for the single-core comparison)",
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "all cores".into())
    );

    let estore = engine_fixture(n, deg);
    let mut cgraph = core_fixture(n, deg);

    println!("  engine_ms     core_ms    ratio      rows  shape");
    let mut wins = 0;
    let mut losses = 0;
    let mut skips = 0;
    let mut worst: Vec<(f64, String)> = Vec::new();
    for s in shapes() {
        if !filter.is_empty() && !s.tag.contains(&filter) {
            continue;
        }
        let e = time_engine(&estore, s.e, REPS);
        let c = time_core(&mut cgraph, s.c, REPS);
        match (e, c) {
            (Ok((e_ms, e_rows)), Ok((c_ms, c_rows))) => {
                let flag = if e_rows == c_rows {
                    ""
                } else {
                    " (ROW COUNT DIFF!)"
                };
                let ratio = c_ms / e_ms;
                if ratio >= 1.0 {
                    wins += 1;
                } else {
                    losses += 1;
                    worst.push((ratio, format!("{:>6.2}x  {}", ratio, s.tag)));
                }
                println!(
                    "{e_ms:>11.3} {c_ms:>11.3} {ratio:>8.2} {e_rows:>9}  {}{flag}",
                    s.tag
                );
            }
            (e, c) => {
                skips += 1;
                let why = match (&e, &c) {
                    (Err(x), Ok(_)) => format!("engine {x}"),
                    (Ok(_), Err(x)) => format!("core {x}"),
                    (Err(x), Err(y)) => format!("both (engine {x} / core {y})"),
                    _ => unreachable!(),
                };
                println!("       SKIP {:>44}  {why}", s.tag);
            }
        }
    }
    worst.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    println!("\n{wins} win, {losses} lose, {skips} skip");
    if !worst.is_empty() {
        println!("losers (slowest first):");
        for (_, line) in &worst {
            println!("  {line}");
        }
    }
}
