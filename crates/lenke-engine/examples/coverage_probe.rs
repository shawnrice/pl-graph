//! Feature-surface + coverage probe: runs a BROAD curated GQL corpus (the
//! `gql_bench` shapes plus wide feature probes) through BOTH engines on an identical
//! graph, and classifies each query:
//!   GAP     — core runs it, the engine cannot (parse/exec error or panic). A real
//!             feature-surface hole: core does something the engine does not.
//!   DIVERGE — both run it but the result multisets differ. A correctness gap.
//!   ONLY-E  — the engine runs it, core cannot (engine is ahead / or a core limit).
//!   BOTH-ERR— both reject it (agreement, fine).
//!   OK      — same result multiset; the perf ratio (core_ms/engine_ms) is reported.
//!
//! The graph is built ONCE as NDJSON and decoded into both engines, so they share a
//! byte-identical fixture. Purpose: prove the engine is a superset-capable, faster
//! replacement across far more shapes than `vs_core_bench`'s hand-picked 69 — and
//! surface any hole loudly. Native only.
//!   cargo run --release --manifest-path crates/lenke-engine/Cargo.toml \
//!     --example coverage_probe

use lenke_core::gql::eval::Params as CoreParams;
use std::time::Instant;

const N: usize = 50_000;
const SOFTWARE: usize = 2_000;
const KNOWS_PER: usize = 4;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Build the `gql_bench` social graph in BOTH engines from one shared model and RNG,
/// so the two graphs are identical: core via its NDJSON dialect, the engine via its
/// `Builder`. Person ids 0..N, Software ids N..N+SOFTWARE (insertion order).
fn build_both() -> (lenke_core::graph::Graph, lenke_engine::store::Store) {
    use lenke_engine::store::Builder;
    use lenke_engine::value::Value;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut nd = String::new();
    let mut b = Builder::default();
    for i in 0..N {
        let age = 18 + (i % 62);
        nd.push_str(&format!(
            r#"{{"type":"node","id":"p{i}","labels":["Person"],"properties":{{"age":{age},"name":"name{i}","dept":"d{}"}}}}"#,
            i % 12
        ));
        nd.push('\n');
        b.node(
            &["Person"],
            &[
                ("age", Value::Num(age as f64)),
                ("name", Value::Str(format!("name{i}").into())),
                ("dept", Value::Str(format!("d{}", i % 12).into())),
            ],
        );
    }
    for j in 0..SOFTWARE {
        nd.push_str(&format!(
            r#"{{"type":"node","id":"s{j}","labels":["Software"],"properties":{{"name":"sw{j}"}}}}"#
        ));
        nd.push('\n');
        b.node(
            &["Software"],
            &[("name", Value::Str(format!("sw{j}").into()))],
        );
    }
    let mut e = 0u64;
    for i in 0..N {
        for _ in 0..KNOWS_PER {
            let to = rng.below(N);
            nd.push_str(&format!(
                r#"{{"type":"edge","id":"e{e}","labels":["KNOWS"],"from":"p{i}","to":"p{to}"}}"#
            ));
            nd.push('\n');
            b.edge(i as u32, to as u32, "KNOWS");
            e += 1;
        }
        if i % 2 == 0 {
            let sw = rng.below(SOFTWARE);
            nd.push_str(&format!(
                r#"{{"type":"edge","id":"e{e}","labels":["CREATED"],"from":"p{i}","to":"s{sw}","properties":{{"weight":0.5}}}}"#
            ));
            nd.push('\n');
            b.edge(i as u32, (N + sw) as u32, "CREATED");
            e += 1;
        }
    }
    let cg = lenke_core::ndjson::decode(&nd).expect("core load");
    let mut st = b.build();
    // Weight every CREATED edge 0.5 (KNOWS carry no weight), matching the NDJSON.
    let created = st.etype_id("CREATED");
    for from in 0..N as u32 {
        for a in st.out(from).to_vec() {
            if Some(a.etype) == created {
                st.set_edge_prop(a.eid, "weight", Value::Num(0.5));
            }
        }
    }
    (cg, st)
}

#[derive(PartialEq)]
enum Out {
    Rows(Vec<Vec<String>>),
    Err,
    Parse,
}

fn cell(v: &lenke_engine::value::Value) -> String {
    use lenke_engine::value::Value;
    match v {
        Value::Num(n) => {
            if n.fract() == 0.0 && n.is_finite() {
                format!("i{}", *n as i64)
            } else {
                format!("f{n:.6}")
            }
        }
        Value::Str(s) => format!("s{s}"),
        Value::Bool(b) => format!("b{b}"),
        Value::Null => "null".into(),
        o => format!("{o:?}"),
    }
}
fn ccell(v: &lenke_core::graph::Value) -> String {
    use lenke_core::graph::Value;
    match v {
        Value::Num(n) => {
            if n.fract() == 0.0 && n.is_finite() {
                format!("i{}", *n as i64)
            } else {
                format!("f{n:.6}")
            }
        }
        Value::Str(s) => format!("s{s}"),
        Value::Bool(b) => format!("b{b}"),
        Value::Null => "null".into(),
        o => format!("{o:?}"),
    }
}

fn run_engine(store: &lenke_engine::store::Store, q: &str) -> (Out, f64) {
    let Ok(plan) = lenke_engine::gql::parse(q) else {
        return (Out::Parse, 0.0);
    };
    let plan = lenke_engine::opt::optimize_indexed(plan, store);
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut best = f64::MAX;
        let mut rows = Vec::new();
        for _ in 0..3 {
            let t = Instant::now();
            let o = lenke_engine::exec::try_run(&plan, store);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
            match o {
                Ok(r) => {
                    rows = r
                        .rows
                        .iter()
                        .map(|row| row.iter().map(cell).collect::<Vec<_>>())
                        .collect()
                }
                Err(_) => return (Out::Err, 0.0),
            }
        }
        (Out::Rows(rows), best)
    }));
    res.unwrap_or((Out::Err, 0.0))
}

fn run_core(graph: &mut lenke_core::graph::Graph, q: &str) -> (Out, f64) {
    let Ok(prep) = lenke_core::gql::prepare(q) else {
        return (Out::Parse, 0.0);
    };
    let pa = CoreParams::new();
    let mut best = f64::MAX;
    let mut rows = Vec::new();
    for _ in 0..3 {
        let t = Instant::now();
        match prep.execute(graph, &pa) {
            Ok(rs) => {
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
                rows = rs
                    .rows()
                    .map(|r| r.iter().map(ccell).collect::<Vec<_>>())
                    .collect();
            }
            Err(_) => return (Out::Err, 0.0),
        }
    }
    (Out::Rows(rows), best)
}

fn same(a: &Out, b: &Out) -> bool {
    match (a, b) {
        (Out::Rows(x), Out::Rows(y)) => {
            let mut xs: Vec<String> = x.iter().map(|r| r.join("\u{1}")).collect();
            let mut ys: Vec<String> = y.iter().map(|r| r.join("\u{1}")).collect();
            xs.sort();
            ys.sort();
            xs == ys
        }
        _ => false,
    }
}

fn main() {
    // Give BOTH engines the SAME index (core's `create_vertex_index("name")` ≡ the
    // engine's `create_index("name")`) — no extra index on one side, so the perf
    // comparison is apples-to-apples.
    let (mut cg, mut st) = build_both();
    cg.create_vertex_index("name");
    st.create_index("name");

    let queries = corpus();
    let (mut gap, mut diverge, mut only_e, mut both_err, mut ok) = (0, 0, 0, 0, 0);
    let mut ratios: Vec<f64> = Vec::new();
    println!(
        "{:<52} {:>9} {:>9} {:>7}  verdict",
        "query", "core_ms", "eng_ms", "ratio"
    );
    for q in &queries {
        let (co, cms) = run_core(&mut cg, q);
        let (eo, ems) = run_engine(&st, q);
        let core_ran = matches!(co, Out::Rows(_));
        let eng_ran = matches!(eo, Out::Rows(_));
        let verdict = if core_ran && !eng_ran {
            gap += 1;
            "GAP  <<<<"
        } else if !core_ran && eng_ran {
            only_e += 1;
            "ONLY-E"
        } else if core_ran && eng_ran {
            if same(&co, &eo) {
                ok += 1;
                let r = cms / ems.max(1e-6);
                ratios.push(r);
                if r < 1.0 {
                    "ok LOSE"
                } else {
                    "ok"
                }
            } else {
                diverge += 1;
                "DIVERGE <<<<"
            }
        } else {
            both_err += 1;
            "both-reject"
        };
        let short: String = q.chars().take(50).collect();
        println!(
            "{short:<52} {cms:>9.3} {ems:>9.3} {:>7.2}  {verdict}",
            cms / ems.max(1e-6)
        );
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let wins = ratios.iter().filter(|&&r| r >= 1.0).count();
    println!("\n{} queries: {ok} OK ({wins} eng-faster, {} eng-slower), {gap} GAP, {diverge} DIVERGE, {only_e} ONLY-ENGINE, {both_err} both-reject",
        queries.len(), ok - wins);
    if !ratios.is_empty() {
        let med = ratios[ratios.len() / 2];
        println!(
            "perf on shared shapes: median ratio {med:.2}x, min {:.2}x, max {:.2}x",
            ratios[0],
            ratios[ratios.len() - 1]
        );
    }
}

/// A broad GQL corpus: the `gql_bench` shapes plus wide feature probes.
fn corpus() -> Vec<&'static str> {
    vec![
        // --- gql_bench shapes ---
        "MATCH (n:Person) RETURN count(*) AS c",
        "MATCH (n:Person) WHERE n.age > 50 RETURN count(*) AS c",
        "MATCH (n:Person) RETURN n.name LIMIT 100",
        "MATCH (n:Person) WHERE n.age > 30 RETURN n.name, n.age",
        "MATCH (a:Person)-[:KNOWS]->(b) RETURN count(*) AS c",
        "MATCH (n:Person) RETURN n.dept, count(*) AS c, avg(n.age) AS a",
        "MATCH (n:Person) RETURN n.dept, n.age, count(*) AS c",
        "MATCH (n:Person) WHERE EXISTS { (n)-[:KNOWS]->() } RETURN count(*) AS c",
        "MATCH (a:Person)-[r:CREATED]->(s) WHERE r.weight > 0.4 RETURN count(*) AS c",
        "MATCH (a:Person)-[r:CREATED]->(s) RETURN a.age * 2 + 1 AS x, r.weight + 1 AS w",
        "MATCH (a:Person {name:'name0'})-[:KNOWS]->{1,2}(b) RETURN count(*) AS c",
        "MATCH (n:Person) RETURN n.name ORDER BY n.age DESC LIMIT 20",
        "MATCH (n:Person) RETURN n.age ORDER BY n.age DESC",
        "MATCH (n:Person) RETURN DISTINCT n.dept",
        "MATCH (n:Person) RETURN DISTINCT n.dept, n.age",
        "MATCH (n:Person) WITH n WHERE n.age > 30 RETURN n.name",
        "MATCH (n:Person) WITH n.dept AS d, count(*) AS c WHERE c > 4000 RETURN d, c",
        "MATCH (a:Person) WITH a WHERE a.age > 40 MATCH (a)-[:KNOWS]->(b) RETURN count(*) AS c",
        "MATCH (n:Person) WHERE (n.age * 2 + 1) % 3 = 0 AND n.age > 20 AND abs(n.age - 40) < 15 RETURN count(*) AS c",
        "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN count(*) AS c",
        // --- aggregation surface ---
        "MATCH (n:Person) RETURN sum(n.age) AS s",
        "MATCH (n:Person) RETURN min(n.age) AS lo, max(n.age) AS hi",
        "MATCH (n:Person) RETURN count(DISTINCT n.dept) AS c",
        "MATCH (n:Person) RETURN count(DISTINCT n.age) AS c",
        "MATCH (n:Person) RETURN avg(n.age) AS a, count(*) AS c, min(n.age) AS lo",
        "MATCH (n:Person) RETURN n.dept, collect(n.age) AS ages",
        "MATCH (n:Person) RETURN stdev(n.age) AS sd",
        // --- WHERE surface ---
        "MATCH (n:Person) WHERE n.age >= 30 AND n.age < 40 RETURN count(*) AS c",
        "MATCH (n:Person) WHERE n.age < 25 OR n.age > 75 RETURN count(*) AS c",
        "MATCH (n:Person) WHERE n.age IN [20, 30, 40] RETURN count(*) AS c",
        "MATCH (n:Person) WHERE NOT n.age > 50 RETURN count(*) AS c",
        "MATCH (n:Person) WHERE n.name STARTS WITH 'name1' RETURN count(*) AS c",
        "MATCH (n:Person) WHERE n.name ENDS WITH '9' RETURN count(*) AS c",
        "MATCH (n:Person) WHERE n.name CONTAINS '99' RETURN count(*) AS c",
        "MATCH (n:Person) WHERE n.dept IS NOT NULL RETURN count(*) AS c",
        "MATCH (n:Person) WHERE n.missing IS NULL RETURN count(*) AS c",
        // --- scalar functions ---
        "MATCH (n:Person) RETURN abs(n.age - 40) AS d LIMIT 50",
        "MATCH (n:Person) RETURN floor(n.age / 10.0) AS f LIMIT 50",
        "MATCH (n:Person) RETURN ceil(n.age / 7.0) AS c LIMIT 50",
        "MATCH (n:Person) RETURN round(n.age / 3.0) AS r LIMIT 50",
        "MATCH (n:Person) RETURN upper(n.dept) AS u LIMIT 50",
        "MATCH (n:Person) RETURN lower(n.name) AS l LIMIT 50",
        "MATCH (n:Person) RETURN size(n.name) AS len LIMIT 50",
        "MATCH (n:Person) RETURN substring(n.name, 0, 4) AS sub LIMIT 50",
        "MATCH (n:Person) RETURN coalesce(n.missing, n.name) AS c LIMIT 50",
        "MATCH (n:Person) RETURN n.name || '!' AS x LIMIT 50",
        "MATCH (n:Person) RETURN CASE WHEN n.age > 50 THEN 'old' ELSE 'young' END AS band LIMIT 50",
        "MATCH (n:Person) RETURN trim('  ' || n.dept) AS t LIMIT 20",
        // --- ORDER BY / paging ---
        "MATCH (n:Person) RETURN n.name, n.age ORDER BY n.age ASC, n.name DESC LIMIT 30",
        "MATCH (n:Person) RETURN n.age AS a ORDER BY a DESC LIMIT 10",
        "MATCH (n:Person) RETURN n.name ORDER BY n.name SKIP 100 LIMIT 20",
        "MATCH (n:Person) RETURN DISTINCT n.age ORDER BY n.age LIMIT 15",
        // --- traversal / var-length / path ---
        "MATCH (a:Person {name:'name0'})-[:KNOWS]->{1,3}(b) RETURN count(DISTINCT b) AS c",
        "MATCH (a:Person)-[:KNOWS]->(b) WHERE b.age > a.age RETURN count(*) AS c",
        "MATCH (a:Person {name:'name5'})-[:KNOWS]->(b)<-[:KNOWS]-(c) RETURN count(*) AS c",
        "MATCH p = ANY SHORTEST (a:Person {name:'name0'})-[:KNOWS]->*(b:Person {name:'name7'}) RETURN count(*) AS c",
        // Tiebreak on the name alias so the LIMIT cut is deterministic (many software
        // share a count — without a total order the top-10 among ties is unspecified).
        "MATCH (a:Person)-[:CREATED]->(s:Software) RETURN s.name AS sn, count(*) AS c ORDER BY c DESC, sn LIMIT 10",
        // --- WITH chains ---
        "MATCH (a:Person) WITH a, a.age AS age MATCH (a)-[:KNOWS]->(b) WHERE b.age > age RETURN count(*) AS c",
        "MATCH (n:Person) WITH n.dept AS d, avg(n.age) AS a WHERE a > 45 RETURN d ORDER BY d",
        // --- CALL / algorithms ---
        "CALL degree_centrality() YIELD node, score RETURN count(*) AS c",
        "MATCH (n:Person) CALL { WITH n MATCH (n)-[:KNOWS]->(m) RETURN count(m) AS deg } RETURN n.name, deg LIMIT 10",
        // --- edge shapes ---
        "MATCH (a:Person)-[r:CREATED]->(s) RETURN r.weight AS w, s.name AS who LIMIT 20",
        "MATCH (a:Person)-[r:CREATED]->(s) WHERE r.weight >= 0.5 RETURN count(*) AS c",
    ]
}
