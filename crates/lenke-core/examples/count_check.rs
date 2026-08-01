//! Correctness check for the count fast paths (edge-anchored + parallel): build a
//! graph large enough to cross the parallel seed threshold, print the exact count
//! for each shape. Run the SAME binary built with and without `parallel-query`
//! and diff the output — identical counts prove the parallel path matches serial.
//!   cargo run --release --example count_check > /tmp/serial.txt
//!   cargo run --release --features parallel-query --example count_check > /tmp/par.txt
//!   diff /tmp/serial.txt /tmp/par.txt

use lenke_core::gql::eval::Params;
use lenke_core::gql::prepare;
use lenke_core::graph::{Builder, EdgeRec, Graph, NodeRec, Value};

struct Rng(u64);
impl Rng {
    fn below(&mut self, n: usize) -> usize {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x % n as u64) as usize
    }
}

fn build(n: usize, eper: usize) -> Graph {
    let mut b = Builder::default();
    for i in 0..n {
        // Two labels so a labelled-endpoint filter is exercised on both ends.
        let labels = if i % 3 == 0 {
            vec!["Person".to_string(), "Admin".to_string()]
        } else {
            vec!["Person".to_string()]
        };
        b.nodes.push(NodeRec::owned(
            format!("p{i}"),
            labels,
            vec![
                ("age".to_string(), Value::Num((18 + (i % 62)) as f64)),
                ("name".to_string(), Value::Str(format!("name{i}").into())),
                (
                    "city".to_string(),
                    Value::Str(format!("city{}", i % 50).into()),
                ),
            ],
        ));
    }
    let mut rng = Rng(0x1234_5678_9abc_def0);
    for i in 0..n {
        for _ in 0..eper {
            b.edges.push(EdgeRec::owned(
                format!("p{i}"),
                format!("p{}", rng.below(n)),
                "KNOWS".to_string(),
                vec![],
                None,
            ));
        }
    }
    b.finalize()
}

fn count(g: &mut Graph, q: &str) -> i64 {
    let plan = prepare(q).unwrap();
    let rs = plan.execute(g, &Params::new()).unwrap();
    // The single cell of the single result row.
    match rs.row(0).first() {
        Some(Value::Num(n)) => *n as i64,
        _ => -1,
    }
}

/// A deterministic textual form of a result value, for the serial-vs-parallel diff.
fn fmt_val(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Num(n) => format!("{n:.6}"),
        Value::Str(s) => s.to_string(),
        other => format!("{other:?}"),
    }
}

/// Print every result row, in engine order, one row per line (`|`-joined cells).
fn dump_rows(g: &mut Graph, q: &str) {
    let plan = prepare(q).unwrap();
    let rs = plan.execute(g, &Params::new()).unwrap();
    for r in rs.rows() {
        println!(
            "  {}",
            r.iter().map(fmt_val).collect::<Vec<_>>().join(" | ")
        );
    }
}

fn main() {
    // 20k vertices → 20k seeds, above the parallel MIN_SEEDS threshold (8192).
    let mut g = build(20_000, 6);
    println!("parallel-query={}", cfg!(feature = "parallel-query"));
    for q in [
        "MATCH (a:Person)-[:KNOWS]->(b) RETURN count(*) AS c",
        "MATCH (a:Person)-[:KNOWS]->()-[:KNOWS]->(b) RETURN count(*) AS c",
        "MATCH (a:Person)-[:KNOWS]->(b) WHERE b.age > 40 RETURN count(*) AS c",
        "MATCH (a:Admin)-[:KNOWS]->(b:Person) RETURN count(*) AS c",
        "MATCH (a)-[:KNOWS]->(b) WHERE a.age >= 50 AND b.age < 30 RETURN count(*) AS c",
        "MATCH ()-[:KNOWS]->() RETURN count(*) AS c",
        "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) WHERE c.age > 60 RETURN count(*) AS c",
    ] {
        println!("{:>12}  {q}", count(&mut g, q));
    }

    // Parallel aggregation over a traversal (try_parallel_agg): dump full rows in
    // engine order, so a serial-vs-parallel diff catches any aggregate-value OR
    // first-seen group-order divergence.
    for q in [
        // grouped, no ORDER BY → exercises first-seen group-order preservation
        "MATCH (a:Person)-[:KNOWS]->(b) RETURN b.city AS city, count(*) AS n",
        // grouped, multiple aggregates, ordered
        "MATCH (a:Person)-[:KNOWS]->(b) RETURN b.age AS age, count(*) AS n, avg(a.age) AS aa, min(b.name) AS mn ORDER BY age",
        // global aggregates over a traversal
        "MATCH (a:Person)-[:KNOWS]->(b) RETURN sum(b.age) AS s, avg(b.age) AS av, min(b.age) AS mnn, max(b.age) AS mx, count(*) AS c",
        // 2-hop grouped (the trav2_group shape), first-seen order
        "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN c.city AS city, count(*) AS n",
        // comma-join sharing `a` (the join_multi shape): grouped, first-seen order
        "MATCH (a:Person)-[:KNOWS]->(b), (a)-[:KNOWS]->(c) WHERE b.age > 40 RETURN a.city AS city, count(*) AS n",
        // comma-join, global aggregates over the join
        "MATCH (a:Person)-[:KNOWS]->(b), (a)-[:KNOWS]->(c) WHERE b.age > 60 AND c.age < 25 RETURN count(*) AS c, min(a.name) AS mn, max(c.age) AS mx",
        // var-length grouped aggregation (try_parallel_agg over a quantified segment)
        "MATCH (a:Person)-[:KNOWS]->{1,2}(b) RETURN b.city AS city, count(*) AS n",
        // var-length global aggregates
        "MATCH (a:Person)-[:KNOWS]->{1,2}(b) RETURN count(*) AS c, min(b.age) AS mn, max(b.age) AS mx",
        // Parallel row materialization (try_parallel_scan): a plain traversal
        // projection over a WHERE-filtered join. Order-sensitive dump — a divergent
        // chunk concat order or filter would show up as a row mismatch. Selective
        // WHEREs keep the output small while still seeding all 20k Person (> the
        // parallel threshold, so the parallel path fires).
        "MATCH (a:Person)-[:KNOWS]->(b) WHERE a.age = 79 AND b.age > 74 RETURN a.name AS an, b.city AS bc, b.age AS ba",
        // labeled endpoint (b:Admin) exercises the in-expansion node-label filter
        "MATCH (a:Person)-[:KNOWS]->(b:Admin) WHERE a.age > 76 RETURN a.name AS an, b.name AS bn, b.age AS ba",
    ] {
        println!("--- {q}");
        dump_rows(&mut g, q);
    }

    // Known-answer checks for the interned-string paths (20k vertices, city = i%50).
    let checks = [
        (
            "MATCH (n:Person) WHERE n.name = 'name100' RETURN count(*) AS c",
            1,
        ),
        (
            "MATCH (n:Person) WHERE n.name = 'nope' RETURN count(*) AS c",
            0,
        ),
        (
            "MATCH (n:Person) WHERE n.name <> 'name100' RETURN count(*) AS c",
            19_999,
        ),
        ("MATCH (n:Person) RETURN count(DISTINCT n.city) AS c", 50),
    ];
    for (q, want) in checks {
        let got = count(&mut g, q);
        println!("{got:>12}  (want {want})  {q}");
        assert_eq!(got, want, "MISMATCH on: {q}");
    }

    // ORDER BY + LIMIT partial-sort vs an independently-computed expected result.
    // age = 18 + i%62 ⇒ max age 79 at i%62==61; among those, ORDER BY name ASC.
    let ordered = order_names(
        &mut g,
        "MATCH (n:Person) RETURN n.name AS name ORDER BY n.age DESC, n.name LIMIT 3",
    );
    let mut top_names: Vec<String> = (0..20_000)
        .filter(|i| i % 62 == 61) // the max-age (79) rows
        .map(|i| format!("name{i}"))
        .collect();
    top_names.sort(); // lexicographic, matching the ORDER BY name ASC tiebreak
    let want_order = &top_names[..3];
    println!("order top3: {ordered:?}  (want {want_order:?})");
    assert_eq!(
        ordered, want_order,
        "ORDER BY partial-sort produced wrong rows"
    );

    // Orientation correctness: a pattern and its reverse describe the SAME edges,
    // so — regardless of which end each seeds from — they must return the identical
    // row SET. reverse_path preserving bindings is exactly this equality. Compare
    // sorted so the (legitimately different) row ORDER doesn't matter.
    let orient_pairs = [
        (
            "MATCH (a:Person)-[:KNOWS]->(b:Admin) RETURN a.name AS an, b.name AS bn",
            "MATCH (b:Admin)<-[:KNOWS]-(a:Person) RETURN a.name AS an, b.name AS bn",
        ),
        (
            // 2-hop: forward vs fully reversed (segments + directions flipped).
            "MATCH (a:Admin)-[:KNOWS]->(b)-[:KNOWS]->(c:Admin) RETURN a.name AS x, c.name AS y",
            "MATCH (c:Admin)<-[:KNOWS]-(b)<-[:KNOWS]-(a:Admin) RETURN a.name AS x, c.name AS y",
        ),
        (
            "MATCH (a:Person)-[:KNOWS]->(b:Admin) WHERE a.age > 76 RETURN a.name AS an, b.age AS ba",
            "MATCH (b:Admin)<-[:KNOWS]-(a:Person) WHERE a.age > 76 RETURN a.name AS an, b.age AS ba",
        ),
    ];
    for (fwd, rev) in orient_pairs {
        let mut a = sorted_rows(&mut g, fwd);
        let mut b = sorted_rows(&mut g, rev);
        a.sort();
        b.sort();
        println!("orient pair: {} rows, match={}", a.len(), a == b);
        assert_eq!(a, b, "orientation changed the row SET:\n  {fwd}\n  {rev}");
        assert!(!a.is_empty(), "orient-check query returned no rows: {fwd}");
    }

    // Count-seed correctness: the cardinality-seeded try_count_edges must equal the
    // enumerated row count of the same pattern (both directions, both seed ends).
    for count_vs_rows in [
        "MATCH (a:Person)-[:KNOWS]->(b:Admin)",
        "MATCH (b:Admin)<-[:KNOWS]-(a:Person)",
        "MATCH (a:Admin)-[:KNOWS]->(b:Admin)",
        "MATCH (a:Admin)-[:KNOWS]->(b:Person)",
    ] {
        let c = count(&mut g, &format!("{count_vs_rows} RETURN count(*) AS c"));
        let rows = sorted_rows(
            &mut g,
            &format!("{count_vs_rows} RETURN a.name AS an, b.name AS bn"),
        )
        .len() as i64;
        println!("count-seed: count={c} rows={rows}  {count_vs_rows}");
        assert_eq!(c, rows, "count-seed != enumerated rows: {count_vs_rows}");
    }

    // Semi-join correctness: EXISTS count == count(DISTINCT a) of the join (a wholly
    // different code path — reverse-seed set vs join + distinct fold), and
    // NOT EXISTS count == |La| − that. Admin = i%3==0 (the smaller, seedable end).
    let semi_pairs = [
        (
            "MATCH (a:Person) WHERE EXISTS { (a)-[:KNOWS]->(:Admin) } RETURN count(*) AS c",
            "MATCH (a:Person)-[:KNOWS]->(b:Admin) RETURN count(DISTINCT a) AS c",
        ),
        (
            "MATCH (a:Person) WHERE EXISTS { (a)<-[:KNOWS]-(:Admin) } RETURN count(*) AS c",
            "MATCH (a:Person)<-[:KNOWS]-(b:Admin) RETURN count(DISTINCT a) AS c",
        ),
    ];
    for (exists_q, distinct_q) in semi_pairs {
        let e = count(&mut g, exists_q);
        let d = count(&mut g, distinct_q);
        println!("semi-join: exists={e} distinct={d}  {exists_q}");
        assert_eq!(e, d, "EXISTS count != count(DISTINCT a):\n  {exists_q}");
    }
    // NOT EXISTS = every Person minus those that satisfy EXISTS.
    let people = count(&mut g, "MATCH (a:Person) RETURN count(*) AS c");
    let has = count(
        &mut g,
        "MATCH (a:Person) WHERE EXISTS { (a)-[:KNOWS]->(:Admin) } RETURN count(*) AS c",
    );
    let hasnt = count(
        &mut g,
        "MATCH (a:Person) WHERE NOT EXISTS { (a)-[:KNOWS]->(:Admin) } RETURN count(*) AS c",
    );
    println!("semi-join NOT: people={people} has={has} hasnt={hasnt}");
    assert_eq!(has + hasnt, people, "EXISTS + NOT EXISTS != all Person");

    println!("all string-path + order + orientation + count-seed + semi-join checks OK");
}

/// Every result row as a `|`-joined string (all cells), for set comparison.
fn sorted_rows(g: &mut Graph, q: &str) -> Vec<String> {
    let plan = prepare(q).unwrap();
    let rs = plan.execute(g, &Params::new()).unwrap();
    rs.rows()
        .map(|r| r.iter().map(fmt_val).collect::<Vec<_>>().join("|"))
        .collect()
}

/// Collect the first-column string values of a query's result rows, in order.
fn order_names(g: &mut Graph, q: &str) -> Vec<String> {
    let plan = prepare(q).unwrap();
    let rs = plan.execute(g, &Params::new()).unwrap();
    rs.rows()
        .filter_map(|r| match r.first() {
            Some(Value::Str(s)) => Some(s.to_string()),
            _ => None,
        })
        .collect()
}
