//! **Which graph questions does one engine answer from the IR and the other
//! enumerate?**
//!
//! The shared IR exists so a fix lands once and reaches both languages. That
//! only holds for the operations both engines actually route through it. GQL has
//! eleven count/aggregate shortcuts in `gql/eval/fastpath.rs`; Gremlin has one
//! (`gremlin::exec::try_count`). Each pair below is ONE question in two
//! languages — the answers must match, and a large ratio marks a shortcut that
//! lives on one side of the IR instead of inside it.
//!
//! This is not a "Gremlin is slower than GQL" benchmark. Both engines walk the
//! same storage. A ratio near 1 means the question is answered the same way in
//! both; a big ratio means one of them enumerates rows the other never builds.
//!
//! Run: `cargo run --release --example cross_engine_shortcuts`

use std::time::Instant;

use lenke_core::gql::eval::Params;
use lenke_core::gql::prepare;
use lenke_core::graph::Graph;
use lenke_core::gremlin::parse;

const N: usize = 50_000;
const DEGREE: usize = 3;
const REPS: usize = 5;

/// `N` vertices labelled `V` (a tenth also `W`) with a numeric `n`, each with
/// `DEGREE` out-edges of type `R`. Edges get ids — `encode` emits them, so every
/// reloaded snapshot has them, and omitting them skips the external-id path.
fn fixture() -> Graph {
    let mut lines = String::with_capacity(N * 110);

    for i in 0..N {
        let labels = if i % 10 == 0 {
            r#"["V","W"]"#
        } else {
            r#"["V"]"#
        };
        lines.push_str(&format!(
            r#"{{"type":"node","id":"n{i}","labels":{labels},"properties":{{"n":{}}}}}"#,
            (i * 2_654_435_761) % 1_000
        ));
        lines.push('\n');
    }

    let mut e = 0usize;

    for i in 0..N {
        for d in 0..DEGREE {
            let to = (i * 31 + d * 7 + 1) % N;
            lines.push_str(&format!(
                r#"{{"type":"edge","id":"e{e}","from":"n{i}","to":"n{to}","labels":["R"],"properties":{{"w":{d}}}}}"#
            ));
            lines.push('\n');
            e += 1;
        }
    }

    lenke_core::ndjson::decode(&lines).expect("fixture decodes")
}

/// Min of `REPS` after a warm-up, plus the scalar answer so the two spellings
/// can be checked against each other.
fn time_gql(g: &mut Graph, q: &str) -> (f64, String) {
    let plan = prepare(q).expect("gql plans");
    let p = Params::new();
    let warm = plan.execute(g, &p).expect("gql runs");
    let answer = format!("{:?}", warm.rows().next().map(<[_]>::to_vec));
    let mut best = f64::INFINITY;

    for _ in 0..REPS {
        let t = Instant::now();
        let _ = plan.execute(g, &p).expect("gql runs");
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }

    (best, answer)
}

fn time_gremlin(g: &mut Graph, q: &str) -> (f64, String) {
    let plan = parse(q).unwrap_or_else(|e| panic!("gremlin parses `{q}`: {e}"));
    let answer = format!("{:?}", plan.clone().run(g));
    let mut best = f64::INFINITY;

    for _ in 0..REPS {
        let t = Instant::now();
        let _ = plan.clone().run(g);
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }

    (best, answer)
}

/// OPEN, deliberately deferred: the three rows that still read 2-3x are all
/// SUB-100-MICROSECOND label questions —
///
/// ```text
///   count of a label                 0.000ms  0.000ms  3.2x
///   far-end label decides the seed   0.043ms  0.095ms  2.2x
///   far-end label, edge spelled apart 0.043ms 0.096ms  2.2x
/// ```
///
/// — so the ratio is large and the absolute gap is ~50us. GQL answers the first
/// from a label-bucket length (`try_count_star`, O(1)) and the other two by
/// orienting onto the smaller label end; Gremlin reaches the same oriented plan
/// but pays a fixed setup per query that dominates at this size.
///
/// Worth returning to for the per-query overhead itself, which is what these
/// measure — not for the plans, which already agree. Everything with real
/// absolute cost in this table is now within 10%.
fn main() {
    let mut g = fixture();

    // (question, GQL, Gremlin, the GQL shortcut that answers it)
    let pairs: &[(&str, &str, &str, &str)] = &[
        (
            "count of a label",
            "MATCH (a:W) RETURN count(*) AS c",
            "g.V().hasLabel('W').count()",
            "try_count_star",
        ),
        (
            "count of a typed edge",
            "MATCH ()-[:R]->() RETURN count(*) AS c",
            "g.V().outE('R').count()",
            "try_count_edges",
        ),
        (
            "count of a 2-hop",
            "MATCH ()-[:R]->()-[:R]->() RETURN count(*) AS c",
            "g.V().out('R').out('R').count()",
            "try_count_two_hop",
        ),
        (
            // Exactly two hops on both sides. `->{1,2}` counts one-hop AND
            // two-hop rows while `times(2)` counts only two, so the loose
            // spelling made this pair two different questions.
            //
            // `->{2,2}` is also NOT the same query as `()-[:R]->()-[:R]->()`,
            // however tempting the gap between them looks. A QUANTIFIED walk is a
            // trail (no repeated edge); a fixed chain is not. A self-loop
            // traversed twice is one row for the chain and no row for `{2,2}` —
            // measured 5 vs 4 on a 3-edge graph with one self-loop.
            //
            // So they are not two spellings of one question and there is nothing
            // here to desugar: a desugaring preserves meaning by definition, and
            // rewriting `{n,n}` into n segments would MISTRANSLATE it into a
            // different query that happens to agree on loop-free graphs.
            //
            // The gap WAS a missing shortcut, and is now `try_count_varlen_upto_2`
            // — the degree-product pass already computed the length-2 count and
            // its trail correction for `{1,2}` and simply refused to answer the
            // narrower range. 16.3ms to 1.03ms.
            "count of a fixed 2-hop walk",
            "MATCH ()-[:R]->{2,2}() RETURN count(*) AS c",
            "g.V().repeat(out('R')).times(2).count()",
            "try_count_varlen_upto_2",
        ),
        (
            "distinct endpoints of a hop",
            "MATCH ()-[:R]->(b) RETURN count(DISTINCT b) AS c",
            "g.V().out('R').dedup().count()",
            "try_count_distinct_endpoint",
        ),
        (
            "filtered 1-hop count",
            "MATCH (a:V)-[:R]->(b) WHERE a.n > 900 RETURN count(*) AS c",
            "g.V().hasLabel('V').has('n', gt(900)).out('R').count()",
            "the columnar frame",
        ),
        (
            "group a hop by an endpoint value",
            "MATCH ()-[:R]->(b) RETURN b.n AS k, count(*) AS c",
            "g.V().out('R').values('n').groupCount()",
            "try_grouped_2hop",
        ),
        (
            // The pattern-level equivalence, and the biggest gap on this page.
            // GQL plans the WHOLE pattern and orients — seeds the selective end
            // and walks its adjacency backwards. Gremlin executes steps in
            // written order, so a filter that lands AFTER the hop cannot inform
            // the seed and the whole vertex set gets expanded first.
            //
            // Same question either way: `g.V().out(R).hasLabel(W)` IS
            // `MATCH ()-[:R]->(b:W)`.
            "far-end label decides the seed",
            "MATCH ()-[:R]->(b:W) RETURN count(*) AS c",
            "g.V().out('R').hasLabel('W').count()",
            "try_orient_node_seed",
        ),
        (
            // An EDGE-SOURCE traversal. `g.E()` desugars to `g.V().outE()`, which
            // is what lets it reach the planner at all — declined it made one
            // traverser per edge and filtered afterwards, at 23.2ms against
            // 0.07ms for the GQL spelling of the same question.
            "edge source, far-end label",
            "MATCH ()-[:R]->(b:W) RETURN count(*) AS c",
            "g.E().hasLabel('R').inV().hasLabel('W').count()",
            "try_orient_node_seed (via the E() desugar)",
        ),
        (
            "far-end property decides the seed",
            "MATCH ()-[:R]->(b) WHERE b.n = 7 RETURN count(*) AS c",
            "g.V().out('R').has('n', 7).count()",
            "try_orient_node_seed + the property index",
        ),
        (
            "sum a property over a label",
            "MATCH (a:V) RETURN sum(a.n) AS s",
            "g.V().hasLabel('V').values('n').sum()",
            "the columnar aggregate",
        ),
        (
            "sum a property over a hop",
            "MATCH ()-[:R]->(b) RETURN sum(b.n) AS s",
            "g.V().out('R').values('n').sum()",
            "the columnar aggregate",
        ),
        (
            // Read this row with the fixture in mind: `n` takes only 1000
            // distinct values over 50k vertices, so the top-k partition is
            // almost all TIES. Over 50k DISTINCT values the two engines measure
            // 0.98ms against 0.96ms — the same. The gap here is how each breaks
            // ties, and GQL breaks them deterministically (the comparator falls
            // through to the row index, so ties resolve to scan order); that is
            // a deliberate cost, not a missing shortcut.
            "top-k by a property",
            "MATCH (a:V) RETURN a.n AS x ORDER BY x DESC LIMIT 5",
            "g.V().hasLabel('V').values('n').order().by(desc).limit(5)",
            "the ORDER BY + LIMIT top-k",
        ),
        (
            "rows of a filtered hop",
            "MATCH (a:V)-[:R]->(b) WHERE a.n > 900 RETURN b.n AS x",
            "g.V().hasLabel('V').has('n', gt(900)).out('R').values('n')",
            "the columnar frame",
        ),
        (
            "does any edge of a type exist",
            "MATCH (a:V) WHERE EXISTS { (a)-[:R]->() } RETURN count(*) AS c",
            "g.V().hasLabel('V').where(outE('R')).count()",
            "try_count_semi_join",
        ),
        (
            // The edge spelled apart. `outE('R').inV()` IS `out('R')`, so this is
            // the same question as "far-end label decides the seed" written the
            // other way — and the two Gremlin spellings must not diverge.
            "far-end label, edge spelled apart",
            "MATCH ()-[:R]->(b:W) RETURN count(*) AS c",
            "g.V().outE('R').inV().hasLabel('W').count()",
            "try_orient_node_seed",
        ),
        (
            // Stopping ON the edge, which no vertex hop expresses.
            // An edge PROPERTY counts toward the orientation gate — the planner
            // can seed an edge property index where a left-to-right walk cannot
            // — and it pays even with no index on the fixture: 2.85ms here
            // against 7.68ms when the gate ignores `rel.props`.
            "edges filtered by their own property",
            "MATCH ()-[r:R]->() WHERE r.w = 1 RETURN count(*) AS c",
            "g.V().outE('R').has('w', 1).count()",
            "the edge property seek",
        ),
        (
            // Undirected, where the two languages genuinely differ on a
            // self-loop — so the shared segment has to CARRY the difference
            // rather than assume one. The fixture has none, so the answers match
            // and only the cost is in question.
            "far-end label over an undirected hop",
            "MATCH ()-[:R]-(b:W) RETURN count(*) AS c",
            "g.V().both('R').hasLabel('W').count()",
            "try_orient_node_seed",
        ),
        (
            "distinct property values",
            "MATCH (a:V) RETURN count(DISTINCT a.n) AS c",
            "g.V().hasLabel('V').values('n').dedup().count()",
            "the columnar DISTINCT",
        ),
    ];

    println!(
        "{N} vertices, degree {DEGREE}, min of {REPS}\n\n{:<34} {:>9} {:>10} {:>8}  GQL shortcut",
        "question", "GQL", "Gremlin", "ratio"
    );

    for (name, gql, grem, shortcut) in pairs {
        let (a, ans_a) = time_gql(&mut g, gql);
        let (b, ans_b) = time_gremlin(&mut g, grem);

        // The answers are printed, not asserted: the two languages render a
        // count differently (a GQL row vs a Gremlin scalar) and `groupCount`
        // returns a whole map. What matters here is the SHAPE of the cost.
        let same = ans_a.contains(&ans_b.trim_matches(|c| c == '[' || c == ']').to_string())
            || ans_b.contains(&ans_a.trim_matches(|c| c == '[' || c == ']').to_string());

        println!(
            "{name:<34} {a:>8.3}ms {b:>9.3}ms {:>7.1}x  {shortcut}{}",
            b / a,
            if same {
                ""
            } else {
                "   (answers differ in shape)"
            }
        );
    }
}
