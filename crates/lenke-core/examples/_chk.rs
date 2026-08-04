use lenke_core::gql::eval::Params;
use lenke_core::gql::prepare;
use std::time::Instant;
fn main() {
    let n = 50_000usize;
    let mut lines = String::new();
    for i in 0..n {
        lines.push_str(&format!(
            r#"{{"type":"node","id":"n{i}","labels":["V"],"properties":{{"n":{}}}}}"#,
            (i * 2654435761) % 1000
        ));
        lines.push('\n');
    }
    for i in 0..n {
        for d in 0..3 {
            let to = (i * 31 + d * 7 + 1) % n;
            lines.push_str(&format!(r#"{{"type":"edge","id":"e{i}_{d}","from":"n{i}","to":"n{to}","labels":["R"],"properties":{{}}}}"#));
            lines.push('\n');
        }
    }
    let mut g = lenke_core::ndjson::decode(&lines).unwrap();
    let gr = |q: &str, g: &mut lenke_core::graph::Graph| {
        let p = lenke_core::gremlin::parse(q).unwrap();
        let _ = p.clone().run(g);
        let mut b = f64::INFINITY;
        for _ in 0..7 {
            let t = Instant::now();
            let _ = p.clone().run(g);
            b = b.min(t.elapsed().as_secs_f64() * 1e3);
        }
        println!("  {b:>8.3}ms  {q}");
    };
    let gq = |q: &str, g: &mut lenke_core::graph::Graph| {
        let p = prepare(q).unwrap();
        let pr = Params::new();
        let _ = p.execute(g, &pr).unwrap();
        let mut b = f64::INFINITY;
        for _ in 0..7 {
            let t = Instant::now();
            let _ = p.execute(g, &pr).unwrap();
            b = b.min(t.elapsed().as_secs_f64() * 1e3);
        }
        println!("  {b:>8.3}ms  {q}");
    };
    println!("-- dedup breakdown");
    gr("g.V().hasLabel('V').values('n').count()", &mut g);
    gr("g.V().hasLabel('V').values('n').dedup().count()", &mut g);
    gq("MATCH (a:V) RETURN count(a.n) AS c", &mut g);
    gq("MATCH (a:V) RETURN count(DISTINCT a.n) AS c", &mut g);
    println!("-- groupCount breakdown");
    gr("g.V().out('R').count()", &mut g);
    gr("g.V().out('R').values('n').count()", &mut g);
    gr("g.V().out('R').values('n').groupCount()", &mut g);
    gq("MATCH ()-[:R]->(b) RETURN b.n AS k, count(*) AS c", &mut g);
}
