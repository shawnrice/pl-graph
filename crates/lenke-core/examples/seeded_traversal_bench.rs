//! **What does a traversal cost when its anchor seeks from an index?**
//!
//! `gql_bench`'s seeding section is single-node only, and `bench:usage` measures
//! traversals but not the anchor's seed size. This is the gap between them: the
//! same pattern with a point anchor (seed of 1), a narrow range (seed of ~500)
//! and a wide one (seed of ~49,500), plus the target-anchored mirror where the
//! orientation chooser has to decide which end to lead with.
//!
//! It exists because of a change that MEASURED NEUTRAL. Recognition runs per
//! execution, and several callers only wanted the boolean "is this seekable?"
//! yet answered it by performing the whole seek and discarding the ids — three
//! recogniser calls per traversal, at a cost proportional to the seed. Making
//! those callers ask a boolean (`ElementSeek::can_seek`) removed that work and
//! changed nothing measurable:
//!
//! ```text
//!                        before     after
//!   point anchor, 1 hop   1.0 us    0.9 us
//!   RANGE anchor, 1 hop   120.2     120.8
//!   RANGE anchor, 2 hop   227.6     228.3
//!   wide range anchor    1193.2    1181.8
//!   TARGET-anchored wide 2064.9    2058.0
//!   both ends unindexed   423.0     423.6
//! ```
//!
//! The reason is that the duplicate calls mostly land on UNINDEXED endpoints,
//! where recognition fails without touching the index at all, and the
//! orientation chooser returns as soon as the start end is seekable. Two seeks
//! on the same large seed needs BOTH ends indexed, which is rare. So the
//! redundancy is real and the waste is not.
//!
//! Keep this benchmark pointed at any future change to seeding: it is the shape
//! where a bad seed choice is most expensive.
//!
//! Run: `cargo run --release --example seeded_traversal_bench`
use lenke_core::gql::eval::{Params, Val};
use lenke_core::gql::prepare;
use std::time::Instant;

fn main() {
    let n = 50_000usize;
    let mut lines = Vec::new();
    for i in 0..n {
        lines.push(format!(
            r#"{{"type":"node","id":"u{i}","labels":["User"],"properties":{{"name":"user{i}","age":{}}}}}"#,
            i % 100
        ));
    }
    for i in 0..n {
        lines.push(format!(
            r#"{{"type":"edge","id":"e{i}","labels":["R"],"from":"u{i}","to":"u{}","properties":{{}}}}"#,
            (i * 7919) % n
        ));
    }
    let mut g = lenke_core::ndjson::decode(&lines.join("\n")).unwrap();
    g.create_vertex_index("name");
    g.create_vertex_index("age");
    let mut p = Params::new();
    p.insert("n".to_string(), Val::Str("user5".into()));

    for (label, q) in [
        (
            "point anchor, 1 hop",
            "MATCH (u:User)-[:R]->(x) WHERE u.name = $n RETURN count(*) AS c",
        ),
        (
            "point anchor, 2 hop",
            "MATCH (u:User)-[:R]->(x)-[:R]->(y) WHERE u.name = $n RETURN count(*) AS c",
        ),
        (
            "RANGE anchor, 1 hop",
            "MATCH (u:User)-[:R]->(x) WHERE u.age >= 90 RETURN count(*) AS c",
        ),
        (
            "RANGE anchor, 2 hop",
            "MATCH (u:User)-[:R]->(x)-[:R]->(y) WHERE u.age >= 90 RETURN count(*) AS c",
        ),
        (
            "wide range anchor",
            "MATCH (u:User)-[:R]->(x) WHERE u.age >= 1 RETURN count(*) AS c",
        ),
        (
            "TARGET-anchored wide",
            "MATCH (a:User)-[:R]->(m:User) WHERE m.age >= 1 RETURN count(*) AS c",
        ),
        (
            "TARGET-anchored point",
            "MATCH (a:User)-[:R]->(m:User) WHERE m.name = $n RETURN count(*) AS c",
        ),
        (
            "both ends unindexed",
            "MATCH (a:User)-[:R]->(m:User) RETURN count(*) AS c",
        ),
    ] {
        let plan = prepare(q).unwrap();
        let mut best = f64::MAX;
        for _ in 0..7 {
            let t = Instant::now();
            std::hint::black_box(plan.execute(&mut g, &p).unwrap());
            best = best.min(t.elapsed().as_secs_f64() * 1e6);
        }
        println!("{label:<22}{best:>10.1} us");
    }
}
