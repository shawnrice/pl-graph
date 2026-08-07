//! In-engine graph algorithms — degree centrality, connected components, label
//! propagation, peer pressure (community detection), PageRank, and shortest path —
//! run natively over `&Graph` in a single call (no per-iteration FFI round-trip), so
//! a whole PageRank/CC/label-prop computation stays in the engine instead of JS.
//!
//! Each algorithm is a pure function over the **public** `Graph` surface
//! (`vertex_indices`, `out_adj`/`in_adj`, `vid`, `set_vertex_prop`, …) so a user can
//! write their own the same way — the built-ins double as worked examples. The
//! [`run`] driver dispatches by name, optionally writes the per-vertex result back
//! to a vertex property, and returns a `RowSet` of `(node, <result>)`.
//!
//! Cross-engine determinism: results are computed in dense-vertex-id order (= NDJSON
//! insertion order, which the TS engine also iterates), summing a vertex's
//! neighbours in adjacency (edge insertion) order — no sorting — so the TS mirror in
//! `@lenke/core` is byte-identical, including PageRank's f64 arithmetic.
//!
//! Parallelism (behind the `parallel` feature; serial twins for wasm) is applied
//! only where it preserves that contract: PageRank fans out **across targets** (each
//! target still sums its own fixed-order contribution list) and label propagation
//! **across vertices** per round (each reads only the frozen snapshot; the winner is
//! order-independent). The dangling reduction and per-source weight sums stay serial
//! — reordering those f64 additions would change the last bits. So parallel-native
//! == serial-native == serial-TS, verified by the differential conformance suite.

use crate::graph::{Graph, Value};
#[cfg(feature = "ndjson")]
use crate::json;
use crate::rowset::RowSet;

mod centrality;
mod components;
mod degree;
mod label_prop;
mod neighbor_aggregate;
mod pagerank;
mod peer_pressure;
mod scc;
mod shortest_path;

/// Parsed algorithm configuration (a superset; each algorithm reads the fields it
/// needs). Deserialized from the JSON object handed across the FFI boundary.
#[derive(Debug, Default, Clone)]
pub struct AlgoConfig {
    /// Restrict traversal to one edge type (`None` = every type).
    pub edge_label: Option<String>,
    /// `"out"` (default) / `"in"` / `"both"` — for degree.
    pub direction: Option<String>,
    /// Numeric edge property to weight by (PageRank / weighted shortest path).
    pub weight_property: Option<String>,
    /// PageRank damping factor (default 0.85).
    pub damping_factor: Option<f64>,
    /// Fixed iteration count (PageRank / label propagation).
    pub iterations: Option<u32>,
    /// Sample-source count for approximate betweenness. When set (and `< |V|`),
    /// Brandes runs from a deterministic evenly-spaced sample of `pivots` sources
    /// and scales the result by `|V|/pivots` — turning the O(V·E) exact pass into
    /// O(pivots·E). Omitted → exact.
    pub pivots: Option<u32>,
    /// Seed/anchor property for label propagation: a vertex carrying a **non-null**
    /// value for this key keeps its own label forever, so communities form around
    /// the seeds instead of collapsing to one on a hubby/scale-free graph. Omitted →
    /// unsupervised (the prior behaviour).
    pub seed_property: Option<String>,
    /// Source vertex external id (shortest path).
    pub source: Option<String>,
    /// Seed vertex external ids for personalized PageRank / random-walk-with-restart
    /// (the restart set). `None`/empty → degenerates to global PageRank.
    pub source_nodes: Option<Vec<String>>,
    /// Target vertex external id (goal-directed shortest path).
    pub target: Option<String>,
    /// If set, write each vertex's result to this property before returning.
    pub write_property: Option<String>,
    /// Shortest-path backend: `"dijkstra"` (default, full SSSP) / `"astar"`
    /// (goal-directed, needs a `target`).
    pub algorithm: Option<String>,
    /// Admissible-heuristic vertex property for A\*.
    pub heuristic_property: Option<String>,
    /// Source list-valued property holding each vertex's feature vector
    /// (`neighborAggregate`). Required for that algorithm.
    pub feature: Option<String>,
    /// Element-wise aggregation for `neighborAggregate`: `"mean"` (default),
    /// `"sum"`, `"max"`, `"min"`.
    pub op: Option<String>,
    /// Include the vertex's own feature vector in its aggregate (`neighborAggregate`).
    pub include_self: Option<bool>,
    /// Normalization for `neighborAggregate`: `"none"` (default) or `"gcn"` — the
    /// symmetric GCN operator, weighting each contributor `j` of `i` by
    /// `1/sqrt(deg_i · deg_j)`. Composes with `weightProperty` (coefficient =
    /// `weight · norm`). `sum`/`mean` only.
    pub norm: Option<String>,
}

impl AlgoConfig {
    #[cfg(feature = "ndjson")]
    fn from_json(s: &str) -> Result<Self, ()> {
        if s.trim().is_empty() {
            return Ok(Self::default());
        }
        let j = json::parse(s)?;
        let string = |k: &str| j.get(k).and_then(json::Json::as_str).map(str::to_string);
        let num = |k: &str| j.get(k).and_then(json::Json::as_f64);
        // A string array (personalized-PageRank seed set): keep only the string
        // elements, dropping any non-string; an absent/non-array key → None.
        let string_array = |k: &str| {
            j.get(k).and_then(json::Json::as_array).map(|a| {
                a.iter()
                    .filter_map(|e| e.as_str().map(str::to_string))
                    .collect()
            })
        };
        Ok(Self {
            edge_label: string("edgeLabel"),
            direction: string("direction"),
            weight_property: string("weightProperty"),
            damping_factor: num("dampingFactor"),
            iterations: num("iterations").map(|n| n as u32),
            pivots: num("pivots").map(|n| n as u32),
            seed_property: string("seedProperty"),
            source: string("source"),
            source_nodes: string_array("sourceNodes"),
            target: string("target"),
            write_property: string("writeProperty"),
            algorithm: string("algorithm"),
            heuristic_property: string("heuristicProperty"),
            feature: string("feature"),
            op: string("op"),
            include_self: j.get("includeSelf").and_then(json::Json::as_bool),
            norm: string("norm"),
        })
    }

    /// Resolve `edge_label` to an etype filter: `Some(None)` = every type,
    /// `Some(Some(id))` = a known type, `None` = a *named but unknown* type (no edge
    /// matches — the algorithm treats the graph as edgeless for that relationship).
    fn etype(&self, graph: &Graph) -> Option<Option<u32>> {
        match &self.edge_label {
            None => Some(None),
            Some(name) => graph.etype.get(name).map(Some),
        }
    }
}

/// Does edge `ei` pass an algorithm's edge-label filter?
///
/// `None` means every type. Otherwise ANY of the edge's labels may match — edges
/// are multi-label, and every algorithm here tested `e_type[ei]`, which is only
/// the FIRST. With the filtered label stored second, `degree` returned zero for
/// every vertex: the algorithms saw an edgeless graph.
#[must_use]
pub fn edge_type_ok(graph: &Graph, etype: Option<u32>, ei: u32) -> bool {
    etype.is_none_or(|t| graph.edge_has_label(ei, t))
}

/// The same test for an adjacency entry, which already carries the first label —
/// so the common case is one compare and the rest is only consulted when some
/// edge carries the wanted label as a secondary.
#[must_use]
pub fn adj_type_ok(graph: &Graph, etype: Option<u32>, a: &crate::graph::Adj) -> bool {
    match etype {
        None => true,
        Some(t) => a.etype == t || graph.edge_has_label(a.eidx, t),
    }
}

/// The accepted `CALL <algo>({...})` config keys — every field [`AlgoConfig`]
/// carries. A `CALL` config map is validated against this set (unknown key →
/// error), so a typo or a wrong key no longer silently no-ops. The order is fixed
/// so the "did you mean" tie-break is deterministic and identical to the TS engine.
pub const CONFIG_KEYS: &[&str] = &[
    "edgeLabel",
    "direction",
    "weightProperty",
    "dampingFactor",
    "iterations",
    "pivots",
    "seedProperty",
    "source",
    "sourceNodes",
    "target",
    "writeProperty",
    "algorithm",
    "heuristicProperty",
    "feature",
    "op",
    "includeSelf",
    "norm",
];

/// Case-insensitive Levenshtein edit distance — a plain DP over `char`s. Shared by
/// the config-key "did you mean" so both engines suggest identically.
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().flat_map(char::to_lowercase).collect();
    let b: Vec<char> = b.chars().flat_map(char::to_lowercase).collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// For an unknown config key, the closest known key within edit distance 2 (else
/// `None`) — the `CALL` config analogue of [`crate::gql::plan::suggest_procedure`].
/// Scans [`CONFIG_KEYS`] in order so ties resolve to the earliest, identically on
/// both engines.
pub fn suggest_config_key(name: &str) -> Option<&'static str> {
    CONFIG_KEYS
        .iter()
        .map(|k| (edit_distance(name, k), *k))
        .filter(|(d, _)| *d <= 2)
        .min_by_key(|(d, _)| *d)
        .map(|(_, k)| k)
}

/// Per-edge numeric weights (indexed by edge id) for property `key`: resolve the
/// key to a dense id once and read each edge via `value_id` — no per-edge key-string
/// hashing and one flat allocation the weighted algorithms can index directly.
/// Absent/non-numeric values read as `0.0`, exactly as `Properties::value` would.
pub(super) fn edge_weights(graph: &Graph, key: &str) -> Vec<f64> {
    let kid = graph.edge_props.keys.get(key);
    (0..graph.edge_slots())
        .map(|ei| match kid {
            Some(k) => match graph.edge_props.value_id(ei, k, &graph.strs) {
                Value::Num(x) => x,
                _ => 0.0,
            },
            None => 0.0,
        })
        .collect()
}

/// An algorithm's output: the result column name + `(dense vertex id, value)` rows.
pub type AlgoOutput = (&'static str, Vec<(u32, Value)>);
/// Pending `writeProperty` writes: the property key + per-vertex `(dense id, value)`s.
#[cfg(feature = "ndjson")]
type PendingWrites = Option<(String, Vec<(u32, Value)>)>;

/// Dispatch by name to the pure `&Graph -> Vec<(vertex, Value)>` algorithm, returning
/// the result column name alongside. Read-only. Unknown `name` → `Err`.
fn dispatch(graph: &Graph, name: &str, cfg: &AlgoConfig) -> Result<AlgoOutput, String> {
    // `writeProperty` reaches the store without passing through a query, so it
    // never meets the commit-time name check that covers the GQL/Gremlin write
    // paths. An empty key here built a graph the engine could serialize but not
    // read back — `graphFromNdjson` rejects an empty key — so reject it up front.
    // Checked here rather than in a caller because both the sync (`run_columns`)
    // and off-thread (`compute_parts`) paths funnel through this one function.
    if let Some(prop) = &cfg.write_property {
        crate::graph::validate_prop_key(prop).map_err(|e| e.message.clone())?;
    }

    Ok(match name {
        "degree" => ("degree", degree::degree(graph, cfg)),
        "connectedComponents" => ("componentId", components::connected_components(graph, cfg)),
        "stronglyConnectedComponents" => (
            "componentId",
            scc::strongly_connected_components(graph, cfg),
        ),
        "onCycle" => ("onCycle", scc::on_cycle(graph, cfg)),
        "labelPropagation" => ("label", label_prop::label_propagation(graph, cfg)),
        "peerPressure" => ("cluster", peer_pressure::peer_pressure(graph, cfg)),
        "pagerank" => ("score", pagerank::pagerank(graph, cfg)),
        "personalizedPagerank" => ("score", pagerank::personalized_pagerank(graph, cfg)),
        "betweenness" => ("centrality", centrality::betweenness(graph, cfg)),
        "closeness" => ("centrality", centrality::closeness(graph, cfg)),
        "shortestPath" => {
            // Dijkstra (and A*) require NON-NEGATIVE weights. With a negative edge
            // the relaxation can keep finding a cheaper path forever, so a negative
            // cycle never terminates — and a negative self-loop is enough: one node
            // and one edge hung the engine indefinitely. The precondition was
            // documented on `dijkstra` but never enforced, which turned a caller's
            // mistake into an unbounded spin instead of an error. Rejected for ANY
            // negative weight, not just a cycle: Dijkstra can also settle a vertex
            // too early and return a silently wrong distance, so "no cycle, so it
            // terminated" is luck rather than a correct answer.
            if let Some(k) = cfg.weight_property.as_deref() {
                // NaN is rejected alongside negatives: it makes every relaxation
                // comparison false and strands the search just as effectively.
                if edge_weights(graph, k)
                    .iter()
                    .any(|w| w.is_nan() || *w < 0.0)
                {
                    return Err(format!(
                        "shortestPath `weightProperty` ({k}) must hold non-negative numbers — Dijkstra does not admit negative weights"
                    ));
                }
            }

            ("distance", shortest_path::shortest_path(graph, cfg))
        }
        "neighborAggregate" => (
            "vector",
            neighbor_aggregate::neighbor_aggregate(graph, cfg)?,
        ),
        other => return Err(format!("unknown algorithm: {other}")),
    })
}

/// Materialize `(vertex, value)` results into a `(node, <column>)` RowSet, mapping
/// each dense vertex id to its external id. Read-only.
fn build_rowset(graph: &Graph, column: &str, results: &[(u32, Value)]) -> RowSet {
    let mut rs = RowSet::new(vec!["node".to_string(), column.to_string()]);
    for (v, val) in results {
        rs.push_row([Value::Str(graph.vid.arc(*v)), val.clone()]);
    }
    rs
}

#[cfg(feature = "ndjson")]
/// Run algorithm `name` with a JSON `config`, optionally write each vertex's result
/// to `config.writeProperty`, and return the result rows `(node, <result-column>)`
/// where `node` is the external vertex id. Unknown `name` → `Err`.
pub fn run(graph: &mut Graph, name: &str, config: &str) -> Result<RowSet, String> {
    let cfg =
        AlgoConfig::from_json(config).map_err(|()| "invalid algorithm config JSON".to_string())?;
    run_with(graph, name, &cfg)
}

/// Like [`run`] but taking a pre-built [`AlgoConfig`] (no JSON round-trip) — the entry
/// used by the in-query Gremlin algorithm steps, which build the config from their
/// step modulators directly.
pub fn run_with(graph: &mut Graph, name: &str, cfg: &AlgoConfig) -> Result<RowSet, String> {
    let (column, results) = run_columns(graph, name, cfg)?;

    Ok(build_rowset(graph, column, &results))
}

/// Run an algorithm and return the raw `(dense vertex id, result value)` rows plus
/// the result column name — applying `writeProperty` but WITHOUT materializing a
/// `RowSet`. The GQL `CALL` path uses this so it can bind `node` as a live
/// `Val::Node` handle (deferring the `{id, labels, properties}` hydration to the
/// rows that actually survive to output) instead of a pre-stringified id.
pub fn run_columns(graph: &mut Graph, name: &str, cfg: &AlgoConfig) -> Result<AlgoOutput, String> {
    let (column, results) = dispatch(graph, name, cfg)?;

    if let Some(prop) = &cfg.write_property {
        for (v, val) in &results {
            graph.set_vertex_prop(*v, prop, val.clone());
        }
    }

    Ok((column, results))
}

/// Read-only counterpart of [`run`] for the async (off-thread) path: compute the
/// result RowSet and return any pending `writeProperty` writes for the caller to
/// apply back on the main thread (where `&mut Graph` is exclusive again). The whole
/// computation touches only `&Graph`, so it is safe to run off the JS thread.
#[cfg(feature = "ndjson")]
pub fn compute_parts(
    graph: &Graph,
    name: &str,
    config: &str,
) -> Result<(RowSet, PendingWrites), String> {
    let cfg =
        AlgoConfig::from_json(config).map_err(|()| "invalid algorithm config JSON".to_string())?;
    let (column, results) = dispatch(graph, name, &cfg)?;
    let rs = build_rowset(graph, column, &results);
    let writes = cfg.write_property.map(|prop| (prop, results));
    Ok((rs, writes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ndjson;

    /// The TinkerPop "modern" graph. 1=marko 2=vadas 3=lop 4=josh 5=ripple 6=peter.
    /// KNOWS: marko→vadas, marko→josh. CREATED: marko→lop, josh→ripple, josh→lop,
    /// peter→lop.
    fn modern() -> Graph {
        let lines = [
            r#"{"type":"node","id":"1","labels":["Person"],"properties":{"name":"marko"}}"#,
            r#"{"type":"node","id":"2","labels":["Person"],"properties":{"name":"vadas"}}"#,
            r#"{"type":"node","id":"3","labels":["Software"],"properties":{"name":"lop"}}"#,
            r#"{"type":"node","id":"4","labels":["Person"],"properties":{"name":"josh"}}"#,
            r#"{"type":"node","id":"5","labels":["Software"],"properties":{"name":"ripple"}}"#,
            r#"{"type":"node","id":"6","labels":["Person"],"properties":{"name":"peter"}}"#,
            r#"{"type":"edge","from":"1","to":"2","labels":["KNOWS"]}"#,
            r#"{"type":"edge","from":"1","to":"4","labels":["KNOWS"]}"#,
            r#"{"type":"edge","from":"1","to":"3","labels":["CREATED"]}"#,
            r#"{"type":"edge","from":"4","to":"5","labels":["CREATED"]}"#,
            r#"{"type":"edge","from":"4","to":"3","labels":["CREATED"]}"#,
            r#"{"type":"edge","from":"6","to":"3","labels":["CREATED"]}"#,
        ];
        ndjson::decode(&lines.join("\n")).unwrap()
    }

    /// A tiny feature graph: a→b, b→c, a→c with list features h.
    /// a=[1,2], b=[3,4], c=[5,6].
    fn featured() -> Graph {
        let lines = [
            r#"{"type":"node","id":"a","labels":["N"],"properties":{"h":[1,2]}}"#,
            r#"{"type":"node","id":"b","labels":["N"],"properties":{"h":[3,4]}}"#,
            r#"{"type":"node","id":"c","labels":["N"],"properties":{"h":[5,6]}}"#,
            r#"{"type":"edge","from":"a","to":"b","labels":["R"]}"#,
            r#"{"type":"edge","from":"b","to":"c","labels":["R"]}"#,
            r#"{"type":"edge","from":"a","to":"c","labels":["R"]}"#,
        ];
        ndjson::decode(&lines.join("\n")).unwrap()
    }

    /// `(external id, aggregate vector)` rows from `neighborAggregate`.
    fn aggregates(g: &mut Graph, cfg: &str) -> Vec<(String, Vec<f64>)> {
        let rs = run(g, "neighborAggregate", cfg).unwrap();
        rs.rows()
            .map(|r| match (&r[0], &r[1]) {
                (Value::Str(id), Value::List(items)) => (
                    id.to_string(),
                    items
                        .iter()
                        .map(|x| match x {
                            Value::Num(n) => *n,
                            _ => panic!("aggregate element not a number"),
                        })
                        .collect(),
                ),
                _ => panic!("unexpected aggregate row shape"),
            })
            .collect()
    }

    #[test]
    fn neighbor_aggregate_mean_sum_out() {
        let mut g = featured();
        // out-neighbours: a→{b,c}, b→{c}, c→{}. mean, no self.
        assert_eq!(
            aggregates(&mut g, r#"{"feature":"h","op":"mean","direction":"out"}"#),
            vec![
                ("a".into(), vec![4.0, 5.0]), // mean([3,4],[5,6])
                ("b".into(), vec![5.0, 6.0]), // c
                ("c".into(), vec![0.0, 0.0]), // no out-neighbours → zero vector
            ]
        );
        assert_eq!(
            aggregates(&mut g, r#"{"feature":"h","op":"sum","direction":"out"}"#),
            vec![
                ("a".into(), vec![8.0, 10.0]),
                ("b".into(), vec![5.0, 6.0]),
                ("c".into(), vec![0.0, 0.0]),
            ]
        );
    }

    #[test]
    fn neighbor_aggregate_include_self_and_both() {
        let mut g = featured();
        // includeSelf, both directions: every vertex sees a,b,c (fully connected here
        // once direction is undirected), so each mean == mean(a,b,c) = [3,4].
        assert_eq!(
            aggregates(
                &mut g,
                r#"{"feature":"h","op":"mean","direction":"both","includeSelf":true}"#
            ),
            vec![
                ("a".into(), vec![3.0, 4.0]),
                ("b".into(), vec![3.0, 4.0]),
                ("c".into(), vec![3.0, 4.0]),
            ]
        );
    }

    #[test]
    fn neighbor_aggregate_max_min_and_write() {
        let mut g = featured();
        // max over out-neighbours: a sees b,c → max = [5,6]; b sees c → [5,6].
        assert_eq!(
            aggregates(&mut g, r#"{"feature":"h","op":"max","direction":"out"}"#),
            vec![
                ("a".into(), vec![5.0, 6.0]),
                ("b".into(), vec![5.0, 6.0]),
                ("c".into(), vec![0.0, 0.0]),
            ]
        );
        // writeProperty stores the aggregate list; reading it back as the feature
        // of a *second* aggregate confirms the write round-trips as a list value.
        run(
            &mut g,
            "neighborAggregate",
            r#"{"feature":"h","op":"sum","direction":"out","writeProperty":"agg"}"#,
        )
        .unwrap();
        // `agg` now holds a=[8,10], b=[5,6], c=[0,0]; sum over out-neighbours of `agg`.
        assert_eq!(
            aggregates(&mut g, r#"{"feature":"agg","op":"sum","direction":"out"}"#),
            vec![
                ("a".into(), vec![5.0, 6.0]), // b.agg + c.agg = [5,6]+[0,0]
                ("b".into(), vec![0.0, 0.0]), // c.agg = [0,0]
                ("c".into(), vec![0.0, 0.0]),
            ]
        );
    }

    #[test]
    fn neighbor_aggregate_config_errors() {
        let mut g = featured();
        assert!(run(&mut g, "neighborAggregate", r#"{"op":"mean"}"#).is_err()); // no feature
        assert!(run(
            &mut g,
            "neighborAggregate",
            r#"{"feature":"h","op":"nope"}"#
        )
        .is_err());
        assert!(run(
            &mut g,
            "neighborAggregate",
            r#"{"feature":"h","direction":"sideways"}"#
        )
        .is_err());
    }

    /// A weighted feature graph: a[1,2]→b[3,4] (w2), a→c[5,6] (w1), b→c (w3).
    fn weighted_featured() -> Graph {
        let lines = [
            r#"{"type":"node","id":"a","labels":["N"],"properties":{"h":[1,2]}}"#,
            r#"{"type":"node","id":"b","labels":["N"],"properties":{"h":[3,4]}}"#,
            r#"{"type":"node","id":"c","labels":["N"],"properties":{"h":[5,6]}}"#,
            r#"{"type":"edge","from":"a","to":"b","labels":["R"],"properties":{"w":2.0}}"#,
            r#"{"type":"edge","from":"a","to":"c","labels":["R"],"properties":{"w":1.0}}"#,
            r#"{"type":"edge","from":"b","to":"c","labels":["R"],"properties":{"w":3.0}}"#,
        ];
        ndjson::decode(&lines.join("\n")).unwrap()
    }

    #[test]
    fn neighbor_aggregate_weighted_sum_and_mean() {
        let mut g = weighted_featured();
        // Weighted SUM over out-neighbours: a = 2·[3,4] + 1·[5,6] = [11,14]; b = 3·[5,6].
        assert_eq!(
            aggregates(
                &mut g,
                r#"{"feature":"h","op":"sum","direction":"out","weightProperty":"w"}"#
            ),
            vec![
                ("a".into(), vec![11.0, 14.0]),
                ("b".into(), vec![15.0, 18.0]),
                ("c".into(), vec![0.0, 0.0]),
            ]
        );
        // Weighted MEAN divides by the WEIGHT sum, not the count: a = [11,14]/3.
        assert_eq!(
            aggregates(
                &mut g,
                r#"{"feature":"h","op":"mean","direction":"out","weightProperty":"w"}"#
            ),
            vec![
                ("a".into(), vec![11.0 / 3.0, 14.0 / 3.0]),
                ("b".into(), vec![5.0, 6.0]), // 3·[5,6]/3
                ("c".into(), vec![0.0, 0.0]),
            ]
        );
    }

    #[test]
    fn neighbor_aggregate_gcn_norm() {
        // Two nodes a[1,2]→b[3,4], `both` + includeSelf → each has degree 2 (one neighbour
        // + self), so every GCN coefficient is 1/sqrt(2·2) = 1/2: sum = ½([1,2]+[3,4]) = [2,3].
        let lines = [
            r#"{"type":"node","id":"a","labels":["N"],"properties":{"h":[1,2]}}"#,
            r#"{"type":"node","id":"b","labels":["N"],"properties":{"h":[3,4]}}"#,
            r#"{"type":"edge","from":"a","to":"b","labels":["R"]}"#,
        ];
        let mut g = ndjson::decode(&lines.join("\n")).unwrap();
        assert_eq!(
            aggregates(
                &mut g,
                r#"{"feature":"h","op":"sum","direction":"both","includeSelf":true,"norm":"gcn"}"#
            ),
            vec![("a".into(), vec![2.0, 3.0]), ("b".into(), vec![2.0, 3.0])]
        );
    }

    #[test]
    fn neighbor_aggregate_weight_norm_reject_maxmin() {
        let mut g = weighted_featured();
        // A weight or a `gcn` norm scales contributions → meaningless for max/min: reject.
        assert!(run(
            &mut g,
            "neighborAggregate",
            r#"{"feature":"h","op":"max","weightProperty":"w"}"#
        )
        .is_err());
        assert!(run(
            &mut g,
            "neighborAggregate",
            r#"{"feature":"h","op":"min","norm":"gcn"}"#
        )
        .is_err());
        // An unknown `norm` value → loud error.
        assert!(run(
            &mut g,
            "neighborAggregate",
            r#"{"feature":"h","op":"sum","norm":"nope"}"#
        )
        .is_err());
    }

    /// `(external id, degree)` rows in engine order.
    fn degrees(g: &mut Graph, cfg: &str) -> Vec<(String, i64)> {
        let rs = run(g, "degree", cfg).unwrap();
        rs.rows()
            .map(|r| match (&r[0], &r[1]) {
                (Value::Str(id), Value::Num(d)) => (id.to_string(), *d as i64),
                _ => panic!("unexpected degree row shape"),
            })
            .collect()
    }

    #[test]
    fn degree_out_in_both_and_typed() {
        let mut g = modern();
        // Rows are in insertion (dense-id) order: nodes "1".."6".
        assert_eq!(
            degrees(&mut g, r#"{"direction":"out"}"#),
            vec![
                ("1".into(), 3),
                ("2".into(), 0),
                ("3".into(), 0),
                ("4".into(), 2),
                ("5".into(), 0),
                ("6".into(), 1),
            ],
        );
        assert_eq!(
            degrees(&mut g, r#"{"direction":"in"}"#),
            vec![
                ("1".into(), 0),
                ("2".into(), 1),
                ("3".into(), 3),
                ("4".into(), 1),
                ("5".into(), 1),
                ("6".into(), 0),
            ],
        );
        assert_eq!(
            degrees(&mut g, r#"{"direction":"both"}"#),
            vec![
                ("1".into(), 3),
                ("2".into(), 1),
                ("3".into(), 3),
                ("4".into(), 3),
                ("5".into(), 1),
                ("6".into(), 1),
            ],
        );
        // Typed: out KNOWS — only marko (→vadas,→josh) = 2.
        assert_eq!(
            degrees(&mut g, r#"{"direction":"out","edgeLabel":"KNOWS"}"#)[0],
            ("1".into(), 2),
        );
        // in CREATED of lop("3") = marko, josh, peter = 3.
        assert_eq!(
            degrees(&mut g, r#"{"direction":"in","edgeLabel":"CREATED"}"#)[2],
            ("3".into(), 3),
        );
        // Unknown edge type → all zero.
        assert!(degrees(&mut g, r#"{"edgeLabel":"NOPE"}"#)
            .iter()
            .all(|(_, d)| *d == 0));
    }

    #[test]
    fn write_property_and_unknown_algo() {
        let mut g = modern();
        run(
            &mut g,
            "degree",
            r#"{"direction":"out","writeProperty":"deg"}"#,
        )
        .unwrap();
        // marko's written degree property is now 3.
        let rs = crate::gql::prepare("MATCH (n) WHERE n.name = 'marko' RETURN n.deg AS d")
            .unwrap()
            .execute(&mut g, &crate::gql::eval::Params::new())
            .unwrap();
        assert_eq!(rs.row(0)[0], Value::Num(3.0));
        assert!(run(&mut g, "nope", "{}").is_err());
    }

    /// `(external id, componentId)` rows in engine order.
    fn components(g: &mut Graph, cfg: &str) -> Vec<(String, String)> {
        let rs = run(g, "connectedComponents", cfg).unwrap();
        rs.rows()
            .map(|r| match (&r[0], &r[1]) {
                (Value::Str(id), Value::Str(c)) => (id.to_string(), c.to_string()),
                _ => panic!("unexpected component row shape"),
            })
            .collect()
    }

    /// Two disjoint components (1–2–3 and 5–4) plus an isolated vertex (6). Node
    /// insertion order 1,2,3,4,5,6 makes the component roots the min-index member:
    /// {1,2,3}→"1", {4,5}→"4", {6}→"6". Edge `5→4` also proves undirected union.
    fn two_components() -> Graph {
        let lines = [
            r#"{"type":"node","id":"1","labels":["N"]}"#,
            r#"{"type":"node","id":"2","labels":["N"]}"#,
            r#"{"type":"node","id":"3","labels":["N"]}"#,
            r#"{"type":"node","id":"4","labels":["N"]}"#,
            r#"{"type":"node","id":"5","labels":["N"]}"#,
            r#"{"type":"node","id":"6","labels":["N"]}"#,
            r#"{"type":"edge","from":"1","to":"2","labels":["E"]}"#,
            r#"{"type":"edge","from":"2","to":"3","labels":["E"]}"#,
            r#"{"type":"edge","from":"5","to":"4","labels":["E"]}"#,
        ];
        ndjson::decode(&lines.join("\n")).unwrap()
    }

    #[test]
    fn wcc_roots_are_min_index_member() {
        let mut g = two_components();
        assert_eq!(
            components(&mut g, "{}"),
            vec![
                ("1".into(), "1".into()),
                ("2".into(), "1".into()),
                ("3".into(), "1".into()),
                ("4".into(), "4".into()),
                ("5".into(), "4".into()),
                ("6".into(), "6".into()),
            ],
        );
        // The whole modern graph is one weakly-connected component rooted at "1".
        let mut m = modern();
        assert!(components(&mut m, "{}").iter().all(|(_, c)| c == "1"));
        // A named-but-unknown edge type → every vertex is its own component.
        assert_eq!(
            components(&mut g, r#"{"edgeLabel":"NOPE"}"#),
            (1..=6)
                .map(|i| (i.to_string(), i.to_string()))
                .collect::<Vec<_>>(),
        );
    }

    /// `(external id, componentId)` rows for SCC in engine order.
    fn scc(g: &mut Graph, cfg: &str) -> Vec<(String, String)> {
        let rs = run(g, "stronglyConnectedComponents", cfg).unwrap();
        rs.rows()
            .map(|r| match (&r[0], &r[1]) {
                (Value::Str(id), Value::Str(c)) => (id.to_string(), c.to_string()),
                _ => panic!("unexpected SCC row shape"),
            })
            .collect()
    }

    #[test]
    fn scc_finds_directed_cycles() {
        // 1→2→3→1 is one SCC; 4→3 and 5→4 are their own singletons (no path back).
        let g = ndjson::decode(
            &[
                r#"{"type":"node","id":"1","labels":["N"]}"#,
                r#"{"type":"node","id":"2","labels":["N"]}"#,
                r#"{"type":"node","id":"3","labels":["N"]}"#,
                r#"{"type":"node","id":"4","labels":["N"]}"#,
                r#"{"type":"node","id":"5","labels":["N"]}"#,
                r#"{"type":"edge","from":"1","to":"2","labels":["E"]}"#,
                r#"{"type":"edge","from":"2","to":"3","labels":["E"]}"#,
                r#"{"type":"edge","from":"3","to":"1","labels":["E"]}"#,
                r#"{"type":"edge","from":"4","to":"3","labels":["E"]}"#,
                r#"{"type":"edge","from":"5","to":"4","labels":["E"]}"#,
            ]
            .join("\n"),
        )
        .unwrap();
        let mut g = g;
        assert_eq!(
            scc(&mut g, "{}"),
            vec![
                ("1".into(), "1".into()),
                ("2".into(), "1".into()),
                ("3".into(), "1".into()),
                ("4".into(), "4".into()),
                ("5".into(), "5".into()),
            ],
        );

        // Direction matters: the modern graph is one WCC but has NO directed cycle,
        // so every vertex is its own SCC (id = own external id).
        let mut m = modern();
        assert_eq!(
            scc(&mut m, "{}"),
            (1..=6)
                .map(|i| (i.to_string(), i.to_string()))
                .collect::<Vec<_>>(),
        );

        // A named-but-unknown edge type → no edges → every vertex its own component.
        assert_eq!(
            scc(&mut g, r#"{"edgeLabel":"NOPE"}"#),
            (1..=5)
                .map(|i| (i.to_string(), i.to_string()))
                .collect::<Vec<_>>(),
        );

        // A 2-cycle nested with a self-referential chain: {1,2} strongly connected,
        // and a longer 3→4→5→3 cycle, sharing the min-index id per component.
        let mut two = ndjson::decode(
            &[
                r#"{"type":"node","id":"1","labels":["N"]}"#,
                r#"{"type":"node","id":"2","labels":["N"]}"#,
                r#"{"type":"node","id":"3","labels":["N"]}"#,
                r#"{"type":"node","id":"4","labels":["N"]}"#,
                r#"{"type":"node","id":"5","labels":["N"]}"#,
                r#"{"type":"edge","from":"1","to":"2","labels":["E"]}"#,
                r#"{"type":"edge","from":"2","to":"1","labels":["E"]}"#,
                r#"{"type":"edge","from":"3","to":"4","labels":["E"]}"#,
                r#"{"type":"edge","from":"4","to":"5","labels":["E"]}"#,
                r#"{"type":"edge","from":"5","to":"3","labels":["E"]}"#,
                r#"{"type":"edge","from":"2","to":"3","labels":["E"]}"#,
            ]
            .join("\n"),
        )
        .unwrap();
        assert_eq!(
            scc(&mut two, "{}"),
            vec![
                ("1".into(), "1".into()),
                ("2".into(), "1".into()),
                ("3".into(), "3".into()),
                ("4".into(), "3".into()),
                ("5".into(), "3".into()),
            ],
        );
    }

    #[test]
    fn on_cycle_flags_cycle_members_and_self_loops() {
        // 1→2→3→1 (a cycle); 4→5 (a chain, not on a cycle); 6→6 (a self-loop).
        let mut g = ndjson::decode(
            &[
                r#"{"type":"node","id":"1","labels":["N"]}"#,
                r#"{"type":"node","id":"2","labels":["N"]}"#,
                r#"{"type":"node","id":"3","labels":["N"]}"#,
                r#"{"type":"node","id":"4","labels":["N"]}"#,
                r#"{"type":"node","id":"5","labels":["N"]}"#,
                r#"{"type":"node","id":"6","labels":["N"]}"#,
                r#"{"type":"edge","from":"1","to":"2","labels":["E"]}"#,
                r#"{"type":"edge","from":"2","to":"3","labels":["E"]}"#,
                r#"{"type":"edge","from":"3","to":"1","labels":["E"]}"#,
                r#"{"type":"edge","from":"4","to":"5","labels":["E"]}"#,
                r#"{"type":"edge","from":"6","to":"6","labels":["E"]}"#,
            ]
            .join("\n"),
        )
        .unwrap();
        let rs = run(&mut g, "onCycle", "{}").unwrap();
        let got: Vec<(String, bool)> = rs
            .rows()
            .map(|r| match (&r[0], &r[1]) {
                (Value::Str(id), Value::Bool(b)) => (id.to_string(), *b),
                _ => panic!("unexpected onCycle row"),
            })
            .collect();
        assert_eq!(
            got,
            vec![
                ("1".into(), true), // SCC {1,2,3}
                ("2".into(), true),
                ("3".into(), true),
                ("4".into(), false), // chain
                ("5".into(), false),
                ("6".into(), true), // self-loop
            ],
        );
        // A named-but-unknown edge type → no edges → nothing is on a cycle.
        let none = run(&mut g, "onCycle", r#"{"edgeLabel":"NOPE"}"#).unwrap();
        assert!(none.rows().all(|r| matches!(&r[1], Value::Bool(false))));
    }

    /// `(external id, label)` rows in engine order.
    fn labels(g: &mut Graph, cfg: &str) -> Vec<(String, String)> {
        let rs = run(g, "labelPropagation", cfg).unwrap();
        rs.rows()
            .map(|r| match (&r[0], &r[1]) {
                (Value::Str(id), Value::Str(l)) => (id.to_string(), l.to_string()),
                _ => panic!("unexpected label row shape"),
            })
            .collect()
    }

    /// Two disjoint triangles {1,2,3} and {4,5,6}. A triangle is a clique, so
    /// synchronous LPA converges each to its smallest-id label within a couple of
    /// rounds and stays there — {1,2,3}→"1", {4,5,6}→"4".
    fn two_triangles() -> Graph {
        let lines = [
            r#"{"type":"node","id":"1","labels":["N"]}"#,
            r#"{"type":"node","id":"2","labels":["N"]}"#,
            r#"{"type":"node","id":"3","labels":["N"]}"#,
            r#"{"type":"node","id":"4","labels":["N"]}"#,
            r#"{"type":"node","id":"5","labels":["N"]}"#,
            r#"{"type":"node","id":"6","labels":["N"]}"#,
            r#"{"type":"edge","from":"1","to":"2","labels":["E"]}"#,
            r#"{"type":"edge","from":"2","to":"3","labels":["E"]}"#,
            r#"{"type":"edge","from":"1","to":"3","labels":["E"]}"#,
            r#"{"type":"edge","from":"4","to":"5","labels":["E"]}"#,
            r#"{"type":"edge","from":"5","to":"6","labels":["E"]}"#,
            r#"{"type":"edge","from":"4","to":"6","labels":["E"]}"#,
        ];
        ndjson::decode(&lines.join("\n")).unwrap()
    }

    #[test]
    fn label_prop_triangles_converge_to_min_label() {
        let mut g = two_triangles();
        assert_eq!(
            labels(&mut g, "{}"),
            vec![
                ("1".into(), "1".into()),
                ("2".into(), "1".into()),
                ("3".into(), "1".into()),
                ("4".into(), "4".into()),
                ("5".into(), "4".into()),
                ("6".into(), "4".into()),
            ],
        );
        // Zero iterations → every vertex keeps its own external id as its label.
        assert_eq!(
            labels(&mut g, r#"{"iterations":0}"#),
            (1..=6)
                .map(|i| (i.to_string(), i.to_string()))
                .collect::<Vec<_>>(),
        );
        // A named-but-unknown edge type → no propagation, labels stay = own id.
        assert_eq!(
            labels(&mut g, r#"{"edgeLabel":"NOPE"}"#),
            (1..=6)
                .map(|i| (i.to_string(), i.to_string()))
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn label_prop_seed_anchors_pin_communities() {
        // Triangle {1,2,3}. Unsupervised, it collapses to the single min label "1".
        // Anchoring vertex 3 keeps it pinned to "3", which breaks the collapse — the
        // deterministic result is three distinct communities instead of one.
        let mut g = ndjson::decode(
            &[
                r#"{"type":"node","id":"1","labels":["N"]}"#,
                r#"{"type":"node","id":"2","labels":["N"]}"#,
                r#"{"type":"node","id":"3","labels":["N"],"properties":{"anchor":true}}"#,
                r#"{"type":"edge","from":"1","to":"2","labels":["E"]}"#,
                r#"{"type":"edge","from":"2","to":"3","labels":["E"]}"#,
                r#"{"type":"edge","from":"1","to":"3","labels":["E"]}"#,
            ]
            .join("\n"),
        )
        .unwrap();
        assert_eq!(
            labels(&mut g, "{}"),
            vec![
                ("1".into(), "1".into()),
                ("2".into(), "1".into()),
                ("3".into(), "1".into()),
            ],
        );
        assert_eq!(
            labels(&mut g, r#"{"seedProperty":"anchor"}"#),
            vec![
                ("1".into(), "1".into()),
                ("2".into(), "2".into()),
                ("3".into(), "3".into()), // the seed keeps its own id
            ],
        );
        // A seed key no vertex carries → unsupervised (every value reads null).
        assert_eq!(
            labels(&mut g, r#"{"seedProperty":"nope"}"#),
            labels(&mut g, "{}"),
        );
    }

    /// `(external id, score)` rows in engine order.
    fn scores(g: &mut Graph, cfg: &str) -> Vec<(String, f64)> {
        let rs = run(g, "pagerank", cfg).unwrap();
        rs.rows()
            .map(|r| match (&r[0], &r[1]) {
                (Value::Str(id), Value::Num(s)) => (id.to_string(), *s),
                _ => panic!("unexpected pagerank row shape"),
            })
            .collect()
    }

    #[test]
    fn pagerank_two_cycle_is_uniform_and_mass_conserving() {
        // 1↔2 symmetric cycle → exactly [0.5, 0.5] (a fixed point of the iteration).
        let two_cycle = ndjson::decode(
            &[
                r#"{"type":"node","id":"1","labels":["N"]}"#,
                r#"{"type":"node","id":"2","labels":["N"]}"#,
                r#"{"type":"edge","from":"1","to":"2","labels":["E"]}"#,
                r#"{"type":"edge","from":"2","to":"1","labels":["E"]}"#,
            ]
            .join("\n"),
        )
        .unwrap();
        let mut g = two_cycle;
        assert_eq!(
            scores(&mut g, "{}"),
            vec![("1".into(), 0.5), ("2".into(), 0.5)]
        );

        // Mass conservation: the scores form a probability distribution summing to 1
        // (dangling redistribution keeps total rank constant). The modern graph has
        // dangling sinks (lop, vadas, ripple), so this exercises that path.
        let mut m = modern();
        let total: f64 = scores(&mut m, "{}").iter().map(|(_, s)| s).sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "PageRank mass not conserved: {total}"
        );

        // The most-created software (lop, id 3: in-degree 3) outranks a leaf (vadas).
        let s = scores(&mut m, "{}");
        let by = |id: &str| s.iter().find(|(v, _)| v == id).unwrap().1;
        assert!(by("3") > by("2"));
    }

    /// `(external id, score)` rows for personalized PageRank in engine order.
    fn pscores(g: &mut Graph, cfg: &str) -> Vec<(String, f64)> {
        let rs = run(g, "personalizedPagerank", cfg).unwrap();
        rs.rows()
            .map(|r| match (&r[0], &r[1]) {
                (Value::Str(id), Value::Num(s)) => (id.to_string(), *s),
                _ => panic!("unexpected personalizedPagerank row shape"),
            })
            .collect()
    }

    #[test]
    fn personalized_pagerank_restarts_to_the_seed_set() {
        let mut m = modern();

        // Mass conservation: a personalized distribution still sums to 1.
        let seeded = pscores(&mut m, r#"{"sourceNodes":["1"]}"#);
        let total: f64 = seeded.iter().map(|(_, s)| s).sum();
        assert!((total - 1.0).abs() < 1e-9, "mass not conserved: {total}");

        // Restarting at marko (id 1) concentrates rank near marko relative to a
        // global run: marko outscores peter (id 6), who is unreachable from marko.
        let by = |v: &[(String, f64)], id: &str| v.iter().find(|(x, _)| x == id).unwrap().1;
        assert!(by(&seeded, "1") > by(&seeded, "6"));
        // Personalizing to peter instead flips it: now peter outscores marko.
        let to_peter = pscores(&mut m, r#"{"sourceNodes":["6"]}"#);
        assert!(by(&to_peter, "6") > by(&to_peter, "1"));

        // Damping 0 → no propagation: the result is exactly the personalization
        // vector (all mass split evenly across the two distinct seeds).
        let d0 = pscores(&mut m, r#"{"sourceNodes":["1","4"],"dampingFactor":0}"#);
        for (id, s) in &d0 {
            let expect = if id == "1" || id == "4" { 0.5 } else { 0.0 };
            assert_eq!(*s, expect, "vertex {id}");
        }
        // A repeated seed doesn't double-weight (distinct set): ["1","1"] == ["1"].
        assert_eq!(pscores(&mut m, r#"{"sourceNodes":["1","1"]}"#), seeded);
        // An unknown seed id is dropped: ["1","999"] still == ["1"].
        assert_eq!(pscores(&mut m, r#"{"sourceNodes":["1","999"]}"#), seeded);

        // No resolvable seed degenerates to global PageRank (mathematically; the
        // teleport arithmetic differs from global's base in the last f64 bits).
        let global = scores(&mut m, "{}");
        for cfg in [r#"{"sourceNodes":[]}"#, r#"{"sourceNodes":["nope"]}"#] {
            for ((_, a), (_, b)) in pscores(&mut m, cfg).iter().zip(&global) {
                assert!((a - b).abs() < 1e-12, "empty-seed != global: {a} vs {b}");
            }
        }
    }

    /// `(external id, cluster)` rows in engine order.
    fn clusters(g: &mut Graph, cfg: &str) -> Vec<(String, String)> {
        let rs = run(g, "peerPressure", cfg).unwrap();
        rs.rows()
            .map(|r| match (&r[0], &r[1]) {
                (Value::Str(id), Value::Str(c)) => (id.to_string(), c.to_string()),
                _ => panic!("unexpected peerPressure row shape"),
            })
            .collect()
    }

    /// Two directed cliques {1,2,3} and {4,5,6} (all 6 intra-triangle edges each) —
    /// peer pressure converges each to its smallest-id cluster ("1" and "4").
    fn two_cliques() -> Graph {
        let mut lines: Vec<String> = (1..=6)
            .map(|i| format!(r#"{{"type":"node","id":"{i}","labels":["N"]}}"#))
            .collect();
        for &(a, b) in &[(1, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 6)] {
            lines.push(format!(
                r#"{{"type":"edge","from":"{a}","to":"{b}","labels":["E"]}}"#
            ));
            lines.push(format!(
                r#"{{"type":"edge","from":"{b}","to":"{a}","labels":["E"]}}"#
            ));
        }
        ndjson::decode(&lines.join("\n")).unwrap()
    }

    #[test]
    fn peer_pressure_cliques_converge_to_min_cluster() {
        let mut g = two_cliques();
        assert_eq!(
            clusters(&mut g, "{}"),
            vec![
                ("1".into(), "1".into()),
                ("2".into(), "1".into()),
                ("3".into(), "1".into()),
                ("4".into(), "4".into()),
                ("5".into(), "4".into()),
                ("6".into(), "4".into()),
            ],
        );
        // A named-but-unknown edge type → no votes, every vertex its own cluster.
        assert_eq!(
            clusters(&mut g, r#"{"edgeLabel":"NOPE"}"#),
            (1..=6)
                .map(|i| (i.to_string(), i.to_string()))
                .collect::<Vec<_>>(),
        );
    }

    /// `(external id, distance)` rows in engine order.
    fn paths(g: &mut Graph, cfg: &str) -> Vec<(String, f64)> {
        let rs = run(g, "shortestPath", cfg).unwrap();
        rs.rows()
            .map(|r| match (&r[0], &r[1]) {
                (Value::Str(id), Value::Num(d)) => (id.to_string(), *d),
                _ => panic!("unexpected shortest-path row shape"),
            })
            .collect()
    }

    /// 1→2 (w1), 2→3 (w2), 1→3 (w5); node 4 isolated (unreachable from 1).
    fn weighted_chain() -> Graph {
        let lines = [
            r#"{"type":"node","id":"1","labels":["N"]}"#,
            r#"{"type":"node","id":"2","labels":["N"]}"#,
            r#"{"type":"node","id":"3","labels":["N"]}"#,
            r#"{"type":"node","id":"4","labels":["N"]}"#,
            r#"{"type":"edge","from":"1","to":"2","labels":["E"],"properties":{"w":1.0}}"#,
            r#"{"type":"edge","from":"2","to":"3","labels":["E"],"properties":{"w":2.0}}"#,
            r#"{"type":"edge","from":"1","to":"3","labels":["E"],"properties":{"w":5.0}}"#,
        ];
        ndjson::decode(&lines.join("\n")).unwrap()
    }

    #[test]
    fn shortest_path_bfs_and_dijkstra() {
        let mut g = weighted_chain();
        // Unweighted BFS from 1: 1→3 is a direct edge, so node 3 is 1 hop.
        assert_eq!(
            paths(&mut g, r#"{"source":"1"}"#),
            vec![("1".into(), 0.0), ("2".into(), 1.0), ("3".into(), 1.0)],
        );
        // Weighted Dijkstra from 1: 1→2→3 (1+2=3) beats the direct 1→3 (5).
        assert_eq!(
            paths(&mut g, r#"{"source":"1","weightProperty":"w"}"#),
            vec![("1".into(), 0.0), ("2".into(), 1.0), ("3".into(), 3.0)],
        );
        // From 2 (weighted): only 2 and 3 reachable; node 1 is upstream, omitted.
        assert_eq!(
            paths(&mut g, r#"{"source":"2","weightProperty":"w"}"#),
            vec![("2".into(), 0.0), ("3".into(), 2.0)],
        );
        // Unknown source → no rows; unknown edge type → only the source at 0.
        assert!(paths(&mut g, r#"{"source":"99"}"#).is_empty());
        assert_eq!(
            paths(&mut g, r#"{"source":"1","edgeLabel":"NOPE"}"#),
            vec![("1".into(), 0.0)],
        );
    }

    /// `(external id, f64 value)` rows in engine order for a centrality algorithm.
    fn centrality(g: &mut Graph, name: &str, cfg: &str) -> Vec<(String, f64)> {
        let rs = run(g, name, cfg).unwrap();
        rs.rows()
            .map(|r| match (&r[0], &r[1]) {
                (Value::Str(id), Value::Num(x)) => (id.to_string(), *x),
                _ => panic!("unexpected centrality row shape"),
            })
            .collect()
    }

    /// A directed path 1→2→3→4: the two interior vertices lie on shortest paths.
    fn directed_path() -> Graph {
        let lines = [
            r#"{"type":"node","id":"1","labels":["N"]}"#,
            r#"{"type":"node","id":"2","labels":["N"]}"#,
            r#"{"type":"node","id":"3","labels":["N"]}"#,
            r#"{"type":"node","id":"4","labels":["N"]}"#,
            r#"{"type":"edge","from":"1","to":"2","labels":["E"]}"#,
            r#"{"type":"edge","from":"2","to":"3","labels":["E"]}"#,
            r#"{"type":"edge","from":"3","to":"4","labels":["E"]}"#,
        ];
        ndjson::decode(&lines.join("\n")).unwrap()
    }

    #[test]
    fn betweenness_directed_path_and_diamond() {
        // Path 1→2→3→4 (directed): vertex 2 is on paths (1,3),(1,4); vertex 3 on
        // (1,4),(2,4). CB[2]=2, CB[3]=2, endpoints 0.
        let mut g = directed_path();
        assert_eq!(
            centrality(&mut g, "betweenness", "{}"),
            vec![
                ("1".into(), 0.0),
                ("2".into(), 2.0),
                ("3".into(), 2.0),
                ("4".into(), 0.0),
            ],
        );

        // Diamond 1→2→4, 1→3→4 (two disjoint shortest 1→4 paths): 2 and 3 each carry
        // half of the single (1,4) pair → CB = 0.5 each; sinks/sources 0.
        let diamond = ndjson::decode(
            &[
                r#"{"type":"node","id":"1","labels":["N"]}"#,
                r#"{"type":"node","id":"2","labels":["N"]}"#,
                r#"{"type":"node","id":"3","labels":["N"]}"#,
                r#"{"type":"node","id":"4","labels":["N"]}"#,
                r#"{"type":"edge","from":"1","to":"2","labels":["E"]}"#,
                r#"{"type":"edge","from":"1","to":"3","labels":["E"]}"#,
                r#"{"type":"edge","from":"2","to":"4","labels":["E"]}"#,
                r#"{"type":"edge","from":"3","to":"4","labels":["E"]}"#,
            ]
            .join("\n"),
        )
        .unwrap();
        let mut d = diamond;
        assert_eq!(
            centrality(&mut d, "betweenness", "{}"),
            vec![
                ("1".into(), 0.0),
                ("2".into(), 0.5),
                ("3".into(), 0.5),
                ("4".into(), 0.0),
            ],
        );
        // A named-but-unknown edge type → no paths → every score 0.
        assert!(centrality(&mut d, "betweenness", r#"{"edgeLabel":"NOPE"}"#)
            .iter()
            .all(|(_, x)| *x == 0.0));
    }

    #[test]
    fn betweenness_sampled_pivots() {
        let mut g = directed_path(); // 1→2→3→4; exact CB = [0, 2, 2, 0]
        let exact = centrality(&mut g, "betweenness", "{}");

        // `pivots` ≥ |V| (or 0) is exactly the exact pass — no sampling, no scaling.
        assert_eq!(centrality(&mut g, "betweenness", r#"{"pivots":4}"#), exact);
        assert_eq!(centrality(&mut g, "betweenness", r#"{"pivots":99}"#), exact);

        // A real sample (2 of 4 sources, evenly spaced → vertices 1 and 3) scales the
        // summed dependencies by 4/2 = 2. Deterministic: same input → same estimate.
        let sampled = centrality(&mut g, "betweenness", r#"{"pivots":2}"#);
        assert_eq!(
            sampled,
            centrality(&mut g, "betweenness", r#"{"pivots":2}"#)
        );
        // The estimate is finite and non-negative everywhere (endpoints stay 0).
        assert!(sampled.iter().all(|(_, x)| x.is_finite() && *x >= 0.0));
        assert_eq!(sampled[0].1, 0.0); // vertex 1 is never an interior node
    }

    #[test]
    fn closeness_directed_path_unnormalized() {
        // Path 1→2→3→4: 1/(1+2+3)=1/6, 1/(1+2)=1/3, 1/1=1, sink 4 reaches nothing → 0.
        let mut g = directed_path();
        assert_eq!(
            centrality(&mut g, "closeness", "{}"),
            vec![
                ("1".into(), 1.0 / 6.0),
                ("2".into(), 1.0 / 3.0),
                ("3".into(), 1.0),
                ("4".into(), 0.0),
            ],
        );
        // Weighted: put w=2 on each edge → distances double, closeness halves.
        let weighted = ndjson::decode(
            &[
                r#"{"type":"node","id":"1","labels":["N"]}"#,
                r#"{"type":"node","id":"2","labels":["N"]}"#,
                r#"{"type":"node","id":"3","labels":["N"]}"#,
                r#"{"type":"edge","from":"1","to":"2","labels":["E"],"properties":{"w":2.0}}"#,
                r#"{"type":"edge","from":"2","to":"3","labels":["E"],"properties":{"w":2.0}}"#,
            ]
            .join("\n"),
        )
        .unwrap();
        let mut w = weighted;
        assert_eq!(
            centrality(&mut w, "closeness", r#"{"weightProperty":"w"}"#),
            vec![
                ("1".into(), 1.0 / 6.0),
                ("2".into(), 1.0 / 2.0),
                ("3".into(), 0.0),
            ],
        );
    }

    #[test]
    fn astar_matches_dijkstra_target_distance() {
        let mut g = weighted_chain();
        // A* to node 3 returns just the target's distance, identical to Dijkstra (3).
        assert_eq!(
            paths(
                &mut g,
                r#"{"source":"1","target":"3","weightProperty":"w","algorithm":"astar"}"#,
            ),
            vec![("3".into(), 3.0)],
        );
        // For every reachable target, A* agrees with Dijkstra's distance.
        let dijkstra = paths(&mut g, r#"{"source":"1","weightProperty":"w"}"#);
        for (id, dist) in dijkstra {
            let astar = paths(
                &mut g,
                &format!(
                    r#"{{"source":"1","target":"{id}","weightProperty":"w","algorithm":"astar"}}"#
                ),
            );
            assert_eq!(astar, vec![(id, dist)]);
        }
        // Unreachable target (upstream) → no rows.
        assert!(paths(
            &mut g,
            r#"{"source":"3","target":"1","weightProperty":"w","algorithm":"astar"}"#,
        )
        .is_empty());
    }

    // -----------------------------------------------------------------------
    // Negative weights are REJECTED, not run.
    //
    // Dijkstra's precondition was documented on `dijkstra` and never enforced, so
    // a graph that violated it did not fail — it spun forever. A negative
    // self-loop (ONE node, ONE edge) was enough to hang the engine, reachable
    // from the public `shortestPath` API and from GQL `CALL shortest_path`.
    // Found by the randomized algorithm differential, whose weight corpus
    // includes negatives.
    // -----------------------------------------------------------------------

    fn weighted(edges: &[(&str, &str, f64)]) -> Graph {
        let mut lines: Vec<String> = ["a", "b", "c"]
            .iter()
            .map(|id| format!(r#"{{"type":"node","id":"{id}","labels":["P"],"properties":{{}}}}"#))
            .collect();
        for (i, (from, to, w)) in edges.iter().enumerate() {
            lines.push(format!(
                r#"{{"type":"edge","id":"e{i}","labels":["R"],"from":"{from}","to":"{to}","properties":{{"w":{w}}}}}"#
            ));
        }
        ndjson::decode(&lines.join("\n")).expect("fixture decodes")
    }

    fn sp(graph: &Graph, weighted_by: Option<&str>) -> Result<AlgoOutput, String> {
        let cfg = AlgoConfig::from_json(&match weighted_by {
            Some(k) => format!(r#"{{"source":"a","weightProperty":"{k}"}}"#),
            None => r#"{"source":"a"}"#.to_string(),
        })
        .expect("config parses");
        dispatch(graph, "shortestPath", &cfg)
    }

    #[test]
    fn negative_self_loop_is_rejected_not_run_forever() {
        // The minimal hang: one node, one edge.
        let g = weighted(&[("a", "a", -1.0)]);

        assert!(
            sp(&g, Some("w")).is_err(),
            "a negative self-loop must be rejected"
        );
    }

    #[test]
    fn negative_cycle_is_rejected() {
        let g = weighted(&[("a", "b", -1.0), ("b", "a", 0.0)]);

        assert!(
            sp(&g, Some("w")).is_err(),
            "a negative cycle must be rejected"
        );
    }

    #[test]
    fn a_single_negative_edge_is_rejected_even_without_a_cycle() {
        // This one terminated before the fix and returned -1. Dijkstra can settle a
        // vertex before a cheaper negative path reaches it, so an acyclic negative
        // graph terminating is luck, not a correct answer.
        let g = weighted(&[("a", "b", -1.0)]);

        assert!(sp(&g, Some("w")).is_err());
    }

    #[test]
    fn the_rejection_names_the_property() {
        let g = weighted(&[("a", "b", -1.0)]);
        let err = sp(&g, Some("w")).expect_err("should reject");

        assert!(err.contains("weightProperty"), "got: {err}");
        assert!(
            err.contains('w'),
            "the offending key should be named: {err}"
        );
    }

    #[test]
    fn non_negative_weights_still_run() {
        let g = weighted(&[("a", "b", 1.0), ("b", "c", 2.5), ("a", "c", 0.0)]);

        assert!(sp(&g, Some("w")).is_ok());
        // And zero weights (the value a missing property takes) are fine.
        let z = weighted(&[("a", "b", 0.0)]);

        assert!(sp(&z, Some("w")).is_ok());
    }

    #[test]
    fn an_unweighted_run_ignores_negative_weights_entirely() {
        // BFS has no such precondition, so the guard must not fire without
        // `weightProperty`.
        let g = weighted(&[("a", "b", -5.0)]);

        assert!(sp(&g, None).is_ok());
    }
}

#[cfg(test)]
mod multi_label_edge_sweep {
    //! EVERY algorithm, over an edge-label filter, on a multi-label graph.
    //!
    //! The reference is a metamorphic one: two graphs identical in every way
    //! except whether the filtered label is stored FIRST or SECOND on each edge.
    //! An algorithm that tests `e_type[ei] == t` — an edge's first label — sees a
    //! different graph in the second case, so any such filter shows up as a
    //! disagreement without needing a hand-computed expected answer per
    //! algorithm.
    //!
    //! Written as a sweep rather than per-algorithm because checking these one at
    //! a time is exactly how the same bug survived three passes in the query
    //! engines.
    use super::{run_with, AlgoConfig};
    use crate::graph::Graph;

    /// A fixed little graph; `first` decides whether the interesting label `R` is
    /// each edge's first label or its second. `NOISE` is a label no query asks
    /// for, present only to occupy the other slot.
    fn build(first: bool) -> Graph {
        let labels = |r_first: bool| {
            if r_first {
                r#"["R","NOISE"]"#
            } else {
                r#"["NOISE","R"]"#
            }
        };
        let mut lines: Vec<String> = (0..8)
            .map(|i| {
                // `f` is a feature vector, read only by `neighborAggregate`.
                format!(
                    r#"{{"type":"node","id":"n{i}","labels":["V"],"properties":{{"f":[{i}.0]}}}}"#
                )
            })
            .collect();

        // A shape with a cycle, a branch and a tail, so components / SCC /
        // centrality / pagerank all have something to distinguish.
        for (i, (a, b)) in [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (5, 6)]
            .into_iter()
            .enumerate()
        {
            lines.push(format!(
                r#"{{"type":"edge","id":"e{i}","from":"n{a}","to":"n{b}","labels":{},"properties":{{"w":1.0}}}}"#,
                labels(first)
            ));
        }

        crate::ndjson::decode(&lines.join("\n")).expect("fixture decodes")
    }

    #[test]
    fn every_algorithm_filters_on_all_of_an_edges_labels() {
        const ALGOS: &[&str] = &[
            "degree",
            "connectedComponents",
            "stronglyConnectedComponents",
            "onCycle",
            "labelPropagation",
            "peerPressure",
            "pagerank",
            "betweenness",
            "closeness",
        ];

        for name in ALGOS {
            let cfg = AlgoConfig {
                edge_label: Some("R".to_string()),
                ..AlgoConfig::default()
            };

            let mut first = build(true);
            let mut second = build(false);
            let a =
                run_with(&mut first, name, &cfg).unwrap_or_else(|e| panic!("`{name}` failed: {e}"));
            let b = run_with(&mut second, name, &cfg)
                .unwrap_or_else(|e| panic!("`{name}` failed: {e}"));

            assert_eq!(
                a, b,
                "`{name}` gives a different answer when `R` is an edge's SECOND \
                 label — it is testing only the first"
            );

            // Agreement is worthless if the filter matched nothing on BOTH sides
            // (two edgeless graphs agree perfectly). Pin that `R` really selects
            // edges by checking it against a label no edge carries.
            let none = AlgoConfig {
                edge_label: Some("ABSENT".to_string()),
                ..AlgoConfig::default()
            };
            let empty = run_with(&mut build(true), name, &none)
                .unwrap_or_else(|e| panic!("`{name}` failed: {e}"));
            assert_ne!(
                a, empty,
                "`{name}` on `R` matches its answer on a label no edge carries — \
                 the sweep would agree vacuously"
            );
        }
    }

    /// The two algorithms the sweep above cannot reach: both need config beyond
    /// an edge label, and both filtered on an adjacency entry's first label.
    #[test]
    fn configured_algorithms_filter_on_all_of_an_edges_labels() {
        let cases: Vec<(&str, AlgoConfig)> = vec![
            (
                "neighborAggregate",
                AlgoConfig {
                    edge_label: Some("R".to_string()),
                    feature: Some("f".to_string()),
                    op: Some("sum".to_string()),
                    ..AlgoConfig::default()
                },
            ),
            (
                "shortestPath",
                AlgoConfig {
                    edge_label: Some("R".to_string()),
                    source: Some("n0".to_string()),
                    target: Some("n4".to_string()),
                    ..AlgoConfig::default()
                },
            ),
        ];

        for (name, cfg) in cases {
            let (mut first, mut second) = (build(true), build(false));
            let a =
                run_with(&mut first, name, &cfg).unwrap_or_else(|e| panic!("`{name}` failed: {e}"));
            let b = run_with(&mut second, name, &cfg)
                .unwrap_or_else(|e| panic!("`{name}` failed: {e}"));
            assert_eq!(
                a, b,
                "`{name}` gives a different answer when `R` is an edge's SECOND label"
            );

            let mut none = cfg.clone();
            none.edge_label = Some("ABSENT".to_string());
            let mut third = build(true);
            let empty = run_with(&mut third, name, &none)
                .unwrap_or_else(|e| panic!("`{name}` failed: {e}"));
            assert_ne!(a, empty, "`{name}` on `R` matches a label no edge carries");
        }
    }
}
