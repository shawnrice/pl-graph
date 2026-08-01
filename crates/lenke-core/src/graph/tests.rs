//! Storage-layer unit tests, extracted from graph.rs. Each inner module is a
//! focused test group over the columnar store, constraints, and transactions.
#![cfg(test)]

#[cfg(test)]
mod wellformed_names {
    //! Labels/edge-types must be non-empty and `::`-free (GraphSON's multi-label
    //! separator); property keys must be non-empty. Enforced at ingestion.
    use crate::graph::*;

    #[test]
    fn label_rules() {
        assert!(validate_label("Person").is_ok());
        assert!(validate_label("a:b").is_ok()); // a single colon is fine
        assert!(validate_label("").is_err()); // empty collapses to "no labels"
        assert!(validate_label("a::b").is_err()); // GraphSON multi-label separator
        assert!(validate_label("::").is_err());
    }

    #[test]
    fn key_rules() {
        assert!(validate_prop_key("name").is_ok());
        assert!(validate_prop_key("a::b").is_ok()); // keys are never `::`-joined
        assert!(validate_prop_key("").is_err());
    }
}

#[cfg(test)]
mod vector_column {
    //! The typed fixed-dim numeric-vector column (`Column::Vec`): an all-numeric
    //! fixed-length list packs into a de-boxed f64 column, invisibly to callers
    //! (`value` reconstructs the identical `Value::List`), and a non-conforming
    //! write promotes it to `Mixed` losslessly.
    use crate::graph::*;

    fn decode(s: &str) -> Graph {
        crate::ndjson::decode(s).unwrap()
    }

    /// Which column variant backs `key` in the vertex store.
    fn col_name(g: &Graph, key: &str) -> &'static str {
        match g.props.col(key) {
            Some(Column::Vec { .. }) => "vec",
            Some(Column::Mixed { .. }) => "mixed",
            Some(Column::Num { .. }) => "num",
            Some(Column::Record { .. }) => "record",
            _ => "other",
        }
    }

    #[test]
    fn numeric_list_packs_into_a_vec_column_and_reads_back_identically() {
        let g = decode(
            r#"{"type":"node","id":"a","labels":["N"],"properties":{"h":[1.5,2.5,3.5]}}
{"type":"node","id":"b","labels":["N"],"properties":{"h":[9,8,7]}}"#,
        );
        assert_eq!(
            col_name(&g, "h"),
            "vec",
            "an all-numeric fixed-len list is a Vec column"
        );
        // `value` reconstructs the exact `Value::List` a caller would have seen.
        let a = g.vid.get("a").unwrap() as usize;
        assert_eq!(
            g.props.value(a, "h", &g.strs),
            Value::List(vec![Value::Num(1.5), Value::Num(2.5), Value::Num(3.5)])
        );
        // Zero-copy slice accessor.
        assert_eq!(g.props.vector(a, "h"), Some(&[1.5, 2.5, 3.5][..]));
        // NDJSON round-trips byte-for-byte (the Vec column encodes like a boxed list).
        let round = crate::ndjson::encode(&g);
        let g2 = crate::ndjson::decode(&round).unwrap();
        assert_eq!(crate::ndjson::encode(&g2), round);
        assert_eq!(col_name(&g2, "h"), "vec");
    }

    #[test]
    fn a_ragged_or_non_numeric_list_stays_mixed() {
        // Different lengths under one key → Mixed.
        let g = decode(
            r#"{"type":"node","id":"a","labels":["N"],"properties":{"h":[1,2,3]}}
{"type":"node","id":"b","labels":["N"],"properties":{"h":[1,2]}}"#,
        );
        assert_eq!(col_name(&g, "h"), "mixed");
        // A non-numeric element → Mixed.
        let g2 = decode(r#"{"type":"node","id":"a","labels":["N"],"properties":{"h":[1,"x",3]}}"#);
        assert_eq!(col_name(&g2, "h"), "mixed");
        assert!(g2
            .props
            .vector(g2.vid.get("a").unwrap() as usize, "h")
            .is_none());
    }

    #[test]
    fn a_mismatched_set_promotes_the_vec_column_to_mixed_losslessly() {
        let mut g = decode(
            r#"{"type":"node","id":"a","labels":["N"],"properties":{"h":[1.0,2.0]}}
{"type":"node","id":"b","labels":["N"],"properties":{"h":[3.0,4.0]}}"#,
        );
        assert_eq!(col_name(&g, "h"), "vec");
        // Overwrite b's vector with a length-3 list → promotes the whole column.
        let b = g.vid.get("b").unwrap();
        g.set_vertex_prop(b, "h", Value::List(vec![Value::Num(5.0); 3]));
        assert_eq!(
            col_name(&g, "h"),
            "mixed",
            "a dim mismatch promotes to Mixed"
        );
        // a's original vector survives the promotion.
        let a = g.vid.get("a").unwrap() as usize;
        assert_eq!(
            g.props.value(a, "h", &g.strs),
            Value::List(vec![Value::Num(1.0), Value::Num(2.0)])
        );
        assert_eq!(
            g.props.value(b as usize, "h", &g.strs),
            Value::List(vec![Value::Num(5.0), Value::Num(5.0), Value::Num(5.0)])
        );
    }

    #[test]
    fn vec_column_uses_less_heap_than_a_boxed_list() {
        // 8 B/f64 contiguous vs a ~40 B Option<Value> slot per element (plus the
        // uncounted heap Vec of boxed Nums that a Mixed list also carries).
        let g = decode(r#"{"type":"node","id":"a","labels":["N"],"properties":{"h":[1,2,3,4]}}"#);
        match g.props.col("h").unwrap() {
            Column::Vec { data, dim, .. } => {
                assert_eq!(*dim, 4);
                assert_eq!(data.len(), 4); // one element × dim
            }
            _ => panic!("expected a Vec column"),
        }
    }

    #[test]
    fn removing_a_vector_clears_presence_but_a_reset_repopulates() {
        let mut g =
            decode(r#"{"type":"node","id":"a","labels":["N"],"properties":{"h":[1.0,2.0]}}"#);
        let a = g.vid.get("a").unwrap();
        g.remove_vertex_prop(a, "h");
        assert!(!g.props.is_present(a as usize, "h"));
        assert_eq!(g.props.value(a as usize, "h", &g.strs), Value::Null);
        // Re-set with a conforming vector — the column is still a Vec.
        g.set_vertex_prop(a, "h", Value::List(vec![Value::Num(7.0), Value::Num(8.0)]));
        assert_eq!(g.props.vector(a as usize, "h"), Some(&[7.0, 8.0][..]));
    }
}

#[cfg(test)]
mod last_write_scope {
    //! The content-derived CDC value-scope of the most recent committed write.
    use crate::graph::*;

    fn run(g: &mut Graph, q: &str) {
        crate::gql::parse(q)
            .unwrap()
            .execute(g, &crate::gql::eval::Params::new())
            .unwrap();
    }

    #[test]
    fn scope_reflects_the_last_write_touched_values() {
        let mut g = crate::ndjson::decode(
            r#"{"type":"node","id":"a","labels":["Msg"],"properties":{"room":1}}"#,
        )
        .unwrap();

        // An INSERT into room 42 → scope ["42"] (a number renders without `.0`).
        run(&mut g, "INSERT (:Msg {room: 42, body: 'hi'})");
        assert_eq!(g.last_write_scope("room"), vec!["42".to_string()]);

        // A SET touching the seed vertex (room 1) → scope ["1"].
        run(&mut g, "MATCH (m:Msg {room: 1}) SET m.body = 'edited'");
        assert_eq!(g.last_write_scope("room"), vec!["1".to_string()]);

        // A write touching two rooms → both, distinct, in touch order.
        run(
            &mut g,
            "INSERT (:Msg {room: 7}), (:Msg {room: 7}), (:Msg {room: 9})",
        );
        let scope = g.last_write_scope("room");
        assert_eq!(scope.len(), 2);
        assert!(scope.contains(&"7".to_string()) && scope.contains(&"9".to_string()));

        // A string scope key renders verbatim; a missing key contributes nothing.
        run(&mut g, "INSERT (:Msg {tenant: 'acme', body: 'x'})");
        assert_eq!(g.last_write_scope("tenant"), vec!["acme".to_string()]);
        assert!(g.last_write_scope("room").is_empty()); // that write set no room
    }
}

#[cfg(test)]
mod null_is_first_class {
    //! `null` is a stored, present property value — NOT sugar for removal. These
    //! lock in the semantics `set_value`/`is_present`/`remove_value` agree on,
    //! and guard against a regression back to the old "SET null removes" model
    //! (a deliberate divergence from Cypher/TinkerPop).
    use crate::graph::*;

    fn props(len: usize) -> Properties {
        let mut p = Properties::default();
        for _ in 0..len {
            p.push_element();
        }
        p
    }

    #[test]
    fn a_stored_null_is_present_and_distinct_from_absent() {
        let mut strs = Dict::default();
        let mut p = props(2);
        p.set_value(0, "k", Value::Null, &mut strs); // row 0: present null; row 1: untouched

        assert!(p.is_present(0, "k"), "a stored null is present");
        assert!(
            matches!(p.value(0, "k", &strs), Value::Null),
            "and reads back as Null"
        );
        assert!(!p.is_present(1, "k"), "an unset key is absent");
        assert!(
            matches!(p.value(1, "k", &strs), Value::Null),
            "absent also reads as Null"
        );
    }

    #[test]
    fn setting_null_stores_it_without_disturbing_a_typed_column() {
        // A Num key set to null on another row keeps both — the column promotes
        // to Mixed rather than the null vanishing.
        let mut strs = Dict::default();
        let mut p = props(2);
        p.set_value(0, "k", Value::Num(5.0), &mut strs);
        p.set_value(1, "k", Value::Null, &mut strs);

        assert!(matches!(p.value(0, "k", &strs), Value::Num(n) if n == 5.0));
        assert!(p.is_present(1, "k"));
        assert!(matches!(p.value(1, "k", &strs), Value::Null));
    }

    #[test]
    fn remove_value_deletes_even_a_stored_null() {
        let mut strs = Dict::default();
        let mut p = props(1);
        p.set_value(0, "k", Value::Null, &mut strs);
        assert!(p.is_present(0, "k"));

        p.remove_value(0, "k"); // explicit removal is the ONLY way to unset it
        assert!(!p.is_present(0, "k"));
    }
}

#[cfg(test)]
mod storable_maps {
    //! A `Value::Map` is a first-class STORED property (boxed in a `Mixed`
    //! column, like a non-numeric list), canonicalized to sorted keys on the way
    //! in — the substrate foundation for GQL records / Gremlin maps.
    use crate::graph::*;

    fn s(x: &str) -> Value {
        Value::Str(x.into())
    }
    fn map(pairs: &[(&str, Value)]) -> Value {
        Value::Map(
            pairs
                .iter()
                .map(|(k, v)| ((*k).into(), v.clone()))
                .collect(),
        )
    }

    #[test]
    fn stored_map_roundtrips_with_keys_sorted() {
        let mut strs = Dict::default();
        let mut p = Properties::default();
        p.push_element();
        // Author keys OUT of order; storage must canonicalize to sorted.
        p.set_value(
            0,
            "meta",
            map(&[("name", s("marko")), ("age", Value::Num(29.0))]),
            &mut strs,
        );
        assert!(p.is_present(0, "meta"));
        assert_eq!(
            p.value(0, "meta", &strs),
            map(&[("age", Value::Num(29.0)), ("name", s("marko"))]),
        );
    }

    #[test]
    fn nested_maps_and_lists_are_canonicalized_recursively() {
        let mut strs = Dict::default();
        let mut p = Properties::default();
        p.push_element();
        p.set_value(
            0,
            "m",
            map(&[
                ("z", Value::Num(1.0)),
                (
                    "a",
                    Value::List(vec![map(&[("y", Value::Num(2.0)), ("x", Value::Num(3.0))])]),
                ),
            ]),
            &mut strs,
        );
        assert_eq!(
            p.value(0, "m", &strs),
            map(&[
                (
                    "a",
                    Value::List(vec![map(&[("x", Value::Num(3.0)), ("y", Value::Num(2.0))])]),
                ),
                ("z", Value::Num(1.0)),
            ]),
        );
    }

    #[test]
    fn duplicate_field_names_collapse_last_wins() {
        let mut strs = Dict::default();
        let mut p = Properties::default();
        p.push_element();
        p.set_value(
            0,
            "m",
            Value::Map(vec![
                ("k".into(), Value::Num(1.0)),
                ("k".into(), Value::Num(2.0)),
            ]),
            &mut strs,
        );
        assert_eq!(p.value(0, "m", &strs), map(&[("k", Value::Num(2.0))]));
    }

    #[test]
    fn map_null_field_is_preserved_and_distinct_from_absence() {
        // A present field with a null value survives the round-trip (null is a
        // first-class value inside a record, mirroring the top-level policy).
        let mut strs = Dict::default();
        let mut p = Properties::default();
        p.push_element();
        p.set_value(0, "m", map(&[("k", Value::Null)]), &mut strs);
        assert_eq!(p.value(0, "m", &strs), map(&[("k", Value::Null)]));
    }

    #[test]
    fn a_map_key_coexists_with_scalar_keys_via_mixed() {
        let mut strs = Dict::default();
        let mut p = Properties::default();
        p.push_element();
        p.set_value(0, "n", Value::Num(1.0), &mut strs);
        p.set_value(0, "m", map(&[("a", Value::Num(1.0))]), &mut strs);
        assert!(matches!(p.value(0, "n", &strs), Value::Num(n) if n == 1.0));
        assert_eq!(p.value(0, "m", &strs), map(&[("a", Value::Num(1.0))]));
    }

    // A three-vertex graph with a nested `meta.city` field, for the dotted-path
    // index. Never index the map — index the scalar leaf at the path.
    fn city_graph() -> Graph {
        crate::ndjson::decode(
            "{\"type\":\"node\",\"id\":\"a\",\"labels\":[\"P\"],\"properties\":{\"meta\":{\"city\":\"NYC\"}}}\n\
             {\"type\":\"node\",\"id\":\"b\",\"labels\":[\"P\"],\"properties\":{\"meta\":{\"city\":\"LA\"}}}\n\
             {\"type\":\"node\",\"id\":\"c\",\"labels\":[\"P\"],\"properties\":{\"meta\":{\"city\":\"NYC\"}}}",
        )
        .unwrap()
    }

    #[test]
    fn dotted_path_index_seeks_a_nested_field() {
        let mut g = city_graph();
        g.create_vertex_index("meta.city");
        // Both NYC vertices (a=0, c=2), the one LA vertex (b=1).
        let nyc = g
            .vertices_by_prop("meta.city", &IdxKey::Str("NYC".into()))
            .unwrap();
        assert_eq!(nyc, &[0, 2]);
        let la = g
            .vertices_by_prop("meta.city", &IdxKey::Str("LA".into()))
            .unwrap();
        assert_eq!(la, &[1]);
        // A city with no vertex → an empty (but present) bucket.
        assert_eq!(
            g.vertices_by_prop("meta.city", &IdxKey::Str("SF".into())),
            Some(&[][..]),
        );
    }

    #[test]
    fn dotted_path_index_maintained_on_write() {
        let mut g = city_graph();
        g.create_vertex_index("meta.city");
        // Move vertex b (LA → NYC): the index must follow. Bucket order is
        // unspecified, so compare as a set.
        g.set_vertex_prop(1, "meta", map(&[("city", s("NYC"))]));
        let mut nyc = g
            .vertices_by_prop("meta.city", &IdxKey::Str("NYC".into()))
            .unwrap()
            .to_vec();
        nyc.sort_unstable();
        assert_eq!(nyc, vec![0, 1, 2]);
        assert_eq!(
            g.vertices_by_prop("meta.city", &IdxKey::Str("LA".into())),
            Some(&[][..]),
        );
    }

    #[test]
    fn dotted_path_index_skips_absent_or_nonscalar_leaves() {
        let mut g = city_graph();
        // An index into a field that doesn't exist on any vertex → empty index.
        g.create_vertex_index("meta.zip");
        assert_eq!(
            g.vertices_by_prop("meta.zip", &IdxKey::Str("10001".into())),
            Some(&[][..]),
        );
        // Point one vertex's `meta.zip` at a nested map (non-scalar) → not indexed.
        g.set_vertex_prop(
            0,
            "meta",
            map(&[("city", s("NYC")), ("zip", map(&[("k", s("v"))]))]),
        );
        assert_eq!(
            g.vertices_by_prop("meta.zip", &IdxKey::Str("10001".into())),
            Some(&[][..]),
        );
    }
}

#[cfg(test)]
mod transactions {
    //! Transactions: an explicit transaction over the GQL eval mutation path must roll
    //! back to byte-identical prior state, and commit must persist. The eval layer
    //! wraps each statement in its own auto-commit frame, so these tests exercise
    //! the *nested* case (explicit begin → statements → rollback/commit), where
    //! the inner per-statement frames join the outer one.
    use crate::gql::eval::Params;
    use crate::gql::parse;
    use crate::graph::*;
    use crate::ndjson;

    fn run(g: &mut Graph, q: &str) {
        parse(q)
            .unwrap()
            .execute(g, &Params::new())
            .unwrap_or_else(|e| panic!("query failed: {q}: {e:?}"));
    }

    #[test]
    fn rollback_restores_exact_prior_state() {
        let mut g = ndjson::decode("").unwrap();
        // Seed committed data (outside any explicit transaction).
        run(&mut g, "INSERT (:User {name: 'Seed', age: 1})");
        let before = ndjson::encode(&g);
        let vc_before = g.vertex_count();

        g.begin_tx();
        // A brand-new vertex (insert) and a mutation of the seed (property write).
        run(&mut g, "INSERT (:User {name: 'A'})");
        run(
            &mut g,
            "MATCH (u:User {name: 'Seed'}) SET u.name = 'Changed', u.age = 99",
        );
        // Read-your-writes: the staged inserts are visible inside the transaction.
        assert_eq!(g.vertex_count(), vc_before + 1);

        g.rollback_tx();

        assert_eq!(g.vertex_count(), vc_before, "vertex_count restored");
        assert_eq!(ndjson::encode(&g), before, "serialization byte-identical");
        // The seed's property values are exactly as before.
        let rows = parse("MATCH (u:User {name: 'Seed'}) RETURN u.age")
            .unwrap()
            .execute(&mut g, &Params::new())
            .unwrap();
        assert_eq!(
            rows.rows().count(),
            1,
            "the changed-then-rolled-back seed is back"
        );
    }

    #[test]
    fn commit_persists() {
        let mut g = ndjson::decode("").unwrap();
        g.begin_tx();
        run(&mut g, "INSERT (:User {name: 'A'})");
        assert!(matches!(g.commit_tx(), Ok(())));
        assert_eq!(g.vertex_count(), 1, "the committed insert persists");
        assert!(!g.in_transaction());
    }

    #[test]
    fn rollback_restores_deleted_vertex_and_its_edge() {
        // DETACH DELETE cascades an edge removal; rollback must un-tombstone both
        // the vertex and the edge in place (byte-identical serialization).
        let mut g = ndjson::decode("").unwrap();
        run(
            &mut g,
            "INSERT (:User {name: 'A'})-[:KNOWS {since: 2020}]->(:User {name: 'B'})",
        );
        let before = ndjson::encode(&g);
        let (vc, ec) = (g.vertex_count(), g.edge_count());

        g.begin_tx();
        run(&mut g, "MATCH (u:User {name: 'A'}) DETACH DELETE u");
        assert_eq!(g.vertex_count(), vc - 1);
        assert_eq!(g.edge_count(), ec - 1);

        g.rollback_tx();

        assert_eq!(g.vertex_count(), vc, "vertex restored");
        assert_eq!(g.edge_count(), ec, "cascaded edge restored");
        assert_eq!(ndjson::encode(&g), before, "serialization byte-identical");
    }

    #[test]
    fn per_statement_atomicity_leaves_no_partial_write() {
        // A single INSERT of two rows whose second collides under a unique
        // constraint must leave ZERO rows — the whole statement rolls back.
        let mut g = ndjson::decode("").unwrap();
        g.create_unique_constraint("Acct", "email").unwrap();
        let err = parse("INSERT (:Acct {email: 'a@x.io'}), (:Acct {email: 'a@x.io'})")
            .unwrap()
            .execute(&mut g, &Params::new())
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::ConstraintViolation);
        assert_eq!(g.vertex_count(), 0, "the faulting statement left no trace");
    }
}

#[cfg(test)]
mod cardinality {
    //! Cardinality constraints (degree bounds), exercised over the GQL eval
    //! path (each statement is an auto-commit frame, so max AND min land at the
    //! per-statement commit). Byte-identical to the TS core.
    use crate::gql::eval::Params;
    use crate::gql::parse;
    use crate::graph::*;
    use crate::ndjson;

    fn run(g: &mut Graph, q: &str) -> CodeResult<()> {
        parse(q).unwrap().execute(g, &Params::new()).map(|_| ())
    }

    #[test]
    fn exactly_one_via_gql_commit() {
        let mut g = ndjson::decode("").unwrap();
        g.create_cardinality_constraint("Purchase", "PLACED_BY", 0, 1, Some(1))
            .unwrap();

        // Node + mandatory edge in one INSERT (one auto-commit frame) satisfies it.
        run(
            &mut g,
            "INSERT (:Purchase {id: 'o1'})-[:PLACED_BY]->(:Customer {id: 'c1'})",
        )
        .unwrap();
        assert_eq!(g.vertex_count(), 2);

        // A bare Purchase with no PLACED_BY out-edge is degree 0 < min → rejected, and
        // the statement rolls back (no trace).
        let err = run(&mut g, "INSERT (:Purchase {id: 'o2'})").unwrap_err();
        assert_eq!(err.code, ErrorCode::ConstraintViolation);
        assert_eq!(g.vertex_count(), 2, "the rejected INSERT left no trace");
    }

    #[test]
    fn over_max_is_rejected_at_commit() {
        let mut g = ndjson::decode("").unwrap();
        g.create_cardinality_constraint("Purchase", "PLACED_BY", 0, 0, Some(1))
            .unwrap();
        run(
            &mut g,
            "INSERT (:Purchase {id: 'o1'})-[:PLACED_BY]->(:Customer {id: 'c1'})",
        )
        .unwrap();
        // A second PLACED_BY out-edge from o1 pushes its out-degree to 2 > max 1.
        let err = run(
            &mut g,
            "MATCH (o:Purchase {id: 'o1'}), (c:Customer {id: 'c1'}) INSERT (o)-[:PLACED_BY]->(c)",
        )
        .unwrap_err();
        assert_eq!(err.code, ErrorCode::ConstraintViolation);
        assert_eq!(g.edge_count(), 1, "the over-max edge rolled back");
    }

    #[test]
    fn remove_edge_below_min_rolls_back() {
        let mut g = ndjson::decode("").unwrap();
        run(
            &mut g,
            "INSERT (:Purchase {id: 'o1'})-[:PLACED_BY]->(:Customer {id: 'c1'})",
        )
        .unwrap();
        g.create_cardinality_constraint("Purchase", "PLACED_BY", 0, 1, Some(1))
            .unwrap();
        // Deleting the only PLACED_BY edge drops o1 to degree 0 < min → rejected.
        let err = run(&mut g, "MATCH (:Purchase)-[r:PLACED_BY]->() DELETE r").unwrap_err();
        assert_eq!(err.code, ErrorCode::ConstraintViolation);
        assert_eq!(g.edge_count(), 1, "the delete rolled back");
    }

    #[test]
    fn declare_time_scan_and_self_loop_degree() {
        let mut g = ndjson::decode("").unwrap();
        run(&mut g, "INSERT (:Purchase {id: 'o1'})").unwrap(); // degree 0
                                                               // min:1 over existing degree-0 data → rejected at declare time.
        let err = g
            .create_cardinality_constraint("Purchase", "PLACED_BY", 0, 1, Some(1))
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::ConstraintViolation);

        // A self-loop counts once for out and once for in.
        run(
            &mut g,
            "MATCH (o:Purchase {id: 'o1'}) INSERT (o)-[:SELF]->(o)",
        )
        .unwrap();
        // The sole Purchase vertex is index 0 (first inserted); `id` is a property,
        // not the external vertex identity, so degree is read by index here.
        assert_eq!(g.out_degree(0, "SELF"), 1);
        assert_eq!(g.in_degree(0, "SELF"), 1);
    }

    #[test]
    fn drop_and_introspection() {
        let mut g = ndjson::decode("").unwrap();
        g.create_cardinality_constraint("Purchase", "PLACED_BY", 0, 1, Some(1))
            .unwrap();
        g.create_cardinality_constraint("Customer", "PRIMARY", 1, 0, Some(1))
            .unwrap();
        assert_eq!(
            g.cardinality_constraints(),
            vec![
                ("Customer".into(), "PRIMARY".into(), 1, 0, Some(1)),
                ("Purchase".into(), "PLACED_BY".into(), 0, 1, Some(1)),
            ]
        );
        // Re-declaring replaces the bounds (not a second entry).
        g.create_cardinality_constraint("Purchase", "PLACED_BY", 0, 0, None)
            .unwrap();
        assert_eq!(g.cardinality_constraints().len(), 2);
        g.drop_cardinality_constraint("Purchase", "PLACED_BY", 0);
        assert_eq!(
            g.cardinality_constraints(),
            vec![("Customer".into(), "PRIMARY".into(), 1, 0, Some(1))]
        );
        g.drop_cardinality_constraint("Purchase", "PLACED_BY", 0); // idempotent
    }
}

#[cfg(test)]
mod validator {
    //! Custom validator constraints (a GQL boolean predicate per label),
    //! exercised over the GQL eval path (each statement is an auto-commit frame,
    //! so the predicate is re-checked against every touched element at the
    //! per-statement commit). SQL-`CHECK` semantics — a definite `false` fails, a
    //! null/unknown passes. Byte-identical to the TS `createValidator`.
    use crate::gql::eval::Params;
    use crate::gql::parse;
    use crate::graph::*;
    use crate::ndjson;

    fn run(g: &mut Graph, q: &str) -> CodeResult<()> {
        parse(q).unwrap().execute(g, &Params::new()).map(|_| ())
    }

    #[test]
    fn per_write_reject_accept_and_null_passes() {
        let mut g = ndjson::decode("").unwrap();
        g.create_validator("User", "u", "u.age >= 0 AND u.age < 150")
            .unwrap();

        let err = run(&mut g, "INSERT (:User {age: -5})").unwrap_err();
        assert_eq!(err.code, ErrorCode::ConstraintViolation);
        assert_eq!(g.vertex_count(), 0, "the rejected INSERT left no trace");

        run(&mut g, "INSERT (:User {age: 20})").unwrap();
        // No `age` → `u.age` is null → predicate UNKNOWN → passes (SQL-CHECK).
        run(&mut g, "INSERT (:User {name: 'Ada'})").unwrap();
        run(&mut g, "INSERT (:User {age: null, name: 'Bo'})").unwrap();
        assert_eq!(g.vertex_count(), 3);
    }

    #[test]
    fn declare_time_scan_rejects_violating_data() {
        let mut g = ndjson::decode("").unwrap();
        run(&mut g, "INSERT (:User {age: -5})").unwrap();

        let err = g.create_validator("User", "u", "u.age >= 0").unwrap_err();
        assert_eq!(err.code, ErrorCode::ConstraintViolation);
        // The rejected declaration registered nothing.
        assert!(g.validators().is_empty());
    }

    #[test]
    fn deferred_within_a_transaction() {
        // Briefly-invalid-then-fixed across an explicit multi-statement frame → the
        // final state satisfies the validator, so the transaction commits.
        let mut g2 = ndjson::decode("").unwrap();
        g2.create_validator("User", "u", "u.age >= 0").unwrap();
        g2.begin_tx();
        parse("INSERT (:User {id: 'a', age: -5})")
            .unwrap()
            .execute(&mut g2, &Params::new())
            .unwrap();
        parse("MATCH (u:User {id: 'a'}) SET u.age = 5")
            .unwrap()
            .execute(&mut g2, &Params::new())
            .unwrap();
        assert!(g2.commit_tx().is_ok(), "final state valid → commits");
        assert_eq!(g2.vertex_count(), 1);

        // Left invalid across the frame → the whole transaction rolls back.
        let mut g3 = ndjson::decode("").unwrap();
        g3.create_validator("User", "u", "u.age >= 0").unwrap();
        g3.begin_tx();
        parse("INSERT (:User {id: 'b', age: -1})")
            .unwrap()
            .execute(&mut g3, &Params::new())
            .unwrap();
        let err = g3.commit_tx().unwrap_err();
        assert!(matches!(err, TxCommitError::Validator(_)));
        g3.rollback_tx();
        assert_eq!(g3.vertex_count(), 0, "rolled back");
    }

    #[test]
    fn edge_validator() {
        let mut g = ndjson::decode("").unwrap();
        g.create_validator("KNOWS", "r", "r.weight >= 0").unwrap();

        let err = run(
            &mut g,
            "INSERT (:P {name: 'a'})-[:KNOWS {weight: -1}]->(:P {name: 'b'})",
        )
        .unwrap_err();
        assert_eq!(err.code, ErrorCode::ConstraintViolation);
        assert_eq!(g.edge_count(), 0, "rejected edge left no trace");

        run(
            &mut g,
            "INSERT (:P {name: 'a'})-[:KNOWS {weight: 5}]->(:P {name: 'b'})",
        )
        .unwrap();
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn drop_and_introspection() {
        let mut g = ndjson::decode("").unwrap();
        g.create_validator("User", "u", "u.age >= 0").unwrap();
        g.create_validator("User", "u", "u.age < 150").unwrap();

        assert_eq!(
            g.validators(),
            vec![
                ("User".into(), "u".into(), "u.age < 150".into()),
                ("User".into(), "u".into(), "u.age >= 0".into()),
            ]
        );

        g.drop_validator("User");
        assert!(g.validators().is_empty());
        // No validator left → a previously-rejected write now succeeds.
        run(&mut g, "INSERT (:User {age: -5})").unwrap();
        assert_eq!(g.vertex_count(), 1);
    }

    #[test]
    fn unparseable_predicate_is_a_syntax_error() {
        let mut g = ndjson::decode("").unwrap();
        assert_eq!(
            g.create_validator("User", "u", "u.age >>>")
                .unwrap_err()
                .code,
            ErrorCode::Syntax
        );
        assert_eq!(
            g.create_validator("User", "u", "").unwrap_err().code,
            ErrorCode::Syntax
        );
        // A predicate smuggling in an extra clause is rejected too.
        assert_eq!(
            g.create_validator("User", "u", "true RETURN 1")
                .unwrap_err()
                .code,
            ErrorCode::Syntax
        );
    }

    #[test]
    fn predicate_referencing_the_wrong_variable_is_rejected_at_declare_time() {
        let mut g = ndjson::decode("").unwrap();
        // The predicate references `x`, but the element binds to `u` — `x.age` is
        // unbound → the predicate reads UNKNOWN → the SQL-CHECK never fires and the
        // validator would silently do nothing. Reject it at DECLARE time (Syntax).
        assert_eq!(
            g.create_validator("User", "u", "x.age >= 0")
                .unwrap_err()
                .code,
            ErrorCode::Syntax
        );
        // A bare unbound name (no dotted property) is rejected too.
        assert_eq!(
            g.create_validator("User", "u", "age >= 0")
                .unwrap_err()
                .code,
            ErrorCode::Syntax
        );
        // The rejected declarations registered nothing.
        assert!(g.validators().is_empty());

        // The declared variable is fine, and a constant predicate (references NO
        // variable at all) is legitimately allowed.
        g.create_validator("User", "u", "u.age >= 0").unwrap();
        g.create_validator("User", "u", "1 = 1").unwrap();
        assert_eq!(g.validators().len(), 2);

        // A sub-query pattern variable is bound *within* the sub-query, so a
        // predicate that references only `u` and its own sub-pattern vars is fine.
        g.create_validator("User", "u", "EXISTS { (v) WHERE v.age = u.age }")
            .unwrap();
    }
}

#[cfg(test)]
mod invariant {
    //! Graph-level INVARIANTS (cross-write assertions): a whole-graph GQL query
    //! run ONCE per write transaction against the fully-staged graph. `false`-only
    //! -fails — VIOLATED iff a result cell is boolean `false`; everything else
    //! (`true`/`null`/non-boolean/empty) holds. Enforced in `commit_tx` after the
    //! per-element deferred checks, and only when the transaction wrote something.
    //! Byte-identical to the TS `createInvariant`.
    use crate::gql::eval::Params;
    use crate::gql::parse;
    use crate::graph::*;
    use crate::ndjson;

    fn run(g: &mut Graph, q: &str) -> CodeResult<()> {
        parse(q).unwrap().execute(g, &Params::new()).map(|_| ())
    }

    // Two accounts summing to zero; the classic double-entry ledger. The `name`
    // property (not the ndjson node id) is what MATCH patterns key on.
    const LEDGER: &str = "\
{\"type\":\"node\",\"id\":\"a\",\"labels\":[\"Acct\"],\"properties\":{\"name\":\"a\",\"balance\":100}}
{\"type\":\"node\",\"id\":\"b\",\"labels\":[\"Acct\"],\"properties\":{\"name\":\"b\",\"balance\":-100}}";

    #[test]
    fn balanced_transfer_commits_unbalanced_rolls_back() {
        let mut g = ndjson::decode(LEDGER).unwrap();
        g.create_invariant("balanced", "MATCH (a:Acct) RETURN sum(a.balance) = 0")
            .unwrap();

        // A transfer that keeps the sum at zero commits.
        g.begin_tx();
        run(&mut g, "MATCH (a:Acct {name: 'a'}) SET a.balance = 70").unwrap();
        run(&mut g, "MATCH (b:Acct {name: 'b'}) SET b.balance = -70").unwrap();
        assert!(g.commit_tx().is_ok(), "sum still 0 → commits");

        // An unbalanced half-transfer rolls the whole transaction back.
        g.begin_tx();
        run(&mut g, "MATCH (a:Acct {name: 'a'}) SET a.balance = 999").unwrap();
        let err = g.commit_tx().unwrap_err();
        assert!(matches!(err, TxCommitError::Invariant(_)));
        g.rollback_tx();

        // The balances are unchanged from the last good commit (70 / -70).
        let rows = parse("MATCH (a:Acct) RETURN sum(a.balance) AS s")
            .unwrap()
            .execute(&mut g, &Params::new())
            .unwrap();
        assert_eq!(rows.row(0)[0], Value::Num(0.0));
    }

    #[test]
    fn single_statement_unbalanced_write_rejected() {
        // Every GQL statement auto-commits, so a single unbalanced SET trips the
        // invariant at its own commit boundary (no explicit transaction needed).
        let mut g = ndjson::decode(LEDGER).unwrap();
        g.create_invariant("balanced", "MATCH (a:Acct) RETURN sum(a.balance) = 0")
            .unwrap();

        let err = run(&mut g, "MATCH (a:Acct {name: 'a'}) SET a.balance = 5").unwrap_err();
        assert_eq!(err.code, ErrorCode::ConstraintViolation);
        // Rolled back — the balance is still 100.
        let rows = parse("MATCH (a:Acct {name: 'a'}) RETURN a.balance AS b")
            .unwrap()
            .execute(&mut g, &Params::new())
            .unwrap();
        assert_eq!(rows.row(0)[0], Value::Num(100.0));
    }

    #[test]
    fn declare_time_rejects_already_violating_graph() {
        let mut g = ndjson::decode(LEDGER).unwrap();
        run(&mut g, "MATCH (a:Acct {name: 'a'}) SET a.balance = 5").ok(); // no invariant yet → fine
                                                                          // Now the sum is -95, so declaring the invariant must reject.
        let err = g
            .create_invariant("balanced", "MATCH (a:Acct) RETURN sum(a.balance) = 0")
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::ConstraintViolation);
        assert!(
            g.invariants().is_empty(),
            "rejected declaration stored nothing"
        );
    }

    #[test]
    fn count_invariant_at_least_one_admin() {
        let seed = "\
{\"type\":\"node\",\"id\":\"u1\",\"labels\":[\"User\"],\"properties\":{\"name\":\"u1\",\"role\":\"Admin\"}}
{\"type\":\"node\",\"id\":\"u2\",\"labels\":[\"User\"],\"properties\":{\"name\":\"u2\",\"role\":\"Member\"}}";
        let mut g = ndjson::decode(seed).unwrap();
        g.create_invariant(
            "has_admin",
            "MATCH (u:User) WHERE u.role = 'Admin' RETURN count(u) > 0",
        )
        .unwrap();

        // Demote the member → still one admin → holds.
        run(&mut g, "MATCH (u:User {name: 'u2'}) SET u.role = 'Guest'").unwrap();
        // Demote the last admin → count drops to 0 → violated, rolled back.
        let err = run(&mut g, "MATCH (u:User {name: 'u1'}) SET u.role = 'Guest'").unwrap_err();
        assert_eq!(err.code, ErrorCode::ConstraintViolation);
        let rows = parse("MATCH (u:User {role: 'Admin'}) RETURN count(u) AS n")
            .unwrap()
            .execute(&mut g, &Params::new())
            .unwrap();
        assert_eq!(rows.row(0)[0], Value::Num(1.0));
    }

    #[test]
    fn pure_read_transaction_does_not_run_the_invariant() {
        // The gate proof: with the graph in a state that VIOLATES the invariant, a
        // pure-read transaction must still commit (the invariant is not run), while
        // a transaction that writes anything trips it. We break the sum via the
        // direct store API (which bypasses the GQL auto-commit that would catch it)
        // to set up a violating-but-committed state.
        let mut g = ndjson::decode(LEDGER).unwrap();
        g.create_invariant("balanced", "MATCH (a:Acct) RETURN sum(a.balance) = 0")
            .unwrap();

        // Directly skew one balance so the sum is now -50 (invariant would fail).
        let vi = g.vertex_indices().next().unwrap();
        g.set_vertex_prop(vi, "balance", Value::Num(50.0));

        // A pure-read transaction commits — the invariant is skipped (nothing written).
        g.begin_tx();
        parse("MATCH (a:Acct) RETURN a.balance")
            .unwrap()
            .execute(&mut g, &Params::new())
            .unwrap();
        assert!(g.commit_tx().is_ok(), "pure-read commit skips invariants");

        // But a transaction that writes runs the invariant against the (violating)
        // staged graph and rolls back.
        g.begin_tx();
        run(&mut g, "MATCH (a:Acct {name: 'b'}) SET a.balance = -100").unwrap();
        assert!(
            matches!(g.commit_tx().unwrap_err(), TxCommitError::Invariant(_)),
            "a writing commit runs the invariant"
        );
        g.rollback_tx();
    }

    #[test]
    fn drop_and_introspection() {
        let mut g = ndjson::decode(LEDGER).unwrap();
        g.create_invariant("balanced", "MATCH (a:Acct) RETURN sum(a.balance) = 0")
            .unwrap();
        g.create_invariant("has_acct", "MATCH (a:Acct) RETURN count(a) >= 0")
            .unwrap();
        assert_eq!(
            g.invariants(),
            vec![
                (
                    "balanced".into(),
                    "MATCH (a:Acct) RETURN sum(a.balance) = 0".into()
                ),
                (
                    "has_acct".into(),
                    "MATCH (a:Acct) RETURN count(a) >= 0".into()
                ),
            ]
        );

        g.drop_invariant("balanced");
        assert_eq!(
            g.invariants(),
            vec![(
                "has_acct".into(),
                "MATCH (a:Acct) RETURN count(a) >= 0".into()
            )]
        );
        // Dropped → a previously-rejected unbalanced write now succeeds.
        run(&mut g, "MATCH (a:Acct {name: 'a'}) SET a.balance = 5").unwrap();
    }

    #[test]
    fn unparseable_query_is_a_syntax_error() {
        let mut g = ndjson::decode("").unwrap();
        assert_eq!(
            g.create_invariant("bad", "MATCH (a:Acct) RETURN >>>")
                .unwrap_err()
                .code,
            ErrorCode::Syntax
        );
        assert_eq!(
            g.create_invariant("empty", "").unwrap_err().code,
            ErrorCode::Syntax
        );
    }

    #[test]
    fn non_boolean_and_null_and_empty_all_hold() {
        // `false`-only-fails: a non-boolean cell, a null cell, and an empty result
        // set each HOLD (only a literal `false` cell fails).
        let mut g = ndjson::decode(LEDGER).unwrap();
        g.create_invariant("nonbool", "MATCH (a:Acct) RETURN sum(a.balance)")
            .unwrap(); // yields 0 (a number, not false) → holds
        g.create_invariant("nullcell", "MATCH (a:Acct) RETURN a.missing")
            .unwrap(); // null cells → hold
        g.create_invariant("empty", "MATCH (z:NoSuchLabel) RETURN z.x = z.x")
            .unwrap(); // empty result → holds
                       // A write still commits (all three hold regardless of the balance sum).
        run(&mut g, "MATCH (a:Acct {name: 'a'}) SET a.balance = 12345").unwrap();
    }
}

#[cfg(test)]
mod clone_graph {
    //! `Graph: Clone` — the fast fork/branch substrate. A deep, independent copy
    //! of the columnar store (element ids preserved exactly), the native half of
    //! `graph.copy()` over the FFI. Mirrors the TS `Graph.copy()` for parity.
    use crate::gql::eval::Params;
    use crate::gql::parse;
    use crate::graph::*;
    use crate::ndjson;

    fn run(g: &mut Graph, q: &str) -> CodeResult<Vec<Vec<crate::graph::Value>>> {
        parse(q)
            .unwrap()
            .execute(g, &Params::new())
            .map(|rs| rs.rows().map(<[Value]>::to_vec).collect())
    }

    #[test]
    fn clone_is_independent_and_preserves_ids_and_constraints() {
        let mut base = ndjson::decode(
            &[
                r#"{"type":"node","id":"a","labels":["P"],"properties":{"id":"a","v":1}}"#,
                r#"{"type":"node","id":"b","labels":["P"],"properties":{"id":"b","v":2}}"#,
                r#"{"type":"edge","id":"e1","from":"a","to":"b","labels":["R"],"properties":{}}"#,
            ]
            .join("\n"),
        )
        .unwrap();
        base.create_unique_constraint("P", "id").unwrap();

        let mut copy = base.clone();

        // Independent: a write to one is invisible to the other.
        run(&mut copy, "INSERT (:P {id: 'c', v: 3})").unwrap();
        run(&mut base, "MATCH (n:P {id: 'a'}) SET n.v = 99").unwrap();
        assert_eq!(
            run(&mut base, "MATCH (n:P) RETURN count(*) AS c").unwrap(),
            vec![vec![Value::Num(2.0)]]
        );
        assert_eq!(
            run(&mut copy, "MATCH (n:P) RETURN count(*) AS c").unwrap(),
            vec![vec![Value::Num(3.0)]]
        );
        assert_eq!(
            run(&mut copy, "MATCH (n:P {id: 'a'}) RETURN n.v AS v").unwrap(),
            vec![vec![Value::Num(1.0)]] // unaffected by base's SET
        );

        // Ids preserved exactly: the edge still connects the same endpoints.
        assert_eq!(
            run(
                &mut copy,
                "MATCH (:P {id: 'a'})-[:R]->(x:P) RETURN x.id AS id"
            )
            .unwrap(),
            vec![vec![Value::Str("b".into())]]
        );

        // Indexes come along and are functional (declared + populated): the seek
        // path is used, and the index is listed on the copy.
        base.create_vertex_index("v");
        base.create_edge_index("w");
        let mut copy2 = base.clone();
        assert!(copy2.vertex_indexes().contains(&"v".to_string()));
        assert!(copy2.edge_indexes().contains(&"w".to_string()));
        // `b.v` is 2 (untouched); `a.v` was set to 99 earlier in this test.
        assert_eq!(
            run(&mut copy2, "MATCH (n:P {v: 2}) RETURN n.id AS id").unwrap(),
            vec![vec![Value::Str("b".into())]]
        );

        // Every constraint kind is enforced on the copy, not just unique. Required
        // + type on a fresh graph so the checks are unambiguous.
        let mut g2 = ndjson::decode(
            &[r#"{"type":"node","id":"a","labels":["P"],"properties":{"id":"a","age":30}}"#]
                .join("\n"),
        )
        .unwrap();
        g2.create_required_constraint("P", "id").unwrap();
        g2.create_type_constraint("P", "age", "number").unwrap();
        let mut copy3 = g2.clone();
        assert!(run(&mut copy3, "INSERT (:P {age: 1})").is_err()); // missing required id
        assert!(run(&mut copy3, "INSERT (:P {id: 'z', age: 'x'})").is_err()); // wrong type

        // The unique constraint came along: a duplicate id is rejected in the copy.
        assert!(run(&mut copy, "INSERT (:P {id: 'a'})").is_err());
    }
}

#[cfg(test)]
mod record_type_spec {
    use crate::graph::*;

    fn sc(k: &str, t: PropType) -> (Arc<str>, TypeSpec, bool) {
        (k.into(), TypeSpec::Scalar(t), false)
    }
    fn s(x: &str) -> Value {
        Value::Str(x.into())
    }
    fn vmap(pairs: &[(&str, Value)]) -> Value {
        Value::Map(
            pairs
                .iter()
                .map(|(k, v)| ((*k).into(), v.clone()))
                .collect(),
        )
    }

    #[test]
    fn parse_scalar_and_record_types() {
        assert_eq!(
            TypeSpec::parse("string"),
            Some(TypeSpec::Scalar(PropType::Str))
        );
        // Fields canonicalized to sorted order; `::` and `:` both accepted.
        assert_eq!(
            TypeSpec::parse("record { tier :: number, city :: string }"),
            Some(TypeSpec::Record(vec![
                sc("city", PropType::Str),
                sc("tier", PropType::Num)
            ])),
        );
        assert_eq!(
            TypeSpec::parse("record{a:number}"),
            Some(TypeSpec::Record(vec![sc("a", PropType::Num)])),
        );
        // Nested record.
        assert_eq!(
            TypeSpec::parse("record{addr::record{city::string}}"),
            Some(TypeSpec::Record(vec![(
                "addr".into(),
                TypeSpec::Record(vec![sc("city", PropType::Str)]),
                false,
            )])),
        );
        // A `NOT NULL` field parses to the required flag and round-trips.
        let nn = TypeSpec::parse("record{id::string NOT NULL,tier::number}").unwrap();
        assert_eq!(
            nn,
            TypeSpec::Record(vec![
                ("id".into(), TypeSpec::Scalar(PropType::Str), true),
                sc("tier", PropType::Num),
            ]),
        );
        assert_eq!(TypeSpec::parse(&nn.to_name()), Some(nn));
        assert_eq!(TypeSpec::parse("record{id::string NOT}"), None); // NOT without NULL
                                                                     // Round-trips through to_name.
        let t = TypeSpec::parse("record{city::string,tier::number}").unwrap();
        assert_eq!(TypeSpec::parse(&t.to_name()), Some(t));
        // Malformed.
        assert_eq!(TypeSpec::parse("record{a}"), None);
        assert_eq!(TypeSpec::parse("nope"), None);
        assert_eq!(TypeSpec::parse("record{a:string"), None);
    }

    #[test]
    fn value_matches_record_type() {
        let ty = TypeSpec::parse("record{city::string,tier::number}").unwrap();
        // Exact shape matches.
        assert!(value_matches(
            &vmap(&[("city", s("NYC")), ("tier", Value::Num(2.0))]),
            &ty
        ));
        // A null value satisfies any type (REQUIRED is separate).
        assert!(value_matches(&Value::Null, &ty));
        // A null FIELD is allowed (field-level required is separate).
        assert!(value_matches(
            &vmap(&[("city", Value::Null), ("tier", Value::Num(2.0))]),
            &ty
        ));
        // Wrong field type → no match.
        assert!(!value_matches(
            &vmap(&[("city", Value::Num(1.0)), ("tier", Value::Num(2.0))]),
            &ty
        ));
        // A missing NULLABLE field is OK (optional by default); the empty record
        // is OK too. An EXTRA field is rejected (closed on extras).
        assert!(value_matches(&vmap(&[("city", s("NYC"))]), &ty));
        assert!(value_matches(&vmap(&[]), &ty));
        assert!(!value_matches(
            &vmap(&[
                ("city", s("NYC")),
                ("tier", Value::Num(2.0)),
                ("x", Value::Num(1.0))
            ]),
            &ty,
        ));
        // A non-map value → no match.
        assert!(!value_matches(&Value::Num(1.0), &ty));

        // `NOT NULL` makes a field required (present + non-null).
        let req = TypeSpec::parse("record{id::string NOT NULL,tier::number}").unwrap();
        assert!(value_matches(&vmap(&[("id", s("x"))]), &req)); // tier optional
        assert!(!value_matches(&vmap(&[("tier", Value::Num(1.0))]), &req)); // id absent
        assert!(!value_matches(&vmap(&[("id", Value::Null)]), &req)); // id null
        assert!(value_matches(
            &vmap(&[("id", s("x")), ("tier", Value::Null)]),
            &req
        )); // tier nullable null OK
    }
}

#[cfg(test)]
mod record_constraint {
    use crate::graph::*;

    fn s(x: &str) -> Value {
        Value::Str(x.into())
    }
    fn vmap(pairs: &[(&str, Value)]) -> Value {
        Value::Map(
            pairs
                .iter()
                .map(|(k, v)| ((*k).into(), v.clone()))
                .collect(),
        )
    }
    fn base() -> Graph {
        crate::ndjson::decode(
            r#"{"type":"node","id":"a","labels":["Person"],"properties":{"meta":{"city":"NYC","tier":2}}}"#,
        )
        .unwrap()
    }

    #[test]
    fn declare_record_constraint_validates_existing_data() {
        let mut g = base();
        // Matching shape → declares OK.
        assert!(g
            .create_type_constraint("Person", "meta", "record{city::string,tier::number}")
            .is_ok());
        // A conflicting shape against existing data → rejected at declaration.
        let mut g2 = base();
        assert!(g2
            .create_type_constraint("Person", "meta", "record{city::number,tier::number}")
            .is_err());
    }

    #[test]
    fn record_constraint_enforced_on_set_and_insert() {
        let mut g = base();
        g.create_type_constraint("Person", "meta", "record{city::string,tier::number}")
            .unwrap();
        // A well-shaped write passes.
        assert!(!g.type_conflict_on_set(
            0,
            "meta",
            &vmap(&[("city", s("LA")), ("tier", Value::Num(3.0))])
        ));
        // A wrong field type conflicts.
        assert!(g.type_conflict_on_set(
            0,
            "meta",
            &vmap(&[("city", Value::Num(1.0)), ("tier", Value::Num(3.0))])
        ));
        // A missing NULLABLE field does NOT conflict (optional by default); an
        // EXTRA field does (closed on extras).
        assert!(!g.type_conflict_on_set(0, "meta", &vmap(&[("city", s("LA"))])));
        assert!(g.type_conflict_on_set(
            0,
            "meta",
            &vmap(&[
                ("city", s("LA")),
                ("tier", Value::Num(1.0)),
                ("x", Value::Num(9.0))
            ])
        ));
        // A null is exempt.
        assert!(!g.type_conflict_on_set(0, "meta", &Value::Null));
        // A new-vertex insert with a bad meta is a type_violation.
        assert!(g
            .type_violation(
                &["Person".to_string()],
                &[(
                    "meta".to_string(),
                    vmap(&[("city", Value::Num(9.0)), ("tier", Value::Num(1.0))])
                )],
            )
            .is_some());
    }

    #[test]
    fn dropping_the_record_constraint_lifts_enforcement() {
        let mut g = base();
        g.create_type_constraint("Person", "meta", "record{city::string,tier::number}")
            .unwrap();
        g.drop_type_constraint("Person", "meta");
        assert!(!g.type_conflict_on_set(0, "meta", &vmap(&[("city", Value::Num(1.0))])));
    }
}

/// Scalar `NOT NULL` on a type constraint (`string NOT NULL`) — the type-surface
/// spelling of a required (present + non-null) property, mirroring per-field
/// `NOT NULL` in a record.
#[cfg(test)]
mod scalar_not_null {
    use crate::graph::*;

    fn node(props: &str) -> Graph {
        crate::ndjson::decode(&format!(
            r#"{{"type":"node","id":"a","labels":["P"],"properties":{props}}}"#
        ))
        .unwrap()
    }

    #[test]
    fn parse_roundtrips_a_top_level_not_null() {
        assert_eq!(
            TypeSpec::parse_with_not_null("string"),
            Some((TypeSpec::Scalar(PropType::Str), false))
        );
        assert_eq!(
            TypeSpec::parse_with_not_null("string NOT NULL"),
            Some((TypeSpec::Scalar(PropType::Str), true))
        );
        // case-insensitive, whitespace-tolerant
        assert_eq!(
            TypeSpec::parse_with_not_null("number  not null"),
            Some((TypeSpec::Scalar(PropType::Num), true))
        );
        assert_eq!(TypeSpec::parse_with_not_null("string NOT"), None); // NOT without NULL
        assert_eq!(TypeSpec::parse_with_not_null("bogus"), None);
    }

    #[test]
    fn declare_requires_existing_data_present_and_non_null() {
        // present + non-null → OK
        assert!(node(r#"{"name":"marko"}"#)
            .create_type_constraint("P", "name", "string NOT NULL")
            .is_ok());
        // absent → declare fails
        assert!(node("{}")
            .create_type_constraint("P", "name", "string NOT NULL")
            .is_err());
        // stored null → declare fails
        assert!(node(r#"{"name":null}"#)
            .create_type_constraint("P", "name", "string NOT NULL")
            .is_err());
        // a plain (nullable) type constraint stays exempt from absent/null
        assert!(node("{}")
            .create_type_constraint("P", "name", "string")
            .is_ok());
    }

    #[test]
    fn missing_required_folds_in_the_not_null_type_constraint() {
        let mut g = node(r#"{"name":"marko"}"#);
        g.create_type_constraint("P", "name", "string NOT NULL")
            .unwrap();
        let labels = ["P".to_string()];
        // absent / null → a required violation; present non-null → OK.
        assert!(g.missing_required(&labels, &[]).is_some());
        assert!(g
            .missing_required(&labels, &[("name".to_string(), Value::Null)])
            .is_some());
        assert!(g
            .missing_required(&labels, &[("name".to_string(), Value::Str("x".into()))])
            .is_none());
        // A wrong TYPE is still a separate type violation.
        assert!(g
            .type_violation(&labels, &[("name".to_string(), Value::Num(1.0))])
            .is_some());
    }

    #[test]
    fn dropping_leaves_an_independent_required_intact() {
        let mut g = node(r#"{"name":"marko"}"#);
        g.create_required_constraint("P", "name").unwrap(); // declared independently
        g.create_type_constraint("P", "name", "string NOT NULL")
            .unwrap();
        g.drop_type_constraint("P", "name");
        // The type not-null is gone, but the independent required still enforces.
        assert!(!g.v_type_not_null.contains_key("P"));
        assert!(g.missing_required(&["P".to_string()], &[]).is_some());
    }

    #[test]
    fn dump_schema_roundtrips_scalar_not_null() {
        let mut g = node(r#"{"name":"marko"}"#);
        g.create_type_constraint("P", "name", "string NOT NULL")
            .unwrap();
        assert!(g.dump_schema().contains(r#""type":"string NOT NULL""#));
    }

    #[test]
    fn dump_schema_emits_edge_interval_index() {
        // An RI-tree interval index is an accelerator, not derivable from data, so
        // it must appear in dump_schema to survive a snapshot reload and replicate
        // over the CDC schema stream. `applySchemaOp` routes it back to createIndex.
        let mut g = crate::ndjson::decode("").unwrap();
        g.create_edge_interval_index("vf", "vt");
        assert!(
            g.dump_schema()
                .contains(r#"{"op":"createEdgeIntervalIndex","loKey":"vf","hiKey":"vt"}"#),
            "got: {}",
            g.dump_schema()
        );
    }

    #[test]
    fn not_null_on_a_record_type_is_now_supported() {
        // Previously rejected; a whole-record NOT NULL is now a valid constraint
        // (see the `any_record_and_record_not_null` module). Over PRESENT data it
        // declares cleanly.
        assert!(node(r#"{"meta":{"a":1}}"#)
            .create_type_constraint("P", "meta", "record{a::number} NOT NULL")
            .is_ok());
    }

    #[test]
    fn edge_scalar_not_null_enforces_and_roundtrips() {
        let mut g = crate::ndjson::decode(
            concat!(
                r#"{"type":"node","id":"a","labels":["P"],"properties":{}}"#,
                "\n",
                r#"{"type":"node","id":"b","labels":["P"],"properties":{}}"#,
                "\n",
                r#"{"type":"edge","id":"e","from":"a","to":"b","labels":["LINK"],"properties":{"w":1.5}}"#,
            ),
        )
        .unwrap();
        g.create_edge_type_constraint("LINK", "w", "number NOT NULL")
            .unwrap();
        let et = ["LINK".to_string()];
        assert!(g.edge_missing_required(&et, &[]).is_some());
        assert!(g
            .edge_missing_required(&et, &[("w".to_string(), Value::Num(2.0))])
            .is_none());
        assert!(g.dump_schema().contains(r#""type":"number NOT NULL""#));
    }
}

/// ISO `<record type> ::= [ANY] RECORD [<field spec>] [NOT NULL]` — the OPEN
/// record (`ANY RECORD` / bare `RECORD`) and a whole-record `NOT NULL`.
#[cfg(test)]
mod any_record_and_record_not_null {
    use crate::graph::*;

    fn s(x: &str) -> Value {
        Value::Str(x.into())
    }
    fn vmap(pairs: &[(&str, Value)]) -> Value {
        Value::Map(
            pairs
                .iter()
                .map(|(k, v)| ((*k).into(), v.clone()))
                .collect(),
        )
    }
    fn node(props: &str) -> Graph {
        crate::ndjson::decode(&format!(
            r#"{{"type":"node","id":"a","labels":["P"],"properties":{props}}}"#
        ))
        .unwrap()
    }

    #[test]
    fn parse_open_record_forms() {
        // `any record`, bare `record` → the open type; canonical name is `any record`.
        assert_eq!(TypeSpec::parse("any record"), Some(TypeSpec::AnyRecord));
        assert_eq!(TypeSpec::parse("record"), Some(TypeSpec::AnyRecord));
        assert_eq!(TypeSpec::parse("ANY  RECORD"), Some(TypeSpec::AnyRecord));
        assert_eq!(TypeSpec::AnyRecord.to_name(), "any record");
        assert_eq!(TypeSpec::parse("any"), None); // ANY without RECORD
                                                  // The closed form still parses to a Record.
        assert!(matches!(
            TypeSpec::parse("record{a::number}"),
            Some(TypeSpec::Record(_))
        ));
    }

    #[test]
    fn any_record_matches_any_map_but_not_a_scalar() {
        assert!(value_matches(
            &vmap(&[("x", Value::Num(1.0))]),
            &TypeSpec::AnyRecord
        ));
        assert!(value_matches(&vmap(&[]), &TypeSpec::AnyRecord)); // empty map OK
        assert!(value_matches(&Value::Null, &TypeSpec::AnyRecord)); // top-level null exempt
        assert!(!value_matches(&Value::Num(1.0), &TypeSpec::AnyRecord)); // a scalar is not a record
    }

    #[test]
    fn any_record_constraint_enforces_and_does_not_debox() {
        let mut g = node(r#"{"meta":{"city":"NYC"}}"#);
        g.create_type_constraint("P", "meta", "any record").unwrap();
        // Any-shaped map passes; a scalar is a type violation.
        assert!(!g.type_conflict_on_set(0, "meta", &vmap(&[("anything", Value::Num(9.0))])));
        assert!(g.type_conflict_on_set(0, "meta", &Value::Num(1.0)));
        // Open record has no field contract → the column stays boxed (NOT de-boxed).
        assert!(matches!(g.props.col("meta"), Some(Column::Mixed { .. })));
    }

    #[test]
    fn record_level_not_null_parses_and_is_required() {
        assert_eq!(
            TypeSpec::parse_with_not_null("record{a::number} NOT NULL"),
            TypeSpec::parse("record{a::number}").map(|s| (s, true))
        );
        assert_eq!(
            TypeSpec::parse_with_not_null("any record NOT NULL"),
            Some((TypeSpec::AnyRecord, true))
        );

        // A closed record + NOT NULL: present map OK; absent / null → required violation.
        let mut g = node(r#"{"meta":{"city":"NYC"}}"#);
        g.create_type_constraint("P", "meta", "record{city::string} NOT NULL")
            .unwrap();
        let labels = ["P".to_string()];
        assert!(g.missing_required(&labels, &[]).is_some());
        assert!(g
            .missing_required(&labels, &[("meta".to_string(), Value::Null)])
            .is_some());
        assert!(g
            .missing_required(&labels, &[("meta".to_string(), vmap(&[("city", s("LA"))]))])
            .is_none());
        // A NOT NULL closed record STILL de-boxes (presence is orthogonal to shape).
        assert!(matches!(g.props.col("meta"), Some(Column::Record { .. })));
        assert!(g
            .dump_schema()
            .contains(r#""type":"record{city::string} NOT NULL""#));
    }

    #[test]
    fn declaring_record_not_null_over_absent_data_fails() {
        // A label vertex missing the key → the NOT NULL declare is rejected.
        assert!(node("{}")
            .create_type_constraint("P", "meta", "any record NOT NULL")
            .is_err());
        // A present map satisfies it.
        assert!(node(r#"{"meta":{"a":1}}"#)
            .create_type_constraint("P", "meta", "any record NOT NULL")
            .is_ok());
    }

    #[test]
    fn dropping_a_record_not_null_removes_its_requiredness() {
        let mut g = node(r#"{"meta":{"a":1}}"#);
        g.create_type_constraint("P", "meta", "any record NOT NULL")
            .unwrap();
        g.drop_type_constraint("P", "meta");
        assert!(g.missing_required(&["P".to_string()], &[]).is_none());
    }
}

/// Record-typed constraints, step 2: a declared RECORD constraint de-boxes the property key's
/// column into typed per-field sub-columns ([`Column::Record`]). These tests pin
/// the substrate: de-boxing happens on declare, every read round-trips
/// byte-identically to the boxed map, non-conforming values (a shared key across
/// labels) stay correct via the escape overlay, backfill + drop-rebox work, and a
/// field reads straight from its sub-column.
#[cfg(test)]
mod record_debox {
    use crate::graph::*;

    fn s(x: &str) -> Value {
        Value::Str(x.into())
    }
    fn n(x: f64) -> Value {
        Value::Num(x)
    }
    fn vmap(pairs: &[(&str, Value)]) -> Value {
        Value::Map(
            pairs
                .iter()
                .map(|(k, v)| ((*k).into(), v.clone()))
                .collect(),
        )
    }
    /// A graph with `Person` a whose `meta = {city, tier}` and a record constraint
    /// already declared (so `meta` is de-boxed).
    fn declared() -> Graph {
        let mut g = crate::ndjson::decode(
            r#"{"type":"node","id":"a","labels":["Person"],"properties":{"meta":{"city":"NYC","tier":2}}}"#,
        )
        .unwrap();
        g.create_type_constraint("Person", "meta", "record{city::string,tier::number}")
            .unwrap();
        g
    }
    fn read(g: &Graph, idx: usize) -> Value {
        g.props.value(idx, "meta", &g.strs)
    }
    fn write(g: &mut Graph, idx: usize, v: Value) {
        g.props.set_value(idx, "meta", v, &mut g.strs);
    }
    fn col_name(g: &Graph, key: &str) -> &'static str {
        match g.props.col(key) {
            Some(Column::Record { .. }) => "record",
            Some(Column::Mixed { .. }) => "mixed",
            _ => "other",
        }
    }

    #[test]
    fn declaring_deboxes_the_column_and_types_the_fields() {
        let g = declared();
        assert_eq!(col_name(&g, "meta"), "record");
        // The field sub-columns are TYPED (string→Str, number→Num), not boxed.
        let Some(Column::Record {
            field_names,
            fields,
            ..
        }) = g.props.col("meta")
        else {
            panic!("meta should be a de-boxed record column");
        };
        assert_eq!(
            field_names.iter().map(|n| n.as_ref()).collect::<Vec<_>>(),
            ["city", "tier"] // sorted canonical order
        );
        assert!(matches!(fields[0], Column::Str { .. }));
        assert!(matches!(fields[1], Column::Num { .. }));
        // The backfilled value reads back identically to the boxed map.
        assert_eq!(read(&g, 0), vmap(&[("city", s("NYC")), ("tier", n(2.0))]));
    }

    #[test]
    fn a_stored_null_field_keeps_its_sub_column_typed() {
        // 1b: a nullable field set to an explicit null records the null in the
        // per-field `field_nulls` bitset and keeps the sub-column TYPED — it no
        // longer promotes to `Mixed`. The value still round-trips.
        let mut g = declared();
        write(&mut g, 0, vmap(&[("city", s("LA")), ("tier", Value::Null)]));
        assert_eq!(
            read(&g, 0),
            vmap(&[("city", s("LA")), ("tier", Value::Null)])
        );
        let Some(Column::Record {
            fields,
            field_nulls,
            ..
        }) = g.props.col("meta")
        else {
            panic!();
        };
        // `tier` (field 1) stayed a Num column; its null bit is set for element 0.
        assert!(matches!(fields[1], Column::Num { .. }), "tier stayed typed");
        assert!(field_nulls[1].get(0), "tier null recorded in field_nulls");
        assert!(!field_nulls[0].get(0), "city is not null");
        // Overwriting the null with a value clears the null bit.
        write(&mut g, 0, vmap(&[("city", s("LA")), ("tier", n(9.0))]));
        let Some(Column::Record { field_nulls, .. }) = g.props.col("meta") else {
            panic!();
        };
        assert!(
            !field_nulls[1].get(0),
            "null bit cleared on a non-null write"
        );
    }

    #[test]
    fn a_nested_record_field_deboxes_recursively() {
        // 1a: a RECORD-typed field is itself a de-boxed `Column::Record`.
        let mut g = crate::ndjson::decode(
            r#"{"type":"node","id":"a","labels":["P"],"properties":{"addr":{"geo":{"lat":1.0,"lng":2.0}}}}"#,
        )
        .unwrap();
        g.create_type_constraint("P", "addr", "record{geo::record{lat::number,lng::number}}")
            .unwrap();
        let Some(Column::Record { fields, .. }) = g.props.col("addr") else {
            panic!("addr should be de-boxed");
        };
        // The `geo` field is a NESTED record column whose own fields are typed.
        let Column::Record {
            field_names: geo_names,
            fields: geo_fields,
            ..
        } = &fields[0]
        else {
            panic!("geo should be a nested record column, not boxed");
        };
        assert_eq!(
            geo_names.iter().map(|n| n.as_ref()).collect::<Vec<_>>(),
            ["lat", "lng"]
        );
        assert!(matches!(geo_fields[0], Column::Num { .. }));
        // Reads round-trip (whole record + a deep field).
        assert_eq!(
            g.props.value(0, "addr", &g.strs),
            vmap(&[("geo", vmap(&[("lat", n(1.0)), ("lng", n(2.0))]))])
        );
        let kid = g.props.keys.get("addr").unwrap();
        assert_eq!(g.props.field_at(0, kid, &["geo", "lat"], &g.strs), n(1.0));
    }

    #[test]
    fn every_record_shape_roundtrips_byte_identically() {
        let mut g = declared();
        for v in [
            vmap(&[("city", s("LA")), ("tier", n(3.0))]),     // full
            vmap(&[("city", s("SF"))]),                       // nullable field omitted
            vmap(&[("city", s("X")), ("tier", Value::Null)]), // field stored null
            vmap(&[]),                                        // empty map (present, not absent)
        ] {
            write(&mut g, 0, v.clone());
            assert_eq!(read(&g, 0), v, "round-trip mismatch");
            assert!(g.props.is_present(0, "meta"));
        }
        // A stored null at the top level reads back as null but stays PRESENT.
        write(&mut g, 0, Value::Null);
        assert_eq!(read(&g, 0), Value::Null);
        assert!(g.props.is_present(0, "meta"));
        // Removal is distinct from a stored null: absent, not present.
        g.props.remove_value(0, "meta");
        assert_eq!(read(&g, 0), Value::Null);
        assert!(!g.props.is_present(0, "meta"));
    }

    #[test]
    fn nonconforming_values_stay_correct_via_the_escape_overlay() {
        // The column is de-boxed for `Person.meta`, but the key is global — a
        // scalar or a differently-shaped map must still round-trip.
        let mut g = declared();
        let a = g.add_vertex(&["Other".into()], vec![]);
        let b = g.add_vertex(&["Other".into()], vec![]);
        write(&mut g, a as usize, n(42.0)); // a scalar escapes
        write(
            &mut g,
            b as usize,
            vmap(&[("lat", n(1.0)), ("lng", n(2.0))]), // an extra-keyed map escapes
        );
        assert_eq!(read(&g, a as usize), n(42.0));
        assert_eq!(
            read(&g, b as usize),
            vmap(&[("lat", n(1.0)), ("lng", n(2.0))])
        );
        assert!(g.props.is_present(a as usize, "meta"));
        // Overwriting an escapee with a conforming map clears the escape.
        write(&mut g, a as usize, vmap(&[("city", s("NYC"))]));
        assert_eq!(read(&g, a as usize), vmap(&[("city", s("NYC"))]));
        let Some(Column::Record { escaped, .. }) = g.props.col("meta") else {
            panic!();
        };
        assert!(!escaped.contains_key(&a), "escape not cleared on reconform");
        assert!(escaped.contains_key(&b), "b still escaped");
    }

    #[test]
    fn typing_a_mixed_population_succeeds_and_deboxes_each_faithfully() {
        // The scenario: pre-existing vertices, then you type the label. A LABELED
        // vertex that already conforms de-boxes into fields; a labeled vertex that
        // merely LACKS the property is exempt (nullable); a DIFFERENT-label vertex
        // holding a non-conforming value isn't checked and escapes. Declaration
        // succeeds and every value survives byte-identically.
        let mut g = crate::ndjson::decode(
            r#"{"type":"node","id":"a","labels":["Person"],"properties":{"meta":{"city":"NYC","tier":2}}}"#,
        )
        .unwrap();
        let b = g.add_vertex(&["Person".into()], vec![]) as usize; // labeled, meta ABSENT
        let c = g.add_vertex(&["Other".into()], vec![("meta".into(), n(42.0))]) as usize; // other label, scalar
        assert!(g
            .create_type_constraint("Person", "meta", "record{city::string,tier::number}")
            .is_ok());
        assert_eq!(col_name(&g, "meta"), "record");
        assert_eq!(read(&g, 0), vmap(&[("city", s("NYC")), ("tier", n(2.0))])); // scattered
        assert!(!g.props.is_present(b, "meta")); // still absent (nullable, exempt)
        assert_eq!(read(&g, c as usize), n(42.0)); // escaped, unchanged
    }

    #[test]
    fn a_labeled_violator_makes_typing_throw_and_deboxes_nothing() {
        // If ANY live vertex carrying the label already violates the shape, the
        // declaration throws — atomically. No constraint is recorded and the column
        // is NOT de-boxed (no half-applied state, no grandfathered landmine).
        let mut g = crate::ndjson::decode(
            r#"{"type":"node","id":"a","labels":["Person"],"properties":{"meta":{"city":"NYC","tier":2}}}"#,
        )
        .unwrap();
        // A second Person whose meta is a scalar — a violation of the record shape.
        g.add_vertex(&["Person".into()], vec![("meta".into(), n(1.0))]);
        assert!(g
            .create_type_constraint("Person", "meta", "record{city::string,tier::number}")
            .is_err());
        assert_eq!(col_name(&g, "meta"), "mixed", "column untouched on failure");
        // No constraint recorded → a would-be-violating write no longer conflicts.
        assert!(!g.type_conflict_on_set(0, "meta", &n(7.0)));
    }

    #[test]
    fn declaring_before_bulk_append_scatters_directly_never_boxing() {
        // The order the user expects to pay off: declare the constraint on an empty
        // graph, THEN bulk-ingest. `ndjson::append` routes through add_vertex_with_id
        // → set_value → the Record arm, so conforming maps scatter straight into the
        // typed sub-columns — no `Value::Map` is ever boxed, nothing escapes.
        let mut g = crate::ndjson::decode("").unwrap();
        g.create_type_constraint("Person", "meta", "record{city::string,tier::number}")
            .unwrap();
        assert_eq!(col_name(&g, "meta"), "record"); // empty Record column exists up front
        let batch = [
            r#"{"type":"node","id":"p0","labels":["Person"],"properties":{"meta":{"city":"NYC","tier":1}}}"#,
            // nullable `tier` omitted — a conforming partial record
            r#"{"type":"node","id":"p1","labels":["Person"],"properties":{"meta":{"city":"LA"}}}"#,
        ]
        .join("\n");
        crate::ndjson::append(&mut g, &batch).unwrap();

        let Some(Column::Record {
            fields, escaped, ..
        }) = g.props.col("meta")
        else {
            panic!("meta stayed boxed after a declare-then-append");
        };
        assert!(matches!(fields[0], Column::Str { .. }));
        assert!(matches!(fields[1], Column::Num { .. }));
        assert!(
            escaped.is_empty(),
            "a conforming bulk append must never box/escape"
        );
        let p0 = g.vertex_by_id("p0").unwrap() as usize;
        let p1 = g.vertex_by_id("p1").unwrap() as usize;
        assert_eq!(read(&g, p0), vmap(&[("city", s("NYC")), ("tier", n(1.0))]));
        assert_eq!(read(&g, p1), vmap(&[("city", s("LA"))]));
    }

    #[test]
    fn field_at_reads_a_deboxed_field_directly() {
        let mut g = declared();
        let kid = g.props.keys.get("meta").unwrap();
        write(&mut g, 0, vmap(&[("city", s("LA")), ("tier", n(3.0))]));
        assert_eq!(g.props.field_at(0, kid, &["city"], &g.strs), s("LA"));
        assert_eq!(g.props.field_at(0, kid, &["tier"], &g.strs), n(3.0));
        // A field the (nullable) value omits → Null.
        write(&mut g, 0, vmap(&[("city", s("LA"))]));
        assert_eq!(g.props.field_at(0, kid, &["tier"], &g.strs), Value::Null);
        // An undeclared segment → Null (closed record).
        assert_eq!(g.props.field_at(0, kid, &["nope"], &g.strs), Value::Null);
        // On a stored-null / absent record, a field is Null.
        write(&mut g, 0, Value::Null);
        assert_eq!(g.props.field_at(0, kid, &["city"], &g.strs), Value::Null);
        // On an escapee, `field_at` walks the boxed value.
        let e = g.add_vertex(&["Other".into()], vec![]) as usize;
        write(&mut g, e, vmap(&[("lat", n(9.0))]));
        assert_eq!(g.props.field_at(e, kid, &["lat"], &g.strs), n(9.0));
    }

    #[test]
    fn backfill_on_declare_matches_the_boxed_reads() {
        // Store several boxed maps FIRST, snapshot their reads, then declare.
        let mut g = crate::ndjson::decode(
            r#"{"type":"node","id":"a","labels":["Person"],"properties":{"meta":{"city":"NYC","tier":2}}}"#,
        )
        .unwrap();
        let b = g.add_vertex(
            &["Person".into()],
            vec![("meta".into(), vmap(&[("city", s("LA"))]))],
        ) as usize;
        let c = g.add_vertex(
            &["Person".into()],
            vec![("meta".into(), vmap(&[("city", s("SF")), ("tier", n(7.0))]))],
        ) as usize;
        let before: Vec<Value> = (0..=c).map(|i| read(&g, i)).collect();
        assert_eq!(col_name(&g, "meta"), "mixed");
        g.create_type_constraint("Person", "meta", "record{city::string,tier::number}")
            .unwrap();
        assert_eq!(col_name(&g, "meta"), "record");
        let after: Vec<Value> = (0..=c).map(|i| read(&g, i)).collect();
        assert_eq!(before, after, "backfill changed a value");
        assert!(g.props.is_present(b, "meta") && g.props.is_present(c, "meta"));
    }

    #[test]
    fn dropping_the_constraint_reboxes_to_mixed_without_data_loss() {
        let mut g = declared();
        write(&mut g, 0, vmap(&[("city", s("LA")), ("tier", n(3.0))]));
        let before = read(&g, 0);
        g.drop_type_constraint("Person", "meta");
        assert_eq!(col_name(&g, "meta"), "mixed");
        assert_eq!(read(&g, 0), before, "rebox changed the value");
    }

    #[test]
    fn a_second_label_on_the_same_key_keeps_it_deboxed_until_both_drop() {
        let mut g = declared();
        g.create_type_constraint("Company", "meta", "record{city::string,tier::number}")
            .unwrap();
        assert_eq!(col_name(&g, "meta"), "record");
        // Dropping one of two constraints on the key must NOT re-box.
        g.drop_type_constraint("Person", "meta");
        assert_eq!(col_name(&g, "meta"), "record");
        // Dropping the last one re-boxes.
        g.drop_type_constraint("Company", "meta");
        assert_eq!(col_name(&g, "meta"), "mixed");
    }

    #[test]
    fn ndjson_encodes_a_deboxed_record_identically_to_the_boxed_map() {
        // The same graph, boxed vs de-boxed, must serialize to the same NDJSON.
        let boxed = crate::ndjson::decode(
            r#"{"type":"node","id":"a","labels":["Person"],"properties":{"meta":{"city":"NYC","tier":2}}}"#,
        )
        .unwrap();
        let deboxed = declared();
        assert_eq!(col_name(&deboxed, "meta"), "record");
        assert_eq!(
            crate::ndjson::encode(&boxed),
            crate::ndjson::encode(&deboxed)
        );
    }

    #[test]
    fn edge_record_constraint_deboxes_and_roundtrips() {
        let mut g = crate::ndjson::decode(
            concat!(
                r#"{"type":"node","id":"a","labels":["P"],"properties":{}}"#,
                "\n",
                r#"{"type":"node","id":"b","labels":["P"],"properties":{}}"#,
                "\n",
                r#"{"type":"edge","id":"e","from":"a","to":"b","labels":["LINK"],"properties":{"meta":{"w":0.5}}}"#,
            ),
        )
        .unwrap();
        g.create_edge_type_constraint("LINK", "meta", "record{w::number}")
            .unwrap();
        assert!(matches!(
            g.edge_props.col("meta"),
            Some(Column::Record { .. })
        ));
        assert_eq!(
            g.edge_props.value(0, "meta", &g.strs),
            vmap(&[("w", n(0.5))])
        );
        g.drop_edge_type_constraint("LINK", "meta");
        assert!(matches!(
            g.edge_props.col("meta"),
            Some(Column::Mixed { .. })
        ));
        assert_eq!(
            g.edge_props.value(0, "meta", &g.strs),
            vmap(&[("w", n(0.5))])
        );
    }
}

#[cfg(test)]
mod temporal_index_key_tests {
    use crate::graph::*;
    use crate::temporal as t;

    /// The scalar `Temporal::index_key` i128 MUST equal the column's
    /// `monotonic_key` bit-for-bit — otherwise a key built from a query literal
    /// won't match a key built from a stored column and the index silently returns
    /// wrong rows. Guards against the two encodings drifting apart.
    #[test]
    fn temporal_index_key_matches_column() {
        let cases: Vec<(TemporalKind, Vec<Temporal>)> = vec![
            (
                TemporalKind::Date,
                vec![
                    Temporal::Date(t::Date { days: -1000 }),
                    Temporal::Date(t::Date { days: 0 }),
                    Temporal::Date(t::Date { days: 19_723 }),
                ],
            ),
            (
                TemporalKind::DateTime,
                vec![
                    Temporal::DateTime(t::DateTime { secs: -5, nanos: 0 }),
                    Temporal::DateTime(t::DateTime {
                        secs: 1_700_000_000,
                        nanos: 123,
                    }),
                ],
            ),
            (
                TemporalKind::ZonedDateTime,
                vec![
                    Temporal::ZonedDateTime(t::ZonedDateTime {
                        secs: 1_700_000_000,
                        nanos: 5,
                        offset: -120,
                    }),
                    Temporal::ZonedDateTime(t::ZonedDateTime {
                        secs: 1_700_000_000,
                        nanos: 5,
                        offset: 300,
                    }),
                ],
            ),
            (
                TemporalKind::Time,
                vec![Temporal::Time(t::Time {
                    secs: 3600,
                    nanos: 42,
                })],
            ),
            (
                TemporalKind::ZonedTime,
                vec![Temporal::ZonedTime(t::ZonedTime {
                    secs: 3600,
                    nanos: 42,
                    offset: 60,
                })],
            ),
        ];
        for (kind, vals) in cases {
            let mut col = TemporalCol::with_len(kind, vals.len());
            for (i, v) in vals.iter().enumerate() {
                assert!(col.set(i, v), "{kind:?} slot {i}: set kind mismatch");
            }
            for (i, v) in vals.iter().enumerate() {
                let col_key = col.monotonic_key(i).expect("indexable kind has a key");
                assert_eq!(
                    col.get(i).index_key().unwrap().1,
                    col_key,
                    "{kind:?} slot {i}: get→scalar drift"
                );
                assert_eq!(
                    v.index_key().unwrap().1,
                    col_key,
                    "{kind:?} slot {i}: scalar drift"
                );
            }
        }
        // Duration has no monotonic key on either side.
        assert!(Temporal::Duration(t::Duration {
            months: 1,
            days: 2,
            secs: 3,
            nanos: 4
        })
        .index_key()
        .is_none());
    }

    /// Within a kind the key is monotonic with the value's own order; across kinds
    /// the kind rank keeps them disjoint so a range seek never interleaves them.
    #[test]
    fn temporal_index_key_is_monotonic_and_kind_disjoint() {
        let dates: Vec<Temporal> = (0..40)
            .map(|k| Temporal::Date(t::Date { days: k * 9 - 137 }))
            .collect();
        for w in dates.windows(2) {
            assert!(
                w[0].index_key().unwrap() < w[1].index_key().unwrap(),
                "date order broke"
            );
        }
        // A max Date still ranks below a min DateTime (disjoint kind ranges).
        let big_date = Temporal::Date(t::Date { days: i32::MAX })
            .index_key()
            .unwrap();
        let small_dt = Temporal::DateTime(t::DateTime {
            secs: i64::MIN,
            nanos: 0,
        })
        .index_key()
        .unwrap();
        assert!(
            big_date < small_dt,
            "kind ranks must keep Date below DateTime"
        );
    }
}

// ---------------------------------------------------------------------------
// The interning dictionary's open-addressed table.
//
// It replaced a `HashMap` for memory-access reasons (8-byte slots, and a stored
// hash that rejects a non-match without dereferencing the string), so the paths
// a hash map used to handle for us — growth, rehashing, probe termination,
// 32-bit hash collisions — are now ours and need their own cover.
// ---------------------------------------------------------------------------

#[test]
fn dict_interning_is_stable_and_dense() {
    use crate::graph::Dict;

    let mut d = Dict::default();

    assert_eq!(d.intern("a"), 0);
    assert_eq!(d.intern("b"), 1);
    assert_eq!(d.intern("a"), 0, "re-interning must return the first id");
    assert_eq!(d.intern("c"), 2);
    assert_eq!(d.len(), 3);
    assert_eq!(d.text(0), "a");
    assert_eq!(d.text(2), "c");
    assert_eq!(d.get("b"), Some(1));
    assert_eq!(d.get("nope"), None);
}

#[test]
fn dict_survives_many_growths_with_ids_intact() {
    use crate::graph::Dict;

    // Well past several doublings from the default size, so the rehash path runs
    // repeatedly and every previously-assigned id has to survive it.
    let mut d = Dict::default();
    let n = 5_000u32;

    for i in 0..n {
        assert_eq!(d.intern(&format!("key{i}")), i);
    }

    assert_eq!(d.len(), n as usize);
    for i in 0..n {
        assert_eq!(
            d.get(&format!("key{i}")),
            Some(i),
            "id moved during a growth"
        );
        assert_eq!(d.text(i), format!("key{i}"));
    }
}

#[test]
fn dict_with_capacity_matches_a_grown_one() {
    use crate::graph::Dict;

    // Pre-sizing must not change any assigned id.
    let mut grown = Dict::default();
    let mut sized = Dict::with_capacity(1_000);

    for i in 0..1_000u32 {
        let k = format!("k{i}");

        assert_eq!(grown.intern(&k), sized.intern(&k));
    }

    assert_eq!(grown.len(), sized.len());
}

#[test]
fn dict_distinguishes_strings_that_share_a_hash_bucket() {
    use crate::graph::Dict;

    // The table stores a 32-bit hash and only compares the string when that
    // matches, so a bucket shared by different strings must still resolve by
    // content. With thousands of keys in a small table this exercises long probe
    // runs as well.
    let mut d = Dict::default();
    let keys: Vec<String> = (0..2_000)
        .map(|i| format!("{i:width$}", width = 40))
        .collect();

    for (i, k) in keys.iter().enumerate() {
        assert_eq!(d.intern(k), i as u32);
    }
    for (i, k) in keys.iter().enumerate() {
        assert_eq!(d.get(k), Some(i as u32));
    }
}

#[test]
fn dict_handles_empty_and_unicode_keys() {
    use crate::graph::Dict;

    let mut d = Dict::default();

    assert_eq!(d.intern(""), 0);
    assert_eq!(d.intern("😀"), 1);
    assert_eq!(d.intern("中文"), 2);
    assert_eq!(d.intern(""), 0);
    assert_eq!(d.get("😀"), Some(1));
    assert_eq!(d.text(2), "中文");
}

#[test]
fn dict_clone_keeps_lookups_working() {
    use crate::graph::Dict;

    // The table is cloned alongside the strings; a clone that kept one without
    // the other would look empty or panic.
    let mut d = Dict::default();

    for i in 0..500u32 {
        d.intern(&format!("v{i}"));
    }

    let c = d.clone();

    for i in 0..500u32 {
        assert_eq!(c.get(&format!("v{i}")), Some(i));
    }
}
