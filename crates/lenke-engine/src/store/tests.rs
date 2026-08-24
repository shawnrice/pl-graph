use super::*;

#[test]
fn version_bumps_on_every_mutation() {
    let mut s = Store::default();
    assert_eq!(s.version(), 0);
    let a = s.add_node(&["N"], &[]);
    let v1 = s.version();
    assert!(v1 > 0, "add_node bumps version");
    let b = s.add_node(&["N"], &[]);
    assert!(s.version() > v1, "second add_node bumps again");
    let v2 = s.version();
    let e = s.add_edge(a, b, "R");
    assert!(s.version() > v2, "add_edge bumps");
    let v3 = s.version();
    s.set_prop(a, "k", Value::Num(1.0));
    assert!(s.version() > v3, "set_prop bumps");
    let v4 = s.version();
    s.delete_edge(a, b, e);
    assert!(s.version() > v4, "delete_edge bumps");
}

#[test]
fn epoch_bumps_only_the_tokens_a_change_touches() {
    let mut st = Store::default();
    let a = st.add_node(&["Person"], &[("name", s("alice"))]);
    // Both the label and the property key were touched by the add.
    let (p0, n0) = (st.epoch("Person"), st.epoch("name"));
    assert!(p0 > 0 && n0 > 0);
    assert_eq!(st.epoch("Project"), 0, "an untouched token stays 0");

    // A property change bumps only that property's epoch, not the label's.
    st.set_prop(a, "age", Value::Num(30.0));
    assert!(st.epoch("age") > n0);
    assert_eq!(st.epoch("Person"), p0, "unrelated label epoch is unchanged");
    assert_eq!(
        st.epoch("name"),
        n0,
        "unrelated property epoch is unchanged"
    );

    // Epoch never exceeds the global version.
    assert!(st.epoch("age") <= st.version());
}

#[test]
fn clone_is_deep_and_independent() {
    let mut s = Store::default();
    let a = s.add_node(&["N"], &[("k", Value::Num(1.0))]);
    let b = s.add_node(&["N"], &[]);
    s.add_edge(a, b, "R");
    let ver = s.version();
    let snap = s.clone();
    assert_eq!(snap.version(), ver, "clone copies the version");
    // Mutating the original must not touch the clone.
    s.add_node(&["N"], &[]);
    assert_eq!(snap.node_count(), 2, "clone unaffected by later mutation");
    assert_eq!(s.node_count(), 3);
    assert_eq!(snap.edge_count(), 1);
    assert_eq!(snap.version(), ver, "clone version is frozen at copy time");
}

/// The CSR read overlay must (a) match the per-node adjacency exactly after a
/// build, (b) reflect a write IMMEDIATELY (invalidation → Vec fallback), and
/// (c) match again after an explicit rebuild.
#[test]
fn csr_overlay_matches_and_invalidates_on_write() {
    let mut b = Builder::default();
    b.node(&["N"], &[]);
    b.node(&["N"], &[]);
    b.node(&["N"], &[]);
    b.edge(0, 1, "R");
    b.edge(0, 2, "R");
    let mut st = b.build();
    let out = |st: &Store, v: u32| -> Vec<u32> { st.out(v).iter().map(|a| a.nbr).collect() };
    let inc = |st: &Store, v: u32| -> Vec<u32> { st.inc(v).iter().map(|a| a.nbr).collect() };
    // (a) fresh CSR after build: neighbour ORDER preserved.
    assert!(st.csr_fresh);
    assert_eq!(out(&st, 0), vec![1, 2]);
    // (b) a write clears the overlay and is visible at once via the Vec fallback.
    st.add_edge(0, 1, "R");
    assert!(!st.csr_fresh);
    assert_eq!(out(&st, 0), vec![1, 2, 1]);
    assert_eq!(inc(&st, 1), vec![0, 0]);
    // (c) rebuild re-enables the CSR and it still matches.
    st.rebuild_csr();
    assert!(st.csr_fresh);
    assert_eq!(out(&st, 0), vec![1, 2, 1]);
    assert_eq!(inc(&st, 1), vec![0, 0]);
}

fn s(x: &str) -> Value {
    Value::Str(Arc::from(x))
}
fn n(x: f64) -> Value {
    Value::Num(x)
}

#[test]
fn temporal_props_use_a_typed_column_and_promote_on_mixed_kind() {
    use crate::temporal::{Date, Temporal, TemporalKind, Time};
    let d = |iso: &str| Value::Temporal(Temporal::Date(Date::parse(iso).unwrap()));
    let mut b = Builder::default();
    b.node(&["P"], &[("born", d("1990-01-01"))]);
    b.node(&["P"], &[("born", d("2000-01-01"))]);
    let mut st = b.build();
    // Homogeneous Date props de-box into a typed Temporal column of kind Date.
    assert!(matches!(
        st.column("born"),
        Some(Column::Temporal {
            kind: TemporalKind::Date,
            ..
        })
    ));
    match st.prop(0, "born") {
        Value::Temporal(Temporal::Date(x)) => assert_eq!(x.format(), "1990-01-01"),
        o => panic!("expected a Date, got {o:?}"),
    }
    // Writing a DIFFERENT temporal kind promotes the column to Gen; both
    // values still read back correctly.
    st.set_prop(
        1,
        "born",
        Value::Temporal(Temporal::Time(Time::parse("12:00:00").unwrap())),
    );
    assert!(matches!(st.column("born"), Some(Column::Gen { .. })));
    assert!(matches!(
        st.prop(0, "born"),
        Value::Temporal(Temporal::Date(_))
    ));
    assert!(matches!(
        st.prop(1, "born"),
        Value::Temporal(Temporal::Time(_))
    ));
}

#[test]
fn commit_records_the_change_list_rollback_records_nothing() {
    let mut st = Builder::default().build();
    let a = st.add_node(&["P"], &[]); // outside a txn → nothing observed
    assert!(st.last_commit_changes().is_empty());

    // A committed transaction publishes exactly its changes, in order.
    st.begin();
    let b = st.add_node(&["P"], &[("name", s("b"))]);
    st.set_prop(a, "age", n(1.0));
    let eid = st.add_edge(a, b, "R");
    st.commit();
    assert_eq!(
        st.last_commit_changes(),
        &[
            Change::NodeAdded(b),
            Change::NodeProp {
                node: a,
                key: "age".into(),
            },
            Change::EdgeAdded(eid),
        ]
    );

    // A rolled-back transaction publishes nothing: `last_commit` still shows
    // the previous COMMIT, unchanged (rollback is not an event).
    let previous: Vec<Change> = st.last_commit_changes().to_vec();
    st.begin();
    st.set_prop(a, "age", n(2.0));
    st.rollback();
    assert_eq!(st.last_commit_changes(), previous.as_slice());
}

#[test]
fn touched_scopes_are_the_distinct_rooms_a_commit_writes() {
    let str_scopes = |scopes: &[Value]| -> Vec<String> {
        scopes
            .iter()
            .map(|v| match v {
                Value::Str(x) => x.to_string(),
                o => format!("{o:?}"),
            })
            .collect()
    };
    let mut st = Builder::default().build();
    st.begin();
    st.add_node(&["Msg"], &[("room", s("A"))]);
    st.add_node(&["Msg"], &[("room", s("B"))]);
    st.add_node(&["Msg"], &[("room", s("A"))]); // duplicate room A
    st.commit();
    let (scopes, open) = st.touched_scopes("room");
    assert_eq!(str_scopes(&scopes), vec!["A", "B"]); // distinct, sorted
    assert!(!open); // every change was scopable

    // A node with no `room` property → fail-open (visible to all).
    st.begin();
    st.add_node(&["Sys"], &[]);
    st.commit();
    let (scopes2, open2) = st.touched_scopes("room");
    assert!(scopes2.is_empty());
    assert!(open2);
}

#[test]
fn last_write_scope_json_renders_scopes_and_open_flag() {
    let mut st = Builder::default().build();
    st.begin();
    st.add_node(&["Msg"], &[("room", s("A"))]);
    st.add_node(&["Msg"], &[("room", s("B"))]);
    st.commit();
    assert_eq!(
        st.last_write_scope_json("room"),
        r#"{"scopes":["A","B"],"open":false}"#
    );

    // An unscopable change (no `room`) flips open to true.
    st.begin();
    st.add_node(&["Sys"], &[]);
    st.commit();
    assert_eq!(
        st.last_write_scope_json("room"),
        r#"{"scopes":[],"open":true}"#
    );
}

#[test]
fn cdc_reports_delete_as_one_node_deleted() {
    let mut st = Builder::default().build();
    let a = st.add_node(&["P"], &[]);
    let b = st.add_node(&["P"], &[]);
    st.add_edge(a, b, "R");
    st.begin();
    st.delete_node(a); // cascades the edge — reported as one NodeDeleted
    st.commit();
    assert_eq!(st.last_commit_changes(), &[Change::NodeDeleted(a)]);
}

#[test]
fn set_null_no_longer_de_opts_the_column() {
    use crate::value::Value;
    let mut st = Builder::default().build();
    let a = st.add_node(&["P"], &[("age", n(30.0))]);
    let _b = st.add_node(&["P"], &[("age", n(40.0))]);
    assert!(
        matches!(st.column("age"), Some(Column::Num { .. })),
        "starts Num"
    );
    st.set_prop(a, "age", Value::Null); // present-null
    assert!(
        matches!(st.column("age"), Some(Column::Num { .. })),
        "SET null keeps the column TYPED (no Gen de-opt) — the footgun is gone"
    );
    assert!(
        st.has_prop(a, "age") && st.prop(a, "age").is_null(),
        "present-null"
    );
    assert!(
        matches!(st.prop(_b, "age"), Value::Num(x) if x == 40.0),
        "sibling value intact + typed"
    );
    // remove the null -> column stays Num, node absent
    st.remove_prop(a, "age");
    assert!(!st.has_prop(a, "age"));
    assert!(matches!(st.column("age"), Some(Column::Num { .. })));
}

#[test]
fn required_constraint_declared_and_checked() {
    let mut st = Builder::default().build();
    st.add_node(&["User"], &[("email", s("a@x"))]);
    // Every User has email → the constraint declares, and the check passes.
    assert!(st.create_required_constraint("User", "email").is_ok());
    assert!(st.check_required_for_label("User").is_ok());
    // A User missing email → the check fails (present-null would pass; absence
    // is the violation).
    st.add_node(&["User"], &[("name", s("b"))]);
    assert!(st.check_required_for_label("User").is_err());
    // Declaring on already-violating data errors.
    let mut st2 = Builder::default().build();
    st2.add_node(&["User"], &[("name", s("x"))]);
    assert!(st2.create_required_constraint("User", "email").is_err());
}

#[test]
fn dotted_path_index_maintained_through_mutations() {
    use crate::value::make_record;
    let city = |c: &str| make_record(vec![(Arc::from("city"), s(c))]);
    let mut st = Builder::default().build();
    let a = st.add_node(&["P"], &[("meta", city("NYC"))]);
    let b = st.add_node(&["P"], &[("meta", city("LA"))]);
    let c = st.add_node(&["P"], &[("meta", city("NYC"))]);
    // Built from existing data: index on the record sub-field `meta.city`.
    st.create_index("meta.city");
    let nyc = |st: &Store| {
        let mut v = st.index_lookup("meta.city", &s("NYC")).unwrap();
        v.sort_unstable();
        v
    };
    assert_eq!(nyc(&st), vec![a, c]);
    assert_eq!(st.index_lookup("meta.city", &s("LA")).unwrap(), vec![b]);

    // Maintained on a write: change b's city to NYC.
    st.set_prop(b, "meta", city("NYC"));
    assert_eq!(nyc(&st), vec![a, b, c]);
    // …and on delete.
    st.delete_node(a);
    assert_eq!(nyc(&st), vec![b, c]);
    // No index on this path → None (distinct from an empty match).
    assert!(st.index_lookup("meta.zip", &n(1.0)).is_none());
}

/// Build an empty store, add two nodes and an edge, and read it all back —
/// hand-verified: ids 0 and 1, one out-edge 0→1 mirrored as an in-edge.
#[test]
fn add_nodes_and_edge_then_read_back() {
    let mut st = Builder::default().build();
    let a = st.add_node(&["P"], &[("name", s("a"))]);
    let b = st.add_node(&["P"], &[("name", s("b"))]);
    assert_eq!((a, b), (0, 1));
    assert_eq!(st.node_count(), 2);
    st.add_edge(a, b, "R");
    assert_eq!(st.nodes_with_label("P"), &[0, 1]);
    assert_eq!(st.out(a).len(), 1);
    assert_eq!(st.out(a)[0].nbr, b);
    assert_eq!(st.inc(b)[0].nbr, a);
    assert_eq!(st.out(a)[0].eid, st.inc(b)[0].eid); // shared edge id
    assert!(matches!(st.prop(a, "name"), Value::Str(x) if &*x == "a"));
}

/// Adding a node AFTER a property column exists extends that column with an
/// absent slot: the old node keeps its value, the new node reads NULL.
#[test]
fn add_node_extends_existing_columns() {
    let mut st = Builder::default().build();
    let a = st.add_node(&["P"], &[("age", n(30.0))]);
    let b = st.add_node(&["P"], &[]); // no age
    assert!(matches!(st.prop(a, "age"), Value::Num(x) if x == 30.0));
    assert!(st.prop(b, "age").is_null());
}

/// Writing a value of a different type promotes the column to `Gen`; both the
/// old-typed and new value read back correctly.
#[test]
fn set_prop_promotes_on_type_change() {
    let mut st = Builder::default().build();
    let a = st.add_node(&["P"], &[("v", n(1.0))]);
    let b = st.add_node(&["P"], &[("v", n(2.0))]);
    st.set_prop(a, "v", s("two")); // Num column, Str value -> promote to Gen
    assert!(matches!(st.prop(a, "v"), Value::Str(x) if &*x == "two"));
    assert!(matches!(st.prop(b, "v"), Value::Num(x) if x == 2.0)); // preserved
}

/// `remove_prop` makes the property read NULL again; overwriting sets it.
#[test]
fn set_and_remove_prop() {
    let mut st = Builder::default().build();
    let a = st.add_node(&["P"], &[("age", n(30.0))]);
    st.set_prop(a, "age", n(31.0));
    assert!(matches!(st.prop(a, "age"), Value::Num(x) if x == 31.0));
    st.remove_prop(a, "age");
    assert!(st.prop(a, "age").is_null());
}

/// A repeated edge type interns once (same `etype`) but each edge gets a
/// distinct `eid`. Ids continue after a `build()`-created edge.
#[test]
fn edge_type_interns_once_ids_unique() {
    let mut b = Builder::default();
    let x = b.node(&["P"], &[]);
    let y = b.node(&["P"], &[]);
    b.edge(x, y, "R"); // eid 0 at build
    let mut st = b.build();
    st.add_edge(x, y, "R"); // eid 1, same type
    st.add_edge(x, y, "S"); // eid 2, new type
    assert_eq!(st.out(x).len(), 3);
    let eids: Vec<u32> = st.out(x).iter().map(|a| a.eid).collect();
    assert_eq!(eids, vec![0, 1, 2]); // continued, unique
    assert_eq!(st.out(x)[0].etype, st.out(x)[1].etype); // R == R
    assert_ne!(st.out(x)[1].etype, st.out(x)[2].etype); // R != S
}

/// `delete_edge` removes the edge from both endpoints and is idempotent.
#[test]
fn delete_edge_detaches_both_sides() {
    let mut st = Builder::default().build();
    let a = st.add_node(&[], &[]);
    let b = st.add_node(&[], &[]);
    st.add_edge(a, b, "R");
    let eid = st.out(a)[0].eid;
    st.delete_edge(a, b, eid);
    assert!(st.out(a).is_empty());
    assert!(st.inc(b).is_empty());
    st.delete_edge(a, b, eid); // no-op the second time
    assert!(st.out(a).is_empty());
}

/// `delete_node` tombstones the node, detaches its edges from the neighbours'
/// mirror lists, clears its props, and drops it from scans. Hand-traced on
/// a→b, a→c, b→c: deleting b leaves a→c only, c with one incoming (from a).
#[test]
fn label_bucket_stays_sorted_through_delete_rollback() {
    let mut st = Builder::default().build();
    let ids: Vec<u32> = (0..6)
        .map(|i| st.add_node(&["P"], &[("age", n(f64::from(i)))]))
        .collect();
    st.create_index("age");
    // Delete a MIDDLE node in a transaction, then roll back (un-tombstone).
    st.begin();
    st.delete_node(ids[2]);
    st.rollback();
    // The label bucket must be sorted again — the restore re-inserts in place,
    // not appended — so the id-order scan seed and the binary-search label
    // intersection in `index_seek_ids` stay correct.
    let bucket = st.nodes_with_label("P");
    assert_eq!(bucket.len(), 6);
    assert!(
        bucket.windows(2).all(|w| w[0] < w[1]),
        "bucket not sorted after rollback: {bucket:?}"
    );
    // And the hash index still resolves the restored middle node.
    assert_eq!(st.index_lookup("age", &n(2.0)).unwrap(), vec![ids[2]]);
}

#[test]
fn delete_node_tombstones_and_cleans_up() {
    let mut st = Builder::default().build();
    let a = st.add_node(&["P"], &[("name", s("a"))]);
    let b = st.add_node(&["P"], &[("name", s("b"))]);
    let c = st.add_node(&["P"], &[("name", s("c"))]);
    st.add_edge(a, b, "R");
    st.add_edge(a, c, "R");
    st.add_edge(b, c, "R");
    st.delete_node(b);

    assert!(!st.is_alive(b));
    assert_eq!(st.all_nodes(), vec![a, c]);
    assert_eq!(st.nodes_with_label("P"), &[a, c]); // b removed from bucket
    assert_eq!(st.out(a).len(), 1); // a→b gone, a→c stays
    assert_eq!(st.out(a)[0].nbr, c);
    assert_eq!(st.inc(c).len(), 1); // b→c gone, a→c stays
    assert_eq!(st.inc(c)[0].nbr, a);
    assert!(st.out(b).is_empty());
    assert!(st.prop(b, "name").is_null()); // props cleared
    assert!(!st.prop(a, "name").is_null()); // neighbour intact
    st.delete_node(b); // idempotent
    assert_eq!(st.all_nodes(), vec![a, c]);
}

/// A self-loop is detached without panicking when its node is deleted.
#[test]
fn delete_node_with_self_loop() {
    let mut st = Builder::default().build();
    let a = st.add_node(&[], &[]);
    st.add_edge(a, a, "R");
    st.delete_node(a);
    assert!(!st.is_alive(a));
    assert!(st.out(a).is_empty());
    assert!(st.inc(a).is_empty());
}

// --- Transactions ---

/// Commit keeps the changes; the log is discarded.
#[test]
fn commit_keeps_changes() {
    let mut st = Builder::default().build();
    st.begin();
    let a = st.add_node(&["P"], &[("name", s("a"))]);
    st.commit();
    assert_eq!(st.node_count(), 1);
    assert!(matches!(st.prop(a, "name"), Value::Str(x) if &*x == "a"));
}

/// Rolling back an `add_node` truly removes it: node_count returns to 0 and
/// the columns shrink back (not merely tombstoned).
#[test]
fn rollback_add_node_shrinks_back() {
    let mut st = Builder::default().build();
    st.begin();
    st.add_node(&["P"], &[("name", s("a"))]);
    st.add_node(&["P"], &[("name", s("b"))]);
    assert_eq!(st.node_count(), 2);
    st.rollback();
    assert_eq!(st.node_count(), 0);
    assert!(st.all_nodes().is_empty());
    assert!(st.nodes_with_label("P").is_empty());
}

/// Rolling back `set_prop` restores the exact prior cell (present value).
#[test]
fn rollback_set_prop_restores_value() {
    let mut st = Builder::default().build();
    let a = st.add_node(&["P"], &[("age", n(30.0))]); // committed (autocommit)
    st.begin();
    st.set_prop(a, "age", n(99.0));
    st.set_prop(a, "age", s("oops")); // also promotes column to Gen
    assert!(matches!(st.prop(a, "age"), Value::Str(x) if &*x == "oops"));
    st.rollback();
    assert!(matches!(st.prop(a, "age"), Value::Num(x) if x == 30.0));
}

/// Rolling back a newly-set property (absent before) makes it absent again.
#[test]
fn rollback_new_prop_becomes_absent() {
    let mut st = Builder::default().build();
    let a = st.add_node(&["P"], &[]);
    st.begin();
    st.set_prop(a, "age", n(30.0));
    st.rollback();
    assert!(st.prop(a, "age").is_null());
    assert!(!st.has_prop(a, "age"));
}

/// Rolling back `add_edge` removes it from both endpoints.
#[test]
fn rollback_add_edge() {
    let mut st = Builder::default().build();
    let a = st.add_node(&[], &[]);
    let b = st.add_node(&[], &[]);
    st.begin();
    st.add_edge(a, b, "R");
    st.rollback();
    assert!(st.out(a).is_empty());
    assert!(st.inc(b).is_empty());
}

/// Rolling back `delete_node` restores it fully: tombstone, adjacency (its own
/// lists AND the neighbours' mirrors), label membership, and properties.
/// Hand-traced on a→b, b→c: delete b, then roll back → identical to before.
#[test]
fn rollback_delete_node_restores_everything() {
    let mut st = Builder::default().build();
    let a = st.add_node(&["P"], &[("name", s("a"))]);
    let b = st.add_node(&["P"], &[("name", s("b"))]);
    let c = st.add_node(&["P"], &[("name", s("c"))]);
    st.add_edge(a, b, "R");
    st.add_edge(b, c, "R");
    st.begin();
    st.delete_node(b);
    assert!(!st.is_alive(b));
    st.rollback();

    assert!(st.is_alive(b));
    assert_eq!(st.nodes_with_label("P").len(), 3);
    assert!(matches!(st.prop(b, "name"), Value::Str(x) if &*x == "b"));
    // adjacency restored on all three nodes
    assert_eq!(st.out(a).len(), 1); // a→b
    assert_eq!(st.out(a)[0].nbr, b);
    assert_eq!(st.out(b).len(), 1); // b→c
    assert_eq!(st.out(b)[0].nbr, c);
    assert_eq!(st.inc(b).len(), 1); // a→b mirror
    assert_eq!(st.inc(c).len(), 1); // b→c mirror
}

/// `savepoint` + `rollback_to` give per-statement atomicity: the first
/// statement's writes survive, the second's are undone, the transaction stays
/// open, and the final commit keeps only the first.
#[test]
fn savepoint_rolls_back_one_statement() {
    let mut st = Builder::default().build();
    st.begin();
    let a = st.add_node(&["P"], &[("name", s("a"))]); // statement 1
    let mark = st.savepoint();
    let b = st.add_node(&["P"], &[("name", s("b"))]); // statement 2
    st.add_edge(a, b, "R");
    st.rollback_to(mark); // undo statement 2 only
    assert_eq!(st.node_count(), 1); // b popped
    assert!(st.out(a).is_empty()); // edge gone
    st.commit();
    assert_eq!(st.node_count(), 1);
    assert!(matches!(st.prop(a, "name"), Value::Str(x) if &*x == "a"));
}

// --- Edge properties ---

/// Set / read / remove an edge property, keyed by the edge's eid.
#[test]
fn edge_property_set_read_remove() {
    let mut st = Builder::default().build();
    let a = st.add_node(&[], &[]);
    let b = st.add_node(&[], &[]);
    st.add_edge(a, b, "R");
    let eid = st.out(a)[0].eid;
    assert!(st.edge_prop(eid, "weight").is_null()); // absent
    st.set_edge_prop(eid, "weight", n(0.5));
    assert!(st.has_edge_prop(eid, "weight"));
    assert!(matches!(st.edge_prop(eid, "weight"), Value::Num(x) if x == 0.5));
    st.remove_edge_prop(eid, "weight");
    assert!(!st.has_edge_prop(eid, "weight"));
    assert!(st.edge_prop(eid, "weight").is_null());
}

/// The numeric edge overlay tracks the boxed source of truth through every
/// mutation, and demotes a key that gains a non-numeric value.
#[test]
fn edge_num_overlay_tracks_boxed() {
    // Edges via the Builder (as the bench / from_ndjson do), so build() leaves the
    // overlay fresh and the incremental set-maintenance keeps it so.
    let mut b = Builder::default();
    let ids: Vec<u32> = (0..6).map(|_| b.node(&[], &[])).collect();
    for i in 0..5 {
        b.edge(ids[i], ids[i + 1], "R");
    }
    let mut st = b.build();
    let eids: Vec<u32> = (0..5).map(|i| st.out(i)[0].eid).collect();
    // Overlay agrees with edge_prop for every eid, for a numeric key.
    let agree = |st: &Store, key: &str| {
        let Some((data, present)) = st.edge_num_column(key) else {
            return None; // key not overlaid
        };
        for (idx, &eid) in eids.iter().enumerate() {
            let boxed = st.edge_prop(eid, key);
            let ov = if present[eid as usize] {
                Value::Num(data[eid as usize])
            } else {
                Value::Null
            };
            assert!(
                crate::value::equals(&boxed, &ov) || (boxed.is_null() && ov.is_null()),
                "overlay/boxed mismatch at eid {eid} (row {idx})"
            );
        }
        Some(())
    };
    for &eid in &eids {
        st.set_edge_prop(eid, "w", n(f64::from(eid) * 2.0));
    }
    assert!(
        agree(&st, "w").is_some(),
        "w should be overlaid after bulk set"
    );
    // Remove one, mutate another — overlay still agrees.
    st.remove_edge_prop(eids[1], "w");
    st.set_edge_prop(eids[2], "w", n(99.0));
    agree(&st, "w");
    // A non-numeric write demotes the key (readers fall back to boxed).
    st.set_edge_prop(eids[0], "w", s("hello"));
    assert!(
        st.edge_num_column("w").is_none(),
        "a Str value demotes the overlay"
    );
    assert!(matches!(st.edge_prop(eids[0], "w"), Value::Str(ref x) if &**x == "hello"));
    // A separate numeric key stays overlaid and correct.
    st.set_edge_prop(eids[3], "k", n(7.0));
    st.set_edge_prop(eids[4], "k", n(8.0));
    assert!(agree(&st, "k").is_some());
    // A fresh add_edge invalidates the overlay (eid space grew) → boxed fallback.
    st.add_edge(0, 2, "R");
    assert!(
        st.edge_num_column("k").is_none(),
        "add_edge invalidates the overlay"
    );
    assert!(matches!(st.edge_prop(eids[3], "k"), Value::Num(x) if x == 7.0));
    // boxed still right
}

/// An edge property write rolls back with the transaction.
#[test]
fn edge_property_rolls_back() {
    let mut st = Builder::default().build();
    let a = st.add_node(&[], &[]);
    let b = st.add_node(&[], &[]);
    st.add_edge(a, b, "R");
    let eid = st.out(a)[0].eid;
    st.set_edge_prop(eid, "weight", n(1.0)); // committed (autocommit)
    st.begin();
    st.set_edge_prop(eid, "weight", n(2.0));
    st.set_edge_prop(eid, "fresh", s("x"));
    st.rollback();
    assert!(matches!(st.edge_prop(eid, "weight"), Value::Num(x) if x == 1.0)); // restored
    assert!(!st.has_edge_prop(eid, "fresh")); // new key gone
}

// --- Unique constraints ---

/// A unique constraint on already-conforming data is accepted; check passes.
#[test]
fn unique_constraint_accepts_conforming_data() {
    let mut st = Builder::default().build();
    st.add_node(&["User"], &[("email", s("a@x"))]);
    st.add_node(&["User"], &[("email", s("b@x"))]);
    assert!(st.create_unique_constraint("User", &["email"]).is_ok());
    assert!(st.check_unique_for_label("User").is_ok());
}

/// Declaring a constraint the data already violates errors.
#[test]
fn unique_constraint_rejects_existing_duplicate() {
    let mut st = Builder::default().build();
    st.add_node(&["User"], &[("email", s("dup"))]);
    st.add_node(&["User"], &[("email", s("dup"))]);
    assert!(st.create_unique_constraint("User", &["email"]).is_err());
}

/// After a constraint, a duplicate added at the store level is detected by the
/// check (the store primitive itself stays infallible; enforcement is the
/// caller's, as the write statements do).
#[test]
fn unique_check_detects_new_duplicate() {
    let mut st = Builder::default().build();
    st.add_node(&["User"], &[("email", s("x"))]);
    st.create_unique_constraint("User", &["email"]).unwrap();
    st.add_node(&["User"], &[("email", s("x"))]); // primitive allows it
    assert!(st.check_unique_for_label("User").is_err()); // check catches it
}

/// Conflict-target inference: the constraint keys are returned when the
/// pattern's key set covers them.
#[test]
fn unique_keys_for_infers_target() {
    let mut st = Builder::default().build();
    st.create_unique_constraint("User", &["email"]).unwrap();
    assert_eq!(
        st.unique_keys_for("User", &["email".into(), "name".into()]),
        Some(vec!["email".into()])
    );
    assert_eq!(st.unique_keys_for("User", &["name".into()]), None);
    assert_eq!(st.unique_keys_for("Other", &["email".into()]), None);
}

/// `transaction` commits on `Ok` and rolls back on `Err`.
#[test]
fn transaction_commits_ok_rolls_back_err() {
    let mut st = Builder::default().build();
    let r: Result<u32, ()> = st.transaction(|s| Ok(s.add_node(&["P"], &[])));
    assert!(r.is_ok());
    assert_eq!(st.node_count(), 1);

    let r: Result<(), &str> = st.transaction(|s| {
        s.add_node(&["P"], &[]);
        Err("boom")
    });
    assert_eq!(r, Err("boom"));
    assert_eq!(st.node_count(), 1); // the aborted add was rolled back
}

// --- opt-in edge-type index (G5) ---

/// A small multi-type graph: node 0 knows 1 and likes 2; node 1 knows 2.
fn typed_graph() -> Store {
    let mut b = Builder::default();
    for _ in 0..3 {
        b.node(&["V"], &[]);
    }
    b.edge(0, 1, "KNOWS");
    b.edge(0, 2, "LIKES");
    b.edge(1, 2, "KNOWS");
    b.build()
}

/// The typed neighbours of `node` along `etype` (out), as a sorted id list.
fn out_ids(st: &Store, node: u32, ty: &str) -> Vec<u32> {
    let et = st.etype_id(ty).unwrap();
    let mut v: Vec<u32> = st.out_typed(node, et).iter().map(|a| a.nbr).collect();
    v.sort_unstable();
    v
}

/// The index, once built, agrees with a manual type-filter of the flat
/// adjacency for every node and type.
#[test]
fn edge_type_index_matches_flat_scan() {
    let mut st = typed_graph();
    st.create_edge_type_index();
    assert!(st.has_edge_type_index());
    for node in 0..st.node_count() as u32 {
        for ty in ["KNOWS", "LIKES"] {
            let et = st.etype_id(ty).unwrap();
            let mut scan: Vec<u32> = st
                .out(node)
                .iter()
                .filter(|a| a.etype == et)
                .map(|a| a.nbr)
                .collect();
            scan.sort_unstable();
            assert_eq!(out_ids(&st, node, ty), scan, "node {node} type {ty}");
        }
    }
    // node 0: KNOWS -> {1}, LIKES -> {2}
    assert_eq!(out_ids(&st, 0, "KNOWS"), vec![1]);
    assert_eq!(out_ids(&st, 0, "LIKES"), vec![2]);
}

/// add_edge keeps the index current (the O(1) hot path).
#[test]
fn edge_type_index_tracks_add() {
    let mut st = typed_graph();
    st.create_edge_type_index();
    st.add_edge(0, 2, "KNOWS");
    assert_eq!(out_ids(&st, 0, "KNOWS"), vec![1, 2]); // 0 now KNOWS 1 and 2
}

/// delete_edge and delete_node keep the index current, including neighbours'
/// incoming buckets.
#[test]
fn edge_type_index_tracks_delete() {
    let mut st = typed_graph();
    st.create_edge_type_index();
    let et = st.etype_id("KNOWS").unwrap();
    // delete 0-KNOWS->1: gone from 0's out bucket AND 1's in bucket.
    st.delete_edge(0, 1, 0);
    assert_eq!(out_ids(&st, 0, "KNOWS"), Vec::<u32>::new());
    let in1: Vec<u32> = st.in_typed(1, et).iter().map(|a| a.nbr).collect();
    assert_eq!(in1, Vec::<u32>::new());
    // delete node 2: removes 0-LIKES->2 and 1-KNOWS->2 mirrors.
    st.delete_node(2);
    assert_eq!(out_ids(&st, 0, "LIKES"), Vec::<u32>::new());
    assert_eq!(out_ids(&st, 1, "KNOWS"), Vec::<u32>::new());
}

/// Transaction rollback restores the index exactly (a per-node rebuild off the
/// restored flat adjacency, so no delta bookkeeping can drift).
#[test]
fn edge_type_index_survives_rollback() {
    let mut st = typed_graph();
    st.create_edge_type_index();
    st.begin();
    st.add_edge(0, 2, "KNOWS"); // 0 KNOWS {1,2} inside the txn
    st.delete_edge(1, 2, 2); // 1 KNOWS {} inside the txn
    assert_eq!(out_ids(&st, 0, "KNOWS"), vec![1, 2]);
    st.rollback();
    // Back to the committed shape: 0 KNOWS {1}, 1 KNOWS {2}.
    assert_eq!(out_ids(&st, 0, "KNOWS"), vec![1]);
    assert_eq!(out_ids(&st, 1, "KNOWS"), vec![2]);
}

/// A node added AFTER the index exists grows the index and indexes its edges.
#[test]
fn edge_type_index_grows_with_new_node() {
    let mut st = typed_graph();
    st.create_edge_type_index();
    let three = st.add_node(&["V"], &[]);
    st.add_edge(three, 0, "LIKES");
    assert_eq!(out_ids(&st, three, "LIKES"), vec![0]);
}

// --- opt-in edge interval index (G4) ---

/// One Emp node (0) with `degree` HELD edges to role node 1, edge d carrying
/// interval `[d, d+width]`.
fn interval_graph(degree: u32, width: i64) -> Store {
    let mut b = Builder::default();
    b.node(&["Emp"], &[]);
    b.node(&["Role"], &[]);
    let mut st = b.build();
    for d in 0..degree {
        let eid = st.add_edge(0, 1, "HELD");
        st.set_edge_prop(eid, "vf", n(f64::from(d)));
        st.set_edge_prop(eid, "vt", n((i64::from(d) + width) as f64));
    }
    st
}

/// Overlap eids from the index (sorted), vs a brute-force scan of the flat
/// adjacency reading the boxed props.
fn overlap_eids(st: &Store, node: u32, qlo: f64, qhi: f64) -> Vec<u32> {
    let mut v = Vec::new();
    st.for_each_overlap(node, qlo, qhi, |eid, _| v.push(eid));
    v.sort_unstable();
    v
}
fn overlap_bruteforce(st: &Store, node: u32, qlo: f64, qhi: f64) -> Vec<u32> {
    let mut v: Vec<u32> = st
        .out(node)
        .iter()
        .filter(|a| {
            matches!((st.edge_prop(a.eid, "vf"), st.edge_prop(a.eid, "vt")),
                    (Value::Num(lo), Value::Num(hi)) if lo <= qhi && hi >= qlo)
        })
        .map(|a| a.eid)
        .collect();
    v.sort_unstable();
    v
}

/// The seek agrees with a brute-force overlap scan for point queries across the
/// timeline AND for wider interval queries (both seed axes exercised).
#[test]
fn interval_seek_matches_bruteforce() {
    let st = {
        let mut s = interval_graph(64, 4);
        s.create_interval_index("vf", "vt");
        s
    };
    assert!(st.has_interval_index("vf", "vt"));
    // as-of points across the whole timeline (0..=67), incl. the ends where one
    // axis is far more selective than the other.
    for t in 0..=67 {
        let q = f64::from(t);
        assert_eq!(
            overlap_eids(&st, 0, q, q),
            overlap_bruteforce(&st, 0, q, q),
            "point t={t}"
        );
    }
    // wider ranges
    for &(lo, hi) in &[(10.0, 20.0), (0.0, 100.0), (63.0, 63.0), (-5.0, 2.0)] {
        assert_eq!(
            overlap_eids(&st, 0, lo, hi),
            overlap_bruteforce(&st, 0, lo, hi),
            "range [{lo},{hi}]"
        );
    }
}

/// Writes keep the interval index current: a new edge+interval appears, a
/// changed interval moves, a deleted edge vanishes.
#[test]
fn interval_index_tracks_writes() {
    let mut st = interval_graph(4, 2); // edges: [0,2],[1,3],[2,4],[3,5]
    st.create_interval_index("vf", "vt");
    // as-of t=10 → none.
    assert_eq!(overlap_eids(&st, 0, 10.0, 10.0), Vec::<u32>::new());
    // add an edge covering t=10.
    let e = st.add_edge(0, 1, "HELD");
    st.set_edge_prop(e, "vf", n(8.0));
    st.set_edge_prop(e, "vt", n(12.0));
    assert_eq!(overlap_eids(&st, 0, 10.0, 10.0), vec![e]);
    // move it off t=10.
    st.set_edge_prop(e, "vt", n(9.0));
    assert_eq!(overlap_eids(&st, 0, 10.0, 10.0), Vec::<u32>::new());
    // delete it.
    st.set_edge_prop(e, "vt", n(12.0));
    st.delete_edge(0, 1, e);
    assert_eq!(overlap_eids(&st, 0, 10.0, 10.0), Vec::<u32>::new());
}

/// Rollback restores the interval index exactly (a full rebuild against the
/// restored graph, so prop AND adjacency undo ordering can't drift it).
#[test]
fn interval_index_survives_rollback() {
    let mut st = interval_graph(4, 2);
    st.create_interval_index("vf", "vt");
    let before: Vec<Vec<u32>> = (0..8)
        .map(|t| overlap_eids(&st, 0, f64::from(t), f64::from(t)))
        .collect();
    st.begin();
    let e = st.add_edge(0, 1, "HELD");
    st.set_edge_prop(e, "vf", n(0.0));
    st.set_edge_prop(e, "vt", n(100.0)); // covers everything, inside the txn
    st.delete_edge(0, 1, 0); // and drop the first committed edge
    assert!(overlap_eids(&st, 0, 3.0, 3.0).contains(&e));
    st.rollback();
    let after: Vec<Vec<u32>> = (0..8)
        .map(|t| overlap_eids(&st, 0, f64::from(t), f64::from(t)))
        .collect();
    assert_eq!(before, after);
}

/// Updating an edge's interval property must cost the same whether the graph has 2k
/// or 32k nodes — the index is source-partitioned, so only the edge's source node is
/// reindexed. Guards against the O(V+E) full `rebuild_interval` this path used to do
/// on every write (a bitemporal workload that hot-updates edge validity intervals
/// would otherwise repack the whole index per write). Timing/scaling guard, so MIN
/// over reps and `#[ignore]`d — run isolated in release:
///   cargo test -p lenke-engine --release --lib \
///     interval_edge_write_is_independent_of_graph_size -- --ignored
#[test]
#[ignore = "timing/scaling guard — run isolated in release (see doc comment)"]
fn interval_edge_write_is_independent_of_graph_size() {
    use std::time::Instant;

    const REPS: usize = 9;

    // A graph of `nnodes` where node 0 has a FIXED small out-degree of interval
    // edges. The source degree (what an incremental reindex touches) is constant, so
    // only a full-index rebuild would make the write scale with `nnodes`.
    let build = |nnodes: u32| -> (Store, u32) {
        let mut b = Builder::default();
        for _ in 0..nnodes {
            b.node(&["P"], &[]);
        }
        let mut st = b.build();
        st.create_interval_index("vf", "vt");
        let mut first = 0;
        for d in 0..4u32 {
            let eid = st.add_edge(0, (d + 1) % nnodes, "HELD");
            st.set_edge_prop(eid, "vf", n(f64::from(d)));
            st.set_edge_prop(eid, "vt", n(f64::from(d) + 10.0));
            if d == 0 {
                first = eid;
            }
        }
        (st, first)
    };
    let time = |st: &mut Store, eid: u32| {
        for i in 0..20 {
            st.set_edge_prop(eid, "vt", n(f64::from(i) + 10.0)); // warm up
        }
        let mut best = f64::INFINITY;
        for _ in 0..REPS {
            let t = Instant::now();
            for i in 0..100 {
                st.set_edge_prop(eid, "vt", n(f64::from(i) + 10.0));
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
        best
    };

    let (mut small_st, e_small) = build(2_000);
    let small = time(&mut small_st, e_small);
    let (mut large_st, e_large) = build(32_000);
    let large = time(&mut large_st, e_large);

    let ratio = large / small.max(f64::MIN_POSITIVE);
    assert!(
        ratio < 6.0,
        "interval-key edge write scaled with graph size: 16x more nodes cost {ratio:.1}x \
             more time ({small:.5}s -> {large:.5}s, min of {REPS}). `set_edge_prop` is \
             rebuilding the whole interval index instead of reindexing the edge's source node."
    );
}

// --- type / cardinality / edge / drop-index constraints ---------------

#[test]
fn type_constraint_enforced_on_write() {
    let mut st = Store::default();
    assert!(st
        .create_type_constraint("P", "age", "number", false)
        .is_ok());
    st.begin();
    st.add_node(&["P"], &[("age", Value::Str("old".into()))]);
    assert!(
        st.run_deferred_checks().is_err(),
        "string age violates number"
    );
    st.rollback();
    st.begin();
    st.add_node(&["P"], &[("age", n(42.0))]);
    assert!(st.run_deferred_checks().is_ok());
    st.commit();
}

#[test]
fn type_constraint_unknown_name_is_invalid_value() {
    let mut st = Store::default();
    let e = st
        .create_type_constraint("P", "age", "bogus", false)
        .unwrap_err();
    assert!(e.starts_with("E_INVALID_VALUE"), "{e}");
}

#[test]
fn not_null_type_constraint_rejects_absent_and_null() {
    let mut st = Store::default();
    st.create_type_constraint("P", "name", "string NOT NULL", false)
        .unwrap();
    for missing in [vec![], vec![("name", Value::Null)]] {
        st.begin();
        st.add_node(&["P"], &missing);
        assert!(st.run_deferred_checks().is_err(), "NOT NULL must reject");
        st.rollback();
    }
    st.begin();
    st.add_node(&["P"], &[("name", Value::Str("ann".into()))]);
    assert!(st.run_deferred_checks().is_ok());
    st.commit();
}

#[test]
fn cardinality_min_and_max_enforced() {
    let mut st = Store::default();
    // out-degree of KNOWS must be exactly 1.
    st.create_cardinality_constraint("P", "KNOWS", 0, 1, Some(1))
        .unwrap();
    st.begin();
    st.add_node(&["P"], &[]); // 0 edges → below min
    assert!(st.run_deferred_checks().is_err());
    st.rollback();
    st.begin();
    let a = st.add_node(&["P"], &[]);
    let b = st.add_node(&["T"], &[]);
    let c = st.add_node(&["T"], &[]);
    st.add_edge(a, b, "KNOWS");
    st.add_edge(a, c, "KNOWS"); // out-degree 2 → above max
    assert!(st.run_deferred_checks().is_err());
    st.rollback();
    st.begin();
    let a = st.add_node(&["P"], &[]);
    let b = st.add_node(&["T"], &[]);
    st.add_edge(a, b, "KNOWS"); // exactly 1
    assert!(st.run_deferred_checks().is_ok());
    st.commit();
}

#[test]
fn edge_unique_enforced_and_null_exempt() {
    let mut st = Store::default();
    st.create_edge_unique_constraint("PAID", &["ref"]).unwrap();
    st.begin();
    let a = st.add_node(&["A"], &[]);
    let b = st.add_node(&["A"], &[]);
    let e1 = st.add_edge(a, b, "PAID");
    st.set_edge_prop(e1, "ref", Value::Str("x".into()));
    let e2 = st.add_edge(a, b, "PAID");
    st.set_edge_prop(e2, "ref", Value::Str("x".into())); // duplicate
    assert!(st.run_deferred_checks().is_err());
    st.rollback();
    // Two edges with an absent `ref` are exempt (nulls don't collide).
    st.begin();
    let a = st.add_node(&["A"], &[]);
    let b = st.add_node(&["A"], &[]);
    st.add_edge(a, b, "PAID");
    st.add_edge(a, b, "PAID");
    assert!(st.run_deferred_checks().is_ok());
    st.commit();
}

#[test]
fn edge_required_enforced_on_write() {
    let mut st = Store::default();
    st.create_edge_required_constraint("PAID", "amt").unwrap();
    st.begin();
    let a = st.add_node(&["A"], &[]);
    let b = st.add_node(&["A"], &[]);
    st.add_edge(a, b, "PAID"); // missing amt
    assert!(st.run_deferred_checks().is_err());
    st.rollback();
}

#[test]
fn drop_index_removes_and_guards_a_backing_unique() {
    let mut st = Store::default();
    st.add_node(&["P"], &[("age", n(1.0))]);
    st.create_index("age");
    assert!(st.has_hash_index("age"));
    assert!(st.drop_vertex_index("age").is_ok());
    assert!(!st.has_hash_index("age"));
    assert!(st.drop_vertex_index("age").is_ok(), "drop is idempotent");
    // Dropping the index behind a unique constraint is rejected.
    st.create_index("email");
    st.create_unique_constraint("P", &["email"]).unwrap();
    let e = st.drop_vertex_index("email").unwrap_err();
    assert!(e.starts_with("E_INVALID_GRAPH_OP"), "{e}");
}

#[test]
fn declaring_a_constraint_the_data_breaks_is_rejected() {
    let mut st = Store::default();
    st.add_node(&["P"], &[("age", Value::Str("nope".into()))]);
    let e = st
        .create_type_constraint("P", "age", "number", false)
        .unwrap_err();
    assert!(e.starts_with("E_TYPE"), "{e}");
}

/// A CLOSED record type: field types + NOT NULL fields + closed-on-extras, with
/// a nested record. A node whose `m` is a matching record accepts the constraint;
/// each way of breaking the shape rejects it.
#[test]
fn closed_record_type_constraint() {
    let spec = "record{a::number,b::string NOT NULL,c::record{d::boolean}}";
    // `m` built from a nested JSON object → a canonical Value::Record.
    let node = |m: &str| {
        crate::ndjson::from_ndjson(&format!(
            "{{\"id\":\"a\",\"labels\":[\"P\"],\"props\":{{\"m\":{m}}}}}\n"
        ))
        .unwrap()
    };
    let declare = |mut st: Store| st.create_type_constraint("P", "m", spec, false);

    // Conforming (optional `a`/`c` omitted; required `b` present; nested ok).
    assert!(declare(node(r#"{"b":"x"}"#)).is_ok());
    assert!(declare(node(r#"{"a":1,"b":"x","c":{"d":true}}"#)).is_ok());
    // Wrong scalar field type.
    assert!(declare(node(r#"{"a":"nope","b":"x"}"#)).is_err());
    // Missing a NOT NULL field.
    assert!(declare(node(r#"{"a":1}"#)).is_err());
    // Extra field (closed on extras).
    assert!(declare(node(r#"{"b":"x","z":2}"#)).is_err());
    // Nested field wrong type.
    assert!(declare(node(r#"{"b":"x","c":{"d":5}}"#)).is_err());
    // A null property is exempt (the property is nullable without NOT NULL).
    assert!(declare(node("null")).is_ok());
    // A non-record value violates a record type.
    assert!(declare(node("42")).is_err());
}

#[test]
fn any_record_type_constraint() {
    let node = |m: &str| {
        crate::ndjson::from_ndjson(&format!(
            "{{\"id\":\"a\",\"labels\":[\"P\"],\"props\":{{\"m\":{m}}}}}\n"
        ))
        .unwrap()
    };
    // `any record` accepts any record shape but rejects a scalar.
    assert!(node(r#"{"anything":1,"here":true}"#)
        .create_type_constraint("P", "m", "any record", false)
        .is_ok());
    assert!(node("42")
        .create_type_constraint("P", "m", "any record", false)
        .is_err());
}

#[test]
fn record_type_name_round_trips() {
    // A declared record type dumps a spec that parses back to the same rule.
    let mut st = crate::ndjson::from_ndjson(
        "{\"id\":\"a\",\"labels\":[\"P\"],\"props\":{\"m\":{\"a\":1,\"b\":\"x\"}}}\n",
    )
    .unwrap();
    let spec = "record{a::number,b::string NOT NULL}";
    st.create_type_constraint("P", "m", spec, false).unwrap();
    let (_, _, dumped, _) = st.type_constraints().into_iter().next().unwrap();
    // Re-declaring from the dumped name succeeds on the same data (round-trips).
    let mut st2 = crate::ndjson::from_ndjson(
        "{\"id\":\"a\",\"labels\":[\"P\"],\"props\":{\"m\":{\"a\":1,\"b\":\"x\"}}}\n",
    )
    .unwrap();
    assert!(
        st2.create_type_constraint("P", "m", &dumped, false).is_ok(),
        "dumped: {dumped}"
    );
}
