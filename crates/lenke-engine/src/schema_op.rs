//! The schema-DDL op vocabulary behind the C ABI's `lnk_schema_apply` /
//! `lnk_schema_dump` (see `docs/abi.md`).
//!
//! Core exposes one C entry point per schema operation (10 `create_*` + 2
//! `drop_*`); the engine collapses them into a single tagged-JSON dispatcher. One
//! `{"op":…}` object per [`apply`] call selects the operation:
//!
//! ```jsonc
//! { "op": "createIndex", "on": "vertex", "kind": "hash",     "keys": ["age"] }
//! { "op": "createIndex", "on": "vertex", "kind": "range",    "keys": ["score"] }
//! { "op": "createIndex", "on": "edge",   "kind": "interval", "keys": ["lo","hi"] }
//! { "op": "createIndex", "on": "edge",   "kind": "type" }
//! { "op": "unique",   "on": "vertex", "label": "Person", "key": "email" }
//! { "op": "required", "on": "vertex", "label": "Person", "key": "name" }
//! ```
//!
//! [`dump`] emits the same vocabulary for the constraints it can introspect, so a
//! `dump` → `apply` round-trip reconstructs them. Ops the engine's `Store` does
//! not implement yet (`dropIndex`, `type`/`cardinality`/`validator`/`invariant`,
//! and edge unique/required) are reported as [`SchemaError::BadRequest`] rather
//! than silently ignored — that error list is the schema work still to do.

use crate::ndjson::{self, Json};
use crate::store::Store;

/// Why a schema op could not be applied. Each carries the wire code the C ABI
/// reports (via `lnk_schema_apply`), so a host maps a declaration failure to the
/// same `LenkeError` core would raise: `BadRequest`→`E_FFI`(-1),
/// `Invalid`→`E_INVALID_VALUE`(-1), `Syntax`→`E_SYNTAX`(-1),
/// `GraphOp`→`E_INVALID_GRAPH_OP`(-1), `Rejected`→`E_CONSTRAINT_VIOLATION`(-2).
#[derive(Debug)]
pub enum SchemaError {
    /// Malformed JSON, an unknown/missing field, or an unknown op.
    BadRequest(String),
    /// A well-formed op the current data violates (e.g. an existing duplicate).
    Rejected(String),
    /// An out-of-model value (e.g. an unknown type name).
    Invalid(String),
    /// A malformed validator predicate or invariant query.
    Syntax(String),
    /// A graph-op that can't run (e.g. dropping an index that backs a constraint).
    GraphOp(String),
}

/// Map a store-layer error (which prefixes its message with a wire code) to the
/// matching [`SchemaError`], stripping the prefix. An unprefixed / constraint
/// message is a data-violation `Rejected`.
fn store_err(e: String) -> SchemaError {
    if let Some((prefix, msg)) = e.split_once(": ") {
        match prefix {
            "E_INVALID_VALUE" => return SchemaError::Invalid(msg.to_string()),
            "E_SYNTAX" => return SchemaError::Syntax(msg.to_string()),
            "E_INVALID_GRAPH_OP" => return SchemaError::GraphOp(msg.to_string()),
            "E_UNIQUE" | "E_REQUIRED" | "E_TYPE" | "E_CARDINALITY" | "E_VALIDATOR"
            | "E_INVARIANT" => return SchemaError::Rejected(msg.to_string()),
            _ => {}
        }
    }
    SchemaError::Rejected(e)
}

/// Apply one schema op described by `json` (a single tagged `{"op":…}` object).
pub fn apply(store: &mut Store, json: &str) -> Result<(), SchemaError> {
    let parsed = ndjson::parse_json(json).map_err(SchemaError::BadRequest)?;
    let Json::Obj(fields) = &parsed else {
        return Err(SchemaError::BadRequest(
            "schema op must be a JSON object".into(),
        ));
    };
    let op = str_field(fields, "op")?;
    match op.as_str() {
        "createIndex" => create_index(store, fields),
        "dropIndex" => drop_index(store, fields),
        "unique" => unique(store, fields),
        "required" => required(store, fields),
        "type" => type_constraint(store, fields),
        "cardinality" => cardinality(store, fields),
        // Validators and invariants need the query evaluator, so they are declared
        // through the exec layer (see `crate::exec::apply_schema_op`), not here.
        "validator" | "invariant" => Err(SchemaError::BadRequest(format!(
            "schema op `{op}` must be applied through the exec layer"
        ))),
        other => Err(SchemaError::BadRequest(format!(
            "unknown schema op `{other}`"
        ))),
    }
}

/// The edge/vertex target selector — reads `on` (default `vertex`).
fn is_edge(fields: &[(String, Json)]) -> Result<bool, SchemaError> {
    match str_field(fields, "on")?.as_str() {
        "edge" => Ok(true),
        "vertex" => Ok(false),
        other => Err(SchemaError::BadRequest(format!(
            "unknown `on` target `{other}`"
        ))),
    }
}

/// The constrained target name: `label` for a vertex op, `etype` for an edge op.
fn target_field(fields: &[(String, Json)], edge: bool) -> Result<String, SchemaError> {
    str_field(fields, if edge { "etype" } else { "label" })
}

/// Dump the schema — indexes and constraints — in the [`apply`] vocabulary, so a
/// `dump` → `apply` round-trip reconstructs it. Ops are emitted in a stable order
/// (sorted keys) so the dump is deterministic.
pub fn dump(store: &Store) -> String {
    let mut out = String::new();

    // Indexes first (a constraint may rely on its backing index existing).
    let mut hash = store.hash_index_keys();
    hash.sort();
    for key in hash {
        out.push_str("{\"op\":\"createIndex\",\"on\":\"vertex\",\"kind\":\"hash\",\"keys\":");
        ndjson::encode_str_array(&mut out, std::slice::from_ref(&key));
        out.push_str("}\n");
    }
    let mut range = store.range_index_keys();
    range.sort();
    for key in range {
        out.push_str("{\"op\":\"createIndex\",\"on\":\"vertex\",\"kind\":\"range\",\"keys\":");
        ndjson::encode_str_array(&mut out, std::slice::from_ref(&key));
        out.push_str("}\n");
    }
    if let Some((lo, hi)) = store.interval_index_keys() {
        out.push_str("{\"op\":\"createIndex\",\"on\":\"edge\",\"kind\":\"interval\",\"keys\":");
        ndjson::encode_str_array(&mut out, &[lo, hi]);
        out.push_str("}\n");
    }
    if store.has_edge_type_index() {
        out.push_str("{\"op\":\"createIndex\",\"on\":\"edge\",\"kind\":\"type\"}\n");
    }

    let mut uniques = store.unique_constraints();
    uniques.sort();
    for (label, keys) in uniques {
        out.push_str("{\"op\":\"unique\",\"on\":\"vertex\",\"label\":");
        ndjson::encode_string(&mut out, &label);
        out.push_str(",\"keys\":");
        ndjson::encode_str_array(&mut out, &keys);
        out.push_str("}\n");
    }
    let mut required = store.required_constraints();
    required.sort();
    for (label, key) in required {
        out.push_str("{\"op\":\"required\",\"on\":\"vertex\",\"label\":");
        ndjson::encode_string(&mut out, &label);
        out.push_str(",\"key\":");
        ndjson::encode_string(&mut out, &key);
        out.push_str("}\n");
    }

    // Vertex type constraints (scalar / list / record; NOT NULL folded into `type`).
    let mut v_type = store.type_constraints();
    v_type.sort();
    for (label, key, ty, not_null) in v_type {
        push_type_op(&mut out, "vertex", "label", &label, &key, &ty, not_null);
    }

    let mut e_unique = store.edge_unique_constraints();
    e_unique.sort();
    for (etype, keys) in e_unique {
        out.push_str("{\"op\":\"unique\",\"on\":\"edge\",\"etype\":");
        ndjson::encode_string(&mut out, &etype);
        out.push_str(",\"keys\":");
        ndjson::encode_str_array(&mut out, &keys);
        out.push_str("}\n");
    }
    let mut e_required = store.edge_required_constraints();
    e_required.sort();
    for (etype, key) in e_required {
        out.push_str("{\"op\":\"required\",\"on\":\"edge\",\"etype\":");
        ndjson::encode_string(&mut out, &etype);
        out.push_str(",\"key\":");
        ndjson::encode_string(&mut out, &key);
        out.push_str("}\n");
    }
    let mut e_type = store.edge_type_constraints();
    e_type.sort();
    for (etype, key, ty, not_null) in e_type {
        push_type_op(&mut out, "edge", "etype", &etype, &key, &ty, not_null);
    }

    // Cardinality constraints.
    let mut cardinality = store.cardinality_constraints();
    cardinality.sort();
    for (label, etype, direction, min, max) in cardinality {
        out.push_str("{\"op\":\"cardinality\",\"label\":");
        ndjson::encode_string(&mut out, &label);
        out.push_str(",\"edgeType\":");
        ndjson::encode_string(&mut out, &etype);
        out.push_str(",\"direction\":");
        ndjson::encode_string(&mut out, if direction == 1 { "in" } else { "out" });
        out.push_str(&format!(",\"min\":{min},\"max\":"));
        match max {
            Some(m) => out.push_str(&m.to_string()),
            None => out.push_str("null"),
        }
        out.push_str("}\n");
    }

    // Validators and invariants (source text round-trips).
    let mut validators = store.validators();
    validators.sort();
    for (target, var, predicate) in validators {
        out.push_str("{\"op\":\"validator\",\"label\":");
        ndjson::encode_string(&mut out, &target);
        out.push_str(",\"var\":");
        ndjson::encode_string(&mut out, &var);
        out.push_str(",\"predicate\":");
        ndjson::encode_string(&mut out, &predicate);
        out.push_str("}\n");
    }
    let mut invariants = store.invariants();
    invariants.sort();
    for (name, query) in invariants {
        out.push_str("{\"op\":\"invariant\",\"name\":");
        ndjson::encode_string(&mut out, &name);
        out.push_str(",\"query\":");
        ndjson::encode_string(&mut out, &query);
        out.push_str("}\n");
    }
    out
}

/// Emit one `{op:"type", …}` line; `NOT NULL` is folded into the `type` value.
fn push_type_op(
    out: &mut String,
    on: &str,
    target_key: &str,
    target: &str,
    key: &str,
    ty: &str,
    not_null: bool,
) {
    out.push_str("{\"op\":\"type\",\"on\":\"");
    out.push_str(on);
    out.push_str("\",\"");
    out.push_str(target_key);
    out.push_str("\":");
    ndjson::encode_string(out, target);
    out.push_str(",\"key\":");
    ndjson::encode_string(out, key);
    out.push_str(",\"type\":");
    let type_str = if not_null {
        format!("{ty} NOT NULL")
    } else {
        ty.to_string()
    };
    ndjson::encode_string(out, &type_str);
    out.push_str("}\n");
}

// --------------------------------------------------------------- op handlers ---

fn create_index(store: &mut Store, fields: &[(String, Json)]) -> Result<(), SchemaError> {
    let on = str_field(fields, "on")?;
    let kind = str_field(fields, "kind")?;
    match (on.as_str(), kind.as_str()) {
        ("vertex", "hash") => {
            store.create_index(&one_key(&keys_of(fields)?)?);
            Ok(())
        }
        ("vertex", "range") => {
            store.create_range_index(&one_key(&keys_of(fields)?)?);
            Ok(())
        }
        ("edge", "interval") => {
            let keys = keys_of(fields)?;
            let [lo, hi] = keys.as_slice() else {
                return Err(SchemaError::BadRequest(
                    "an interval index needs exactly two keys [lo, hi]".into(),
                ));
            };
            store.create_interval_index(lo, hi);
            Ok(())
        }
        // Engine extension: the opt-in edge-type index (no key).
        ("edge", "type") => {
            store.create_edge_type_index();
            Ok(())
        }
        _ => Err(SchemaError::BadRequest(format!(
            "unsupported index: on={on} kind={kind}"
        ))),
    }
}

fn drop_index(store: &mut Store, fields: &[(String, Json)]) -> Result<(), SchemaError> {
    let key = str_field(fields, "key")?;
    if is_edge(fields)? {
        store.drop_edge_index(&key).map_err(store_err)
    } else {
        store.drop_vertex_index(&key).map_err(store_err)
    }
}

fn unique(store: &mut Store, fields: &[(String, Json)]) -> Result<(), SchemaError> {
    let edge = is_edge(fields)?;
    let target = target_field(fields, edge)?;
    let keys = keys_of(fields)?;
    let key_refs: Vec<&str> = keys.iter().map(String::as_str).collect();
    if edge {
        store
            .create_edge_unique_constraint(&target, &key_refs)
            .map_err(store_err)
    } else {
        store
            .create_unique_constraint(&target, &key_refs)
            .map_err(store_err)
    }
}

fn required(store: &mut Store, fields: &[(String, Json)]) -> Result<(), SchemaError> {
    let edge = is_edge(fields)?;
    let target = target_field(fields, edge)?;
    let key = str_field(fields, "key")?;
    if edge {
        store
            .create_edge_required_constraint(&target, &key)
            .map_err(store_err)
    } else {
        store
            .create_required_constraint(&target, &key)
            .map_err(store_err)
    }
}

fn type_constraint(store: &mut Store, fields: &[(String, Json)]) -> Result<(), SchemaError> {
    let edge = is_edge(fields)?;
    let target = target_field(fields, edge)?;
    let key = str_field(fields, "key")?;
    let type_name = str_field(fields, "type")?;
    store
        .create_type_constraint(&target, &key, &type_name, edge)
        .map_err(store_err)
}

fn cardinality(store: &mut Store, fields: &[(String, Json)]) -> Result<(), SchemaError> {
    let label = str_field(fields, "label")?;
    let etype = str_field(fields, "edgeType")?;
    let direction = match str_field(fields, "direction")?.as_str() {
        "out" => 0,
        "in" => 1,
        other => {
            return Err(SchemaError::BadRequest(format!(
                "cardinality `direction` must be \"out\" or \"in\", got `{other}`"
            )))
        }
    };
    let min = num_field(fields, "min")?.max(0.0) as u32;
    // `max` is a number or JSON null (unbounded).
    let max = match ndjson::field(fields, "max") {
        None | Some(Json::Null) => None,
        Some(Json::Num(n)) => Some(*n as u32),
        Some(_) => {
            return Err(SchemaError::BadRequest(
                "cardinality `max` must be a number or null".into(),
            ))
        }
    };
    store
        .create_cardinality_constraint(&label, &etype, direction, min, max)
        .map_err(store_err)
}

// --------------------------------------------------------------------- helpers ---

/// Read a required string field, wrapping any shape error as a `BadRequest`.
fn str_field(fields: &[(String, Json)], key: &str) -> Result<String, SchemaError> {
    let j = ndjson::req(fields, key).map_err(SchemaError::BadRequest)?;
    ndjson::json_string(j).map_err(|e| SchemaError::BadRequest(format!("field `{key}`: {e}")))
}

/// The index keys, from `keys: [...]` or a single `key: "..."`.
fn keys_of(fields: &[(String, Json)]) -> Result<Vec<String>, SchemaError> {
    if let Some(j) = ndjson::field(fields, "keys") {
        ndjson::json_str_array(j).map_err(|e| SchemaError::BadRequest(format!("field `keys`: {e}")))
    } else if let Some(j) = ndjson::field(fields, "key") {
        Ok(vec![ndjson::json_string(j).map_err(|e| {
            SchemaError::BadRequest(format!("field `key`: {e}"))
        })?])
    } else {
        Err(SchemaError::BadRequest(
            "index op needs `keys` (array) or `key` (string)".into(),
        ))
    }
}

fn one_key(keys: &[String]) -> Result<String, SchemaError> {
    match keys {
        [k] => Ok(k.clone()),
        _ => Err(SchemaError::BadRequest(
            "this index needs exactly one key".into(),
        )),
    }
}

/// Read a required numeric field, wrapping any shape error as a `BadRequest`.
fn num_field(fields: &[(String, Json)], key: &str) -> Result<f64, SchemaError> {
    match ndjson::req(fields, key).map_err(SchemaError::BadRequest)? {
        Json::Num(n) => Ok(*n),
        _ => Err(SchemaError::BadRequest(format!(
            "field `{key}` must be a number"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_people_same_email() -> Store {
        crate::ndjson::from_ndjson(
            "{\"id\":\"1\",\"labels\":[\"P\"],\"props\":{\"email\":\"a@x\"}}\n\
             {\"id\":\"2\",\"labels\":[\"P\"],\"props\":{\"email\":\"a@x\"}}\n",
        )
        .expect("fixture parses")
    }

    #[test]
    fn create_hash_index() {
        let mut s = Store::default();
        apply(
            &mut s,
            r#"{"op":"createIndex","on":"vertex","kind":"hash","keys":["age"]}"#,
        )
        .unwrap();
        assert!(s.has_hash_index("age"));
    }

    #[test]
    fn create_range_index() {
        let mut s = Store::default();
        apply(
            &mut s,
            r#"{"op":"createIndex","on":"vertex","kind":"range","key":"score"}"#,
        )
        .unwrap();
        assert!(s.has_range_index("score"));
    }

    #[test]
    fn create_interval_index_needs_two_keys() {
        let mut s = Store::default();
        apply(
            &mut s,
            r#"{"op":"createIndex","on":"edge","kind":"interval","keys":["lo","hi"]}"#,
        )
        .unwrap();
        assert!(s.has_interval_index("lo", "hi"));

        let mut s2 = Store::default();
        let err = apply(
            &mut s2,
            r#"{"op":"createIndex","on":"edge","kind":"interval","keys":["lo"]}"#,
        )
        .unwrap_err();
        assert!(matches!(err, SchemaError::BadRequest(_)));
    }

    #[test]
    fn create_edge_type_index() {
        let mut s = Store::default();
        apply(&mut s, r#"{"op":"createIndex","on":"edge","kind":"type"}"#).unwrap();
        assert!(s.has_edge_type_index());
    }

    #[test]
    fn full_schema_dump_roundtrips_indexes_and_constraints() {
        let mut s = Store::default();
        for op in [
            r#"{"op":"createIndex","on":"vertex","kind":"hash","keys":["age"]}"#,
            r#"{"op":"createIndex","on":"vertex","kind":"range","keys":["score"]}"#,
            r#"{"op":"createIndex","on":"edge","kind":"interval","keys":["lo","hi"]}"#,
            r#"{"op":"createIndex","on":"edge","kind":"type"}"#,
            r#"{"op":"unique","on":"vertex","label":"Person","key":"email"}"#,
            r#"{"op":"required","on":"vertex","label":"Person","key":"name"}"#,
        ] {
            apply(&mut s, op).unwrap();
        }
        // Replay the dump into a fresh store; both the dump text and the introspected
        // index/constraint state must match.
        let dumped = dump(&s);
        let mut s2 = Store::default();
        for line in dumped.lines().filter(|l| !l.is_empty()) {
            apply(&mut s2, line).unwrap();
        }
        assert_eq!(dump(&s), dump(&s2), "dump -> apply -> dump is stable");
        assert!(s2.has_hash_index("age"));
        assert!(s2.has_range_index("score"));
        assert!(s2.has_interval_index("lo", "hi"));
        assert!(s2.has_edge_type_index());
        assert_eq!(s2.unique_constraints(), s.unique_constraints());
        assert_eq!(s2.required_constraints(), s.required_constraints());
    }

    #[test]
    fn unique_on_clean_data_then_dump_roundtrips() {
        let mut s = Store::default();
        apply(
            &mut s,
            r#"{"op":"unique","on":"vertex","label":"Person","key":"email"}"#,
        )
        .unwrap();
        apply(
            &mut s,
            r#"{"op":"required","on":"vertex","label":"Person","key":"name"}"#,
        )
        .unwrap();

        let dumped = dump(&s);
        // Re-apply the dump into a fresh store; the constraints must reconstruct.
        let mut s2 = Store::default();
        for line in dumped.lines().filter(|l| !l.is_empty()) {
            apply(&mut s2, line).unwrap();
        }
        assert_eq!(dump(&s), dump(&s2), "dump -> apply -> dump is stable");
        assert_eq!(s2.unique_constraints(), s.unique_constraints());
        assert_eq!(s2.required_constraints(), s.required_constraints());
    }

    #[test]
    fn unique_rejected_when_data_violates() {
        let mut s = two_people_same_email();
        let err = apply(
            &mut s,
            r#"{"op":"unique","on":"vertex","label":"P","key":"email"}"#,
        )
        .unwrap_err();
        assert!(matches!(err, SchemaError::Rejected(_)), "got {err:?}");
    }

    #[test]
    fn composite_unique_via_keys_array() {
        let mut s = Store::default();
        apply(
            &mut s,
            r#"{"op":"unique","on":"vertex","label":"P","keys":["a","b"]}"#,
        )
        .unwrap();
        assert_eq!(s.unique_constraints().len(), 1);
    }

    #[test]
    fn edge_unique_and_required_are_declared() {
        let mut s = Store::default();
        apply(
            &mut s,
            r#"{"op":"unique","on":"edge","etype":"K","key":"id"}"#,
        )
        .unwrap();
        apply(
            &mut s,
            r#"{"op":"required","on":"edge","etype":"K","key":"id"}"#,
        )
        .unwrap();
        assert_eq!(s.edge_unique_constraints().len(), 1);
        assert_eq!(s.edge_required_constraints().len(), 1);
    }

    #[test]
    fn type_cardinality_and_drop_index_apply() {
        let mut s = Store::default();
        apply(
            &mut s,
            r#"{"op":"type","on":"vertex","label":"P","key":"age","type":"number"}"#,
        )
        .unwrap();
        assert_eq!(s.type_constraints().len(), 1);
        apply(
            &mut s,
            r#"{"op":"cardinality","label":"P","edgeType":"KNOWS","direction":"out","min":0,"max":null}"#,
        )
        .unwrap();
        assert_eq!(s.cardinality_constraints().len(), 1);
        // dropIndex is a no-op (Ok) when the index is absent.
        apply(&mut s, r#"{"op":"dropIndex","on":"vertex","key":"age"}"#).unwrap();
    }

    #[test]
    fn unknown_type_name_is_invalid() {
        let mut s = Store::default();
        let err = apply(
            &mut s,
            r#"{"op":"type","on":"vertex","label":"P","key":"age","type":"bogus"}"#,
        )
        .unwrap_err();
        assert!(matches!(err, SchemaError::Invalid(_)), "{err:?}");
    }

    #[test]
    fn validator_and_invariant_route_through_exec() {
        // These need the query evaluator, so schema_op rejects them (BadRequest);
        // exec::apply_schema_op declares them instead.
        let mut s = Store::default();
        for op in ["validator", "invariant"] {
            let json = format!(
                r#"{{"op":"{op}","label":"P","name":"x","var":"p","predicate":"true","query":"MATCH (n) RETURN true"}}"#
            );
            assert!(matches!(
                apply(&mut s, &json).unwrap_err(),
                SchemaError::BadRequest(_)
            ));
        }
    }

    #[test]
    fn malformed_input_is_bad_request() {
        let mut s = Store::default();
        assert!(matches!(
            apply(&mut s, "not json").unwrap_err(),
            SchemaError::BadRequest(_)
        ));
        assert!(matches!(
            apply(&mut s, r#"["array","not","object"]"#).unwrap_err(),
            SchemaError::BadRequest(_)
        ));
        assert!(matches!(
            apply(&mut s, r#"{"on":"vertex"}"#).unwrap_err(), // missing `op`
            SchemaError::BadRequest(_)
        ));
        assert!(matches!(
            apply(&mut s, r#"{"op":"nope"}"#).unwrap_err(),
            SchemaError::BadRequest(_)
        ));
    }
}
