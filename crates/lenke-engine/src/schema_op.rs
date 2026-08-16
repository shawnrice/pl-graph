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

/// Why a schema op could not be applied — split so the C ABI returns the right
/// status: [`BadRequest`](SchemaError::BadRequest) → `-1`,
/// [`Rejected`](SchemaError::Rejected) → `-2`.
#[derive(Debug)]
pub enum SchemaError {
    /// Malformed JSON, an unknown/missing field, or an op the engine does not
    /// support yet. The request itself is wrong or unsupported.
    BadRequest(String),
    /// A well-formed op that the current data violates (e.g. a UNIQUE constraint
    /// with existing duplicates).
    Rejected(String),
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
        "unique" => unique(store, fields),
        "required" => required(store, fields),
        // Well-formed vocabulary, but no backing Store method yet — the work-list.
        "dropIndex" | "type" | "cardinality" | "validator" | "invariant" => {
            Err(SchemaError::BadRequest(format!(
                "schema op `{op}` is not supported by the engine yet"
            )))
        }
        other => Err(SchemaError::BadRequest(format!(
            "unknown schema op `{other}`"
        ))),
    }
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
    out
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

fn unique(store: &mut Store, fields: &[(String, Json)]) -> Result<(), SchemaError> {
    require_vertex(fields, "UNIQUE")?;
    let label = str_field(fields, "label")?;
    let keys = keys_of(fields)?;
    let key_refs: Vec<&str> = keys.iter().map(String::as_str).collect();
    store
        .create_unique_constraint(&label, &key_refs)
        .map_err(SchemaError::Rejected)
}

fn required(store: &mut Store, fields: &[(String, Json)]) -> Result<(), SchemaError> {
    require_vertex(fields, "REQUIRED")?;
    let label = str_field(fields, "label")?;
    let key = str_field(fields, "key")?;
    store
        .create_required_constraint(&label, &key)
        .map_err(SchemaError::Rejected)
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

/// The engine currently backs UNIQUE / REQUIRED on vertices only.
fn require_vertex(fields: &[(String, Json)], what: &str) -> Result<(), SchemaError> {
    match str_field(fields, "on")?.as_str() {
        "vertex" => Ok(()),
        other => Err(SchemaError::BadRequest(format!(
            "engine supports {what} on `vertex` only for now, not `{other}`"
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
    fn edge_unique_is_unsupported_bad_request() {
        let mut s = Store::default();
        let err = apply(
            &mut s,
            r#"{"op":"unique","on":"edge","etype":"K","key":"id"}"#,
        )
        .unwrap_err();
        assert!(matches!(err, SchemaError::BadRequest(_)));
    }

    #[test]
    fn deferred_ops_are_bad_request_not_silent() {
        let mut s = Store::default();
        for op in ["dropIndex", "type", "cardinality", "validator", "invariant"] {
            let json = format!(r#"{{"op":"{op}","on":"vertex","label":"P","key":"k"}}"#);
            let err = apply(&mut s, &json).unwrap_err();
            assert!(
                matches!(err, SchemaError::BadRequest(_)),
                "{op} should be BadRequest"
            );
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
