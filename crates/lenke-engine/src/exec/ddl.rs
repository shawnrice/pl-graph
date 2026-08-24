use super::*;
use crate::store::Store;

// -------------------------------------------------- validators / invariants ---
//
// These two constraint kinds need the query evaluator, so — unlike the pure-store
// constraints in `run_deferred_checks` — they are declared and enforced here.
// A validator is checked by the composed query `MATCH (var:target) WHERE NOT
// (pred) …`: three-valued `WHERE` keeps only rows where `pred` is definitely
// FALSE, so a non-empty result is exactly an SQL-`CHECK` violation (null/true
// pass). An invariant is its own whole-graph query; a boolean-`false` cell fails.

/// Apply one schema-DDL op. Validator/invariant ops are declared here (parse,
/// run the declaration-time check over existing data, then store); every other op
/// delegates to the pure-store [`crate::schema_op::apply`]. This is the single
/// schema entry point the C ABI calls.
pub fn apply_schema_op(store: &mut Store, json: &str) -> Result<(), crate::schema_op::SchemaError> {
    use crate::schema_op::SchemaError;
    let parsed = crate::ndjson::parse_json(json).map_err(SchemaError::BadRequest)?;
    let crate::ndjson::Json::Obj(fields) = &parsed else {
        return Err(SchemaError::BadRequest(
            "schema op must be a JSON object".into(),
        ));
    };
    let op = crate::ndjson::field(fields, "op")
        .and_then(|j| crate::ndjson::json_string(j).ok())
        .ok_or_else(|| SchemaError::BadRequest("schema op needs a string `op`".into()))?;
    match op.as_str() {
        "validator" => declare_validator_op(store, fields),
        "invariant" => declare_invariant_op(store, fields),
        _ => crate::schema_op::apply(store, json),
    }
}

/// Read a required string field, as a `BadRequest` on any shape error.
fn schema_str(
    fields: &[(String, crate::ndjson::Json)],
    key: &str,
) -> Result<String, crate::schema_op::SchemaError> {
    use crate::schema_op::SchemaError;
    let j = crate::ndjson::req(fields, key).map_err(SchemaError::BadRequest)?;
    crate::ndjson::json_string(j)
        .map_err(|e| SchemaError::BadRequest(format!("field `{key}`: {e}")))
}

fn declare_validator_op(
    store: &mut Store,
    fields: &[(String, crate::ndjson::Json)],
) -> Result<(), crate::schema_op::SchemaError> {
    let target = schema_str(fields, "label")?;
    let var = schema_str(fields, "var")?;
    let pred = schema_str(fields, "predicate")?;
    declare_validator(store, &target, &var, &pred)
}

/// Declare a validator: compose + parse the vertex/edge check queries, verify the
/// current data conforms, then store the rule. Shared by the schema-op path and
/// the binary-snapshot reload.
pub(crate) fn declare_validator(
    store: &mut Store,
    target: &str,
    var: &str,
    pred: &str,
) -> Result<(), crate::schema_op::SchemaError> {
    use crate::schema_op::SchemaError;
    let vq = format!("MATCH ({var}:{target}) WHERE NOT ({pred}) RETURN {var} LIMIT 1");
    let eq = format!("MATCH ()-[{var}:{target}]->() WHERE NOT ({pred}) RETURN {var} LIMIT 1");
    let vplan = crate::gql::parse(&vq).map_err(SchemaError::Syntax)?;
    let eplan = crate::gql::parse(&eq).map_err(SchemaError::Syntax)?;
    for plan in [&vplan, &eplan] {
        if !try_run(plan, store)
            .map_err(SchemaError::Rejected)?
            .rows
            .is_empty()
        {
            return Err(SchemaError::Rejected(
                "existing data already violates the validator being declared".into(),
            ));
        }
    }
    store.declare_validator(target, var, pred, vec![vplan, eplan]);
    Ok(())
}

fn declare_invariant_op(
    store: &mut Store,
    fields: &[(String, crate::ndjson::Json)],
) -> Result<(), crate::schema_op::SchemaError> {
    let name = schema_str(fields, "name")?;
    let query = schema_str(fields, "query")?;
    declare_invariant(store, &name, &query)
}

/// Declare an invariant: parse the query, verify the current data holds, then
/// store it. Shared by the schema-op path and the binary-snapshot reload.
pub(crate) fn declare_invariant(
    store: &mut Store,
    name: &str,
    query: &str,
) -> Result<(), crate::schema_op::SchemaError> {
    use crate::schema_op::SchemaError;
    let plan = crate::gql::parse(query).map_err(SchemaError::Syntax)?;
    if rows_have_false(&try_run(&plan, store).map_err(SchemaError::Rejected)?) {
        return Err(SchemaError::Rejected(format!(
            "existing data already violates the invariant '{name}'"
        )));
    }
    store.declare_invariant(name, query, plan);
    Ok(())
}

/// Run every validator + invariant after a write statement; the caller rolls the
/// statement back on `Err`. A no-op when none are declared.
pub(crate) fn enforce_expr_constraints(store: &Store) -> Result<(), String> {
    for plan in store.validator_check_plans() {
        if !try_run(plan, store)?.rows.is_empty() {
            return Err("E_VALIDATOR: a validator predicate was violated".to_string());
        }
    }
    for (name, plan) in store.invariant_plans() {
        if rows_have_false(&try_run(plan, store)?) {
            return Err(format!("E_INVARIANT: invariant '{name}' violated"));
        }
    }
    Ok(())
}
