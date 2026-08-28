//! Shared graph serialization codecs (pg-json, pg-text, graphson, csv) over a
//! NEUTRAL graph model ([`GraphData`] / [`Value`]). `lenke-engine` converts its
//! graph to/from the neutral model and calls
//! [`serialize`] / [`deserialize`] here, so the byte-for-byte format logic lives
//! in exactly one place.
//!
//! Error codes are the shared `@lenke/errors` wire strings, so each engine's FFI
//! layer maps a codec failure to the same `LenkeError` (`E_INVALID_JSON`,
//! `E_INVALID_SHAPE`, `E_INVALID_VALUE`, `E_UNKNOWN_FORMAT`, `E_UNSUPPORTED`).

mod csv;
mod decstream;
mod graphson;
mod json;
mod jsonfmt;
mod model;
mod pg_json;
mod pg_text;
mod stream;

pub use decstream::{deserialize_into, DecVal, GraphSink};
pub use jsonfmt::{js_number, push_json_str, push_num, push_value};
pub use model::{Edge, GraphData, Node, Value};
pub use graphson::{EProps, GraphsonSink, LabelJoin, VProps};
pub use pg_text::PgTextSink;
pub use stream::{push_value_ref, Labels, PgJsonSink, Props, ValueRef};

use json::Json;

/// A codec failure carrying a stable wire code (an `@lenke/errors` `E_*` string).
#[derive(Debug, Clone)]
pub struct CodecError {
    pub code: &'static str,
    pub message: String,
}

impl CodecError {
    /// A codec failure with a stable `E_*` wire code. `pub` so a host crate's
    /// GraphData bridge can raise a codec-coded error (e.g. a temporal that failed
    /// to validate) alongside the ones this crate raises.
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

/// The `E_*` wire codes this crate raises, exposed so a host can construct a
/// [`CodecError`] with the matching code from its own bridge logic.
pub mod codes {
    pub const INVALID_JSON: &str = super::E_INVALID_JSON;
    pub const INVALID_SHAPE: &str = super::E_INVALID_SHAPE;
    pub const INVALID_VALUE: &str = super::E_INVALID_VALUE;
    pub const UNKNOWN_FORMAT: &str = super::E_UNKNOWN_FORMAT;
    pub const UNSUPPORTED: &str = super::E_UNSUPPORTED;
}

pub type CodeResult<T> = Result<T, CodecError>;

// Wire codes (mirror @lenke/errors ErrorCode).
pub(crate) const E_INVALID_JSON: &str = "E_INVALID_JSON";
pub(crate) const E_INVALID_SHAPE: &str = "E_INVALID_SHAPE";
pub(crate) const E_INVALID_VALUE: &str = "E_INVALID_VALUE";
pub(crate) const E_UNKNOWN_FORMAT: &str = "E_UNKNOWN_FORMAT";
pub(crate) const E_UNSUPPORTED: &str = "E_UNSUPPORTED";

// --------------------------------------------------------------- decode helpers ---

/// A finite float that is an exact integer — GraphSON `g:Int64` vs `g:Double`,
/// CSV `integer` vs `float` (JS `Number.isInteger`).
pub(crate) fn is_intish(x: f64) -> bool {
    x.is_finite() && x.fract() == 0.0
}

/// A parsed JSON value as a neutral [`Value`]. A non-finite number → `Null`
/// (matching the LPG numeric model); a single-key `{"@date":…}` object → a tagged
/// temporal; any other object → a map.
pub(crate) fn json_to_value(j: &Json) -> CodeResult<Value> {
    Ok(match j {
        Json::Null => Value::Null,
        Json::Bool(b) => Value::Bool(*b),
        Json::Num(n) => {
            if n.is_finite() {
                Value::Num(*n)
            } else {
                Value::Null
            }
        }
        Json::Str(s) => Value::Str(s.to_string()),
        Json::Arr(a) => Value::List(a.iter().map(json_to_value).collect::<CodeResult<_>>()?),
        Json::Obj(pairs) => match json::temporal_from_pairs(pairs) {
            Some((tag, iso)) => Value::Temporal { tag, iso },
            None => Value::Map(
                pairs
                    .iter()
                    .map(|(k, v)| Ok((k.to_string(), json_to_value(v)?)))
                    .collect::<CodeResult<_>>()?,
            ),
        },
    })
}

/// A JSON id field as a string (a string verbatim; a number via its JS text; else
/// `"null"`).
pub(crate) fn json_id(j: &Json) -> String {
    match j {
        Json::Str(s) => s.to_string(),
        Json::Num(n) => js_number(*n),
        Json::Bool(b) => b.to_string(),
        _ => "null".to_string(),
    }
}

/// A JSON array field as `Vec<String>` (non-string elements dropped).
pub(crate) fn json_str_array(field: Option<&Json>) -> Vec<String> {
    field
        .and_then(Json::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|x| match x {
                    Json::Str(s) => Some(s.to_string()),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default()
}

/// A JSON object field as neutral property pairs.
pub(crate) fn json_props(field: Option<&Json>) -> CodeResult<Vec<(String, Value)>> {
    match field.and_then(Json::as_object) {
        Some(m) => m
            .iter()
            .map(|(k, v)| Ok((k.to_string(), json_to_value(v)?)))
            .collect(),
        None => Ok(Vec::new()),
    }
}

/// Parse a JSON document, mapping a syntax failure to `E_INVALID_JSON`.
pub(crate) fn parse_json<'a>(input: &'a str, ctx: &str) -> CodeResult<Json<'a>> {
    json::parse(input).map_err(|()| CodecError::new(E_INVALID_JSON, format!("{ctx}: invalid JSON")))
}

/// Whether any node or edge carries a map/record property (a nested object) —
/// used to reject flat formats that can't represent one.
fn has_map_property(g: &GraphData) -> bool {
    let is_map = |props: &[(String, Value)]| props.iter().any(|(_, v)| matches!(v, Value::Map(_)));
    g.nodes.iter().any(|n| is_map(&n.props)) || g.edges.iter().any(|e| is_map(&e.props))
}

// ------------------------------------------------------------------- dispatch ---

fn unknown_format(format: &str) -> CodecError {
    CodecError::new(
        E_UNKNOWN_FORMAT,
        format!("unknown serialization format '{format}'"),
    )
}

/// Serialize neutral graph data in the named format (`pg-json`; more to come).
pub fn serialize(g: &GraphData, format: &str) -> CodeResult<String> {
    if matches!(format, "pg-text" | "csv") && has_map_property(g) {
        return Err(CodecError::new(
            E_UNSUPPORTED,
            "a map/record property can't be serialized to a flat format (pg-text/csv); \
             use a structured format: ndjson, graphson, or pg-json",
        ));
    }
    match format {
        "pg-json" => Ok(pg_json::encode(g)),
        "graphson" => Ok(graphson::encode(g)),
        "pg-text" => Ok(pg_text::encode(g)),
        "csv" => Ok(csv::encode(g)),
        other => Err(unknown_format(other)),
    }
}

/// Deserialize `input` in the named format into neutral graph data. Endpoint
/// well-formedness (an edge referencing an undeclared node) is the host's concern
/// when it builds its graph from the returned [`GraphData`].
pub fn deserialize(input: &str, format: &str) -> CodeResult<GraphData> {
    match format {
        "pg-json" => pg_json::decode(input),
        "graphson" => graphson::decode(input),
        "pg-text" => Ok(pg_text::decode(input)),
        "csv" => Ok(csv::decode(input)),
        other => Err(unknown_format(other)),
    }
}
